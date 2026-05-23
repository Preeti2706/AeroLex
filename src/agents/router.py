"""
router.py — Retrieval Router for AeroLex Agents

WHAT:
    Executes RetrievalPlan from planner.py — routes queries to the
    correct retrieval pipeline and returns structured results.

WHY:
    Planner decides WHAT to retrieve. Router decides HOW to execute.
    This separation keeps each component single-responsibility:
    - planner.py = strategy (which sources, how many chunks)
    - router.py  = execution (actually run the retrieval)

    Router handles 3 execution modes:
    STANDARD:     Single pipeline call — fast
    MULTI_SOURCE: Parallel calls per source — merged + reranked
    MULTI_HOP:    Sequential calls with hop queries — reasoning chain

HOW:
    1. Receive RetrievalPlan
    2. Execute source configs via RetrievalPipeline
    3. Merge results (MULTI_SOURCE) or chain results (MULTI_HOP)
    4. Return RouterResult with chunks + metadata

MERGING STRATEGY (MULTI_SOURCE):
    Each source returns top_k chunks with rerank scores.
    Merge = weighted score combination:
        final_score = rerank_score × source_weight
    Then global sort + take rerank_top_k.

MULTI_HOP EXECUTION:
    Hop 1: Retrieve on original query
    Hop 2: Retrieve on hop_query_1
    Hop 3: Retrieve on hop_query_2
    Deduplicate by chunk_id → rerank combined pool → top_k

Official Docs:
    LangGraph routing: https://langchain-ai.github.io/langgraph/
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, AgentError
from src.agents.planner import RetrievalPlan, RetrievalStrategy, SourceConfig
from src.retrieval.reranker import RetrievalPipeline

logger = get_logger(__name__)


# ── Router Result ────────────────────────────────────────────────────────────

@dataclass
class RouterResult:
    """
    Complete retrieval result from router.

    Fields:
        query:          Original user query
        strategy:       Which strategy was executed
        chunks:         Final retrieved chunks (dicts)
        total_retrieved:Total chunks before reranking
        final_count:    Chunks after reranking
        sources_used:   Which sources were actually searched
        hop_results:    Per-hop results (MULTI_HOP only)
        latency_ms:     Total retrieval latency
        retrieval_stats:Detailed stats per source
    """
    query:           str
    strategy:        str
    chunks:          list[dict]
    total_retrieved: int
    final_count:     int
    sources_used:    list[str]
    hop_results:     list[dict] = field(default_factory=list)
    latency_ms:      float = 0.0
    retrieval_stats: dict = field(default_factory=dict)


# ── Chunk Deduplication ──────────────────────────────────────────────────────

def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """
    Deduplicate chunks by chunk_id — keeps highest scoring version.

    Called after merging multi-source or multi-hop results.
    Same chunk may appear from different sources or hops.

    Args:
        chunks: List of chunk dicts with chunk_id + rerank_score

    Returns:
        Deduplicated list — one entry per chunk_id
    """
    seen = {}
    for chunk in chunks:
        cid = chunk.get("chunk_id", "")
        if cid not in seen:
            seen[cid] = chunk
        else:
            # Keep higher scoring version
            existing_score = seen[cid].get("rerank_score", 0.0)
            new_score      = chunk.get("rerank_score", 0.0)
            if new_score > existing_score:
                seen[cid] = chunk

    deduped = list(seen.values())
    logger.debug(f"Deduplication: {len(chunks)} → {len(deduped)} chunks")
    return deduped


def _apply_source_weight(chunks: list[dict], weight: float) -> list[dict]:
    """
    Apply source authority weight to rerank scores.

    final_score = rerank_score × source_weight

    This ensures eCFR (weight=1.0) ranks above SKYbrary (weight=0.80)
    when scores are otherwise equal.

    Args:
        chunks: List of chunk dicts
        weight: Source authority weight (0.0 - 1.0)

    Returns:
        Chunks with weighted_score field added
    """
    weighted = []
    for chunk in chunks:
        c = dict(chunk)
        raw_score    = c.get("rerank_score", c.get("rrf_score", 0.0))
        c["weighted_score"] = round(raw_score * weight, 6)
        c["source_weight"]  = weight
        weighted.append(c)
    return weighted


# ── Execution Functions ──────────────────────────────────────────────────────

def _execute_standard(
    plan: RetrievalPlan,
    pipeline: RetrievalPipeline,
) -> RouterResult:
    """
    Execute STANDARD single-source retrieval.

    Args:
        plan:     RetrievalPlan with STANDARD strategy
        pipeline: RetrievalPipeline instance

    Returns:
        RouterResult with retrieved chunks
    """
    start = time.time()
    sc    = plan.source_configs[0]

    logger.info(
        f"STANDARD execution | "
        f"Source: {sc.source} | "
        f"top_k: {sc.top_k}"
    )

    result = pipeline.retrieve(
        query=plan.query,
        top_k=sc.top_k,
        filters=sc.filter,
    )

    chunks   = result.get("chunks", [])
    stats    = result.get("retrieval_stats", {})
    latency  = (time.time() - start) * 1000

    logger.info(
        f"STANDARD complete | "
        f"Chunks: {len(chunks)} | "
        f"Latency: {latency:.0f}ms"
    )

    return RouterResult(
        query=plan.query,
        strategy=RetrievalStrategy.STANDARD.value,
        chunks=chunks[:plan.rerank_top_k],
        total_retrieved=len(chunks),
        final_count=min(len(chunks), plan.rerank_top_k),
        sources_used=[sc.source],
        latency_ms=latency,
        retrieval_stats={"standard": stats},
    )


def _execute_multi_source(
    plan: RetrievalPlan,
    pipeline: RetrievalPipeline,
) -> RouterResult:
    """
    Execute MULTI_SOURCE parallel retrieval.

    Retrieves from each source independently, applies
    source weights, merges and re-sorts globally.

    Args:
        plan:     RetrievalPlan with MULTI_SOURCE strategy
        pipeline: RetrievalPipeline instance

    Returns:
        RouterResult with merged chunks from all sources
    """
    start = time.time()
    all_chunks   = []
    sources_used = []
    all_stats    = {}

    for sc in plan.source_configs:
        logger.info(
            f"MULTI_SOURCE — retrieving from {sc.source} | "
            f"top_k: {sc.top_k}"
        )

        try:
            result = pipeline.retrieve(
                query=plan.query,
                top_k=sc.top_k,
                filters=sc.filter,
            )
            chunks = result.get("chunks", [])
            stats  = result.get("retrieval_stats", {})

            # Apply source authority weight
            weighted_chunks = _apply_source_weight(chunks, sc.weight)

            # Tag source for transparency
            for c in weighted_chunks:
                c["retrieved_from"] = sc.source

            all_chunks.extend(weighted_chunks)
            sources_used.append(sc.source)
            all_stats[sc.source] = stats

            logger.info(
                f"Source {sc.source}: {len(chunks)} chunks retrieved"
            )

        except Exception as e:
            logger.warning(
                f"Source {sc.source} retrieval failed: {e} — skipping"
            )

    # Deduplicate
    all_chunks = _deduplicate_chunks(all_chunks)

    # Global sort by weighted_score
    all_chunks.sort(
        key=lambda x: x.get("weighted_score", 0.0),
        reverse=True
    )

    # Take rerank_top_k
    final_chunks = all_chunks[:plan.rerank_top_k]
    latency      = (time.time() - start) * 1000

    logger.info(
        f"MULTI_SOURCE complete | "
        f"Total: {len(all_chunks)} | "
        f"Final: {len(final_chunks)} | "
        f"Sources: {sources_used} | "
        f"Latency: {latency:.0f}ms"
    )

    return RouterResult(
        query=plan.query,
        strategy=RetrievalStrategy.MULTI_SOURCE.value,
        chunks=final_chunks,
        total_retrieved=len(all_chunks),
        final_count=len(final_chunks),
        sources_used=sources_used,
        latency_ms=latency,
        retrieval_stats=all_stats,
    )


def _execute_multi_hop(
    plan: RetrievalPlan,
    pipeline: RetrievalPipeline,
) -> RouterResult:
    """
    Execute MULTI_HOP sequential retrieval.

    Retrieves on original query + each hop query.
    Deduplicates across all hops, sorts by score.

    This simulates a reasoning chain:
    "What are MEL requirements?" → "What are inoperative equipment rules?"
    → combine → rerank → answer

    Args:
        plan:     RetrievalPlan with MULTI_HOP strategy
        pipeline: RetrievalPipeline instance

    Returns:
        RouterResult with combined hop chunks
    """
    start      = time.time()
    all_chunks = []
    hop_results = []
    sc          = plan.source_configs[0]

    # All queries: original + hop queries
    all_queries = [plan.query] + plan.hop_queries

    for i, query in enumerate(all_queries):
        hop_label = "original" if i == 0 else f"hop_{i}"
        if i > 0:
            time.sleep(20)
        hop_label = "original" if i == 0 else f"hop_{i}"
        logger.info(
            f"MULTI_HOP — {hop_label} | "
            f"Query: '{query[:60]}'"
        )

        try:
            result = pipeline.retrieve(
                query=query,
                top_k=sc.top_k,
                filters=sc.filter,
            )
            chunks = result.get("chunks", [])
            stats  = result.get("retrieval_stats", {})

            # Tag which hop retrieved each chunk
            for c in chunks:
                c["hop_source"] = hop_label
                c["hop_query"]  = query

            all_chunks.extend(chunks)
            hop_results.append({
                "hop":    hop_label,
                "query":  query,
                "chunks": len(chunks),
                "stats":  stats,
            })

            logger.info(
                f"Hop {hop_label}: {len(chunks)} chunks retrieved"
            )

        except Exception as e:
            logger.warning(
                f"Hop {hop_label} retrieval failed: {e} — skipping"
            )

    # Deduplicate across hops
    all_chunks = _deduplicate_chunks(all_chunks)

    # Sort by rerank_score
    all_chunks.sort(
        key=lambda x: x.get("rerank_score", x.get("rrf_score", 0.0)),
        reverse=True
    )

    # Take rerank_top_k
    final_chunks = all_chunks[:plan.rerank_top_k]
    latency      = (time.time() - start) * 1000

    logger.info(
        f"MULTI_HOP complete | "
        f"Total (pre-dedup): {len(all_chunks)} | "
        f"Final: {len(final_chunks)} | "
        f"Hops executed: {len(all_queries)} | "
        f"Latency: {latency:.0f}ms"
    )

    return RouterResult(
        query=plan.query,
        strategy=RetrievalStrategy.MULTI_HOP.value,
        chunks=final_chunks,
        total_retrieved=len(all_chunks),
        final_count=len(final_chunks),
        sources_used=[sc.source for sc in plan.source_configs],
        hop_results=hop_results,
        latency_ms=latency,
        retrieval_stats={"hops": hop_results},
    )


# ── Main Router Class ────────────────────────────────────────────────────────

class RetrievalRouter:
    """
    Executes retrieval plans from planner.py.

    Usage:
        router = RetrievalRouter()
        result = router.route(plan)
        print(result.chunks)        # Final retrieved chunks
        print(result.sources_used)  # Which sources were searched
        print(result.latency_ms)    # Total retrieval time
    """

    def __init__(
        self,
        collection_name: str = "aerolex_voyage",
    ):
        self.collection_name = collection_name
        self.pipeline = RetrievalPipeline(
            collection=collection_name,
            embedding_model="voyage",
            use_reranker=True,
            reranker_type="voyage",
        )
        logger.info(
            f"RetrievalRouter initialized | "
            f"Collection: {collection_name}"
        )

    def route(self, plan: RetrievalPlan) -> RouterResult:
        """
        Execute a RetrievalPlan and return results.

        Routes to STANDARD, MULTI_SOURCE, or MULTI_HOP
        execution based on plan.strategy.

        Args:
            plan: RetrievalPlan from RetrievalPlanner.plan()

        Returns:
            RouterResult with chunks + metadata

        Raises:
            AgentError: If routing fails
        """
        try:
            logger.info(
                f"RetrievalRouter.route() | "
                f"Strategy: {plan.strategy.value} | "
                f"Query: '{plan.query[:60]}'"
            )

            if plan.strategy == RetrievalStrategy.STANDARD:
                result = _execute_standard(plan, self.pipeline)

            elif plan.strategy == RetrievalStrategy.MULTI_SOURCE:
                result = _execute_multi_source(plan, self.pipeline)

            else:  # MULTI_HOP
                result = _execute_multi_hop(plan, self.pipeline)

            logger.info(
                f"Routing complete | "
                f"Final chunks: {result.final_count} | "
                f"Latency: {result.latency_ms:.0f}ms"
            )

            return result

        except Exception as e:
            handle_exception(
                e,
                context="RetrievalRouter.route",
                raise_as=AgentError
            )


def format_router_result(result: RouterResult) -> str:
    """Format RouterResult for CLI output."""

    lines = [
        f"\n{'═'*65}",
        f"ROUTER RESULT — {result.strategy}",
        f"{'═'*65}",
        f"Query          : {result.query}",
        f"Sources Used   : {result.sources_used}",
        f"Total Retrieved: {result.total_retrieved}",
        f"Final Count    : {result.final_count}",
        f"Latency        : {result.latency_ms:.0f}ms",
        f"\nTop Chunks:",
    ]

    for i, chunk in enumerate(result.chunks[:3], 1):
        score = chunk.get("rerank_score", chunk.get("weighted_score", 0.0))
        lines.append(
            f"  [{i}] score={score:.4f} | "
            f"source={chunk.get('source', 'unknown')} | "
            f"section={chunk.get('chunk_id', 'unknown')[:40]}"
        )
        lines.append(
            f"      text={chunk.get('text', '')[:100]}..."
        )

    if result.hop_results:
        lines.append(f"\nHop Summary:")
        for hr in result.hop_results:
            lines.append(
                f"  {hr['hop']}: {hr['chunks']} chunks | "
                f"query='{hr['query'][:50]}'"
            )

    lines.append(f"{'═'*65}")
    return "\n".join(lines)


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== AeroLex Retrieval Router — Test ===\n")

    from src.agents.query_classifier import QueryClassifier
    from src.agents.planner import RetrievalPlanner

    classifier = QueryClassifier()
    planner    = RetrievalPlanner()
    router     = RetrievalRouter()

    test_queries = [
        "What does 14 CFR 91.103 say about preflight requirements?",
        "How do FAA and DGCA preflight rules differ?",
        "Can I fly without an MEL if the altimeter is broken?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        classification = classifier.classify(query)
        plan           = planner.plan(classification)
        result         = router.route(plan)
        print(format_router_result(result))