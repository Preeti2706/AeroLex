"""
planner.py — Retrieval Strategy Planner for AeroLex Agents

WHAT:
    Takes ClassificationResult from query_classifier.py and builds
    a concrete RetrievalPlan — how many chunks, which sources,
    which retrieval strategy, what filters to apply.

WHY:
    Different query types need different retrieval strategies:
    - LOOKUP:     1 source, top_k=5, standard hybrid search
    - COMPARISON: 2+ sources, top_k=5 per source, parallel retrieval
    - ADVISORY:   multi-hop, top_k=8, broader search + reranking

    Without a planner, every query gets the same retrieval —
    wasteful for simple queries, insufficient for complex ones.
    Planner = intelligent resource allocation per query type.

HOW:
    1. Receive ClassificationResult
    2. Select retrieval strategy based on query_type + complexity
    3. Build source-specific filter configs
    4. Return RetrievalPlan consumed by router.py

STRATEGIES:
    STANDARD:    Single source, hybrid BM25+Dense+RRF, top_k=5
    MULTI_SOURCE: Parallel retrieval from 2+ sources, merge + rerank
    MULTI_HOP:   Sequential retrieval — first fetch, then follow-up
                 queries based on initial results

MATH:
    Token budget per query type:
    LOOKUP:     ~500 context tokens  (5 chunks × 100 avg)
    COMPARISON: ~1000 context tokens (10 chunks × 100 avg)
    ADVISORY:   ~1500 context tokens (15 chunks × 100 avg)

Official Docs:
    LangGraph planning: https://langchain-ai.github.io/langgraph/
    RAG strategies: https://arxiv.org/abs/2312.10997
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from qdrant_client.models import Filter

from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, AgentError
from src.agents.query_classifier import ClassificationResult, QueryType
from src.retrieval.metadata_filter import build_metadata_filter

logger = get_logger(__name__)


# ── Retrieval Strategy Enum ──────────────────────────────────────────────────

class RetrievalStrategy(str, Enum):
    """
    Three retrieval strategies mapped to query types.

    STANDARD:     Single source, single pass — LOOKUP queries
    MULTI_SOURCE: Parallel multi-source — COMPARISON queries
    MULTI_HOP:    Sequential reasoning hops — ADVISORY queries
    """
    STANDARD     = "STANDARD"
    MULTI_SOURCE = "MULTI_SOURCE"
    MULTI_HOP    = "MULTI_HOP"


# ── Source Config ────────────────────────────────────────────────────────────

@dataclass
class SourceConfig:
    """
    Configuration for retrieving from one regulatory source.

    Fields:
        source:      Source name — "ecfr", "dgca", "faa_ad", etc.
        top_k:       Number of chunks to retrieve from this source
        filter:      Qdrant Filter object for this source
        weight:      Importance weight for result merging (COMPARISON)
        description: Human-readable description for LLM context
    """
    source:      str
    top_k:       int
    filter:      Optional[Filter]
    weight:      float
    description: str


# ── Retrieval Plan ───────────────────────────────────────────────────────────

@dataclass
class RetrievalPlan:
    """
    Complete retrieval plan for one query.

    This is the output of planner.py — consumed by router.py.

    Fields:
        query:           Original user query
        query_type:      LOOKUP / COMPARISON / ADVISORY
        strategy:        STANDARD / MULTI_SOURCE / MULTI_HOP
        source_configs:  Per-source retrieval configs
        total_top_k:     Total chunks to retrieve across all sources
        token_budget:    Estimated context token budget
        use_reranker:    Whether to apply Voyage reranker
        rerank_top_k:    Final chunks after reranking
        hop_queries:     Follow-up queries for MULTI_HOP strategy
        complexity:      simple / medium / complex
        plan_reasoning:  Why this plan was chosen
    """
    query:          str
    query_type:     QueryType
    strategy:       RetrievalStrategy
    source_configs: list[SourceConfig]
    total_top_k:    int
    token_budget:   int
    use_reranker:   bool
    rerank_top_k:   int
    hop_queries:    list[str]
    complexity:     str
    plan_reasoning: str


# ── Source Descriptions ──────────────────────────────────────────────────────

SOURCE_DESCRIPTIONS = {
    "ecfr":     "US Federal Aviation Regulations (14 CFR) — primary FAA regulatory source",
    "dgca":     "Indian DGCA Civil Aviation Requirements — primary Indian regulatory source",
    "faa_ad":   "FAA Airworthiness Directives — mandatory safety compliance actions",
    "faa_ac":   "FAA Advisory Circulars — guidance and recommended practices",
    "skybrary": "SKYbrary Aviation Safety articles — incident and safety knowledge base",
}

# ── Source Weights for COMPARISON ───────────────────────────────────────────
# Higher weight = results from this source ranked higher in merged output
SOURCE_WEIGHTS = {
    "ecfr":     1.00,
    "faa_ad":   0.95,
    "dgca":     0.90,
    "faa_ac":   0.85,
    "skybrary": 0.80,
}


# ── Plan Builders ────────────────────────────────────────────────────────────

def _build_standard_plan(classification: ClassificationResult) -> RetrievalPlan:
    """
    Build STANDARD plan for LOOKUP queries.

    Single source, single pass retrieval.
    Fast, cheap, sufficient for factual queries.

    Args:
        classification: ClassificationResult from QueryClassifier

    Returns:
        RetrievalPlan with STANDARD strategy
    """
    # Primary source from hints — default eCFR
    primary_source = classification.sources_hint[0] if classification.sources_hint else "ecfr"

    # top_k based on complexity
    top_k_map = {"simple": 5, "medium": 7, "complex": 8}
    top_k = top_k_map.get(classification.complexity, 5)

    qdrant_filter = build_metadata_filter(source=primary_source)

    source_configs = [
        SourceConfig(
            source=primary_source,
            top_k=top_k,
            filter=qdrant_filter,
            weight=SOURCE_WEIGHTS.get(primary_source, 1.0),
            description=SOURCE_DESCRIPTIONS.get(primary_source, primary_source),
        )
    ]

    # Token budget: avg chunk ~100 tokens
    token_budget = top_k * 100

    return RetrievalPlan(
        query=classification.query,
        query_type=classification.query_type,
        strategy=RetrievalStrategy.STANDARD,
        source_configs=source_configs,
        total_top_k=top_k,
        token_budget=token_budget,
        use_reranker=True,
        rerank_top_k=min(top_k, 5),
        hop_queries=[],
        complexity=classification.complexity,
        plan_reasoning=(
            f"LOOKUP query → STANDARD strategy. "
            f"Single source ({primary_source}), top_k={top_k}. "
            f"Token budget: ~{token_budget} tokens."
        )
    )


def _build_multi_source_plan(classification: ClassificationResult) -> RetrievalPlan:
    """
    Build MULTI_SOURCE plan for COMPARISON queries.

    Parallel retrieval from multiple sources.
    Results merged by source weight + reranked globally.

    Args:
        classification: ClassificationResult from QueryClassifier

    Returns:
        RetrievalPlan with MULTI_SOURCE strategy
    """
    # Determine sources to search
    hints = classification.sources_hint

    # COMPARISON always needs at least 2 sources
    # If only 1 hint detected, add complementary source
    if len(hints) < 2:
        if "dgca" in hints:
            hints = ["ecfr", "dgca"]
        elif "faa_ad" in hints:
            hints = ["ecfr", "faa_ad"]
        else:
            hints = ["ecfr", "dgca"]  # Default FAA vs DGCA

    # top_k per source — less than ADVISORY since we merge
    top_k_per_source = 5
    rerank_top_k = 6  # Final after global reranking

    source_configs = []
    for source in hints:
        qdrant_filter = build_metadata_filter(source=source)
        source_configs.append(
            SourceConfig(
                source=source,
                top_k=top_k_per_source,
                filter=qdrant_filter,
                weight=SOURCE_WEIGHTS.get(source, 0.80),
                description=SOURCE_DESCRIPTIONS.get(source, source),
            )
        )

    total_top_k = top_k_per_source * len(source_configs)
    token_budget = total_top_k * 100

    return RetrievalPlan(
        query=classification.query,
        query_type=classification.query_type,
        strategy=RetrievalStrategy.MULTI_SOURCE,
        source_configs=source_configs,
        total_top_k=total_top_k,
        token_budget=token_budget,
        use_reranker=True,
        rerank_top_k=rerank_top_k,
        hop_queries=[],
        complexity=classification.complexity,
        plan_reasoning=(
            f"COMPARISON query → MULTI_SOURCE strategy. "
            f"Sources: {hints}, top_k={top_k_per_source} per source. "
            f"Total chunks: {total_top_k}, global rerank to top {rerank_top_k}. "
            f"Token budget: ~{token_budget} tokens."
        )
    )


def _build_multi_hop_plan(classification: ClassificationResult) -> RetrievalPlan:
    """
    Build MULTI_HOP plan for ADVISORY queries.

    Sequential retrieval — initial fetch + follow-up queries
    based on what was found. Simulates reasoning chain.

    Example:
        Query: "Can I fly without MEL if altimeter broken?"
        Hop 1: "MEL requirements Part 91"
        Hop 2: "inoperative equipment regulations"
        Hop 3: "altimeter airworthiness requirements"

    Args:
        classification: ClassificationResult from QueryClassifier

    Returns:
        RetrievalPlan with MULTI_HOP strategy
    """
    primary_source = classification.sources_hint[0] if classification.sources_hint else "ecfr"

    # ADVISORY gets broader retrieval
    top_k = 8
    rerank_top_k = 5

    # Generate hop queries — decompose the advisory query
    hop_queries = _generate_hop_queries(classification.query)

    # Primary source config — broad search
    qdrant_filter = build_metadata_filter(source=primary_source)
    source_configs = [
        SourceConfig(
            source=primary_source,
            top_k=top_k,
            filter=qdrant_filter,
            weight=SOURCE_WEIGHTS.get(primary_source, 1.0),
            description=SOURCE_DESCRIPTIONS.get(primary_source, primary_source),
        )
    ]

    # Add all sources for advisory — compliance needs broad coverage
    for source in classification.sources_hint[1:]:
        source_configs.append(
            SourceConfig(
                source=source,
                top_k=5,
                filter=build_metadata_filter(source=source),
                weight=SOURCE_WEIGHTS.get(source, 0.80),
                description=SOURCE_DESCRIPTIONS.get(source, source),
            )
        )

    total_top_k = sum(sc.top_k for sc in source_configs)
    token_budget = total_top_k * 100 + len(hop_queries) * 200

    return RetrievalPlan(
        query=classification.query,
        query_type=classification.query_type,
        strategy=RetrievalStrategy.MULTI_HOP,
        source_configs=source_configs,
        total_top_k=total_top_k,
        token_budget=token_budget,
        use_reranker=True,
        rerank_top_k=rerank_top_k,
        hop_queries=hop_queries,
        complexity=classification.complexity,
        plan_reasoning=(
            f"ADVISORY query → MULTI_HOP strategy. "
            f"Primary source: {primary_source}, top_k={top_k}. "
            f"Hop queries: {len(hop_queries)} follow-up retrievals planned. "
            f"Token budget: ~{token_budget} tokens."
        )
    )


def _generate_hop_queries(query: str) -> list[str]:
    """
    Decompose an advisory query into hop sub-queries.

    These sub-queries are used in MULTI_HOP retrieval to
    gather broader regulatory context.

    Strategy: extract key regulatory concepts from query
    and generate targeted sub-queries for each.

    Args:
        query: Original advisory query

    Returns:
        List of 2-3 hop sub-queries
    """
    query_lower = query.lower()
    hops = []

    # Equipment / airworthiness hop
    if any(kw in query_lower for kw in ["mel", "equipment", "inoperative", "broken", "failed"]):
        hops.append("minimum equipment list MEL inoperative instruments requirements Part 91")
        hops.append("airworthiness requirements inoperative equipment flight")

    # Fuel / planning hop
    if any(kw in query_lower for kw in ["fuel", "vfr", "ifr", "flight plan"]):
        hops.append("fuel requirements VFR IFR flight planning Part 91")
        hops.append("preflight planning requirements pilot in command")

    # Weather / visibility hop
    if any(kw in query_lower for kw in ["weather", "visibility", "ceiling", "ifr", "vfr"]):
        hops.append("VFR weather minimums visibility ceiling requirements")

    # Crew / certification hop
    if any(kw in query_lower for kw in ["pilot", "crew", "certificate", "rating", "qualified"]):
        hops.append("pilot certification requirements currency recency")

    # Default hop if nothing specific detected
    if not hops:
        hops.append(f"regulatory requirements {query[:50]}")
        hops.append("Part 91 general operating flight rules compliance")

    return hops[:3]  # Max 3 hops to control cost


# ── Main Planner Class ───────────────────────────────────────────────────────

class RetrievalPlanner:
    """
    Builds retrieval plans from classification results.

    Usage:
        planner = RetrievalPlanner()
        plan = planner.plan(classification_result)
        print(plan.strategy)       # STANDARD / MULTI_SOURCE / MULTI_HOP
        print(plan.total_top_k)    # 5 / 10 / 8
        print(plan.token_budget)   # ~500 / ~1000 / ~1500
    """

    def __init__(self):
        logger.info("RetrievalPlanner initialized")

    def plan(self, classification: ClassificationResult) -> RetrievalPlan:
        """
        Build a RetrievalPlan from a ClassificationResult.

        Routes to the appropriate plan builder based on query_type.

        Args:
            classification: Output from QueryClassifier.classify()

        Returns:
            RetrievalPlan consumed by router.py

        Raises:
            AgentError: If planning fails
        """
        try:
            logger.info(
                f"Planning retrieval | "
                f"Type: {classification.query_type.value} | "
                f"Complexity: {classification.complexity} | "
                f"Sources hint: {classification.sources_hint}"
            )

            if classification.query_type == QueryType.LOOKUP:
                plan = _build_standard_plan(classification)

            elif classification.query_type == QueryType.COMPARISON:
                plan = _build_multi_source_plan(classification)

            else:  # ADVISORY
                plan = _build_multi_hop_plan(classification)

            logger.info(
                f"Plan built | "
                f"Strategy: {plan.strategy.value} | "
                f"Total top_k: {plan.total_top_k} | "
                f"Token budget: {plan.token_budget} | "
                f"Hops: {len(plan.hop_queries)}"
            )

            return plan

        except Exception as e:
            handle_exception(
                e,
                context="RetrievalPlanner.plan",
                raise_as=AgentError
            )


def format_plan(plan: RetrievalPlan) -> str:
    """Format RetrievalPlan for CLI output."""

    strategy_icons = {
        RetrievalStrategy.STANDARD:     "⚡ STANDARD",
        RetrievalStrategy.MULTI_SOURCE: "🔀 MULTI_SOURCE",
        RetrievalStrategy.MULTI_HOP:    "🔁 MULTI_HOP",
    }

    lines = [
        f"\n{'─'*65}",
        f"RETRIEVAL PLAN",
        f"{'─'*65}",
        f"Query      : {plan.query}",
        f"Type       : {plan.query_type.value}",
        f"Strategy   : {strategy_icons[plan.strategy]}",
        f"Total top_k: {plan.total_top_k}",
        f"Rerank to  : {plan.rerank_top_k}",
        f"Token Budget: ~{plan.token_budget} tokens",
        f"Complexity : {plan.complexity}",
        f"\nSource Configs:",
    ]

    for i, sc in enumerate(plan.source_configs, 1):
        lines.append(
            f"  [{i}] {sc.source} | top_k={sc.top_k} | "
            f"weight={sc.weight:.2f} | {sc.description}"
        )

    if plan.hop_queries:
        lines.append(f"\nHop Queries ({len(plan.hop_queries)}):")
        for i, hq in enumerate(plan.hop_queries, 1):
            lines.append(f"  Hop {i}: {hq}")

    lines += [
        f"\nReasoning  : {plan.plan_reasoning}",
        f"{'─'*65}",
    ]

    return "\n".join(lines)


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== AeroLex Retrieval Planner — Test ===\n")

    from src.agents.query_classifier import QueryClassifier

    classifier = QueryClassifier()
    planner    = RetrievalPlanner()

    test_queries = [
        "What does 14 CFR 91.103 say about preflight requirements?",
        "How do FAA and DGCA preflight rules differ?",
        "Can I fly without an MEL if the altimeter is broken?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        classification = classifier.classify(query)
        plan = planner.plan(classification)
        print(format_plan(plan))