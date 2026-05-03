"""
rag_chain.py — Core RAG Chain: Retrieval + LLM Answer Generation

WHAT:
    Orchestrates the full RAG pipeline — takes a user query,
    retrieves relevant chunks via hybrid search, builds a
    structured prompt, calls Claude/GPT, returns answer + citations.

WHY:
    LLMs hallucinate without context. RAG grounds the LLM in
    real retrieved regulatory text — making answers trustworthy,
    traceable, and citation-backed.

HOW:
    1. Build metadata filter (optional, from query intent)
    2. Run RetrievalPipeline (hybrid BM25 + dense + RRF + rerank)
    3. Build structured prompt with retrieved chunks as context
    4. Call Claude (primary) or GPT (fallback)
    5. Return RAGResponse — answer + sources + confidence + cost

MATH:
    Final answer = LLM( query + top_k retrieved chunks )
    where chunks = RetrievalPipeline(query, filter, top_k=5)

Official Docs:
    Anthropic: https://docs.anthropic.com/en/api/messages
    LangSmith:  https://docs.smith.langchain.com/
    MLflow:     https://mlflow.org/docs/latest/tracking.html
"""

import time
from dataclasses import dataclass, field
from typing import Optional
import mlflow
import anthropic
from openai import OpenAI

from src.retrieval.reranker import RetrievalPipeline
from src.retrieval.metadata_filter import build_metadata_filter, get_filter_for_query_intent
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, RAGError
#from src.monitoring.cost_tracker import CostTracker
#from src.monitoring.langsmith_tracker import LangSmithTracker
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


# ── Response Schema ──────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """Single retrieved chunk with metadata."""
    text: str
    source: str
    doc_type: str
    part_number: str
    chunk_id: str
    similarity_score: float


@dataclass
class RAGResponse:
    """
    Structured RAG output — answer + full provenance.

    Every field here is intentional:
    - answer:       LLM generated response grounded in context
    - sources:      List of chunks used — for citation_builder.py
    - confidence:   Proxy metric — how many chunks were retrieved
    - model_used:   Claude or GPT — for cost comparison in MLflow
    - latency_ms:   End-to-end pipeline latency
    - input_tokens: For cost tracking
    - output_tokens: For cost tracking
    - cost_usd:     Total cost of this RAG call
    """
    answer: str
    sources: list[RetrievedChunk]
    confidence: float
    model_used: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    query: str
    error: Optional[str] = None


# ── Prompt Builder ───────────────────────────────────────────────────────────

def build_rag_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """
    Build structured prompt with retrieved chunks as context.

    Design decisions:
    - Numbered sources → LLM can cite [Source 1], [Source 2]
    - Explicit instruction to NOT hallucinate beyond context
    - Aviation domain persona for precise regulatory language
    - Confidence instruction — say 'I don't know' if context insufficient

    Args:
        query:  User's original question
        chunks: Top-K retrieved chunks from RetrievalPipeline

    Returns:
        Complete prompt string ready for LLM
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        block = (
            f"[Source {i}]\n"
            f"Regulation: {chunk.source} | Type: {chunk.doc_type} | Part: {chunk.part_number}\n"
            f"Text: {chunk.text}\n"
        )
        context_blocks.append(block)

    context = "\n---\n".join(context_blocks)

    prompt = f"""You are AeroLex — an expert Aviation Regulatory Compliance Assistant.
You answer questions strictly based on the provided regulatory context.

RULES:
1. Answer ONLY from the context below — do not hallucinate
2. Cite sources as [Source N] inline in your answer
3. If context is insufficient, say: "The provided regulatory context does not contain enough information to answer this question."
4. Be precise — aviation regulations are safety-critical
5. Use regulatory language — exact section numbers, part references

REGULATORY CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    return prompt


# ── LLM Callers ─────────────────────────────────────────────────────────────

def call_claude(prompt: str) -> tuple[str, int, int, float]:
    """
    Call Anthropic Claude API.

    Args:
        prompt: Complete RAG prompt

    Returns:
        Tuple of (answer_text, input_tokens, output_tokens, cost_usd)
    """
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    # Claude sonnet-4 pricing (per million tokens)
    cost_usd = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000

    logger.info(
        f"Claude response — "
        f"input_tokens={input_tokens}, "
        f"output_tokens={output_tokens}, "
        f"cost=${cost_usd:.6f}"
    )
    return answer, input_tokens, output_tokens, cost_usd


def call_gpt(prompt: str) -> tuple[str, int, int, float]:
    """
    Call OpenAI GPT API (fallback model).

    Args:
        prompt: Complete RAG prompt

    Returns:
        Tuple of (answer_text, input_tokens, output_tokens, cost_usd)
    """
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    # GPT-4o-mini pricing (per million tokens)
    cost_usd = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000

    logger.info(
        f"GPT response — "
        f"input_tokens={input_tokens}, "
        f"output_tokens={output_tokens}, "
        f"cost=${cost_usd:.6f}"
    )
    return answer, input_tokens, output_tokens, cost_usd


# ── Main RAG Chain ───────────────────────────────────────────────────────────

class RAGChain:
    """
    Full RAG pipeline orchestrator.

    Usage:
        chain = RAGChain()
        response = chain.run(query="What must a pilot do before flight?")
        print(response.answer)
        print(response.sources)
    """

    def __init__(
        self,
        collection_name: str = "aerolex_voyage",
        top_k: int = 5,
        use_claude: bool = True,
        auto_filter: bool = True,
    ):
        """
        Args:
            collection_name: Qdrant collection to search
            top_k:           Number of chunks to retrieve
            use_claude:      True = Claude, False = GPT-4o-mini
            auto_filter:     Auto-detect metadata filter from query
        """
        self.collection_name = collection_name
        self.top_k = top_k
        self.use_claude = use_claude
        self.auto_filter = auto_filter
        self.retrieval_pipeline = RetrievalPipeline(
            collection=collection_name,   # collection_name → collection
            embedding_model="voyage",
            use_reranker=True,
            reranker_type="voyage",
        )
        logger.info(
            f"RAGChain initialized — "
            f"collection={collection_name}, "
            f"top_k={top_k}, "
            f"model={'Claude' if use_claude else 'GPT-4o-mini'}"
        )

    def run(
        self,
        query: str,
        source: Optional[str] = None,
        doc_type: Optional[str] = None,
        part_number: Optional[str] = None,
    ) -> RAGResponse:
        """
        Execute full RAG pipeline for a query.

        Args:
            query:       User question
            source:      Optional manual filter — "eCFR", "DGCA", etc.
            doc_type:    Optional manual filter — "regulation", etc.
            part_number: Optional manual filter — "91", "121", etc.

        Returns:
            RAGResponse with answer, sources, cost, latency
        """
        start_time = time.time()
        model_name = "claude-sonnet-4-20250514" if self.use_claude else "gpt-4o-mini"

        logger.info(f"RAGChain.run() — query='{query[:80]}...'")

        with mlflow.start_run(run_name=f"rag_chain_{model_name}", nested=True):
            mlflow.log_param("query", query[:250])
            mlflow.log_param("model", model_name)
            mlflow.log_param("collection", self.collection_name)
            mlflow.log_param("top_k", self.top_k)

            try:
                # ── Step 1: Build metadata filter ──
                qdrant_filter = None
                if source or doc_type or part_number:
                    qdrant_filter = build_metadata_filter(
                        source=source,
                        doc_type=doc_type,
                        part_number=part_number
                    )
                elif self.auto_filter:
                    qdrant_filter = get_filter_for_query_intent(query)

                # ── Step 2: Retrieve chunks ──
                retrieved_data = self.retrieval_pipeline.retrieve(
                    query=query,
                    top_k=self.top_k,
                    filters=None,
                )
                raw_results = retrieved_data.get("chunks", [])

                if not raw_results:
                    logger.warning("No chunks retrieved — returning empty response")
                    return RAGResponse(
                        answer="No relevant regulatory context found for this query.",
                        sources=[],
                        confidence=0.0,
                        model_used=model_name,
                        latency_ms=0.0,
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=0.0,
                        query=query,
                        error="No chunks retrieved"
                    )

                # ── Step 3: Convert to RetrievedChunk objects ──
                chunks = []
                for r in raw_results:
                    chunks.append(RetrievedChunk(
                        text=r.get("text", ""),
                        source=r.get("source", "unknown"),
                        doc_type=r.get("doc_type", "unknown"),
                        part_number=r.get("part_number", "unknown"),
                        chunk_id=r.get("chunk_id", "unknown"),
                        similarity_score=r.get("rerank_score", r.get("rrf_score", 0.0))
                    ))

                logger.info(f"Retrieved {len(chunks)} chunks for RAG context")
                mlflow.log_metric("chunks_retrieved", len(chunks))

                # ── Step 4: Build prompt ──
                prompt = build_rag_prompt(query, chunks)
                logger.debug(f"Prompt length: {len(prompt)} chars")

                # ── Step 5: Call LLM ──
                if self.use_claude:
                    answer, input_tokens, output_tokens, cost_usd = call_claude(prompt)
                else:
                    answer, input_tokens, output_tokens, cost_usd = call_gpt(prompt)

                # ── Step 6: Confidence proxy ──
                # Simple heuristic: more chunks + higher scores = higher confidence
                avg_score = sum(c.similarity_score for c in chunks) / len(chunks)
                confidence = min(avg_score * (len(chunks) / self.top_k), 1.0)

                # ── Step 7: Latency ──
                latency_ms = (time.time() - start_time) * 1000

                # ── Step 8: MLflow logging ──
                mlflow.log_metric("input_tokens", input_tokens)
                mlflow.log_metric("output_tokens", output_tokens)
                mlflow.log_metric("cost_usd", cost_usd)
                mlflow.log_metric("confidence", confidence)
                mlflow.log_metric("latency_ms", latency_ms)

                logger.info(
                    f"RAGChain complete — "
                    f"latency={latency_ms:.0f}ms, "
                    f"confidence={confidence:.3f}, "
                    f"cost=${cost_usd:.6f}"
                )

                return RAGResponse(
                    answer=answer,
                    sources=chunks,
                    confidence=confidence,
                    model_used=model_name,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    query=query
                )

            except Exception as e:
                handle_exception(
                    e,
                    context="RAGChain.run",
                    raise_as=RAGError
                )


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== AeroLex RAG Chain — Test ===\n")

    chain = RAGChain(
        collection_name="aerolex_voyage",
        top_k=5,
        use_claude=True,
        auto_filter=True
    )

    test_queries = [
        "What must a pilot do before beginning a flight?",
        "What are the fuel requirements for VFR flight under Part 91?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        response = chain.run(query=query)
        print(f"Answer:\n{response.answer}")
        print(f"\nSources used: {len(response.sources)}")
        for i, src in enumerate(response.sources, 1):
            score = f"{response.sources[i-1].similarity_score:.4f}"
            print(f"  [{i}] {src.source} | Part {src.part_number} | score={score}")
        print(f"\nConfidence: {response.confidence:.3f}")
        print(f"Model: {response.model_used}")
        print(f"Cost: ${response.cost_usd:.6f}")
        print(f"Latency: {response.latency_ms:.0f}ms")
        print("=" * 60)