"""
AeroLex — Cross-Encoder Re-Ranker

Re-ranks retrieved chunks using a cross-encoder model for
higher precision before sending to LLM.

Retrieval Pipeline:
    Query → Hybrid Search (Top-20) → Re-ranker (Top-5) → LLM

Why Re-ranking?
- Bi-encoders (embedding models): encode query + doc SEPARATELY
  Fast but less accurate — no direct query-doc interaction
- Cross-encoders: read query + doc TOGETHER
  Slow but very accurate — full attention between query and doc

Think of it like:
- Bi-encoder: "Does this resume LOOK relevant?" (quick scan)
- Cross-encoder: "READ this resume carefully for THIS job" (deep evaluation)

Cross-Encoder Architecture:
- Input: [CLS] query [SEP] document [SEP]
- BERT processes BOTH together with full attention
- Output: single relevance score (0-1)
- Full attention = query words attend to document words directly

Official Docs:
- sentence-transformers cross-encoders:
  https://www.sbert.net/docs/cross_encoder/pretrained_models.html
- MS-MARCO model:
  https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Trained on MS-MARCO passage ranking dataset
- 6-layer MiniLM — fast cross-encoder
- Good balance: quality vs speed
- Size: ~90MB
"""

import time
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, RetrievalError
from config.settings import settings

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_LENGTH     = 512   # Max tokens for cross-encoder input


class CrossEncoderReranker:
    """
    Re-ranks retrieved chunks using cross-encoder model.

    Input:  Query + Top-K chunks from hybrid retriever
    Output: Same chunks re-sorted by cross-encoder relevance score

    Performance characteristics:
    - ~50-100ms per (query, chunk) pair on CPU
    - Top-10 reranking: ~500ms-1sec total
    - Worth it: significantly improves precision
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self.model      = None  # Lazy load
        logger.info(f"CrossEncoderReranker initialized | Model: {model_name}")

    def _load_model(self):
        """Lazy load cross-encoder model."""
        if self.model is None:
            logger.info(f"Loading cross-encoder: {self.model_name}")
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(
                self.model_name,
                max_length=MAX_LENGTH,
            )
            logger.info("Cross-encoder loaded")
        return self.model

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Re-rank chunks using cross-encoder.

        Args:
            query: User's search query
            chunks: List of chunk dicts (from hybrid retriever)
            top_k: Number of chunks to return after reranking

        Returns:
            list[dict]: Chunks sorted by cross-encoder score
        """
        if not chunks:
            return []

        model = self._load_model()

        logger.info(
            f"Reranking {len(chunks)} chunks | "
            f"Query: '{query[:50]}' | "
            f"Top-K: {top_k}"
        )

        start_time = time.time()

        # Build (query, document) pairs for cross-encoder
        # Cross-encoder reads BOTH together — full attention
        pairs = [
            (query, chunk.get("text", ""))
            for chunk in chunks
        ]

        # Score all pairs
        # Returns array of relevance scores
        scores = model.predict(pairs, show_progress_bar=False)

        # Attach scores to chunks
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            chunk_copy = dict(chunk)
            chunk_copy["rerank_score"]    = float(score)
            chunk_copy["original_score"]  = chunk.get("score", 0)
            scored_chunks.append(chunk_copy)

        # Sort by rerank score — highest first
        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Return top-K
        result = scored_chunks[:top_k]

        elapsed = time.time() - start_time
        top_score = f"{result[0]['rerank_score']:.4f}" if result else "0"
        logger.info(
            f"Reranking complete | "
            f"Time: {elapsed:.2f}s | "
            f"Top score: {top_score}"
        )

        return result

    def rerank_with_threshold(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """
        Rerank and filter by minimum score threshold.

        Args:
            query: Search query
            chunks: Retrieved chunks
            top_k: Max results to return
            score_threshold: Minimum rerank score
                            Scores can be negative for cross-encoders!
                            0.0 = filter out clearly irrelevant
                            Use -10.0 to keep everything

        Returns:
            list[dict]: Filtered and reranked chunks
        """
        reranked = self.rerank(query, chunks, top_k=top_k)

        # Filter by threshold
        filtered = [c for c in reranked if c["rerank_score"] >= score_threshold]

        if len(filtered) < len(reranked):
            logger.info(
                f"Threshold filtering: {len(reranked)} → {len(filtered)} chunks "
                f"(threshold: {score_threshold})"
            )

        return filtered

class VoyageReranker:
    """
    Voyage AI Reranker — better for RAG than MS-MARCO cross-encoder.

    Official Docs: https://docs.voyageai.com/docs/reranker
    Model: rerank-2 — Voyage's latest reranking model

    Why Voyage Reranker?
    - Trained specifically for retrieval/RAG tasks
    - Works best with Voyage embeddings (same training distribution)
    - Returns scores 0-1 (interpretable unlike cross-encoder logits)
    - Anthropic recommended for use with Claude
    """

    def __init__(self):
        import voyageai
        self.client     = voyageai.Client(api_key=settings.VOYAGE_API_KEY)
        self.model      = "rerank-2"
        logger.info(f"VoyageReranker initialized | Model: {self.model}")

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank chunks using Voyage rerank-2 API.

        Args:
            query: User query
            chunks: Retrieved chunks
            top_k: Final number to return

        Returns:
            list[dict]: Reranked chunks with voyage_rerank_score
        """
        if not chunks:
            return []

        documents = [c.get("text", "") for c in chunks]

        logger.info(
            f"Voyage reranking {len(chunks)} chunks | "
            f"Query: '{query[:50]}'"
        )

        start_time = time.time()

        result = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_k=top_k,
        )

        # Map results back to chunks
        reranked = []
        for item in result.results:
            chunk = dict(chunks[item.index])
            chunk["rerank_score"]         = item.relevance_score
            chunk["voyage_rerank_score"]  = item.relevance_score
            chunk["original_rank"]        = item.index
            reranked.append(chunk)

        elapsed = time.time() - start_time
        top_score = f"{reranked[0]['rerank_score']:.4f}" if reranked else "0"
        logger.info(
            f"Voyage reranking complete | "
            f"Time: {elapsed:.2f}s | "
            f"Top score: {top_score} | "
            f"Tokens used: {result.total_tokens}"
        )

        return reranked
    

class RetrievalPipeline:
    """
    Complete retrieval pipeline:
    Query → Hybrid Search → Re-ranking → Final Results

    This is what gets called from the RAG chain.
    """

    def __init__(
        self,
        collection: str = "aerolex_voyage",
        embedding_model: str = "voyage",
        use_reranker: bool = True,
        reranker_type: str = "voyage",  # "voyage" or "crossencoder"
    ):
        from src.retrieval.hybrid_retriever import HybridRetriever

        self.hybrid = HybridRetriever(
            collection=collection,
            embedding_model=embedding_model,
        )

        if use_reranker:
            if reranker_type == "voyage":
                self.reranker = VoyageReranker()
            else:
                self.reranker = CrossEncoderReranker()
        else:
            self.reranker = None

        self.use_reranker = use_reranker
        logger.info(
            f"RetrievalPipeline initialized | "
            f"Collection: {collection} | "
            f"Reranker: {reranker_type if use_reranker else 'None'}"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        hybrid_candidates: int = 20,
        filters: dict = None,
    ) -> dict:
        """
        Full retrieval pipeline.

        Args:
            query: User query
            top_k: Final number of chunks for LLM
            hybrid_candidates: Pool size for hybrid search
            filters: Metadata filters

        Returns:
            dict: {
                query, chunks, context, citations,
                retrieval_stats
            }
        """
        start_time = time.time()

        # Step 1: Hybrid retrieval
        hybrid_results = self.hybrid.retrieve(
            query=query,
            top_k=hybrid_candidates,
            filters=filters,
        )

        # Step 2: Re-ranking (if enabled)
        if self.use_reranker and hybrid_results:
            final_chunks = self.reranker.rerank(
                query=query,
                chunks=hybrid_results,
                top_k=top_k,
            )
        else:
            final_chunks = hybrid_results[:top_k]

        # Step 3: Format context
        citations     = []
        context_parts = []

        for i, chunk in enumerate(final_chunks):
            citation = chunk.get("citation", "Unknown")
            text     = chunk.get("text", "")
            citations.append(citation)

            # Include both scores for transparency
            rrf_score    = chunk.get("rrf_score", chunk.get("score", 0))
            rerank_score = chunk.get("rerank_score", "N/A")

            context_parts.append(
                f"[{i+1}] SOURCE: {citation}\n"
                f"CONTENT: {text}"
            )

        total_time = time.time() - start_time

        return {
            "query":      query,
            "chunks":     final_chunks,
            "context":    "\n\n---\n\n".join(context_parts),
            "citations":  citations,
            "num_chunks": len(final_chunks),
            "retrieval_stats": {
                "hybrid_candidates": len(hybrid_results),
                "final_chunks":      len(final_chunks),
                "total_time_sec":    round(total_time, 2),
                "reranker_used":     self.use_reranker,
            }
        }


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔍 AeroLex — Full Retrieval Pipeline Test")
    print("  Hybrid Search → Re-ranking")
    print("="*60)

    # Test with full pipeline
    pipeline = RetrievalPipeline(
        collection="aerolex_voyage",
        embedding_model="voyage",
        use_reranker=True,
        reranker_type="voyage",
    )

    test_queries = [
        "What must a pilot do before beginning a flight?",
        "fuel requirements for VFR flight",
    ]

    for query in test_queries:
        print(f"\n{'─'*60}")
        print(f"📝 Query: {query}")
        print(f"{'─'*60}")

        result = pipeline.retrieve(
            query=query,
            top_k=3,
            hybrid_candidates=10,
        )

        stats = result["retrieval_stats"]
        print(f"\n📊 Retrieval Stats:")
        print(f"   Hybrid candidates : {stats['hybrid_candidates']}")
        print(f"   Final chunks      : {stats['final_chunks']}")
        print(f"   Total time        : {stats['total_time_sec']}s")
        print(f"   Reranker used     : {stats['reranker_used']}")

        print(f"\n📋 Final Results (after reranking):\n")
        for i, chunk in enumerate(result["chunks"]):
            print(f"  [{i+1}] Citation     : {chunk['citation']}")
            print(f"       RRF Score    : {chunk.get('rrf_score', 0):.6f}")
            print(f"       Rerank Score : {chunk.get('rerank_score', 'N/A')}")
            if isinstance(chunk.get('rerank_score'), float):
                print(f"       Rerank Score : {chunk['rerank_score']:.4f}")
            print(f"       Text         : {chunk['text'][:120]}...")
            print()

    print("="*60)
    print("✅ Full Retrieval Pipeline working!")
    print("="*60)