"""
AeroLex — Hybrid Retriever (Dense + BM25 Sparse)

Combines dense vector search with BM25 keyword search
using Reciprocal Rank Fusion (RRF) for best retrieval quality.

Why Hybrid Search?
- Dense alone misses exact regulation numbers (e.g., "91.103", "AD 2024-15")
- BM25 alone misses semantic meaning ("pilot duties" ≠ "aviator responsibilities")
- Hybrid = best of both worlds

What is BM25?
- Best Match 25 — probabilistic keyword ranking algorithm
- Used by Elasticsearch, Solr, Lucene (industry standard since 1994)
- Ranks documents by term frequency + inverse document frequency
- Still state-of-the-art for keyword search!

Official Docs:
- Qdrant Hybrid Search: https://qdrant.tech/documentation/concepts/hybrid-queries/
- BM25 Paper: https://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf
- RRF Paper: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

RRF Formula:
    RRF_score(d) = Σ 1/(k + rank_i(d))
    where k=60 (constant), rank_i = position in i-th result list
    Documents appearing in BOTH lists get highest scores!

BM25 Formula:
    BM25(q,d) = Σ IDF(qi) * (f(qi,d) * (k1+1)) / (f(qi,d) + k1*(1-b+b*|d|/avgdl))
    where:
    - IDF = inverse document frequency (rare terms get higher weight)
    - f(qi,d) = term frequency in document
    - k1 = 1.2-2.0 (term saturation)
    - b = 0.75 (length normalization)
    - |d| = document length, avgdl = average document length
"""

import math
from collections import defaultdict
from typing import Optional
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, RetrievalError
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.qdrant_store import QdrantStore
from config.settings import settings

logger = get_logger(__name__)

# ── RRF Constant ──────────────────────────────────────────────────────────────
# k=60 is standard — from original RRF paper
# Higher k = smoother score distribution
RRF_K = 60


class BM25:
    """
    In-memory BM25 implementation for sparse keyword search.

    Why in-memory BM25?
    - Qdrant sparse vectors support exists but requires fastembed
    - For our corpus size (600-3000 chunks), in-memory is fast enough
    - Production: use Qdrant sparse vectors or Elasticsearch

    Parameters:
        k1=1.5: Term frequency saturation — how much repeated terms help
        b=0.75: Length normalization — penalize long documents
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1   = k1
        self.b    = b
        self.docs = []          # List of tokenized documents
        self.doc_data = []      # Original doc dicts
        self.df   = defaultdict(int)  # Document frequency per term
        self.idf  = {}          # IDF scores
        self.avgdl = 0          # Average document length

    def tokenize(self, text: str) -> list[str]:
        """
        Simple tokenizer — lowercase + split.
        Production: use NLTK or spaCy for better tokenization.
        """
        import re
        # Remove punctuation except hyphens (important for "91-103" style refs)
        text = text.lower()
        text = re.sub(r'[^\w\s\-]', ' ', text)
        tokens = text.split()
        # Remove very short tokens
        return [t for t in tokens if len(t) > 1]

    def fit(self, documents: list[dict]) -> None:
        """
        Build BM25 index from documents.

        Args:
            documents: List of chunk dicts with 'text' field
        """
        self.doc_data = documents
        self.docs     = []

        total_length = 0
        for doc in documents:
            # Combine text + citation for better matching
            # "91.103" in citation helps match exact regulation references
            combined = f"{doc.get('text', '')} {doc.get('citation', '')}"
            tokens   = self.tokenize(combined)
            self.docs.append(tokens)
            total_length += len(tokens)

            # Count document frequency
            for term in set(tokens):
                self.df[term] += 1

        # Calculate average document length
        self.avgdl = total_length / len(documents) if documents else 0

        # Calculate IDF for each term
        n = len(documents)
        for term, df in self.df.items():
            # BM25 IDF formula
            self.idf[term] = math.log(
                (n - df + 0.5) / (df + 0.5) + 1
            )

        logger.info(
            f"BM25 index built | "
            f"Documents: {len(documents)} | "
            f"Vocab size: {len(self.df)} | "
            f"Avg length: {self.avgdl:.1f} tokens"
        )

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        BM25 search — returns (doc_index, score) pairs.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            list: (doc_index, bm25_score) sorted by score descending
        """
        query_tokens = self.tokenize(query)
        scores       = []

        for doc_idx, doc_tokens in enumerate(self.docs):
            score    = 0.0
            doc_len  = len(doc_tokens)
            token_freq = defaultdict(int)

            for token in doc_tokens:
                token_freq[token] += 1

            for term in query_tokens:
                if term not in self.idf:
                    continue

                tf  = token_freq.get(term, 0)
                idf = self.idf[term]

                # BM25 TF component
                tf_component = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                )
                score += idf * tf_component

            if score > 0:
                scores.append((doc_idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridRetriever:
    """
    Hybrid retrieval combining Dense + BM25 with RRF fusion.

    Architecture:
    1. Dense search → Top-20 candidates (semantic)
    2. BM25 search  → Top-20 candidates (keyword)
    3. RRF fusion   → Combine rankings → Top-K final results

    Why RRF over simple score combination?
    - Dense scores (0-1) and BM25 scores (0-∞) are not comparable
    - RRF uses RANK not score — scale-independent!
    - Documents in BOTH lists get big boost
    - Proven to work better than weighted sum in practice
    """

    def __init__(
        self,
        collection: str = "aerolex_voyage",
        embedding_model: str = "voyage",
    ):
        self.collection      = collection
        self.embedding_model = embedding_model

        # Dense retriever
        self.dense_retriever = DenseRetriever(
            collection=collection,
            embedding_model=embedding_model,
        )

        # BM25 index — built lazily when first query arrives
        self.bm25        = BM25()
        self.bm25_fitted = False
        self.all_chunks  = []

        logger.info(
            f"HybridRetriever initialized | "
            f"Collection: {collection} | "
            f"Embedder: {embedding_model}"
        )

    def _build_bm25_index(self) -> None:
        """
        Build BM25 index by scrolling all chunks from Qdrant.

        Why scroll? Qdrant search requires a query vector.
        To get ALL documents for BM25, we scroll the collection.
        """
        if self.bm25_fitted:
            return

        logger.info("Building BM25 index from Qdrant collection...")

        store  = QdrantStore(collection=self.collection)
        client = store.client

        # Scroll all points from Qdrant
        all_points = []
        offset     = None

        while True:
            result, offset = client.scroll(
                collection_name=self.collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(result)

            if offset is None:
                break

        # Convert to chunk dicts
        self.all_chunks = [dict(p.payload) for p in all_points]

        # Fit BM25
        self.bm25.fit(self.all_chunks)
        self.bm25_fitted = True

        logger.info(f"BM25 index ready | Documents: {len(self.all_chunks)}")

    def _rrf_fusion(
        self,
        dense_results: list[dict],
        bm25_results: list[tuple[int, float]],
        k: int = RRF_K,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion — combine dense + BM25 rankings.

        RRF Formula:
            score(d) = Σ 1/(k + rank_i(d))

        Args:
            dense_results: List of chunk dicts with 'score' field
            bm25_results: List of (chunk_index, bm25_score) tuples
            k: RRF constant (default 60 from paper)

        Returns:
            list[dict]: Chunks sorted by RRF score
        """
        rrf_scores = defaultdict(float)

        # Dense results contribution
        # rank 1 → 1/(60+1) = 0.0164
        # rank 2 → 1/(60+2) = 0.0161
        for rank, chunk in enumerate(dense_results, 1):
            chunk_id = chunk.get("chunk_id", "")
            rrf_scores[chunk_id] += 1.0 / (k + rank)

        # BM25 results contribution
        for rank, (doc_idx, _) in enumerate(bm25_results, 1):
            if doc_idx < len(self.all_chunks):
                chunk_id = self.all_chunks[doc_idx].get("chunk_id", "")
                rrf_scores[chunk_id] += 1.0 / (k + rank)

        # Build result list with RRF scores
        # Merge dense results + BM25-only results
        chunk_map = {}

        for chunk in dense_results:
            chunk_id = chunk.get("chunk_id", "")
            chunk_map[chunk_id] = chunk

        for doc_idx, _ in bm25_results:
            if doc_idx < len(self.all_chunks):
                chunk = self.all_chunks[doc_idx]
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = chunk

        # Sort by RRF score
        results = []
        for chunk_id, rrf_score in sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        ):
            if chunk_id in chunk_map:
                result = dict(chunk_map[chunk_id])
                result["rrf_score"]    = round(rrf_score, 6)
                result["score"]        = round(rrf_score, 6)
                results.append(result)

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        dense_candidates: int = 20,
        bm25_candidates: int = 20,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Hybrid retrieval with RRF fusion.

        Args:
            query: User search query
            top_k: Final number of results
            dense_candidates: Dense search pool size
            bm25_candidates: BM25 search pool size
            filters: Metadata filters for dense search

        Returns:
            list[dict]: Top-K chunks sorted by RRF score
        """
        logger.info(
            f"Hybrid retrieval | "
            f"Query: '{query[:50]}' | "
            f"Top-K: {top_k}"
        )

        # Build BM25 index if needed
        self._build_bm25_index()

        # Step 1: Dense search
        try:
            dense_results = self.dense_retriever.retrieve(
                query=query,
                top_k=dense_candidates,
                score_threshold=0.0,  # No threshold for candidates
                filters=filters,
            )
            logger.info(f"Dense candidates: {len(dense_results)}")
        except Exception as e:
            logger.warning(f"Dense search failed — using BM25 only: {e}")
            dense_results = []

        # Step 2: BM25 search
        bm25_results = self.bm25.search(query, top_k=bm25_candidates)
        logger.info(f"BM25 candidates: {len(bm25_results)}")

        # Step 3: RRF fusion
        fused = self._rrf_fusion(dense_results, bm25_results)

        # Return top-K
        final = fused[:top_k]

        top_rrf = f"{final[0]['rrf_score']:.6f}" if final else "0"
        logger.info(
            f"Hybrid retrieval complete | "
            f"Final results: {len(final)} | "
            f"Top RRF score: {top_rrf}"
        )

        return final

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> dict:
        """Retrieve and format as RAG context."""
        chunks    = self.retrieve(query, top_k=top_k, filters=filters)
        citations = []
        context_parts = []

        for i, chunk in enumerate(chunks):
            citation = chunk.get("citation", "Unknown")
            text     = chunk.get("text", "")
            score    = chunk.get("rrf_score", 0)
            citations.append(citation)
            context_parts.append(
                f"[{i+1}] {citation} (RRF score: {score:.4f})\n{text}"
            )

        return {
            "query":      query,
            "chunks":     chunks,
            "context":    "\n\n".join(context_parts),
            "citations":  citations,
            "num_chunks": len(chunks),
        }


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔍 AeroLex — Hybrid Retrieval Test")
    print("="*60)

    test_queries = [
        "What must a pilot do before beginning a flight?",
        "91.103 preflight requirements",
        "fuel requirements VFR flight",
    ]

    retriever = HybridRetriever(
        collection="aerolex_voyage",
        embedding_model="voyage",
    )

    for query in test_queries:
        print(f"\n{'─'*60}")
        print(f"📝 Query: {query}")
        print(f"{'─'*60}")

        result = retriever.retrieve_with_context(query=query, top_k=3)

        print(f"Found {result['num_chunks']} chunks:\n")
        for i, chunk in enumerate(result["chunks"]):
            print(f"  [{i+1}] RRF Score: {chunk['rrf_score']:.6f}")
            print(f"       Citation : {chunk['citation']}")
            print(f"       Text     : {chunk['text'][:120]}...")
            print()

    print("="*60)
    print("✅ Hybrid Retriever working!")
    print("="*60)