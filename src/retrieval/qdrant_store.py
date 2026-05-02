"""
AeroLex — Qdrant Vector Store

Central interface for all Qdrant operations:
- Store embedded chunks (upsert)
- Dense vector search
- Metadata filtering
- Collection management

Official Docs: https://qdrant.tech/documentation/
Python Client: https://github.com/qdrant/qdrant-client

Key Concepts:
- Point: One vector + payload (metadata) = one chunk stored
- Collection: Group of points with same vector size
- Payload: Metadata attached to each point (citation, source, etc.)
- Upsert: Insert if new, Update if exists (idempotent)

Why Upsert not Insert?
- If same chunk re-ingested (regulation updated), old vector replaced
- No duplicates — same chunk_id = same point_id in Qdrant
- Safe to run ingestion pipeline multiple times

HNSW — How Vector Search Works:
- HNSW builds a graph of vectors
- Search: start from entry point, navigate graph greedily
- At each step: move to neighbor most similar to query
- Returns approximate nearest neighbors (very fast!)
- Approximate = 99%+ accuracy in practice

Distance Metrics:
- Cosine: measures angle between vectors (what we use)
- Dot Product: measures magnitude + direction
- Euclidean: measures straight-line distance
For normalized embeddings: Cosine = Dot Product (same result)
"""

import uuid
import json
from pathlib import Path
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
    ScoredPoint,
)
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, VectorStoreError
from config.settings import settings

logger = get_logger(__name__)

# ── Collection → Model mapping ────────────────────────────────────────────────
COLLECTION_MODEL_MAP = {
    "aerolex_voyage": "voyage-3-large",
    "aerolex_openai": "text-embedding-3-small",
    "aerolex_e5":     "intfloat/e5-large-v2",
    "aerolex_bge":    "BAAI/bge-m3",
}

# ── Default collection ────────────────────────────────────────────────────────
DEFAULT_COLLECTION = "aerolex_voyage"


class QdrantStore:
    """
    AeroLex vector store — all Qdrant operations in one place.

    Responsibilities:
    1. Store embedded chunks as Qdrant points
    2. Search by vector similarity (dense search)
    3. Filter by metadata (source, part_number, doc_type, etc.)
    4. Manage collections
    """

    def __init__(self, collection: str = DEFAULT_COLLECTION):
        """
        Args:
            collection: Which Qdrant collection to use
                       Default: aerolex_voyage (primary)
        """
        self.collection = collection
        self.client     = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=30,
        )
        logger.info(
            f"QdrantStore initialized | "
            f"Collection: {collection} | "
            f"Host: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
        )

    # ── Chunk ID → Point ID conversion ───────────────────────────────────────

    def _chunk_id_to_point_id(self, chunk_id: str) -> str:
        """
        Convert chunk_id string to UUID for Qdrant.

        Qdrant requires UUID or integer as point ID.
        We use UUID5 (deterministic) from chunk_id string.
        Same chunk_id → always same UUID → enables upsert (no duplicates).

        Args:
            chunk_id: String chunk ID from chunker

        Returns:
            str: UUID string
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    # ── Store Chunks ──────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        embedded_chunks: list[dict],
        batch_size: int = 100
    ) -> dict:
        """
        Store embedded chunks in Qdrant.

        Uses UPSERT (not insert) — safe to run multiple times.
        Same chunk_id = same point_id = old vector replaced.

        Args:
            embedded_chunks: List of EmbeddedChunk dicts
            batch_size: Points per upsert call

        Returns:
            dict: Upload stats
        """
        stats = {"uploaded": 0, "failed": 0, "batches": 0}

        logger.info(
            f"Starting upsert | "
            f"Chunks: {len(embedded_chunks)} | "
            f"Collection: {self.collection}"
        )

        for i in range(0, len(embedded_chunks), batch_size):
            batch     = embedded_chunks[i:i + batch_size]
            batch_num = i // batch_size + 1

            try:
                points = []
                for chunk in batch:
                    # Build point payload — all metadata
                    payload = {
                        "chunk_id":          chunk.get("chunk_id", ""),
                        "text":              chunk.get("text", ""),
                        "char_count":        chunk.get("char_count", 0),
                        "word_count":        chunk.get("word_count", 0),
                        "chunk_index":       chunk.get("chunk_index", 0),
                        "total_chunks":      chunk.get("total_chunks", 0),
                        "part_number":       chunk.get("part_number", ""),
                        "part_title":        chunk.get("part_title", ""),
                        "subpart":           chunk.get("subpart", ""),
                        "subpart_title":     chunk.get("subpart_title", ""),
                        "section":           chunk.get("section", ""),
                        "section_title":     chunk.get("section_title", ""),
                        "citation":          chunk.get("citation", ""),
                        "hierarchy":         chunk.get("hierarchy", ""),
                        "source":            chunk.get("source", ""),
                        "doc_type":          chunk.get("doc_type", ""),
                        "chunking_strategy": chunk.get("chunking_strategy", ""),
                        "embedding_model":   chunk.get("embedding_model", ""),
                    }

                    point = PointStruct(
                        id      = self._chunk_id_to_point_id(chunk["chunk_id"]),
                        vector  = chunk["embedding"],
                        payload = payload,
                    )
                    points.append(point)

                # Upsert batch
                self.client.upsert(
                    collection_name=self.collection,
                    points=points,
                    wait=True,  # Wait for indexing to complete
                )

                stats["uploaded"] += len(batch)
                stats["batches"]  += 1

                logger.info(
                    f"Batch {batch_num} upserted | "
                    f"Points: {len(batch)} | "
                    f"Total: {stats['uploaded']}"
                )

            except Exception as e:
                handle_exception(
                    e,
                    context=f"QdrantStore.upsert_chunks batch {batch_num}"
                )
                stats["failed"] += len(batch)
                continue

        logger.info(
            f"Upsert complete | "
            f"Uploaded: {stats['uploaded']} | "
            f"Failed: {stats['failed']}"
        )
        return stats

    # ── Dense Vector Search ───────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[dict] = None,
    ) -> list[dict]:    
    # Build filter if provided
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)

        try:
            from qdrant_client.models import Query
            results = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
                with_vectors=False,
            ).points

            formatted = []
            for r in results:
                result = dict(r.payload)
                result["score"]    = r.score
                result["point_id"] = str(r.id)
                formatted.append(result)

            logger.info(
                f"Search complete | "
                f"Results: {len(formatted)} | "
                f"Collection: {self.collection}"
            )
            return formatted

        except Exception as e:
            raise VectorStoreError(
                message="Vector search failed",
                context="QdrantStore.search()",
                original_error=e
            )

    # ── Collection Info ───────────────────────────────────────────────────────

    def get_collection_info(self) -> dict:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection)
            return {
                "collection":  self.collection,
                "points":      info.points_count or 0,
                "status":      str(info.status),
                "model":       COLLECTION_MODEL_MAP.get(self.collection, "unknown"),
            }
        except Exception as e:
            handle_exception(e, context="QdrantStore.get_collection_info()")
            return {}

    def count_points(self) -> int:
        """Count total points in collection."""
        try:
            result = self.client.count(collection_name=self.collection)
            return result.count
        except Exception as e:
            logger.warning(f"Count failed: {e}")
            return 0


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("\n--- Testing Qdrant Store ---\n")

    # Test connection
    store = QdrantStore(collection="aerolex_voyage")
    info  = store.get_collection_info()
    print(f"Collection: {info}")

    # Load embedded chunks (from Phase 2 test)
    # We'll use voyage embeddings if available, else skip
    voyage_path = Path("data/embeddings/part_91_voyage_test.json")

    if not voyage_path.exists():
        print("\n⚠️  No voyage embeddings found — creating test embeddings first...")
        print("   Run: python src/embeddings/voyage_embedder.py")
        print("   Then save output to data/embeddings/part_91_voyage_test.json")
        print("\n   For now, testing with dummy vectors...")

        # Create dummy test data
        import random
        test_chunks = []
        for i in range(5):
            test_chunks.append({
                "chunk_id":          f"test_chunk_{i}",
                "text":              f"Test regulation text {i} — pilot must verify system {i}",
                "char_count":        50,
                "word_count":        10,
                "chunk_index":       i,
                "total_chunks":      5,
                "part_number":       "91",
                "part_title":        "General Operating Rules",
                "subpart":           "A",
                "subpart_title":     "General",
                "section":           f"91.{i}",
                "section_title":     f"Test Section {i}",
                "citation":          f"14 CFR Part 91, Section 91.{i}",
                "hierarchy":         f"Title 14 > Part 91 > Section 91.{i}",
                "source":            "ecfr",
                "doc_type":          "regulation",
                "chunking_strategy": "recursive",
                "embedding_model":   "voyage-3-large",
                "embedding":         [random.uniform(-0.1, 0.1) for _ in range(1024)],
            })

        # Upsert test data
        stats = store.upsert_chunks(test_chunks)
        print(f"\n📊 Upsert Stats: {stats}")

        # Test search with random query vector
        query_vec = [random.uniform(-0.1, 0.1) for _ in range(1024)]
        results   = store.search(query_vec, top_k=3)

        print(f"\n📋 Search Results (top 3):")
        for r in results:
            print(f"   Score: {r['score']:.4f} | Citation: {r['citation']}")
            print(f"   Text: {r['text'][:80]}...")
            print()

    print(f"\n📊 Collection Info:")
    print(f"   Points: {store.count_points()}")
    print(f"   Status: {store.get_collection_info()}")

    print(f"\n✅ Qdrant Store working!")
    print(f"   Dashboard: http://localhost:6333/dashboard")