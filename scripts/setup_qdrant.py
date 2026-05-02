"""
AeroLex — Qdrant Collection Setup

Creates vector collections in Qdrant for AeroLex.

What is a Collection?
- Qdrant mein 'collection' = database mein 'table' jaisa
- Har collection ek specific vector size ke liye hoti hai
- AeroLex mein alag alag embedding models ke liye alag collections

Collections we create:
1. aerolex_voyage    — voyage-3-large vectors (1024 dims) [PRIMARY]
2. aerolex_openai    — text-embedding-3-small (1536 dims)
3. aerolex_e5        — e5-large-v2 (1024 dims) [FALLBACK]
4. aerolex_bge       — bge-m3 (1024 dims) [MULTILINGUAL]

Why separate collections per model?
- Different models = different vector dimensions
- Cannot mix 1024-dim and 1536-dim in same collection
- Allows A/B testing between models in production

HNSW Index — What is it?
- HNSW = Hierarchical Navigable Small World
- Algorithm for approximate nearest neighbor search
- Makes vector search fast even with millions of vectors
- Trade-off: m and ef_construct parameters control speed vs accuracy

Official Docs: https://qdrant.tech/documentation/concepts/collections/
HNSW Paper: https://arxiv.org/abs/1603.09320
"""

from distro import info
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
)
from streamlit import status
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, VectorStoreError
from config.settings import settings

logger = get_logger(__name__)

# ── Collection Definitions ────────────────────────────────────────────────────
COLLECTIONS = {
    "aerolex_voyage": {
        "size":        1024,
        "distance":    Distance.COSINE,
        "description": "voyage-3-large — PRIMARY production collection",
        "model":       "voyage-3-large",
    },
    "aerolex_openai": {
        "size":        1536,
        "distance":    Distance.COSINE,
        "description": "text-embedding-3-small — highest dims",
        "model":       "text-embedding-3-small",
    },
    "aerolex_e5": {
        "size":        1024,
        "distance":    Distance.COSINE,
        "description": "e5-large-v2 — free asymmetric fallback",
        "model":       "intfloat/e5-large-v2",
    },
    "aerolex_bge": {
        "size":        1024,
        "distance":    Distance.COSINE,
        "description": "BAAI/bge-m3 — multilingual (DGCA Hindi docs)",
        "model":       "BAAI/bge-m3",
    },
}

# ── HNSW Config ───────────────────────────────────────────────────────────────
# HNSW = Hierarchical Navigable Small World graph
# m: number of edges per node — higher = better accuracy, more memory
# ef_construct: search depth during indexing — higher = better index, slower build
HNSW_CONFIG = HnswConfigDiff(
    m=16,              # 16 edges per node — good balance
    ef_construct=100,  # 100 candidates during build — good accuracy
)

# ── Optimizer Config ──────────────────────────────────────────────────────────
OPTIMIZER_CONFIG = OptimizersConfigDiff(
    indexing_threshold=10000,  # Start building index after 10K vectors
    memmap_threshold=50000,    # Use memory-mapped files after 50K vectors
)


def create_collection(
    client: QdrantClient,
    name: str,
    config: dict
) -> bool:
    """
    Create a single Qdrant collection.

    Args:
        client: Qdrant client
        name: Collection name
        config: Collection config dict

    Returns:
        bool: True if created/exists
    """
    try:
        # Check if already exists
        existing = [c.name for c in client.get_collections().collections]

        if name in existing:
            logger.info(f"Collection already exists: {name} — skipping")
            return True

        # Create collection
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=config["size"],
                distance=config["distance"],
                # Store vectors on disk for large collections
                on_disk=False,  # In memory for dev — True for production
            ),
            hnsw_config=HNSW_CONFIG,
            optimizers_config=OPTIMIZER_CONFIG,
        )

        logger.info(
            f"Collection created: {name} | "
            f"Dims: {config['size']} | "
            f"Distance: {config['distance']} | "
            f"Model: {config['model']}"
        )
        return True

    except Exception as e:
        raise VectorStoreError(
            message=f"Failed to create collection: {name}",
            context="setup_qdrant.create_collection()",
            original_error=e
        )


def setup_qdrant() -> bool:
    """
    Setup all AeroLex Qdrant collections.

    Returns:
        bool: True if all collections ready
    """
    logger.info(f"Connecting to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")

    try:
        client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=30,
        )

        # Verify connection
        client.get_collections()
        logger.info("Qdrant connection verified!")

        # Create all collections
        success_count = 0
        for name, config in COLLECTIONS.items():
            try:
                success = create_collection(client, name, config)
                if success:
                    success_count += 1
            except VectorStoreError as e:
                handle_exception(e, context=f"setup_qdrant — {name}")
                continue

        logger.info(
            f"Qdrant setup complete | "
            f"Collections ready: {success_count}/{len(COLLECTIONS)}"
        )
        return success_count == len(COLLECTIONS)

    except Exception as e:
        raise VectorStoreError(
            message="Failed to connect to Qdrant",
            context="setup_qdrant.setup_qdrant()",
            original_error=e
        )


def verify_collections(client: QdrantClient) -> None:
    """Print all collection details."""
    collections = client.get_collections().collections
    print(f"\n{'='*60}")
    print(f"QDRANT COLLECTIONS ({len(collections)} total)")
    print(f"{'='*60}")
    for col in collections:
        info = client.get_collection(col.name)
        print(f"\n  📦 {col.name}")
        points = info.points_count if hasattr(info, 'points_count') and info.points_count else 0
        status = info.status if hasattr(info, 'status') else 'unknown'
        print(f"     Points   : {points:,}")
        print(f"     Status   : {status}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("\n--- AeroLex Qdrant Setup ---\n")

    try:
        success = setup_qdrant()

        if success:
            client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
            )
            verify_collections(client)
            print("✅ Qdrant setup complete!")
            print(f"   Dashboard: http://localhost:6333/dashboard")
        else:
            print("❌ Some collections failed — check logs")

    except VectorStoreError as e:
        print(f"❌ Qdrant setup failed: {e}")
        print("   Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")