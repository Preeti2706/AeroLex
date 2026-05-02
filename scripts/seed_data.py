"""
AeroLex — Seed Data Script

Embeds Part 91 chunks using Voyage AI and stores in Qdrant.
This is the FIRST real data going into our vector database!

Pipeline:
1. Load parsed Part 91 sections (from Phase 1)
2. Chunk using Hierarchical strategy (best for regulations)
3. Embed using Voyage AI (best RAG quality)
4. Store in Qdrant aerolex_voyage collection

Run this once to populate the vector database.
After this, real RAG queries will work!
"""

import json
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception
from src.chunking.hierarchical_chunker import HierarchicalChunker
from src.chunking.recursive_chunker import RecursiveChunker
from src.embeddings.voyage_embedder import VoyageEmbedder
from src.retrieval.qdrant_store import QdrantStore
from dataclasses import asdict

logger = get_logger(__name__)


def seed_ecfr_part91():
    """
    Seed Part 91 data into Qdrant.
    Uses hierarchical chunking + Voyage AI embeddings.
    """
    print("\n" + "="*60)
    print("🌱 AeroLex — Seeding Part 91 into Qdrant")
    print("="*60)

    # ── Step 1: Load parsed sections ─────────────────────────────
    parsed_path = Path("data/processed/part_91_parsed.json")
    if not parsed_path.exists():
        print("❌ Part 91 parsed data not found!")
        print("   Run: python src/parsing/xml_parser.py")
        return

    with open(parsed_path, encoding="utf-8") as f:
        sections = json.load(f)

    print(f"\n✅ Step 1: Loaded {len(sections)} sections from Part 91")

    # ── Step 2: Chunk ─────────────────────────────────────────────
    print(f"\n⚙️  Step 2: Chunking with Hierarchical strategy...")
    chunker = HierarchicalChunker()
    chunks  = chunker.chunk_sections(sections)
    print(f"✅ {len(chunks)} chunks created")

    # Convert to dicts for embedder
    chunk_dicts = [asdict(c) for c in chunks]

    # ── Step 3: Embed with Voyage AI ──────────────────────────────
    print(f"\n⚙️  Step 3: Embedding with Voyage AI voyage-3-large...")
    print(f"   Estimated cost: ${len(chunk_dicts) * 0.00000006:.4f}")
    print(f"   Estimated time: ~{len(chunk_dicts) // 100 + 1} seconds")

    embedder = VoyageEmbedder()
    embedded = embedder.embed_chunks(chunk_dicts, input_type="document")

    print(f"✅ {len(embedded)} chunks embedded")
    print(f"   Total tokens : {embedder.stats['total_tokens']:,}")
    print(f"   Total cost   : ${embedder.stats['total_cost_usd']:.4f}")

    # Convert to dicts for Qdrant
    embedded_dicts = [asdict(e) for e in embedded]

    # ── Step 4: Store in Qdrant ───────────────────────────────────
    print(f"\n⚙️  Step 4: Storing in Qdrant aerolex_voyage collection...")

    store = QdrantStore(collection="aerolex_voyage")
    stats = store.upsert_chunks(embedded_dicts, batch_size=100)

    print(f"✅ Qdrant upsert complete!")
    print(f"   Uploaded : {stats['uploaded']}")
    print(f"   Failed   : {stats['failed']}")

    # ── Step 5: Verify ────────────────────────────────────────────
    total_points = store.count_points()
    print(f"\n✅ Step 5: Verification")
    print(f"   Total points in Qdrant: {total_points:,}")

    print(f"\n{'='*60}")
    print(f"🎉 Part 91 seeded successfully!")
    print(f"   Collection : aerolex_voyage")
    print(f"   Points     : {total_points:,}")
    print(f"   Model      : voyage-3-large")
    print(f"   Dashboard  : http://localhost:6333/dashboard")
    print(f"{'='*60}\n")

    return embedded_dicts


if __name__ == "__main__":
    seed_ecfr_part91()