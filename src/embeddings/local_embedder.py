"""
AeroLex — Local Embedding Model (BGE-M3)

Generates embeddings locally using BAAI/bge-m3 via sentence-transformers.
Zero API cost, works offline, data never leaves your machine.

Why BGE-M3?
- State-of-the-art open source embedding model
- Multi-lingual (supports Hindi + English — useful for DGCA docs)
- 1024 dimensions — rich representation
- Comparable quality to OpenAI text-embedding-3-small
- Free forever

vs all-MiniLM-L6-v2 (used in semantic chunker):
- MiniLM: 384 dims, very fast, lower quality
- BGE-M3: 1024 dims, slower, much better quality
- For production RAG: always use BGE-M3

Model size: ~570MB (downloads once, cached locally)

Usage:
    from src.embeddings.local_embedder import LocalEmbedder
    embedder = LocalEmbedder()
    embeddings = embedder.embed_chunks(chunks)
"""

import json
import time
import mlflow
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, EmbeddingError
from config.settings import settings

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "BAAI/bge-m3"
DIMENSIONS  = 1024
BATCH_SIZE  = 32    # Smaller batch — local GPU/CPU memory constraint

# ── Output directory ──────────────────────────────────────────────────────────
EMBEDDINGS_DIR = Path("data/embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class EmbeddedChunk:
    """Chunk with local embedding — same structure as OpenAI version."""
    chunk_id:          str
    text:              str
    char_count:        int
    word_count:        int
    chunk_index:       int
    total_chunks:      int
    part_number:       str
    part_title:        str
    subpart:           str
    subpart_title:     str
    section:           str
    section_title:     str
    citation:          str
    hierarchy:         str
    source:            str
    doc_type:          str
    chunking_strategy: str
    chunk_size:        int
    chunk_overlap:     int
    embedding:         list[float]
    embedding_model:   str
    embedding_dims:    int


class LocalEmbedder:
    """
    Generates embeddings using BAAI/bge-m3 locally.

    First run downloads model (~570MB) — cached after that.
    Subsequent runs load from cache — no internet needed.
    """

    def __init__(self, batch_size: int = BATCH_SIZE):
        self.batch_size = batch_size
        self.model      = None  # Lazy load
        self.stats = {
            "chunks_embedded": 0,
            "batches":         0,
            "total_time_sec":  0.0,
            "avg_time_per_chunk": 0.0,
            "failed":          0,
        }
        logger.info(f"LocalEmbedder initialized | Model: {MODEL_NAME}")

    def _load_model(self) -> None:
        """Load BGE-M3 model — downloads if not cached."""
        if self.model is None:
            logger.info(f"Loading BGE-M3 model: {MODEL_NAME}")
            logger.info("First run: downloading ~570MB model...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(MODEL_NAME)
            logger.info(f"BGE-M3 loaded | Dimensions: {DIMENSIONS}")

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts using local model.

        Args:
            texts: List of strings to embed

        Returns:
            list: List of embedding vectors
        """
        try:
            # encode() returns numpy array
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,  # L2 normalize — better cosine similarity
            )
            # Convert numpy to Python list for JSON serialization
            return embeddings.tolist()

        except Exception as e:
            raise EmbeddingError(
                message=f"Local embedding batch failed",
                context="LocalEmbedder._embed_batch()",
                original_error=e
            )

    def embed_chunks(self, chunks: list[dict]) -> list[EmbeddedChunk]:
        """
        Embed all chunks using local BGE-M3 model.

        Args:
            chunks: List of chunk dicts

        Returns:
            list[EmbeddedChunk]: Chunks with local embeddings
        """
        self._load_model()

        logger.info(
            f"Starting local embedding | "
            f"Chunks: {len(chunks)} | "
            f"Model: {MODEL_NAME}"
        )

        start_time     = time.time()
        embedded_chunks = []

        for i in range(0, len(chunks), self.batch_size):
            batch       = chunks[i:i + self.batch_size]
            batch_texts = [c["text"] for c in batch]
            batch_num   = i // self.batch_size + 1
            total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

            logger.info(
                f"Embedding batch {batch_num}/{total_batches} | "
                f"Chunks {i+1}-{min(i+self.batch_size, len(chunks))}"
            )

            try:
                embeddings = self._embed_batch(batch_texts)
                self.stats["batches"] += 1

                for chunk, embedding in zip(batch, embeddings):
                    embedded = EmbeddedChunk(
                        chunk_id          = chunk.get("chunk_id", ""),
                        text              = chunk.get("text", ""),
                        char_count        = chunk.get("char_count", 0),
                        word_count        = chunk.get("word_count", 0),
                        chunk_index       = chunk.get("chunk_index", 0),
                        total_chunks      = chunk.get("total_chunks", 0),
                        part_number       = chunk.get("part_number", ""),
                        part_title        = chunk.get("part_title", ""),
                        subpart           = chunk.get("subpart", ""),
                        subpart_title     = chunk.get("subpart_title", ""),
                        section           = chunk.get("section", ""),
                        section_title     = chunk.get("section_title", ""),
                        citation          = chunk.get("citation", ""),
                        hierarchy         = chunk.get("hierarchy", ""),
                        source            = chunk.get("source", "ecfr"),
                        doc_type          = chunk.get("doc_type", "regulation"),
                        chunking_strategy = chunk.get("chunking_strategy", ""),
                        chunk_size        = chunk.get("chunk_size", 0),
                        chunk_overlap     = chunk.get("chunk_overlap", 0),
                        embedding         = embedding,
                        embedding_model   = MODEL_NAME,
                        embedding_dims    = len(embedding),
                    )
                    embedded_chunks.append(embedded)
                    self.stats["chunks_embedded"] += 1

            except EmbeddingError as e:
                handle_exception(e, context=f"LocalEmbedder batch {batch_num}")
                self.stats["failed"] += len(batch)
                continue

        # Calculate timing stats
        total_time = time.time() - start_time
        self.stats["total_time_sec"]     = round(total_time, 2)
        self.stats["avg_time_per_chunk"] = round(
            total_time / max(self.stats["chunks_embedded"], 1), 4
        )

        logger.info(
            f"Local embedding complete | "
            f"Embedded: {self.stats['chunks_embedded']} | "
            f"Time: {self.stats['total_time_sec']}s | "
            f"Avg: {self.stats['avg_time_per_chunk']}s/chunk"
        )
        return embedded_chunks

    def save_embeddings(
        self,
        embedded_chunks: list[EmbeddedChunk],
        output_path: str
    ) -> None:
        """Save embeddings to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in embedded_chunks]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Embeddings saved | Path: {path} | Size: {size_mb:.1f}MB")

    def log_to_mlflow(
        self,
        experiment_name: str = "aerolex-embedding-experiments"
    ) -> None:
        """Log local embedding experiment to MLflow."""
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"local_bge_m3"):
                mlflow.log_params({
                    "model":      MODEL_NAME,
                    "dimensions": DIMENSIONS,
                    "batch_size": self.batch_size,
                    "provider":   "local_sentence_transformers",
                    "cost_usd":   0.0,
                })
                mlflow.log_metrics({
                    "chunks_embedded":    self.stats["chunks_embedded"],
                    "total_time_sec":     self.stats["total_time_sec"],
                    "avg_time_per_chunk": self.stats["avg_time_per_chunk"],
                    "cost_usd":           0.0,
                    "failed":             self.stats["failed"],
                })
                logger.info("Local embedding experiment logged to MLflow")

        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("\n--- Testing Local BGE-M3 Embedder ---")
    print("⚠️  First run downloads BGE-M3 model (~570MB)")
    print("⚠️  Zero API cost — runs completely locally\n")

    with open(
        "data/processed/chunks/part_91_recursive_512.json",
        encoding="utf-8"
    ) as f:
        all_chunks = json.load(f)

    # Test with first 10 chunks
    test_chunks = all_chunks[:10]
    print(f"Testing with {len(test_chunks)} chunks\n")

    embedder = LocalEmbedder()
    embedded = embedder.embed_chunks(test_chunks)

    print(f"\n📊 Local Embedding Stats:")
    print(f"   Chunks embedded    : {embedder.stats['chunks_embedded']}")
    print(f"   Total time         : {embedder.stats['total_time_sec']}s")
    print(f"   Avg per chunk      : {embedder.stats['avg_time_per_chunk']}s")
    print(f"   Cost               : $0.00 (FREE!)")

    print(f"\n📋 Sample Embedded Chunk:")
    if embedded:
        e = embedded[0]
        print(f"   Citation  : {e.citation}")
        print(f"   Model     : {e.embedding_model}")
        print(f"   Dims      : {e.embedding_dims}")
        print(f"   Vector[0:5]: {[round(v,4) for v in e.embedding[:5]]}")

    embedder.log_to_mlflow()

    print(f"\n📊 COMPARISON — OpenAI vs Local BGE-M3:")
    print(f"   OpenAI  : 1536 dims | $0.000018 for 10 chunks | ~4s")
    print(f"   BGE-M3  : {DIMENSIONS} dims | $0.00 | {embedder.stats['total_time_sec']}s")

    print(f"\n✅ Local BGE-M3 Embedder working!")