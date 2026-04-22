"""
AeroLex — OpenAI Embedding Model

Generates vector embeddings using OpenAI's text-embedding-3-small model.

Why text-embedding-3-small?
- Best quality/cost ratio in OpenAI's lineup
- 1536 dimensions — rich semantic representation
- $0.02 per 1M tokens — very cheap
- For entire Part 91 corpus (~500K tokens) = ~$0.01

vs text-embedding-3-large:
- 3072 dimensions — higher quality
- $0.13 per 1M tokens — 6.5x more expensive
- Marginal quality gain for our use case

vs text-embedding-ada-002 (older):
- 1536 dimensions
- $0.10 per 1M tokens — 5x more expensive
- Worse quality than 3-small

Usage:
    from src.embeddings.openai_embedder import OpenAIEmbedder
    embedder = OpenAIEmbedder()
    embeddings = embedder.embed_chunks(chunks)
"""

import json
import time
import mlflow
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from openai import OpenAI
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, EmbeddingError
from src.monitoring.cost_tracker import CostTracker
from config.settings import settings

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "text-embedding-3-small"
DIMENSIONS   = 1536
BATCH_SIZE   = 100   # Embed 100 chunks per API call — reduces API calls
MAX_RETRIES  = 3     # Retry on API errors
RETRY_DELAY  = 2     # Seconds between retries

# ── Output directory ──────────────────────────────────────────────────────────
EMBEDDINGS_DIR = Path("data/embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class EmbeddedChunk:
    """A chunk with its vector embedding attached."""
    # All chunk fields
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

    # Embedding fields
    embedding:         list[float]  # The actual vector
    embedding_model:   str          # Model used
    embedding_dims:    int          # Vector dimensions


class OpenAIEmbedder:
    """
    Generates embeddings using OpenAI text-embedding-3-small.

    Batch processing strategy:
    - Groups chunks into batches of 100
    - Single API call per batch = fewer requests = faster + cheaper
    - Retries on failure with exponential backoff
    """

    def __init__(self):
        self.client       = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.cost_tracker = CostTracker()
        self.stats = {
            "chunks_embedded": 0,
            "batches":         0,
            "total_tokens":    0,
            "total_cost_usd":  0.0,
            "failed":          0,
        }
        logger.info(f"OpenAIEmbedder initialized | Model: {MODEL_NAME}")

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts in one API call.

        Args:
            texts: List of text strings to embed

        Returns:
            list: List of embedding vectors
        """
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.embeddings.create(
                    model=MODEL_NAME,
                    input=texts,
                    dimensions=DIMENSIONS
                )

                # Track tokens used
                tokens_used = response.usage.total_tokens
                cost = (tokens_used / 1_000_000) * 0.02  # $0.02 per 1M tokens

                self.stats["total_tokens"] += tokens_used
                self.stats["total_cost_usd"] += cost

                logger.debug(
                    f"Batch embedded | Texts: {len(texts)} | "
                    f"Tokens: {tokens_used} | Cost: ${cost:.6f}"
                )

                # Extract embeddings in order
                embeddings = [item.embedding for item in response.data]
                return embeddings

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Embedding batch failed (attempt {attempt+1}) — "
                        f"retrying in {wait}s: {e}"
                    )
                    time.sleep(wait)
                else:
                    raise EmbeddingError(
                        message=f"Failed to embed batch after {MAX_RETRIES} attempts",
                        context="OpenAIEmbedder._embed_batch()",
                        original_error=e
                    )

    def embed_chunks(self, chunks: list[dict]) -> list[EmbeddedChunk]:
        """
        Embed all chunks in batches.

        Args:
            chunks: List of chunk dicts (from chunker output)

        Returns:
            list[EmbeddedChunk]: Chunks with embeddings attached
        """
        logger.info(
            f"Starting OpenAI embedding | "
            f"Chunks: {len(chunks)} | "
            f"Batch size: {BATCH_SIZE}"
        )

        embedded_chunks = []

        # Process in batches
        for i in range(0, len(chunks), BATCH_SIZE):
            batch        = chunks[i:i + BATCH_SIZE]
            batch_texts  = [c["text"] for c in batch]
            batch_num    = i // BATCH_SIZE + 1
            total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

            logger.info(
                f"Embedding batch {batch_num}/{total_batches} | "
                f"Chunks {i+1}-{min(i+BATCH_SIZE, len(chunks))}"
            )

            try:
                embeddings = self._embed_batch(batch_texts)
                self.stats["batches"] += 1

                for chunk, embedding in zip(batch, embeddings):
                    embedded = EmbeddedChunk(
                        # Copy all chunk fields
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

                        # Embedding fields
                        embedding         = embedding,
                        embedding_model   = MODEL_NAME,
                        embedding_dims    = DIMENSIONS,
                    )
                    embedded_chunks.append(embedded)
                    self.stats["chunks_embedded"] += 1

            except EmbeddingError as e:
                handle_exception(e, context=f"OpenAIEmbedder batch {batch_num}")
                self.stats["failed"] += len(batch)
                continue

            # Small delay between batches — avoid rate limits
            time.sleep(0.1)

        logger.info(
            f"OpenAI embedding complete | "
            f"Embedded: {self.stats['chunks_embedded']} | "
            f"Tokens: {self.stats['total_tokens']:,} | "
            f"Cost: ${self.stats['total_cost_usd']:.4f}"
        )
        return embedded_chunks

    def save_embeddings(
        self,
        embedded_chunks: list[EmbeddedChunk],
        output_path: str
    ) -> None:
        """Save embedded chunks to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Note: embeddings are large — JSON not ideal for production
        # Phase 3 mein Qdrant mein store karenge
        data = [asdict(e) for e in embedded_chunks]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(
            f"Embeddings saved | Path: {path} | "
            f"Size: {size_mb:.1f}MB"
        )

    def log_to_mlflow(
        self,
        experiment_name: str = "aerolex-embedding-experiments"
    ) -> None:
        """Log embedding experiment to MLflow."""
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"openai_{MODEL_NAME}"):
                mlflow.log_params({
                    "model":       MODEL_NAME,
                    "dimensions":  DIMENSIONS,
                    "batch_size":  BATCH_SIZE,
                    "provider":    "openai",
                })
                mlflow.log_metrics({
                    "chunks_embedded": self.stats["chunks_embedded"],
                    "total_tokens":    self.stats["total_tokens"],
                    "total_cost_usd":  self.stats["total_cost_usd"],
                    "batches":         self.stats["batches"],
                    "failed":          self.stats["failed"],
                })
                logger.info("Embedding experiment logged to MLflow")

        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("\n--- Testing OpenAI Embedder ---")
    print("⚠️  This makes real OpenAI API calls — costs ~$0.001\n")

    # Load chunks — use small set for testing
    with open(
        "data/processed/chunks/part_91_recursive_512.json",
        encoding="utf-8"
    ) as f:
        all_chunks = json.load(f)

    # Test with first 10 chunks only
    test_chunks = all_chunks[:10]
    print(f"Testing with {len(test_chunks)} chunks\n")

    embedder = OpenAIEmbedder()
    embedded = embedder.embed_chunks(test_chunks)

    print(f"\n📊 Embedding Stats:")
    print(f"   Chunks embedded : {embedder.stats['chunks_embedded']}")
    print(f"   Total tokens    : {embedder.stats['total_tokens']:,}")
    print(f"   Total cost      : ${embedder.stats['total_cost_usd']:.6f}")
    print(f"   Batches         : {embedder.stats['batches']}")

    print(f"\n📋 Sample Embedded Chunk:")
    if embedded:
        e = embedded[0]
        print(f"   Citation  : {e.citation}")
        print(f"   Model     : {e.embedding_model}")
        print(f"   Dims      : {e.embedding_dims}")
        print(f"   Vector[0:5]: {e.embedding[:5]}")

    # Log to MLflow
    embedder.log_to_mlflow()

    print(f"\n✅ OpenAI Embedder working!")