"""
AeroLex — Voyage AI Embedding Model

Voyage AI is specifically designed for RAG applications.
Anthropic officially recommends Voyage AI embeddings for use with Claude.

Why Voyage AI for RAG?
- Trained specifically on retrieval tasks (not general purpose)
- voyage-3-large outperforms OpenAI text-embedding-3-large on MTEB retrieval
- Lower cost than OpenAI 3-large while matching/exceeding quality
- voyage-code-2 and voyage-law-2 specialized models available

Model Comparison:
    voyage-3-large:  1024 dims | $0.06/1M | Best quality
    voyage-3:        1024 dims | $0.06/1M | Balanced
    voyage-3-lite:   512 dims  | $0.02/1M | Fast + cheap
    voyage-law-2:    1024 dims | $0.12/1M | Legal docs (regulations!)

Why voyage-3-large for AeroLex?
- Aviation regulations = legal/technical text
- voyage-3-large trained on legal, technical, scientific corpora
- Best MTEB retrieval scores in its price range

Usage:
    from src.embeddings.voyage_embedder import VoyageEmbedder
    embedder = VoyageEmbedder()
    embedded = embedder.embed_chunks(chunks)
"""

import json
import time
import mlflow
from pathlib import Path
from dataclasses import dataclass, asdict
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, EmbeddingError
from config.settings import settings

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "voyage-3-large"
DIMENSIONS  = 1024
BATCH_SIZE  = 128    # Voyage allows larger batches than OpenAI
MAX_RETRIES = 3
RETRY_DELAY = 2

# ── Output directory ──────────────────────────────────────────────────────────
EMBEDDINGS_DIR = Path("data/embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class EmbeddedChunk:
    """Chunk with Voyage AI embedding."""
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


class VoyageEmbedder:
    """
    Generates embeddings using Voyage AI voyage-3-large.

    Key difference from OpenAI:
    - input_type parameter: "document" for indexing, "query" for search
    - This asymmetric embedding is specifically designed for RAG
    - Documents embedded differently from queries = better retrieval

    Why asymmetric embeddings matter:
    - Documents: longer, detailed, full context
    - Queries: shorter, question-like
    - Training them differently = better matching
    """

    def __init__(self, model: str = MODEL_NAME):
        import voyageai
        self.client = voyageai.Client(api_key=settings.VOYAGE_API_KEY)
        self.model  = model
        self.stats  = {
            "chunks_embedded": 0,
            "batches":         0,
            "total_tokens":    0,
            "total_cost_usd":  0.0,
            "failed":          0,
        }
        logger.info(f"VoyageEmbedder initialized | Model: {model}")

    def _embed_batch(
        self,
        texts: list[str],
        input_type: str = "document"
    ) -> list[list[float]]:
        """
        Embed a batch using Voyage AI.

        Args:
            texts: List of texts to embed
            input_type: "document" for indexing, "query" for search queries
                        This asymmetric approach improves retrieval quality!

        Returns:
            list: Embedding vectors
        """
        for attempt in range(MAX_RETRIES):
            try:
                result = self.client.embed(
                    texts,
                    model=self.model,
                    input_type=input_type,  # KEY DIFFERENCE from OpenAI!
                )

                # Track usage
                tokens_used = result.total_tokens
                # voyage-3-large = $0.06 per 1M tokens
                cost = (tokens_used / 1_000_000) * 0.06

                self.stats["total_tokens"] += tokens_used
                self.stats["total_cost_usd"] += cost

                logger.debug(
                    f"Voyage batch embedded | Texts: {len(texts)} | "
                    f"Tokens: {tokens_used} | Cost: ${cost:.6f}"
                )

                return result.embeddings

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Voyage batch failed (attempt {attempt+1}) — retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise EmbeddingError(
                        message=f"Voyage embedding failed after {MAX_RETRIES} attempts",
                        context="VoyageEmbedder._embed_batch()",
                        original_error=e
                    )

    def embed_chunks(
        self,
        chunks: list[dict],
        input_type: str = "document"
    ) -> list[EmbeddedChunk]:
        """
        Embed all chunks using Voyage AI.

        Args:
            chunks: List of chunk dicts
            input_type: "document" for storing, "query" for search

        Returns:
            list[EmbeddedChunk]: Chunks with Voyage embeddings
        """
        logger.info(
            f"Starting Voyage embedding | "
            f"Chunks: {len(chunks)} | "
            f"Model: {self.model} | "
            f"input_type: {input_type}"
        )

        embedded_chunks = []

        for i in range(0, len(chunks), BATCH_SIZE):
            batch       = chunks[i:i + BATCH_SIZE]
            batch_texts = [c["text"] for c in batch]
            batch_num   = i // BATCH_SIZE + 1
            total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

            logger.info(
                f"Embedding batch {batch_num}/{total_batches} | "
                f"Chunks {i+1}-{min(i+BATCH_SIZE, len(chunks))}"
            )

            try:
                embeddings = self._embed_batch(batch_texts, input_type)
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
                        embedding_model   = self.model,
                        embedding_dims    = len(embedding),
                    )
                    embedded_chunks.append(embedded)
                    self.stats["chunks_embedded"] += 1

            except EmbeddingError as e:
                handle_exception(e, context=f"VoyageEmbedder batch {batch_num}")
                self.stats["failed"] += len(batch)
                continue

            time.sleep(0.1)

        logger.info(
            f"Voyage embedding complete | "
            f"Embedded: {self.stats['chunks_embedded']} | "
            f"Tokens: {self.stats['total_tokens']:,} | "
            f"Cost: ${self.stats['total_cost_usd']:.6f}"
        )
        return embedded_chunks

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single search query.
        Uses input_type='query' — different from document embedding!

        Args:
            query: User's search query

        Returns:
            list[float]: Query embedding vector
        """
        result = self.client.embed(
            [query],
            model=self.model,
            input_type="query",  # Asymmetric — query vs document
        )
        logger.debug(f"Query embedded | Model: {self.model}")
        return result.embeddings[0]

    def save_embeddings(self, embedded_chunks: list[EmbeddedChunk], output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in embedded_chunks], f, ensure_ascii=False)
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Embeddings saved | Path: {path} | Size: {size_mb:.1f}MB")

    def log_to_mlflow(
        self,
        experiment_name: str = "aerolex-embedding-experiments"
    ) -> None:
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"voyage_{self.model}"):
                mlflow.log_params({
                    "model":       self.model,
                    "dimensions":  DIMENSIONS,
                    "batch_size":  BATCH_SIZE,
                    "provider":    "voyage_ai",
                    "input_type":  "document",
                    "asymmetric":  True,
                })
                mlflow.log_metrics({
                    "chunks_embedded": self.stats["chunks_embedded"],
                    "total_tokens":    self.stats["total_tokens"],
                    "total_cost_usd":  self.stats["total_cost_usd"],
                    "batches":         self.stats["batches"],
                    "failed":          self.stats["failed"],
                })
                logger.info("Voyage embedding experiment logged to MLflow")

        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("\n--- Testing Voyage AI Embedder ---")
    print("⚠️  This makes real Voyage AI API calls — costs ~$0.0001\n")

    with open(
        "data/processed/chunks/part_91_recursive_512.json",
        encoding="utf-8"
    ) as f:
        all_chunks = json.load(f)

    # Test with first 10 chunks
    test_chunks = all_chunks[:10]
    print(f"Testing with {len(test_chunks)} chunks\n")

    embedder = VoyageEmbedder()

    # Embed as documents (for indexing)
    embedded = embedder.embed_chunks(test_chunks, input_type="document")

    print(f"\n📊 Voyage Embedding Stats:")
    print(f"   Chunks embedded : {embedder.stats['chunks_embedded']}")
    print(f"   Total tokens    : {embedder.stats['total_tokens']:,}")
    print(f"   Total cost      : ${embedder.stats['total_cost_usd']:.6f}")
    print(f"   Batches         : {embedder.stats['batches']}")

    # Test query embedding
    print(f"\n🔍 Testing query embedding (asymmetric)...")
    query_vec = embedder.embed_query("What are APU MEL requirements for Boeing 787?")
    print(f"   Query vector dims : {len(query_vec)}")
    print(f"   Query vector[0:5] : {[round(v,4) for v in query_vec[:5]]}")

    print(f"\n📋 Sample Embedded Chunk:")
    if embedded:
        e = embedded[0]
        print(f"   Citation  : {e.citation}")
        print(f"   Model     : {e.embedding_model}")
        print(f"   Dims      : {e.embedding_dims}")
        print(f"   Vector[0:5]: {[round(v,4) for v in e.embedding[:5]]}")

    embedder.log_to_mlflow()

    print(f"\n📊 FULL COMPARISON — All 3 Models:")
    print(f"   OpenAI 3-small : 1536 dims | $0.000018/10 chunks | ~4s")
    print(f"   BGE-M3 Local   : 1024 dims | $0.00       | 6.87s")
    print(f"   Voyage-3-large : {len(embedded[0].embedding)} dims  | ${embedder.stats['total_cost_usd']:.6f}/10 chunks | asymmetric RAG-optimized")

    print(f"\n✅ Voyage AI Embedder working!")