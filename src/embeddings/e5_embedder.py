"""
AeroLex — E5-Large-v2 Local Embedding Model

Microsoft Research ka E5 (text Embeddings from bidirEctional
Encoder representations) model. Free, local, English-optimized.

Official paper: https://arxiv.org/abs/2212.03533
HuggingFace:    https://huggingface.co/intfloat/e5-large-v2

How E5 is different from other models:
- Uses ASYMMETRIC prefixes (like Voyage AI but FREE):
  Documents: prepend "passage: " before text
  Queries:   prepend "query: "   before text
- This prefix tells the model HOW to embed the text
- Trained specifically on retrieval tasks using contrastive learning

Mathematics behind E5:
- Based on BERT architecture (bidirectional transformer)
- Fine-tuned using contrastive loss:
  Loss = -log(sim(q, d+) / sum(sim(q, d-)))
  Where: q=query, d+=relevant doc, d-=irrelevant docs
  Meaning: push query closer to relevant docs, away from irrelevant
- 1024 dimensions — same as BGE-M3 and Voyage

Why use E5 alongside BGE-M3?
- E5 = English-optimized (aviation regs are English)
- BGE-M3 = multi-lingual (future Hindi/DGCA support)
- E5 is 1.74x smaller than BGE-M3 (1.3GB vs 2.27GB)
- Different training = different strengths = good comparison

MTEB Retrieval Score (higher = better):
- text-embedding-3-large:  54.9
- voyage-3-large:          54.1  
- E5-large-v2:             50.6
- BGE-M3:                  48.8
- text-embedding-3-small:  44.9

Usage:
    from src.embeddings.e5_embedder import E5Embedder
    embedder = E5Embedder()
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
MODEL_NAME  = "intfloat/e5-large-v2"
DIMENSIONS  = 1024
BATCH_SIZE  = 32

# ── E5 Prefix Convention — CRITICAL ──────────────────────────────────────────
# E5 model REQUIRES these prefixes for correct operation
# Without prefix → model performs significantly worse
DOCUMENT_PREFIX = "passage: "   # For indexing chunks
QUERY_PREFIX    = "query: "     # For search queries

EMBEDDINGS_DIR = Path("data/embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class EmbeddedChunk:
    """Chunk with E5 embedding."""
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


class E5Embedder:
    """
    Generates embeddings using intfloat/e5-large-v2 locally.

    Key concept — Asymmetric Embedding with Prefixes:

    Normal embedders (OpenAI, BGE-M3):
        embed("pilot must verify checklist") → vector
        embed("what must pilot do?")         → different vector
        These are in same space but trained symmetrically

    E5 (asymmetric):
        embed("passage: pilot must verify checklist") → document vector
        embed("query: what must pilot do?")           → query vector
        Trained so query vectors align with relevant passage vectors
        = Better retrieval quality!

    This is the same concept as Voyage AI's input_type="document"/"query"
    but completely FREE and local.
    """

    def __init__(self, batch_size: int = BATCH_SIZE):
        self.batch_size = batch_size
        self.model      = None  # Lazy load
        self.stats = {
            "chunks_embedded":    0,
            "batches":            0,
            "total_time_sec":     0.0,
            "avg_time_per_chunk": 0.0,
            "failed":             0,
        }
        logger.info(f"E5Embedder initialized | Model: {MODEL_NAME}")

    def _load_model(self) -> None:
        """
        Lazy load E5 model from HuggingFace.
        Downloads ~1.3GB on first run, cached after.
        """
        if self.model is None:
            logger.info(f"Loading E5 model: {MODEL_NAME}")
            logger.info("First run: downloading ~1.3GB model...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(MODEL_NAME)
            logger.info(f"E5 loaded | Dimensions: {DIMENSIONS}")

    def _add_prefix(self, texts: list[str], prefix: str) -> list[str]:
        """
        Add E5 required prefix to all texts.

        E5 was trained with these prefixes — they are NOT optional.
        The model learned different representations for:
        - "passage: ..." → how to encode a document for storage
        - "query: ..."   → how to encode a question for search

        Args:
            texts: Raw texts
            prefix: "passage: " or "query: "

        Returns:
            list: Texts with prefix prepended
        """
        return [f"{prefix}{text}" for text in texts]

    def _embed_batch(
        self,
        texts: list[str],
        is_query: bool = False
    ) -> list[list[float]]:
        """
        Embed a batch with appropriate prefix.

        Args:
            texts: Raw texts (prefix added internally)
            is_query: True for queries, False for documents

        Returns:
            list: Embedding vectors
        """
        try:
            prefix = QUERY_PREFIX if is_query else DOCUMENT_PREFIX
            prefixed_texts = self._add_prefix(texts, prefix)

            embeddings = self.model.encode(
                prefixed_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,  # L2 normalize for cosine similarity
            )
            return embeddings.tolist()

        except Exception as e:
            raise EmbeddingError(
                message="E5 embedding batch failed",
                context="E5Embedder._embed_batch()",
                original_error=e
            )

    def embed_chunks(self, chunks: list[dict]) -> list[EmbeddedChunk]:
        """
        Embed all chunks as documents (passage prefix).

        Args:
            chunks: List of chunk dicts

        Returns:
            list[EmbeddedChunk]: Chunks with E5 embeddings
        """
        self._load_model()

        logger.info(
            f"Starting E5 embedding | "
            f"Chunks: {len(chunks)} | "
            f"Prefix: '{DOCUMENT_PREFIX}'"
        )

        start_time      = time.time()
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
                embeddings = self._embed_batch(batch_texts, is_query=False)
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
                handle_exception(e, context=f"E5Embedder batch {batch_num}")
                self.stats["failed"] += len(batch)
                continue

        total_time = time.time() - start_time
        self.stats["total_time_sec"]     = round(total_time, 2)
        self.stats["avg_time_per_chunk"] = round(
            total_time / max(self.stats["chunks_embedded"], 1), 4
        )

        logger.info(
            f"E5 embedding complete | "
            f"Embedded: {self.stats['chunks_embedded']} | "
            f"Time: {self.stats['total_time_sec']}s"
        )
        return embedded_chunks

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single search query with query prefix.

        IMPORTANT: Always use this method for queries, not embed_chunks!
        Different prefix = different vector space alignment.

        Args:
            query: User search query

        Returns:
            list[float]: Query embedding vector
        """
        self._load_model()
        prefixed = f"{QUERY_PREFIX}{query}"
        embedding = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        logger.debug(f"Query embedded with E5 | prefix='{QUERY_PREFIX}'")
        return embedding[0].tolist()

    def save_embeddings(
        self,
        embedded_chunks: list[EmbeddedChunk],
        output_path: str
    ) -> None:
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

            with mlflow.start_run(run_name="local_e5_large_v2"):
                mlflow.log_params({
                    "model":            MODEL_NAME,
                    "dimensions":       DIMENSIONS,
                    "batch_size":       self.batch_size,
                    "provider":         "local_sentence_transformers",
                    "document_prefix":  DOCUMENT_PREFIX,
                    "query_prefix":     QUERY_PREFIX,
                    "asymmetric":       True,
                    "cost_usd":         0.0,
                })
                mlflow.log_metrics({
                    "chunks_embedded":    self.stats["chunks_embedded"],
                    "total_time_sec":     self.stats["total_time_sec"],
                    "avg_time_per_chunk": self.stats["avg_time_per_chunk"],
                    "cost_usd":           0.0,
                    "failed":             self.stats["failed"],
                })
                logger.info("E5 embedding experiment logged to MLflow")

        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("\n--- Testing E5-Large-v2 Embedder ---")
    print("⚠️  First run downloads E5 model (~1.3GB)")
    print("⚠️  Zero API cost — runs completely locally\n")

    with open(
        "data/processed/chunks/part_91_recursive_512.json",
        encoding="utf-8"
    ) as f:
        all_chunks = json.load(f)

    test_chunks = all_chunks[:10]
    print(f"Testing with {len(test_chunks)} chunks\n")

    embedder = E5Embedder()
    embedded = embedder.embed_chunks(test_chunks)

    print(f"\n📊 E5 Embedding Stats:")
    print(f"   Chunks embedded    : {embedder.stats['chunks_embedded']}")
    print(f"   Total time         : {embedder.stats['total_time_sec']}s")
    print(f"   Avg per chunk      : {embedder.stats['avg_time_per_chunk']}s")
    print(f"   Cost               : $0.00 (FREE!)")

    # Test query embedding
    print(f"\n🔍 Testing asymmetric query embedding...")
    query_vec = embedder.embed_query("What are APU MEL requirements for Boeing 787?")
    print(f"   Query dims    : {len(query_vec)}")
    print(f"   Query vec[0:5]: {[round(v,4) for v in query_vec[:5]]}")

    print(f"\n📋 Sample Embedded Chunk:")
    if embedded:
        e = embedded[0]
        print(f"   Citation  : {e.citation}")
        print(f"   Model     : {e.embedding_model}")
        print(f"   Dims      : {e.embedding_dims}")
        print(f"   Vec[0:5]  : {[round(v,4) for v in e.embedding[:5]]}")

    embedder.log_to_mlflow()

    print(f"\n📊 COMPLETE MODEL COMPARISON:")
    print(f"{'Model':<25} {'Dims':>6} {'Cost/10':>10} {'Type':<20} {'MTEB'}")
    print("-"*75)
    print(f"{'voyage-3-large':<25} {'1024':>6} {'$0.000065':>10} {'Asymmetric API':20} {'54.1'}")
    print(f"{'text-embedding-3-small':<25} {'1536':>6} {'$0.000018':>10} {'Symmetric API':20} {'44.9'}")
    print(f"{'e5-large-v2 (local)':<25} {'1024':>6} {'$0.00':>10} {'Asymmetric Local':20} {'50.6'}")
    print(f"{'BAAI/bge-m3 (local)':<25} {'1024':>6} {'$0.00':>10} {'Symmetric Local':20} {'48.8'}")

    print(f"\n✅ E5-Large-v2 Embedder working!")