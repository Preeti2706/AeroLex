"""
AeroLex — Semantic Text Chunker

Strategy 2 of 3: Semantic Chunking

What it does:
- Uses sentence embeddings to find natural topic boundaries
- Compares consecutive sentences — when meaning "shifts" significantly,
  that's a chunk boundary
- Results in chunks that are semantically coherent — each chunk
  discusses one complete idea

How it works:
1. Split text into individual sentences
2. Embed each sentence using a local embedding model
3. Calculate cosine similarity between consecutive sentences
4. When similarity drops below threshold → new chunk starts
5. Merge small chunks to avoid tiny fragments

Why better than Recursive for regulations?
- Aviation regulations have clear topic shifts:
  "Section 91.103 — Preflight action" has multiple sub-topics
  (weather, NOTAMs, fuel requirements, etc.)
- Semantic chunker naturally separates these sub-topics
- Each chunk = one complete regulatory concept

Pros:
- Semantically coherent chunks
- Natural boundaries — better RAG retrieval quality
- Context not lost at boundaries

Cons:
- SLOW — requires embedding every sentence
- Requires embedding model loaded in memory
- Chunk sizes vary significantly — some very short, some long
- Harder to control exact chunk size

Usage:
    from src.chunking.semantic_chunker import SemanticChunker
    chunker = SemanticChunker()
    chunks = chunker.chunk_sections(sections)
"""

import json
import time
import mlflow
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, ChunkingError
from config.settings import settings

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# Similarity threshold — when cosine similarity drops below this,
# a new chunk starts. Lower = more chunks. Higher = fewer, bigger chunks.
DEFAULT_BREAKPOINT_THRESHOLD = 0.5

# Minimum chunk size — merge chunks smaller than this
MIN_CHUNK_CHARS = 100

# Local embedding model for semantic similarity
# Using sentence-transformers — free, no API cost
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality, 384 dims

# ── Output directory ──────────────────────────────────────────────────────────
CHUNKS_DIR = Path("data/processed/chunks")
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Chunk:
    """Same structure as RecursiveChunker for consistent downstream processing."""
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
    chunk_size:        int   # 0 for semantic — variable size
    chunk_overlap:     int   # 0 for semantic — no fixed overlap


class SemanticChunker:
    """
    Chunks text based on semantic meaning shifts using sentence embeddings.

    Algorithm:
    1. Split into sentences
    2. Embed each sentence (local model — no API cost)
    3. Find breakpoints where similarity drops
    4. Group sentences into semantically coherent chunks
    """

    def __init__(
        self,
        breakpoint_threshold: float = DEFAULT_BREAKPOINT_THRESHOLD,
        min_chunk_chars: int = MIN_CHUNK_CHARS
    ):
        """
        Args:
            breakpoint_threshold: Cosine similarity threshold (0-1)
                                  Lower = more chunk breaks
                                  Higher = fewer, larger chunks
            min_chunk_chars: Minimum chunk size — merge if smaller
        """
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_chars      = min_chunk_chars
        self.model                = None  # Lazy load — only when needed
        self.stats = {
            "sections_processed": 0,
            "total_chunks":       0,
            "avg_chunk_size":     0,
            "min_chunk_size":     float("inf"),
            "max_chunk_size":     0,
            "skipped_short":      0,
            "total_sentences":    0,
        }
        logger.info(
            f"SemanticChunker initialized | "
            f"threshold={breakpoint_threshold} | "
            f"min_chars={min_chunk_chars}"
        )

    def _load_model(self) -> None:
        """
        Lazy load the sentence transformer model.

        Why lazy loading?
        - Model takes ~500MB memory
        - Loading at import time = slow startup for everything
        - Load only when chunking actually needed
        """
        if self.model is None:
            logger.info(f"Loading sentence transformer: {EMBEDDING_MODEL}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Sentence transformer loaded")

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Simple approach: split by '. ' and '; '
        Aviation regulations use these consistently.

        Args:
            text: Input text

        Returns:
            list: Individual sentences
        """
        import re

        # Split by sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

        return sentences

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Cosine similarity = dot product / (magnitude1 × magnitude2)
        Range: -1 to 1 (1 = identical, 0 = unrelated, -1 = opposite)

        For text embeddings, typically 0.3-1.0 range.
        """
        dot_product = np.dot(vec1, vec2)
        magnitude   = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if magnitude == 0:
            return 0.0
        return float(dot_product / magnitude)

    def _find_breakpoints(self, sentences: list[str]) -> list[int]:
        """
        Find indices where semantic breaks should occur.

        Algorithm:
        1. Embed all sentences
        2. Compare each sentence with next
        3. If similarity < threshold → breakpoint here

        Args:
            sentences: List of sentences

        Returns:
            list: Indices where new chunks should start
        """
        if len(sentences) <= 1:
            return []

        # Embed all sentences at once (batch = faster)
        embeddings = self.model.encode(sentences, show_progress_bar=False)

        breakpoints = []
        for i in range(len(sentences) - 1):
            similarity = self._cosine_similarity(embeddings[i], embeddings[i + 1])

            if similarity < self.breakpoint_threshold:
                breakpoints.append(i + 1)
                logger.debug(
                    f"Breakpoint at sentence {i+1} | "
                    f"Similarity: {similarity:.3f} < {self.breakpoint_threshold}"
                )

        return breakpoints

    def chunk_section(self, section: dict) -> list[Chunk]:
        """
        Semantically chunk a single section.

        Args:
            section: ParsedSection dict

        Returns:
            list[Chunk]: Semantically coherent chunks
        """
        text     = section.get("text", "").strip()
        citation = section.get("citation", "unknown")

        if len(text) < 50:
            self.stats["skipped_short"] += 1
            return []

        try:
            self._load_model()

            # Step 1: Split into sentences
            sentences = self._split_into_sentences(text)
            self.stats["total_sentences"] += len(sentences)

            if len(sentences) <= 1:
                # Single sentence — return as one chunk
                return [self._make_chunk(text, 0, 0, 1, section)]

            # Step 2: Find semantic breakpoints
            breakpoints = self._find_breakpoints(sentences)

            # Step 3: Group sentences into chunks
            chunk_texts = []
            current_group = []
            for i, sentence in enumerate(sentences):
                if i in breakpoints and current_group:
                    chunk_texts.append(" ".join(current_group))
                    current_group = [sentence]
                else:
                    current_group.append(sentence)
            if current_group:
                chunk_texts.append(" ".join(current_group))

            # Step 4: Merge small chunks
            merged = []
            buffer = ""
            for ct in chunk_texts:
                if len(buffer) + len(ct) < self.min_chunk_chars:
                    buffer += " " + ct
                else:
                    if buffer:
                        merged.append(buffer.strip())
                    buffer = ct
            if buffer:
                merged.append(buffer.strip())

            # Step 5: Create Chunk objects
            chunks = []
            for i, chunk_text in enumerate(merged):
                if not chunk_text.strip():
                    continue
                chunk = self._make_chunk(chunk_text, i, 0, len(merged), section)
                chunks.append(chunk)

                # Update stats
                self.stats["max_chunk_size"] = max(
                    self.stats["max_chunk_size"], len(chunk_text)
                )
                self.stats["min_chunk_size"] = min(
                    self.stats["min_chunk_size"], len(chunk_text)
                )

            return chunks

        except Exception as e:
            raise ChunkingError(
                message=f"Semantic chunking failed: {citation}",
                context="SemanticChunker.chunk_section()",
                original_error=e
            )

    def _make_chunk(
        self,
        text: str,
        index: int,
        overlap: int,
        total: int,
        section: dict
    ) -> Chunk:
        """Helper to create a Chunk object with section metadata."""
        citation = section.get("citation", "unknown")
        return Chunk(
            chunk_id          = f"{citation.replace(' ', '_').replace(',', '')}_sem_{index}",
            text              = text,
            char_count        = len(text),
            word_count        = len(text.split()),
            chunk_index       = index,
            total_chunks      = total,
            part_number       = section.get("part_number", ""),
            part_title        = section.get("part_title", ""),
            subpart           = section.get("subpart", ""),
            subpart_title     = section.get("subpart_title", ""),
            section           = section.get("section", ""),
            section_title     = section.get("section_title", ""),
            citation          = citation,
            hierarchy         = section.get("hierarchy", ""),
            source            = section.get("source", "ecfr"),
            doc_type          = section.get("doc_type", "regulation"),
            chunking_strategy = "semantic",
            chunk_size        = 0,
            chunk_overlap     = 0,
        )

    def chunk_sections(self, sections: list[dict]) -> list[Chunk]:
        """Chunk all sections."""
        all_chunks = []
        logger.info(f"Starting semantic chunking | Sections: {len(sections)}")

        for i, section in enumerate(sections):
            try:
                chunks = self.chunk_section(section)
                all_chunks.extend(chunks)
                self.stats["sections_processed"] += 1

                if (i + 1) % 50 == 0:
                    logger.info(f"Progress: {i+1}/{len(sections)} sections")

            except ChunkingError as e:
                handle_exception(e, context="SemanticChunker.chunk_sections()")
                continue

        # Final stats
        self.stats["total_chunks"] = len(all_chunks)
        if all_chunks:
            self.stats["avg_chunk_size"] = int(
                sum(c.char_count for c in all_chunks) / len(all_chunks)
            )
        if self.stats["min_chunk_size"] == float("inf"):
            self.stats["min_chunk_size"] = 0

        logger.info(
            f"Semantic chunking complete | "
            f"Sections: {self.stats['sections_processed']} | "
            f"Chunks: {self.stats['total_chunks']} | "
            f"Avg size: {self.stats['avg_chunk_size']} chars"
        )
        return all_chunks

    def save_chunks(self, chunks: list[Chunk], output_path: str) -> None:
        """Save chunks to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in chunks], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(chunks)} chunks to: {path}")

    def log_to_mlflow(self, experiment_name: str = "aerolex-chunking-experiments") -> None:
        """Log semantic chunking experiment to MLflow."""
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"semantic_{self.breakpoint_threshold}"):
                mlflow.log_params({
                    "strategy":              "semantic",
                    "breakpoint_threshold":  self.breakpoint_threshold,
                    "min_chunk_chars":       self.min_chunk_chars,
                    "embedding_model":       EMBEDDING_MODEL,
                })
                mlflow.log_metrics({
                    "total_chunks":       self.stats["total_chunks"],
                    "avg_chunk_size":     self.stats["avg_chunk_size"],
                    "min_chunk_size":     self.stats["min_chunk_size"],
                    "max_chunk_size":     self.stats["max_chunk_size"],
                    "total_sentences":    self.stats["total_sentences"],
                    "sections_processed": self.stats["sections_processed"],
                })
                logger.info("Semantic chunking experiment logged to MLflow")

        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("\n--- Testing Semantic Chunker ---")
    print("⚠️  First run downloads sentence-transformer model (~90MB)\n")

    with open("data/processed/part_91_parsed.json", encoding="utf-8") as f:
        sections = json.load(f)

    # Test on first 20 sections only — semantic is slow
    print(f"Testing on first 20 sections (full = {len(sections)})\n")
    test_sections = sections[:20]

    chunker = SemanticChunker(breakpoint_threshold=0.5)
    chunks  = chunker.chunk_sections(test_sections)

    output_path = "data/processed/chunks/part_91_semantic_05.json"
    chunker.save_chunks(chunks, output_path)
    chunker.log_to_mlflow()

    print(f"\n📊 Semantic Chunking Stats:")
    print(f"   Sections processed : {chunker.stats['sections_processed']}")
    print(f"   Total sentences    : {chunker.stats['total_sentences']}")
    print(f"   Total chunks       : {chunker.stats['total_chunks']}")
    print(f"   Avg chunk size     : {chunker.stats['avg_chunk_size']} chars")
    print(f"   Min chunk size     : {chunker.stats['min_chunk_size']} chars")
    print(f"   Max chunk size     : {chunker.stats['max_chunk_size']} chars")

    print(f"\n📋 Sample Chunks (compare with recursive):")
    for chunk in chunks[:3]:
        print(f"   [{chunk.citation}] Chunk {chunk.chunk_index+1}/{chunk.total_chunks}")
        print(f"   Size: {chunk.char_count} chars | Words: {chunk.word_count}")
        print(f"   Text: {chunk.text[:150]}...")
        print()

    print(f"✅ Semantic chunker working!")