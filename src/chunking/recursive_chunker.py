"""
AeroLex — Recursive Character Text Chunker

Strategy 1 of 3: Recursive Character Splitting

What it does:
- Splits text by trying separators in order: paragraph → sentence → word → character
- Tries to split at natural boundaries first (paragraphs, then sentences)
- Falls back to smaller separators only when chunk is still too big
- Adds overlap between chunks so context is not lost at boundaries

Why "Recursive"?
- First tries to split by "\n\n" (paragraphs)
- If chunk still too big, tries "\n" (newlines)
- If still too big, tries ". " (sentences)
- If still too big, splits by character
- Hence "recursive" — keeps trying smaller splits

Pros:
- Fast, simple, reliable
- Works on any text
- LangChain built-in — battle tested

Cons:
- Does NOT understand meaning — splits by characters, not semantics
- May split a regulation mid-sentence if forced
- Chunk boundaries may be arbitrary

When to use:
- Large documents where speed matters
- When semantic understanding not critical
- As baseline to compare against other strategies

Usage:
    from src.chunking.recursive_chunker import RecursiveChunker
    chunker = RecursiveChunker()
    chunks = chunker.chunk_sections(sections)
"""

import json
import time
import mlflow
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, ChunkingError
from config.settings import settings

logger = get_logger(__name__)

# ── Chunking Config ───────────────────────────────────────────────────────────
# These values will be experimented with in MLflow
DEFAULT_CHUNK_SIZE    = 512   # Characters per chunk
DEFAULT_CHUNK_OVERLAP = 50    # Overlap between chunks

# Separators tried in order — most natural to least natural
SEPARATORS = [
    "\n\n",   # Paragraph break — most natural
    "\n",     # Line break
    ". ",     # Sentence end
    ", ",     # Clause boundary
    " ",      # Word boundary
    "",       # Character — last resort
]

# ── Output directory ──────────────────────────────────────────────────────────
CHUNKS_DIR = Path("data/processed/chunks")
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Chunk:
    """
    Represents one chunk of text ready for embedding.

    Every chunk carries its parent section's metadata — this is
    critical for RAG retrieval filtering and citation generation.
    """
    chunk_id:       str     # Unique ID: section_citation + chunk index
    text:           str     # The actual chunk text
    char_count:     int     # Character count
    word_count:     int     # Word count
    chunk_index:    int     # Position within parent section (0, 1, 2...)
    total_chunks:   int     # Total chunks from parent section

    # ── Inherited metadata from parent section ────────────────────────
    # These come from ParsedSection — used for filtering in Qdrant
    part_number:    str
    part_title:     str
    subpart:        str
    subpart_title:  str
    section:        str
    section_title:  str
    citation:       str     # "14 CFR Part 91, Section 91.103"
    hierarchy:      str     # "Title 14 > Part 91 > Subpart B > Section 91.103"
    source:         str     # "ecfr"
    doc_type:       str     # "regulation"

    # ── Chunking metadata ─────────────────────────────────────────────
    chunking_strategy: str  # "recursive", "semantic", "hierarchical"
    chunk_size:     int     # Config used
    chunk_overlap:  int     # Config used


class RecursiveChunker:
    """
    Chunks aviation regulation text using recursive character splitting.

    Design philosophy:
    - Preserves ALL parent section metadata in each chunk
    - Tracks chunking statistics for MLflow experiment logging
    - Outputs chunks to JSON for downstream embedding
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        """
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Characters to overlap between adjacent chunks

        Why overlap?
        Imagine a regulation says: "The pilot must... [chunk boundary] ...complete
        the checklist before departure." Without overlap, chunk 1 ends mid-thought
        and chunk 2 starts without context. Overlap ensures both chunks have
        the complete sentence.
        """
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize LangChain splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=SEPARATORS,
            length_function=len,  # Measure by characters
        )

        self.stats = {
            "sections_processed": 0,
            "total_chunks":       0,
            "avg_chunk_size":     0,
            "min_chunk_size":     float("inf"),
            "max_chunk_size":     0,
            "skipped_short":      0,
        }

        logger.info(
            f"RecursiveChunker initialized | "
            f"chunk_size={chunk_size} | overlap={chunk_overlap}"
        )

    def chunk_section(self, section: dict) -> list[Chunk]:
        """
        Chunk a single ParsedSection into Chunk objects.

        Args:
            section: ParsedSection dict from xml_parser output

        Returns:
            list[Chunk]: List of chunks with full metadata
        """
        text     = section.get("text", "").strip()
        citation = section.get("citation", "unknown")

        # Skip very short sections — not worth chunking
        if len(text) < 50:
            logger.debug(f"Skipping short section: {citation} ({len(text)} chars)")
            self.stats["skipped_short"] += 1
            return []

        try:
            # Split text into chunks
            text_chunks = self.splitter.split_text(text)

            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue

                chunk = Chunk(
                    chunk_id       = f"{citation.replace(' ', '_').replace(',', '')}_chunk_{i}",
                    text           = chunk_text,
                    char_count     = len(chunk_text),
                    word_count     = len(chunk_text.split()),
                    chunk_index    = i,
                    total_chunks   = len(text_chunks),

                    # Metadata from parent section
                    part_number    = section.get("part_number", ""),
                    part_title     = section.get("part_title", ""),
                    subpart        = section.get("subpart", ""),
                    subpart_title  = section.get("subpart_title", ""),
                    section        = section.get("section", ""),
                    section_title  = section.get("section_title", ""),
                    citation       = citation,
                    hierarchy      = section.get("hierarchy", ""),
                    source         = section.get("source", "ecfr"),
                    doc_type       = section.get("doc_type", "regulation"),

                    # Chunking metadata
                    chunking_strategy = "recursive",
                    chunk_size     = self.chunk_size,
                    chunk_overlap  = self.chunk_overlap,
                )
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
                message=f"Failed to chunk section: {citation}",
                context="RecursiveChunker.chunk_section()",
                original_error=e
            )

    def chunk_sections(self, sections: list[dict]) -> list[Chunk]:
        """
        Chunk all sections from a parsed document.

        Args:
            sections: List of ParsedSection dicts

        Returns:
            list[Chunk]: All chunks from all sections
        """
        all_chunks = []
        logger.info(f"Starting recursive chunking | Sections: {len(sections)}")

        for section in sections:
            try:
                chunks = self.chunk_section(section)
                all_chunks.extend(chunks)
                self.stats["sections_processed"] += 1

            except ChunkingError as e:
                handle_exception(e, context="RecursiveChunker.chunk_sections()")
                continue

        # Calculate averages
        self.stats["total_chunks"] = len(all_chunks)
        if all_chunks:
            self.stats["avg_chunk_size"] = int(
                sum(c.char_count for c in all_chunks) / len(all_chunks)
            )
        if self.stats["min_chunk_size"] == float("inf"):
            self.stats["min_chunk_size"] = 0

        logger.info(
            f"Recursive chunking complete | "
            f"Sections: {self.stats['sections_processed']} | "
            f"Chunks: {self.stats['total_chunks']} | "
            f"Avg size: {self.stats['avg_chunk_size']} chars"
        )
        return all_chunks

    def save_chunks(self, chunks: list[Chunk], output_path: str) -> None:
        """Save chunks to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in chunks], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(chunks)} chunks to: {path}")

    def log_to_mlflow(self, experiment_name: str = "aerolex-chunking-experiments") -> None:
        """
        Log chunking experiment results to MLflow.

        Tracks: strategy, chunk_size, overlap, total_chunks, avg_size
        This allows comparing recursive vs semantic vs hierarchical strategies.
        """
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"recursive_{self.chunk_size}"):
                # Log parameters (what settings were used)
                mlflow.log_params({
                    "strategy":     "recursive",
                    "chunk_size":   self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "separators":   str(SEPARATORS[:3]),
                })
                # Log metrics (what results were achieved)
                mlflow.log_metrics({
                    "total_chunks":    self.stats["total_chunks"],
                    "avg_chunk_size":  self.stats["avg_chunk_size"],
                    "min_chunk_size":  self.stats["min_chunk_size"],
                    "max_chunk_size":  self.stats["max_chunk_size"],
                    "sections_processed": self.stats["sections_processed"],
                })
                logger.info("Chunking experiment logged to MLflow")

        except Exception as e:
            logger.warning(f"MLflow logging failed (non-critical): {e}")


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("\n--- Testing Recursive Chunker ---\n")

    # Load parsed sections
    with open("data/processed/part_91_parsed.json", encoding="utf-8") as f:
        sections = json.load(f)

    print(f"Loaded {len(sections)} sections from Part 91\n")

    # Test with default settings
    chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
    chunks  = chunker.chunk_sections(sections)

    # Save chunks
    output_path = "data/processed/chunks/part_91_recursive_512.json"
    chunker.save_chunks(chunks, output_path)

    # Log to MLflow
    chunker.log_to_mlflow()

    # Print results
    print(f"\n📊 Chunking Stats:")
    print(f"   Sections processed : {chunker.stats['sections_processed']}")
    print(f"   Total chunks       : {chunker.stats['total_chunks']}")
    print(f"   Avg chunk size     : {chunker.stats['avg_chunk_size']} chars")
    print(f"   Min chunk size     : {chunker.stats['min_chunk_size']} chars")
    print(f"   Max chunk size     : {chunker.stats['max_chunk_size']} chars")
    print(f"   Skipped (short)    : {chunker.stats['skipped_short']}")

    print(f"\n📋 Sample Chunks:")
    for chunk in chunks[:3]:
        print(f"   [{chunk.citation}] Chunk {chunk.chunk_index+1}/{chunk.total_chunks}")
        print(f"   Size: {chunk.char_count} chars | Words: {chunk.word_count}")
        print(f"   Text: {chunk.text[:150]}...")
        print()

    print(f"✅ Recursive chunker working!")
    print(f"   Output: {output_path}")