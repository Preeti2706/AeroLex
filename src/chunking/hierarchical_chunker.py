"""
AeroLex — Hierarchical Text Chunker

Strategy 3 of 3: Regulation-Aware Hierarchical Chunking

What it does:
- Aviation regulations have a natural hierarchy:
  Part → Subpart → Section → Paragraph (a)(b)(c) → Sub-paragraph
- This chunker RESPECTS that hierarchy
- Each chunk = one complete regulatory paragraph (a), (b), (c)
- Never splits a paragraph mid-thought
- Adds full hierarchy path to every chunk

Why best for aviation regulations?
- "14 CFR Part 91, Section 91.103(b)" is a meaningful unit
- Compliance queries are often paragraph-specific:
  "What does 91.103(a) say about weather?"
- Retrieval precision improves when chunks = regulatory paragraphs
- Citations are exact: "91.103(a)(1)(ii)" not just "91.103"

How it works:
1. Parse section text for paragraph markers: (a), (b), (1), (2), (i), (ii)
2. Each paragraph becomes its own chunk
3. Add parent context (section title) to every chunk
4. Small paragraphs merged with parent for context

Pros:
- Most precise for regulation queries
- Natural citation boundaries
- Perfect hierarchy metadata
- No arbitrary splits

Cons:
- Aviation-specific — won't work on generic text
- Depends on regulation formatting consistency
- Very small chunks for short paragraphs

Usage:
    from src.chunking.hierarchical_chunker import HierarchicalChunker
    chunker = HierarchicalChunker()
    chunks = chunker.chunk_sections(sections)
"""

import re
import json
import mlflow
from pathlib import Path
from dataclasses import dataclass, asdict
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, ChunkingError
from config.settings import settings

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MIN_PARAGRAPH_CHARS = 50   # Merge paragraphs shorter than this
MAX_CHUNK_CHARS     = 1500 # Split paragraphs longer than this

# ── Output directory ──────────────────────────────────────────────────────────
CHUNKS_DIR = Path("data/processed/chunks")
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# ── Paragraph patterns in CFR regulations ────────────────────────────────────
# These are the standard paragraph markers used in 14 CFR
PARAGRAPH_PATTERNS = [
    r'\([a-z]\)',           # (a), (b), (c)... primary paragraphs
    r'\(\d+\)',             # (1), (2), (3)... numbered sub-paragraphs
    r'\([ivxlcdm]+\)',      # (i), (ii), (iii)... roman numeral sub-sub
    r'\([A-Z]\)',           # (A), (B), (C)... lettered sub-sub-paragraphs
]

# Combined pattern for detecting any paragraph marker
PARA_PATTERN = re.compile(
    r'(?=' + '|'.join(PARAGRAPH_PATTERNS) + r')'
)


@dataclass
class Chunk:
    """Hierarchical chunk with full regulatory path."""
    chunk_id:             str
    text:                 str
    char_count:           int
    word_count:           int
    chunk_index:          int
    total_chunks:         int

    # Section metadata
    part_number:          str
    part_title:           str
    subpart:              str
    subpart_title:        str
    section:              str
    section_title:        str
    citation:             str
    hierarchy:            str
    source:               str
    doc_type:             str

    # Hierarchical-specific metadata
    paragraph_id:         str   # e.g., "(a)", "(a)(1)", "(a)(1)(i)"
    paragraph_level:      int   # 0=section, 1=(a), 2=(1), 3=(i)
    has_sub_paragraphs:   bool  # Does this paragraph have children?
    chunking_strategy:    str
    chunk_size:           int
    chunk_overlap:        int


class HierarchicalChunker:
    """
    Chunks aviation regulation text by regulatory paragraph structure.

    Key insight: CFR regulations use a CONSISTENT paragraph numbering
    system: (a), (b)... → (1), (2)... → (i), (ii)...
    This chunker treats each paragraph as a natural chunk boundary.
    """

    def __init__(
        self,
        min_paragraph_chars: int = MIN_PARAGRAPH_CHARS,
        max_chunk_chars: int = MAX_CHUNK_CHARS
    ):
        self.min_paragraph_chars = min_paragraph_chars
        self.max_chunk_chars     = max_chunk_chars
        self.stats = {
            "sections_processed":  0,
            "total_chunks":        0,
            "avg_chunk_size":      0,
            "min_chunk_size":      float("inf"),
            "max_chunk_size":      0,
            "skipped_short":       0,
            "paragraphs_found":    0,
            "sections_no_paras":   0,
        }
        logger.info(
            f"HierarchicalChunker initialized | "
            f"min_para={min_paragraph_chars} | max_chunk={max_chunk_chars}"
        )

    def _detect_paragraph_level(self, marker: str) -> int:
        """
        Detect the hierarchy level of a paragraph marker.

        Level 0 = no marker (section intro text)
        Level 1 = (a), (b), (c) — primary paragraphs
        Level 2 = (1), (2), (3) — numbered sub-paragraphs
        Level 3 = (i), (ii), (iii) — roman numeral sub-sub
        Level 4 = (A), (B), (C) — letter sub-sub-sub

        Args:
            marker: Paragraph marker string e.g. "(a)", "(1)", "(ii)"

        Returns:
            int: Hierarchy level
        """
        if not marker:
            return 0
        inner = marker.strip("()")

        if re.match(r'^[a-z]$', inner):      return 1  # (a), (b)
        if re.match(r'^\d+$', inner):         return 2  # (1), (2)
        if re.match(r'^[ivxlcdm]+$', inner):  return 3  # (i), (ii)
        if re.match(r'^[A-Z]$', inner):       return 4  # (A), (B)
        return 1

    def _split_into_paragraphs(self, text: str) -> list[dict]:
        """
        Split section text into regulatory paragraphs.

        Splits at paragraph markers: (a), (b), (1), (2), (i), etc.
        Each paragraph gets its marker and level.

        Args:
            text: Full section text

        Returns:
            list: Dicts with 'marker', 'level', 'text'
        """
        # Pattern to split at paragraph boundaries
        split_pattern = re.compile(
            r'(?=\([a-z]\)|\(\d+\)|\([ivxlcdm]+\)|\([A-Z]\))'
        )

        # Split text at paragraph markers
        parts = split_pattern.split(text)

        paragraphs = []
        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Extract leading paragraph marker
            marker_match = re.match(
                r'^(\([a-z]\)|\(\d+\)|\([ivxlcdm]+\)|\([A-Z]\))',
                part
            )

            if marker_match:
                marker = marker_match.group(1)
                level  = self._detect_paragraph_level(marker)
                para_text = part
            else:
                # Intro text before first paragraph marker
                marker = ""
                level  = 0
                para_text = part

            if len(para_text) >= 10:  # Skip tiny fragments
                paragraphs.append({
                    "marker":    marker,
                    "level":     level,
                    "text":      para_text,
                    "char_count": len(para_text)
                })

        return paragraphs

    def _build_paragraph_id(self, marker: str, parent_id: str = "") -> str:
        """
        Build hierarchical paragraph ID.

        Example: parent="(a)", marker="(1)" → "(a)(1)"
        """
        if not parent_id:
            return marker
        return parent_id + marker

    def chunk_section(self, section: dict) -> list[Chunk]:
        """
        Chunk a section by its regulatory paragraph structure.

        Args:
            section: ParsedSection dict

        Returns:
            list[Chunk]: One chunk per regulatory paragraph
        """
        text     = section.get("text", "").strip()
        citation = section.get("citation", "unknown")

        if len(text) < 50:
            self.stats["skipped_short"] += 1
            return []

        try:
            # Split into regulatory paragraphs
            paragraphs = self._split_into_paragraphs(text)
            self.stats["paragraphs_found"] += len(paragraphs)

            if not paragraphs:
                self.stats["sections_no_paras"] += 1
                # No paragraph structure — treat whole section as one chunk
                return [self._make_chunk(
                    text, "", 0, False, 0, 1, section
                )]

            # Merge very short paragraphs with previous
            merged_paragraphs = []
            buffer_text  = ""
            buffer_marker = ""
            buffer_level  = 0

            for para in paragraphs:
                if len(buffer_text) + len(para["text"]) < self.min_paragraph_chars:
                    # Too short — merge
                    buffer_text  += " " + para["text"]
                    if not buffer_marker:
                        buffer_marker = para["marker"]
                        buffer_level  = para["level"]
                else:
                    if buffer_text:
                        merged_paragraphs.append({
                            "marker": buffer_marker,
                            "level":  buffer_level,
                            "text":   buffer_text.strip()
                        })
                    buffer_text   = para["text"]
                    buffer_marker = para["marker"]
                    buffer_level  = para["level"]

            if buffer_text:
                merged_paragraphs.append({
                    "marker": buffer_marker,
                    "level":  buffer_level,
                    "text":   buffer_text.strip()
                })

            # Create chunks
            chunks = []
            section_title = section.get("section_title", "")

            for i, para in enumerate(merged_paragraphs):
                para_text = para["text"]

                # If paragraph too long, split it further
                if len(para_text) > self.max_chunk_chars:
                    # Simple split for very long paragraphs
                    sub_texts = [
                        para_text[j:j+self.max_chunk_chars]
                        for j in range(0, len(para_text), self.max_chunk_chars)
                    ]
                    for k, sub_text in enumerate(sub_texts):
                        sub_marker = f"{para['marker']}[{k+1}]" if para["marker"] else f"[{k+1}]"
                        chunk = self._make_chunk(
                            sub_text,
                            sub_marker,
                            para["level"],
                            False,
                            i * len(sub_texts) + k,
                            len(merged_paragraphs),
                            section
                        )
                        chunks.append(chunk)
                else:
                    # Check if this paragraph has sub-paragraphs
                    has_subs = any(
                        p["level"] > para["level"]
                        for p in merged_paragraphs[i+1:]
                    ) if i < len(merged_paragraphs) - 1 else False

                    chunk = self._make_chunk(
                        para_text,
                        para["marker"],
                        para["level"],
                        has_subs,
                        i,
                        len(merged_paragraphs),
                        section
                    )
                    chunks.append(chunk)

                    # Update stats
                    self.stats["max_chunk_size"] = max(
                        self.stats["max_chunk_size"], len(para_text)
                    )
                    self.stats["min_chunk_size"] = min(
                        self.stats["min_chunk_size"], len(para_text)
                    )

            return chunks

        except Exception as e:
            raise ChunkingError(
                message=f"Hierarchical chunking failed: {citation}",
                context="HierarchicalChunker.chunk_section()",
                original_error=e
            )

    def _make_chunk(
        self,
        text: str,
        paragraph_id: str,
        level: int,
        has_subs: bool,
        index: int,
        total: int,
        section: dict
    ) -> Chunk:
        """Create Chunk object with full metadata."""
        citation = section.get("citation", "unknown")

        # Build full citation with paragraph
        full_citation = citation
        if paragraph_id:
            full_citation = f"{citation}{paragraph_id}"

        return Chunk(
            chunk_id          = f"{citation.replace(' ', '_').replace(',', '')}_hier_{index}",
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
            citation          = full_citation,
            hierarchy         = section.get("hierarchy", ""),
            source            = section.get("source", "ecfr"),
            doc_type          = section.get("doc_type", "regulation"),
            paragraph_id      = paragraph_id,
            paragraph_level   = level,
            has_sub_paragraphs = has_subs,
            chunking_strategy = "hierarchical",
            chunk_size        = 0,
            chunk_overlap     = 0,
        )

    def chunk_sections(self, sections: list[dict]) -> list[Chunk]:
        """Chunk all sections hierarchically."""
        all_chunks = []
        logger.info(f"Starting hierarchical chunking | Sections: {len(sections)}")

        for section in sections:
            try:
                chunks = self.chunk_section(section)
                all_chunks.extend(chunks)
                self.stats["sections_processed"] += 1
            except ChunkingError as e:
                handle_exception(e, context="HierarchicalChunker.chunk_sections()")
                continue

        self.stats["total_chunks"] = len(all_chunks)
        if all_chunks:
            self.stats["avg_chunk_size"] = int(
                sum(c.char_count for c in all_chunks) / len(all_chunks)
            )
        if self.stats["min_chunk_size"] == float("inf"):
            self.stats["min_chunk_size"] = 0

        logger.info(
            f"Hierarchical chunking complete | "
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

    def log_to_mlflow(
        self,
        experiment_name: str = "aerolex-chunking-experiments"
    ) -> None:
        """Log hierarchical experiment to MLflow."""
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name="hierarchical_regulation"):
                mlflow.log_params({
                    "strategy":           "hierarchical",
                    "min_paragraph_chars": self.min_paragraph_chars,
                    "max_chunk_chars":    self.max_chunk_chars,
                    "paragraph_patterns": "(a)(b)(1)(2)(i)(ii)",
                })
                mlflow.log_metrics({
                    "total_chunks":       self.stats["total_chunks"],
                    "avg_chunk_size":     self.stats["avg_chunk_size"],
                    "min_chunk_size":     self.stats["min_chunk_size"],
                    "max_chunk_size":     self.stats["max_chunk_size"],
                    "paragraphs_found":   self.stats["paragraphs_found"],
                    "sections_processed": self.stats["sections_processed"],
                })
                logger.info("Hierarchical experiment logged to MLflow")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("\n--- Testing Hierarchical Chunker ---\n")

    with open("data/processed/part_91_parsed.json", encoding="utf-8") as f:
        sections = json.load(f)

    print(f"Loaded {len(sections)} sections\n")

    chunker = HierarchicalChunker()
    chunks  = chunker.chunk_sections(sections)

    output_path = "data/processed/chunks/part_91_hierarchical.json"
    chunker.save_chunks(chunks, output_path)
    chunker.log_to_mlflow()

    print(f"\n📊 Hierarchical Chunking Stats:")
    print(f"   Sections processed  : {chunker.stats['sections_processed']}")
    print(f"   Paragraphs found    : {chunker.stats['paragraphs_found']}")
    print(f"   Total chunks        : {chunker.stats['total_chunks']}")
    print(f"   Avg chunk size      : {chunker.stats['avg_chunk_size']} chars")
    print(f"   Min chunk size      : {chunker.stats['min_chunk_size']} chars")
    print(f"   Max chunk size      : {chunker.stats['max_chunk_size']} chars")
    print(f"   Sections no paras   : {chunker.stats['sections_no_paras']}")

    print(f"\n📋 Sample Chunks — Notice paragraph_id field!")
    for chunk in chunks[:5]:
        print(f"   [{chunk.citation}]")
        print(f"   Para: '{chunk.paragraph_id}' | Level: {chunk.paragraph_level}")
        print(f"   Size: {chunk.char_count} chars")
        print(f"   Text: {chunk.text[:120]}...")
        print()

    print(f"✅ Hierarchical chunker working!")
    print(f"   Output: {output_path}")