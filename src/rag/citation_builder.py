"""
citation_builder.py — Structured Citation Generator for RAG Responses

WHAT:
    Transforms raw retrieved chunks into structured, traceable citations.
    Attaches regulatory references to LLM answers for auditability.

WHY:
    Aviation regulations are safety-critical — every claim must be
    traceable to an exact source. "The pilot must do X" is useless
    without "per 14 CFR § 91.103, eCFR Part 91".
    
    FAANG principle: RAG without citations = hallucination risk.
    Citations make answers auditable, trustworthy, and legally defensible.

HOW:
    1. Take RAGResponse (answer + retrieved chunks)
    2. Parse [Source N] markers from LLM answer
    3. Build Citation objects for each referenced chunk
    4. Return CitedResponse — answer + structured citations + metadata

MATH:
    Citation confidence = rerank_score * source_weight
    where source_weight: eCFR=1.0, FAA_AD=0.95, DGCA=0.90, SKYbrary=0.80

Official Docs:
    14 CFR formatting: https://www.ecfr.gov/current/title-14
    FAA AD format:     https://rgl.faa.gov/Regulatory_and_Guidance_Library
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, RAGError
from src.rag.rag_chain import RAGResponse, RetrievedChunk

logger = get_logger(__name__)

# ── Source weights for citation confidence ───────────────────────────────────
# Primary regulatory sources weighted higher than advisory/safety sources
SOURCE_WEIGHTS = {
    "ecfr":      1.00,   # Primary FAA regulations — highest authority
    "faa_ad":    0.95,   # Airworthiness Directives — mandatory compliance
    "dgca":      0.90,   # DGCA CARs — Indian regulatory authority
    "faa_ac":    0.85,   # Advisory Circulars — guidance, not mandatory
    "skybrary":  0.80,   # Safety articles — informational
    "unknown":   0.70,   # Fallback weight
}

# ── Regulation formatter map ─────────────────────────────────────────────────
# Formats source + part_number into human-readable regulatory reference
def format_regulation_ref(source: str, part_number: str, doc_type: str) -> str:
    """
    Format a human-readable regulatory reference string.

    Examples:
        eCFR + 91 → "14 CFR § Part 91"
        FAA_AD    → "FAA Airworthiness Directive"
        DGCA      → "DGCA CAR"
        FAA_AC    → "FAA Advisory Circular"
        SKYbrary  → "SKYbrary Safety Article"

    Args:
        source:      Source system — "ecfr", "faa_ad", etc.
        part_number: Regulation part number — "91", "121", etc.
        doc_type:    Document type — "regulation", "directive", etc.

    Returns:
        Formatted regulatory reference string
    """
    source_lower = source.lower()

    if source_lower == "ecfr":
        if part_number and part_number != "unknown":
            return f"14 CFR § Part {part_number}"
        return "14 CFR (Federal Aviation Regulations)"

    elif source_lower == "faa_ad":
        return "FAA Airworthiness Directive"

    elif source_lower == "dgca":
        return "DGCA Civil Aviation Requirement"

    elif source_lower == "faa_ac":
        return "FAA Advisory Circular"

    elif source_lower == "skybrary":
        return "SKYbrary Aviation Safety Article"

    else:
        return f"{source} {doc_type}".strip()


# ── Citation Dataclass ───────────────────────────────────────────────────────

@dataclass
class Citation:
    """
    Single structured citation — one retrieved chunk.

    Every field is intentional:
    - source_num:       Maps to [Source N] in LLM answer
    - regulation_ref:   Human-readable regulatory reference
    - source:           System source — "ecfr", "faa_ad", etc.
    - part_number:      FAR part number for cross-reference
    - doc_type:         Type of regulatory document
    - chunk_id:         Unique ID for full traceability
    - relevance_score:  Voyage rerank score (0-1)
    - confidence:       relevance_score * source_weight
    - text_snippet:     First 150 chars — preview without full text
    - used_in_answer:   True if LLM cited this source
    """
    source_num:       int
    regulation_ref:   str
    source:           str
    part_number:      str
    doc_type:         str
    chunk_id:         str
    relevance_score:  float
    confidence:       float
    text_snippet:     str
    used_in_answer:   bool = False


@dataclass
class CitedResponse:
    """
    Complete RAG response with structured citations.

    This is the final output of the RAG pipeline —
    answer + citations + metadata ready for API/UI consumption.
    """
    answer:              str
    citations:           list[Citation]
    used_citations:      list[Citation]    # Only citations referenced in answer
    total_chunks:        int
    avg_confidence:      float
    model_used:          str
    latency_ms:          float
    cost_usd:            float
    query:               str
    warning:             Optional[str] = None  # Low confidence warning


# ── Core Builder ─────────────────────────────────────────────────────────────

def build_citations(rag_response: RAGResponse) -> CitedResponse:
    """
    Build structured citations from a RAGResponse.

    Steps:
    1. Parse [Source N] markers from LLM answer text
    2. Build Citation object for each retrieved chunk
    3. Mark which citations were actually used in answer
    4. Calculate confidence scores
    5. Return CitedResponse

    Args:
        rag_response: Output from RAGChain.run()

    Returns:
        CitedResponse with structured citations

    Raises:
        RAGError: If citation building fails
    """
    try:
        logger.info(
            f"Building citations | "
            f"Chunks: {len(rag_response.sources)} | "
            f"Query: '{rag_response.query[:60]}'"
        )

        # ── Step 1: Parse which sources were cited in answer ──
        # LLM uses [Source 1], [Source 2] etc. in answer text
        cited_nums = _parse_cited_source_nums(rag_response.answer)
        logger.debug(f"Source numbers cited in answer: {cited_nums}")

        # ── Step 2: Build Citation objects ──
        citations = []
        for chunk in rag_response.sources:
            citation = _build_single_citation(
                chunk=chunk,
                cited_nums=cited_nums
            )
            citations.append(citation)

        # ── Step 3: Used citations only ──
        used_citations = [c for c in citations if c.used_in_answer]
        logger.info(
            f"Citations built | "
            f"Total: {len(citations)} | "
            f"Used in answer: {len(used_citations)}"
        )

        # ── Step 4: Average confidence ──
        if citations:
            avg_confidence = sum(c.confidence for c in citations) / len(citations)
        else:
            avg_confidence = 0.0

        # ── Step 5: Low confidence warning ──
        warning = None
        if avg_confidence < 0.5:
            warning = (
                f"Low confidence answer (avg={avg_confidence:.2f}). "
                f"Retrieved chunks may not fully cover this query. "
                f"Consider expanding the regulatory corpus."
            )
            logger.warning(warning)

        return CitedResponse(
            answer=rag_response.answer,
            citations=citations,
            used_citations=used_citations,
            total_chunks=len(citations),
            avg_confidence=avg_confidence,
            model_used=rag_response.model_used,
            latency_ms=rag_response.latency_ms,
            cost_usd=rag_response.cost_usd,
            query=rag_response.query,
            warning=warning
        )

    except Exception as e:
        handle_exception(
            e,
            context="citation_builder.build_citations",
            raise_as=RAGError
        )


def _parse_cited_source_nums(answer_text: str) -> set[int]:
    """
    Parse [Source N] markers from LLM answer.

    Handles formats:
        [Source 1], [Source 2], [source 3] (case insensitive)

    Args:
        answer_text: Raw LLM answer string

    Returns:
        Set of source numbers cited (1-indexed)
    """
    pattern = r'\[[Ss]ource\s+(\d+)\]'
    matches = re.findall(pattern, answer_text)
    return {int(m) for m in matches}


def _build_single_citation(
    chunk: RetrievedChunk,
    cited_nums: set[int],
) -> Citation:
    """
    Build a single Citation from a RetrievedChunk.

    Args:
        chunk:      Retrieved chunk with metadata
        cited_nums: Set of source numbers cited in LLM answer

    Returns:
        Citation object
    """
    # Source weight for confidence calculation
    source_lower  = chunk.source.lower()
    source_weight = SOURCE_WEIGHTS.get(source_lower, SOURCE_WEIGHTS["unknown"])

    # Confidence = rerank score * source authority weight
    confidence = round(chunk.similarity_score * source_weight, 4)

    # Human-readable regulatory reference
    regulation_ref = format_regulation_ref(
        source=chunk.source,
        part_number=chunk.part_number,
        doc_type=chunk.doc_type
    )

    # Text snippet — first 150 chars
    text_snippet = chunk.text[:150].strip()
    if len(chunk.text) > 150:
        text_snippet += "..."

    # Was this source cited in the answer?
    used_in_answer = chunk.source_num in cited_nums

    return Citation(
        source_num=chunk.source_num,
        regulation_ref=regulation_ref,
        source=chunk.source,
        part_number=chunk.part_number,
        doc_type=chunk.doc_type,
        chunk_id=chunk.chunk_id,
        relevance_score=round(chunk.similarity_score, 4),
        confidence=confidence,
        text_snippet=text_snippet,
        used_in_answer=used_in_answer
    )


def format_cited_response(cited: CitedResponse) -> str:
    """
    Format CitedResponse as human-readable string for CLI/debug output.

    Args:
        cited: CitedResponse object

    Returns:
        Formatted string output
    """
    lines = []
    lines.append(f"\n{'='*65}")
    lines.append(f"QUERY: {cited.query}")
    lines.append(f"{'='*65}")
    lines.append(f"\nANSWER:\n{cited.answer}")
    lines.append(f"\n{'─'*65}")
    lines.append(f"CITATIONS ({len(cited.citations)} total, {len(cited.used_citations)} used in answer):")
    lines.append(f"{'─'*65}")

    for c in cited.citations:
        used_marker = "✓ CITED" if c.used_in_answer else "  retrieved"
        lines.append(
            f"\n[Source {c.source_num}] {used_marker}"
            f"\n  Regulation : {c.regulation_ref}"
            f"\n  Source     : {c.source} | Part: {c.part_number} | Type: {c.doc_type}"
            f"\n  Confidence : {c.confidence:.4f} (rerank={c.relevance_score:.4f})"
            f"\n  Chunk ID   : {c.chunk_id}"
            f"\n  Snippet    : {c.text_snippet}"
        )

    lines.append(f"\n{'─'*65}")
    lines.append(f"AVG CONFIDENCE : {cited.avg_confidence:.4f}")
    lines.append(f"MODEL          : {cited.model_used}")
    lines.append(f"COST           : ${cited.cost_usd:.6f}")
    lines.append(f"LATENCY        : {cited.latency_ms:.0f}ms")
    if cited.warning:
        lines.append(f"⚠ WARNING      : {cited.warning}")
    lines.append(f"{'='*65}\n")

    return "\n".join(lines)


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== AeroLex Citation Builder — Test ===\n")

    from src.rag.rag_chain import RAGChain

    chain = RAGChain(
        collection_name="aerolex_voyage",
        top_k=5,
        use_claude=True,
        auto_filter=True
    )

    queries = [
        "What must a pilot do before beginning a flight?",
        "What are the fuel requirements for VFR flight under Part 91?",
    ]

    for query in queries:
        rag_response = chain.run(query=query)
        cited = build_citations(rag_response)
        print(format_cited_response(cited))