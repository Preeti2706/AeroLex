"""
metadata_filter.py — Qdrant Metadata Pre-Filter Builder

WHAT:
    Builds Qdrant Filter objects to narrow vector search scope
    before similarity computation begins.

WHY:
    Without pre-filtering, similarity search scans ALL 624+ chunks.
    With filters, Qdrant searches only the relevant subset —
    faster retrieval + higher precision.

HOW:
    Uses Qdrant's native Filter + FieldCondition + MatchValue API.
    Filters are applied at the HNSW graph traversal level —
    not post-retrieval. This means zero wasted compute.

MATH:
    Unfiltered search: O(log N) over all N vectors
    Filtered search:   O(log M) where M << N (filtered subset)

Official Docs:
    https://qdrant.tech/documentation/concepts/filtering/

Filter Fields Available (from seed_data.py payload):
    - source:       "eCFR" | "FAA_AD" | "DGCA" | "SKYbrary" | "FAA_AC"
    - doc_type:     "regulation" | "directive" | "advisory" | "safety_article"
    - part_number:  "91" | "121" | "135" | "61"
    - aircraft_type:"general" | "transport" | "rotorcraft" | "all"
"""

from typing import Optional
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    #Must,
    #Should,
)
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, RetrievalError

logger = get_logger(__name__)


def build_metadata_filter(
    source: Optional[str] = None,
    sources: Optional[list[str]] = None,
    doc_type: Optional[str] = None,
    part_number: Optional[str] = None,
    aircraft_type: Optional[str] = None,
) -> Optional[Filter]:
    """
    Build a Qdrant Filter object from optional metadata fields.

    Args:
        source:        Single source to filter — "eCFR", "FAA_AD", "DGCA", etc.
        sources:       Multiple sources (OR logic) — ["eCFR", "DGCA"]
        doc_type:      Document type — "regulation", "directive", "advisory"
        part_number:   FAA Part number — "91", "121", "135"
        aircraft_type: Aircraft category — "general", "transport", "rotorcraft"

    Returns:
        Qdrant Filter object if any filter specified, else None.
        None means — search entire collection (no restriction).

    Raises:
        RetrievalError: If filter construction fails.

    Example:
        >>> f = build_metadata_filter(source="eCFR", part_number="91")
        >>> # Qdrant will only search FAA Part 91 regulation chunks
    """
    try:
        must_conditions = []

        # --- Source filter (single) ---
        if source:
            must_conditions.append(
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source)
                )
            )
            logger.debug(f"Filter added — source: {source}")

        # --- Source filter (multiple, OR logic) ---
        if sources and len(sources) > 0:
            must_conditions.append(
                FieldCondition(
                    key="source",
                    match=MatchAny(any=sources)
                )
            )
            logger.debug(f"Filter added — sources (any of): {sources}")

        # --- Doc type filter ---
        if doc_type:
            must_conditions.append(
                FieldCondition(
                    key="doc_type",
                    match=MatchValue(value=doc_type)
                )
            )
            logger.debug(f"Filter added — doc_type: {doc_type}")

        # --- Part number filter ---
        if part_number:
            must_conditions.append(
                FieldCondition(
                    key="part_number",
                    match=MatchValue(value=part_number)
                )
            )
            logger.debug(f"Filter added — part_number: {part_number}")

        # --- Aircraft type filter ---
        if aircraft_type:
            must_conditions.append(
                FieldCondition(
                    key="aircraft_type",
                    match=MatchValue(value=aircraft_type)
                )
            )
            logger.debug(f"Filter added — aircraft_type: {aircraft_type}")

        # --- No filters provided ---
        if not must_conditions:
            logger.info("No filters specified — full collection search will run")
            return None

        qdrant_filter = Filter(must=must_conditions)
        logger.info(f"Metadata filter built — {len(must_conditions)} condition(s) applied")
        return qdrant_filter

    except Exception as e:
        handle_exception(
            e,
            context="metadata_filter.build_metadata_filter",
            raise_as=RetrievalError
        )


def get_filter_for_query_intent(query: str) -> Optional[Filter]:
    """
    Heuristic-based auto filter detection from query text.
    Useful when user doesn't explicitly specify filters.

    Args:
        query: Raw user query string

    Returns:
        Best-guess Qdrant Filter, or None if no intent detected.

    Example:
        >>> f = get_filter_for_query_intent("What does DGCA say about maintenance?")
        >>> # Returns filter with source="DGCA"
    """
    try:
        query_lower = query.lower()

        # DGCA intent
        if any(kw in query_lower for kw in ["dgca", "india", "car ", "indian aviation"]):
            logger.info("Query intent detected — DGCA source filter applied")
            return build_metadata_filter(source="DGCA")

        # FAA Airworthiness Directive intent
        if any(kw in query_lower for kw in ["airworthiness directive", " ad ", "faa ad"]):
            logger.info("Query intent detected — FAA_AD source filter applied")
            return build_metadata_filter(source="FAA_AD")

        # Part 91 intent
        if any(kw in query_lower for kw in ["part 91", "91.", "general aviation", "preflight"]):
            logger.info("Query intent detected — Part 91 filter applied")
            return build_metadata_filter(part_number="91")

        # Advisory Circular intent
        if any(kw in query_lower for kw in ["advisory circular", "faa ac", " ac "]):
            logger.info("Query intent detected — FAA_AC source filter applied")
            return build_metadata_filter(source="FAA_AC")

        logger.info("No query intent detected — no auto filter applied")
        return None

    except Exception as e:
        handle_exception(
            e,
            context="metadata_filter.get_filter_for_query_intent",
            raise_as=RetrievalError
        )


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== AeroLex Metadata Filter — Test ===\n")

    # Test 1: Single source
    f1 = build_metadata_filter(source="eCFR", part_number="91")
    print(f"Test 1 — eCFR + Part 91 filter: {f1}\n")

    # Test 2: Multiple sources
    f2 = build_metadata_filter(sources=["eCFR", "DGCA"], doc_type="regulation")
    print(f"Test 2 — Multi-source filter: {f2}\n")

    # Test 3: No filters
    f3 = build_metadata_filter()
    print(f"Test 3 — No filter (should be None): {f3}\n")

    # Test 4: Auto intent detection
    queries = [
        "What does DGCA say about maintenance intervals?",
        "preflight requirements for general aviation",
        "latest airworthiness directive for Boeing 737",
        "what is the weather minima for IFR?"
    ]
    print("Test 4 — Auto intent detection:")
    for q in queries:
        f = get_filter_for_query_intent(q)
        print(f"  Query: '{q[:50]}'\n  Filter: {f}\n")