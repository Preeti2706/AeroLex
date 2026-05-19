"""
answer_validator.py — RAG Answer Quality Validator

WHAT:
    Validates LLM-generated answers for groundedness, confidence,
    and citation coverage. Flags low-quality answers for human review.

WHY:
    Aviation regulations are safety-critical. A hallucinated answer
    about fuel requirements or airworthiness could be dangerous.
    Validation ensures every answer is grounded in retrieved context
    before reaching the end user.

HOW:
    1. Groundedness check — are claims supported by retrieved chunks?
    2. Coverage check — how many sources were actually cited?
    3. Confidence banding — HIGH / MEDIUM / LOW / INSUFFICIENT
    4. HITL flag — should a human review this answer?

MATH:
    groundedness_score = (cited_chunks / total_chunks)
                       × avg_confidence
                       × source_authority_weight

    confidence_band:
        >= 0.75 → HIGH         (safe to serve)
        >= 0.50 → MEDIUM       (serve with warning)
        >= 0.25 → LOW          (flag for review)
        <  0.25 → INSUFFICIENT (block, request human)

HITL Threshold:
    confidence_band in [LOW, INSUFFICIENT] → needs_human_review = True
    "I don't know" pattern detected        → needs_human_review = True
    zero citations used                    → needs_human_review = True

Official Docs:
    RAGAS groundedness: https://docs.ragas.io/en/latest/concepts/metrics/faithfulness.html
    HITL patterns:      https://docs.anthropic.com/en/docs/build-with-claude/agentic-and-multi-agent/hitl
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, RAGError
from src.rag.citation_builder import CitedResponse

logger = get_logger(__name__)


# ── Confidence Band Enum ─────────────────────────────────────────────────────

class ConfidenceBand(str, Enum):
    """
    Four-tier confidence classification.

    HIGH:         Answer is well-grounded, safe to serve directly
    MEDIUM:       Answer is mostly grounded, serve with disclaimer
    LOW:          Answer is weakly grounded, flag for human review
    INSUFFICIENT: Answer has no grounding, block and escalate
    """
    HIGH         = "HIGH"
    MEDIUM       = "MEDIUM"
    LOW          = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


# ── Validation Result ────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Complete validation output for one RAG answer.

    Fields:
        is_valid:              True if answer meets minimum quality bar
        confidence_band:       HIGH / MEDIUM / LOW / INSUFFICIENT
        groundedness_score:    0.0 - 1.0 composite score
        citation_coverage:     cited_chunks / total_chunks ratio
        avg_confidence:        Mean confidence across all citations
        needs_human_review:    True if HITL gate should trigger
        insufficient_context:  True if LLM said "not enough info"
        reasons:               List of validation findings
        recommendation:        What to do with this answer
    """
    is_valid:             bool
    confidence_band:      ConfidenceBand
    groundedness_score:   float
    citation_coverage:    float
    avg_confidence:       float
    needs_human_review:   bool
    insufficient_context: bool
    reasons:              list[str] = field(default_factory=list)
    recommendation:       str = ""


# ── Insufficient Context Patterns ───────────────────────────────────────────

# Patterns that indicate LLM could not answer from context
INSUFFICIENT_PATTERNS = [
    r"does not contain enough information",
    r"cannot be found in the.*context",
    r"context is insufficient",
    r"not enough.*context",
    r"no relevant.*context",
    r"context does not.*contain",
    r"unable to.*answer.*context",
    r"provided.*context.*does not",
]


def _detect_insufficient_context(answer: str) -> bool:
    """
    Detect if LLM answer indicates insufficient context.

    Args:
        answer: Raw LLM answer text

    Returns:
        True if LLM flagged insufficient context
    """
    answer_lower = answer.lower()
    for pattern in INSUFFICIENT_PATTERNS:
        if re.search(pattern, answer_lower):
            return True
    return False


# ── Source Authority Weight ──────────────────────────────────────────────────

SOURCE_AUTHORITY = {
    "ecfr":     1.00,
    "faa_ad":   0.95,
    "dgca":     0.90,
    "faa_ac":   0.85,
    "skybrary": 0.80,
    "unknown":  0.60,
}


def _avg_source_authority(cited_response: CitedResponse) -> float:
    """
    Calculate average source authority weight for used citations.

    Args:
        cited_response: CitedResponse with citation list

    Returns:
        Average authority weight (0.0 - 1.0)
    """
    if not cited_response.used_citations:
        return SOURCE_AUTHORITY["unknown"]

    weights = [
        SOURCE_AUTHORITY.get(c.source.lower(), SOURCE_AUTHORITY["unknown"])
        for c in cited_response.used_citations
    ]
    return sum(weights) / len(weights)


# ── Confidence Band Calculator ───────────────────────────────────────────────

def _calculate_confidence_band(score: float) -> ConfidenceBand:
    """
    Map groundedness score to confidence band.

    Thresholds chosen for aviation safety context:
    - HIGH   >= 0.75: Multiple high-confidence regulatory sources cited
    - MEDIUM >= 0.50: Adequate grounding, minor gaps possible
    - LOW    >= 0.25: Weak grounding, human review recommended
    - INSUF  <  0.25: No reliable grounding, block answer

    Args:
        score: Groundedness score (0.0 - 1.0)

    Returns:
        ConfidenceBand enum value
    """
    if score >= 0.75:
        return ConfidenceBand.HIGH
    elif score >= 0.50:
        return ConfidenceBand.MEDIUM
    elif score >= 0.25:
        return ConfidenceBand.LOW
    else:
        return ConfidenceBand.INSUFFICIENT


# ── Core Validator ───────────────────────────────────────────────────────────

def validate_answer(cited_response: CitedResponse) -> ValidationResult:
    """
    Validate a CitedResponse for groundedness and answer quality.

    Validation pipeline:
    1. Detect insufficient context patterns in answer
    2. Calculate citation coverage ratio
    3. Calculate average confidence from citations
    4. Calculate source authority weight
    5. Compute composite groundedness score
    6. Map to confidence band
    7. Determine HITL flag
    8. Build reasons list + recommendation

    Args:
        cited_response: Output from citation_builder.build_citations()

    Returns:
        ValidationResult with full quality assessment

    Raises:
        RAGError: If validation fails unexpectedly
    """
    try:
        logger.info(
            f"Validating answer | "
            f"Query: '{cited_response.query[:60]}' | "
            f"Citations: {cited_response.total_chunks}"
        )

        reasons = []

        # ── Step 1: Insufficient context detection ──
        insufficient_context = _detect_insufficient_context(cited_response.answer)
        if insufficient_context:
            reasons.append(
                "LLM flagged insufficient context — "
                "answer explicitly states context does not cover query"
            )
            logger.warning("Insufficient context detected in answer")

        # ── Step 2: Citation coverage ──
        total   = cited_response.total_chunks
        used    = len(cited_response.used_citations)
        coverage = used / total if total > 0 else 0.0

        if coverage == 0.0:
            reasons.append("Zero citations used — answer not grounded in any source")
        elif coverage < 0.4:
            reasons.append(
                f"Low citation coverage — only {used}/{total} sources cited"
            )
        else:
            reasons.append(
                f"Citation coverage: {used}/{total} sources cited ({coverage:.0%})"
            )

        # ── Step 3: Average confidence ──
        avg_conf = cited_response.avg_confidence
        if avg_conf < 0.5:
            reasons.append(
                f"Low average confidence ({avg_conf:.3f}) — "
                f"retrieved chunks may not be highly relevant"
            )
        else:
            reasons.append(f"Average confidence: {avg_conf:.3f}")

        # ── Step 4: Source authority ──
        authority = _avg_source_authority(cited_response)
        reasons.append(
            f"Source authority weight: {authority:.2f} "
            f"({'primary regulatory' if authority >= 0.90 else 'mixed/advisory'})"
        )

        # ── Step 5: Groundedness score ──
        # Formula: coverage × avg_confidence × source_authority
        # Each factor penalizes a different quality dimension:
        # - coverage:   did LLM use what was retrieved?
        # - avg_conf:   how relevant were the retrieved chunks?
        # - authority:  how authoritative are the sources?
        if insufficient_context:
            # Hard penalty — LLM itself said context is insufficient
            groundedness_score = min(avg_conf * 0.3, 0.24)
        else:
            groundedness_score = coverage * avg_conf * authority

        groundedness_score = round(groundedness_score, 4)
        reasons.append(f"Groundedness score: {groundedness_score:.4f}")

        # ── Step 6: Confidence band ──
        confidence_band = _calculate_confidence_band(groundedness_score)
        reasons.append(f"Confidence band: {confidence_band.value}")

        # ── Step 7: HITL flag ──
        needs_human_review = (
            confidence_band in [ConfidenceBand.LOW, ConfidenceBand.INSUFFICIENT]
            or insufficient_context
            or coverage == 0.0
        )

        if needs_human_review:
            reasons.append(
                "⚠ HITL TRIGGERED — answer flagged for human review"
            )

        # ── Step 8: is_valid + recommendation ──
        is_valid = confidence_band in [ConfidenceBand.HIGH, ConfidenceBand.MEDIUM]

        recommendation = _build_recommendation(
            confidence_band=confidence_band,
            insufficient_context=insufficient_context,
            needs_human_review=needs_human_review,
            coverage=coverage,
        )

        logger.info(
            f"Validation complete | "
            f"Band: {confidence_band.value} | "
            f"Score: {groundedness_score:.4f} | "
            f"HITL: {needs_human_review}"
        )

        return ValidationResult(
            is_valid=is_valid,
            confidence_band=confidence_band,
            groundedness_score=groundedness_score,
            citation_coverage=coverage,
            avg_confidence=avg_conf,
            needs_human_review=needs_human_review,
            insufficient_context=insufficient_context,
            reasons=reasons,
            recommendation=recommendation,
        )

    except Exception as e:
        handle_exception(
            e,
            context="answer_validator.validate_answer",
            raise_as=RAGError
        )


def _build_recommendation(
    confidence_band: ConfidenceBand,
    insufficient_context: bool,
    needs_human_review: bool,
    coverage: float,
) -> str:
    """
    Build human-readable recommendation string.

    Args:
        confidence_band:      Computed confidence band
        insufficient_context: LLM flagged insufficient context
        needs_human_review:   HITL flag
        coverage:             Citation coverage ratio

    Returns:
        Recommendation string
    """
    if confidence_band == ConfidenceBand.HIGH:
        return (
            "SERVE — Answer is well-grounded in regulatory sources. "
            "Safe to return to user directly."
        )
    elif confidence_band == ConfidenceBand.MEDIUM:
        return (
            "SERVE WITH DISCLAIMER — Answer is adequately grounded "
            "but confidence is moderate. Add disclaimer: "
            "'This answer is based on available regulatory context. "
            "Verify with official sources for compliance decisions.'"
        )
    elif confidence_band == ConfidenceBand.LOW:
        if insufficient_context:
            return (
                "ESCALATE — LLM could not answer from available context. "
                "Expand regulatory corpus or route to human expert."
            )
        return (
            "HUMAN REVIEW — Answer has weak grounding. "
            "Route to aviation compliance expert before serving."
        )
    else:  # INSUFFICIENT
        return (
            "BLOCK — Answer is not grounded in regulatory sources. "
            "Do not serve to user. Escalate to human expert immediately."
        )


def format_validation_result(result: ValidationResult) -> str:
    """
    Format ValidationResult for CLI/debug output.

    Args:
        result: ValidationResult object

    Returns:
        Formatted string
    """
    hitl_marker = "⚠ HITL TRIGGERED" if result.needs_human_review else "✓ No HITL"
    valid_marker = "✓ VALID" if result.is_valid else "✗ INVALID"

    lines = [
        f"\n{'─'*65}",
        f"VALIDATION RESULT — {valid_marker} | {hitl_marker}",
        f"{'─'*65}",
        f"Confidence Band    : {result.confidence_band.value}",
        f"Groundedness Score : {result.groundedness_score:.4f}",
        f"Citation Coverage  : {result.citation_coverage:.0%}",
        f"Avg Confidence     : {result.avg_confidence:.4f}",
        f"Insufficient Ctx   : {result.insufficient_context}",
        f"Needs Human Review : {result.needs_human_review}",
        f"\nReasons:",
    ]
    for r in result.reasons:
        lines.append(f"  • {r}")

    lines += [
        f"\nRecommendation:",
        f"  {result.recommendation}",
        f"{'─'*65}\n",
    ]
    return "\n".join(lines)


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== AeroLex Answer Validator — Test ===\n")

    from src.rag.rag_chain import RAGChain
    from src.rag.citation_builder import build_citations

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
        print(f"\nQuery: {query}")
        print("=" * 65)

        rag_response  = chain.run(query=query)
        cited         = build_citations(rag_response)
        validation    = validate_answer(cited)

        print(f"Answer:\n{cited.answer}")
        print(format_validation_result(validation))