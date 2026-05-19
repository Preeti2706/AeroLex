"""
hitl_gate.py — Human-in-the-Loop Gate for RAG Answers

WHAT:
    Final safety gate before RAG answers reach end users.
    Routes answers to AUTO_APPROVE, HOLD, or BLOCK based on
    ValidationResult from answer_validator.py.

WHY:
    Aviation compliance is safety-critical. Even a well-built RAG
    system can produce low-confidence answers. HITL gate ensures
    a human expert reviews borderline answers before they influence
    real compliance decisions.

    Think of it like ATC clearance — pilot (LLM) is ready, but
    ATC (HITL gate) gives final go/no-go decision.

HOW:
    1. Receive ValidationResult from answer_validator
    2. Apply gate decision logic (HIGH/MEDIUM → approve, LOW → hold, INSUF → block)
    3. Log decision to MLflow for audit trail
    4. Return GateDecision with routing + message

GATE LOGIC:
    HIGH        → AUTO_APPROVE  (serve directly)
    MEDIUM      → AUTO_APPROVE  (serve with disclaimer)
    LOW         → HOLD          (queue for human review)
    INSUFFICIENT→ BLOCK         (never serve, escalate)

Official Docs:
    Anthropic HITL: https://docs.anthropic.com/en/docs/build-with-claude/agentic-and-multi-agent/hitl
    MLflow:         https://mlflow.org/docs/latest/tracking.html
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import mlflow

from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, RAGError
from src.rag.answer_validator import ValidationResult, ConfidenceBand

logger = get_logger(__name__)


# ── Gate Decision Enum ───────────────────────────────────────────────────────

class GateStatus(str, Enum):
    """
    Three possible gate outcomes.

    AUTO_APPROVE: Answer meets quality bar — serve to user
    HOLD:         Answer needs human review — queue it
    BLOCK:        Answer is unsafe — never serve
    """
    AUTO_APPROVE = "AUTO_APPROVE"
    HOLD         = "HOLD"
    BLOCK        = "BLOCK"


# ── Gate Decision Dataclass ──────────────────────────────────────────────────

@dataclass
class GateDecision:
    """
    Complete HITL gate decision for one RAG answer.

    Fields:
        status:           AUTO_APPROVE / HOLD / BLOCK
        gate_id:          Unique ID for audit trail
        query:            Original user query
        answer:           LLM answer (may be replaced with fallback)
        user_message:     What to show the user
        confidence_band:  From ValidationResult
        groundedness:     From ValidationResult
        needs_review:     True if routed to human queue
        reviewer_notes:   Instructions for human reviewer
        timestamp:        Unix timestamp of gate decision
        mlflow_run_id:    MLflow run ID for traceability
    """
    status:           GateStatus
    gate_id:          str
    query:            str
    answer:           str
    user_message:     str
    confidence_band:  ConfidenceBand
    groundedness:     float
    needs_review:     bool
    reviewer_notes:   str = ""
    timestamp:        float = field(default_factory=time.time)
    mlflow_run_id:    str = ""


# ── User Messages ────────────────────────────────────────────────────────────

# What users see for each gate status
USER_MESSAGES = {
    GateStatus.AUTO_APPROVE: (
        "Answer retrieved from AeroLex regulatory database."
    ),
    GateStatus.AUTO_APPROVE: (
        "Answer retrieved from AeroLex regulatory database. "
        "This answer is based on available regulatory context. "
        "Verify with official sources for compliance decisions."
    ),
    GateStatus.HOLD: (
        "Your query requires review by an aviation compliance expert. "
        "A specialist will respond within 24 hours. "
        "For urgent safety matters, contact your local FSDO or DGCA office directly."
    ),
    GateStatus.BLOCK: (
        "AeroLex could not find sufficient regulatory context to answer this query reliably. "
        "Please consult official FAA/DGCA sources or an aviation compliance expert. "
        "FAA: https://www.faa.gov | DGCA: https://www.dgca.gov.in"
    ),
}

# Reviewer instructions for HOLD queue
REVIEWER_NOTES_TEMPLATE = """
HUMAN REVIEW REQUIRED
─────────────────────
Gate ID:          {gate_id}
Query:            {query}
Confidence Band:  {confidence_band}
Groundedness:     {groundedness:.4f}
Insufficient Ctx: {insufficient_context}

LLM Answer:
{answer}

Action Required:
1. Review the LLM answer for regulatory accuracy
2. Check against official FAA/DGCA sources
3. Either APPROVE (send to user) or REJECT (send escalation message)
4. Log decision in MLflow run: {mlflow_run_id}
"""


# ── Core Gate ────────────────────────────────────────────────────────────────

class HITLGate:
    """
    Human-in-the-Loop gate — final safety check before serving answers.

    Usage:
        gate = HITLGate()
        decision = gate.evaluate(validation_result, cited_response)
        if decision.status == GateStatus.AUTO_APPROVE:
            return decision.answer
        else:
            return decision.user_message
    """

    def __init__(self, hitl_threshold: float = 0.85):
        """
        Args:
            hitl_threshold: Confidence score below which HITL triggers.
                           Default 0.85 — conservative for aviation safety.
                           Lower = more answers go to human review.
                           Higher = fewer answers go to human review.
        """
        self.hitl_threshold = hitl_threshold
        logger.info(
            f"HITLGate initialized | "
            f"Threshold: {hitl_threshold}"
        )

    def evaluate(
        self,
        validation: ValidationResult,
        query: str,
        answer: str,
    ) -> GateDecision:
        """
        Evaluate a validated answer and make gate decision.

        Gate logic:
            HIGH        → AUTO_APPROVE
            MEDIUM      → AUTO_APPROVE (with disclaimer in user_message)
            LOW         → HOLD
            INSUFFICIENT→ BLOCK

        Args:
            validation: ValidationResult from answer_validator
            query:      Original user query
            answer:     LLM generated answer

        Returns:
            GateDecision with routing decision + messages

        Raises:
            RAGError: If gate evaluation fails
        """
        try:
            gate_id = str(uuid.uuid4())[:8].upper()
            logger.info(
                f"HITLGate.evaluate() | "
                f"Gate ID: {gate_id} | "
                f"Band: {validation.confidence_band.value} | "
                f"Score: {validation.groundedness_score:.4f}"
            )

            # ── Gate decision logic ──
            status = self._decide_status(validation)

            # ── Build user message ──
            user_message = self._build_user_message(status, validation)

            # ── Build reviewer notes (for HOLD only) ──
            reviewer_notes = ""
            if status == GateStatus.HOLD:
                reviewer_notes = REVIEWER_NOTES_TEMPLATE.format(
                    gate_id=gate_id,
                    query=query,
                    confidence_band=validation.confidence_band.value,
                    groundedness=validation.groundedness_score,
                    insufficient_context=validation.insufficient_context,
                    answer=answer,
                    mlflow_run_id="pending",
                )

            # ── MLflow logging ──
            mlflow_run_id = self._log_to_mlflow(
                gate_id=gate_id,
                status=status,
                validation=validation,
                query=query,
            )

            decision = GateDecision(
                status=status,
                gate_id=gate_id,
                query=query,
                answer=answer if status == GateStatus.AUTO_APPROVE else "",
                user_message=user_message,
                confidence_band=validation.confidence_band,
                groundedness=validation.groundedness_score,
                needs_review=(status == GateStatus.HOLD),
                reviewer_notes=reviewer_notes,
                mlflow_run_id=mlflow_run_id,
            )

            logger.info(
                f"Gate decision: {status.value} | "
                f"Gate ID: {gate_id} | "
                f"Needs review: {decision.needs_review}"
            )

            return decision

        except Exception as e:
            handle_exception(
                e,
                context="HITLGate.evaluate",
                raise_as=RAGError
            )

    def _decide_status(self, validation: ValidationResult) -> GateStatus:
        """
        Map ValidationResult to GateStatus.

        Args:
            validation: ValidationResult from answer_validator

        Returns:
            GateStatus enum value
        """
        band = validation.confidence_band

        if band == ConfidenceBand.HIGH:
            return GateStatus.AUTO_APPROVE

        elif band == ConfidenceBand.MEDIUM:
            return GateStatus.AUTO_APPROVE

        elif band == ConfidenceBand.LOW:
            return GateStatus.HOLD

        else:  # INSUFFICIENT
            return GateStatus.BLOCK

    def _build_user_message(
        self,
        status: GateStatus,
        validation: ValidationResult,
    ) -> str:
        """
        Build appropriate user-facing message.

        HIGH/MEDIUM AUTO_APPROVE: answer + optional disclaimer
        LOW HOLD:                 "under review" message
        INSUFFICIENT BLOCK:       escalation message with official links

        Args:
            status:     Gate status
            validation: ValidationResult

        Returns:
            User-facing message string
        """
        if status == GateStatus.AUTO_APPROVE:
            if validation.confidence_band == ConfidenceBand.MEDIUM:
                return (
                    "Answer retrieved from AeroLex regulatory database. "
                    "Note: This answer is based on available regulatory context. "
                    "Please verify with official FAA/DGCA sources for "
                    "compliance-critical decisions."
                )
            return "Answer retrieved from AeroLex regulatory database."

        elif status == GateStatus.HOLD:
            return USER_MESSAGES[GateStatus.HOLD]

        else:  # BLOCK
            return USER_MESSAGES[GateStatus.BLOCK]

    def _log_to_mlflow(
        self,
        gate_id: str,
        status: GateStatus,
        validation: ValidationResult,
        query: str,
    ) -> str:
        """
        Log gate decision to MLflow for audit trail.

        Args:
            gate_id:    Unique gate decision ID
            status:     Gate status
            validation: ValidationResult
            query:      Original query

        Returns:
            MLflow run ID string
        """
        try:
            with mlflow.start_run(
                run_name=f"hitl_gate_{gate_id}",
                nested=True
            ):
                mlflow.log_param("gate_id", gate_id)
                mlflow.log_param("gate_status", status.value)
                mlflow.log_param("confidence_band", validation.confidence_band.value)
                mlflow.log_param("query", query[:250])
                mlflow.log_metric("groundedness_score", validation.groundedness_score)
                mlflow.log_metric("citation_coverage", validation.citation_coverage)
                mlflow.log_metric("avg_confidence", validation.avg_confidence)
                mlflow.log_metric(
                    "needs_human_review",
                    1.0 if validation.needs_human_review else 0.0
                )
                run_id = mlflow.active_run().info.run_id
                logger.debug(f"MLflow logged | Run ID: {run_id}")
                return run_id

        except Exception as e:
            logger.warning(f"MLflow logging failed (non-critical): {e}")
            return "mlflow_unavailable"


def format_gate_decision(decision: GateDecision) -> str:
    """
    Format GateDecision for CLI/debug output.

    Args:
        decision: GateDecision object

    Returns:
        Formatted string
    """
    status_icons = {
        GateStatus.AUTO_APPROVE: "✅ AUTO_APPROVE",
        GateStatus.HOLD:         "🟡 HOLD",
        GateStatus.BLOCK:        "🔴 BLOCK",
    }

    lines = [
        f"\n{'═'*65}",
        f"HITL GATE DECISION — {status_icons[decision.status]}",
        f"{'═'*65}",
        f"Gate ID        : {decision.gate_id}",
        f"Confidence Band: {decision.confidence_band.value}",
        f"Groundedness   : {decision.groundedness:.4f}",
        f"Needs Review   : {decision.needs_review}",
        f"MLflow Run ID  : {decision.mlflow_run_id}",
        f"\nUser Message:",
        f"  {decision.user_message}",
    ]

    if decision.status == GateStatus.AUTO_APPROVE and decision.answer:
        lines += [
            f"\nAnswer (served to user):",
            f"{decision.answer}",
        ]

    if decision.reviewer_notes:
        lines += [
            f"\nReviewer Notes:",
            f"{decision.reviewer_notes}",
        ]

    lines.append(f"{'═'*65}\n")
    return "\n".join(lines)


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== AeroLex HITL Gate — Test ===\n")

    from src.rag.rag_chain import RAGChain
    from src.rag.citation_builder import build_citations
    from src.rag.answer_validator import validate_answer

    chain = RAGChain(
        collection_name="aerolex_voyage",
        top_k=5,
        use_claude=True,
        auto_filter=True
    )
    gate = HITLGate(hitl_threshold=0.85)

    queries = [
        "What must a pilot do before beginning a flight?",
        "What are the fuel requirements for VFR flight under Part 91?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("─" * 65)

        rag_response = chain.run(query=query)
        cited        = build_citations(rag_response)
        validation   = validate_answer(cited)
        decision     = gate.evaluate(
            validation=validation,
            query=query,
            answer=cited.answer,
        )

        print(format_gate_decision(decision))