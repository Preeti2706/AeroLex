"""
compliance.py — Regulatory Compliance Check Endpoint

WHAT:
    POST /compliance — multi-hop compliance advisory endpoint.
    Takes an operational scenario and returns step-by-step
    compliance analysis with COMPLIANT/NON_COMPLIANT/UNCLEAR verdict.

WHY:
    Compliance queries are the highest-value use case for AeroLex.
    A pilot asking "Can I depart with an inoperative altimeter?"
    needs a structured, traceable answer — not a generic paragraph.
    This endpoint enforces ADVISORY query type + chain-of-thought
    reasoning regardless of how the question is phrased.

Official Docs:
    FastAPI routing: https://fastapi.tiangolo.com/tutorial/bigger-applications/
"""
from typing import Optional
import re
import time
from fastapi import APIRouter, Depends, HTTPException

from src.utils.logger import get_logger
from src.agents.agent_graph import AeroLexAgent
from src.api.dependencies import get_agent
from src.api.schemas import (
    ComplianceRequest,
    ComplianceResponse,
    CitationSchema,
    GateStatusEnum,
    ComplianceVerdictEnum,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/compliance", tags=["Compliance"])


# ── Query Builder ─────────────────────────────────────────────────────────

def _build_compliance_query(req: ComplianceRequest) -> str:
    """
    Build compliance advisory query from structured scenario.

    Forces ADVISORY framing — "Is it compliant to..." phrasing
    ensures QueryClassifier always routes to ADVISORY pipeline
    with chain-of-thought reasoning.

    Args:
        req: ComplianceRequest with scenario + optional part

    Returns:
        ADVISORY-framed query string
    """
    part_ref = f" under FAR Part {req.regulation_part}" if req.regulation_part else ""
    jurisdiction_ref = f" ({req.jurisdiction} regulations)" if req.jurisdiction else ""

    query = (
        f"Is it compliant to: {req.scenario}"
        f"{part_ref}{jurisdiction_ref}? "
        f"Provide step-by-step regulatory analysis and compliance verdict."
    )

    logger.debug(f"Built compliance query: '{query[:100]}'")
    return query


# ── Reasoning Steps Extractor ─────────────────────────────────────────────

def _extract_reasoning_steps(answer: str) -> list[str]:
    """
    Extract chain-of-thought reasoning steps from ADVISORY answer.

    The advisory_chain_of_thought prompt produces structured steps:
    Step 1 — Identify Applicable Regulations
    Step 2 — Analyze Requirements
    Step 3 — Apply to Scenario
    Step 4 — Compliance Verdict
    Step 5 — Recommended Action

    Args:
        answer: LLM-generated advisory answer

    Returns:
        List of reasoning step strings (max 5)
    """
    steps = []

    # Match "**Step N — Title**: content" or "Step N: content"
    step_pattern = re.compile(
        r'\*?\*?Step\s+\d+[:\s—–-]+([^\n]+)',
        re.IGNORECASE
    )
    matches = step_pattern.findall(answer)

    if matches:
        for match in matches:
            clean = match.strip().replace("**", "")
            if len(clean) > 5:
                steps.append(clean)
    else:
        # Fallback: extract numbered items
        lines = answer.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+[.)]\s+.{10,}', line):
                clean = re.sub(r'^\d+[.)]\s*', '', line).strip()
                steps.append(clean)

    return steps[:5]


# ── Escalation Contact ────────────────────────────────────────────────────

def _get_escalation(jurisdiction: str, verdict: str) -> Optional[str]:
    """Get escalation contact for UNCLEAR or NON_COMPLIANT verdicts."""
    from typing import Optional
    if verdict in ["NON_COMPLIANT", "UNCLEAR"]:
        contacts = {
            "FAA":  "FAA FSDO — https://www.faa.gov/contact | 1-866-TELL-FAA",
            "DGCA": "DGCA Regional Airworthiness Office — https://www.dgca.gov.in",
        }
        return contacts.get(jurisdiction, "Contact your local aviation authority")
    return None


# ── Endpoint ──────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=ComplianceResponse,
    summary="Regulatory Compliance Check",
    description="""
    Check whether an operational scenario is compliant with aviation regulations.

    Uses multi-hop chain-of-thought reasoning to:
    1. Identify all applicable regulations
    2. Analyze requirements
    3. Apply to the specific scenario
    4. Return COMPLIANT / NON_COMPLIANT / UNCLEAR verdict

    **Example scenarios:**
    - "Departing with an inoperative altimeter on a VFR flight"
    - "Flying without an MEL when equipment is inoperative"
    - "Operating below VFR weather minimums under special VFR"
    """,
)
async def compliance_check(
    request: ComplianceRequest,
    agent: AeroLexAgent = Depends(get_agent),
) -> ComplianceResponse:
    """POST /compliance — Regulatory compliance advisory endpoint."""
    start_time = time.time()
    logger.info(
        f"POST /compliance | "
        f"scenario='{request.scenario[:60]}' | "
        f"part={request.regulation_part} | "
        f"jurisdiction={request.jurisdiction}"
    )

    try:
        # ── Build query ──
        query = _build_compliance_query(request)

        # ── Run agent ──
        agent_response = agent.run(query)

        # ── Extract reasoning steps ──
        reasoning_steps = _extract_reasoning_steps(
            agent_response.get("answer", "")
        )

        # ── Map status ──
        status_str = agent_response.get("status", "ERROR")
        try:
            status = GateStatusEnum(status_str)
        except ValueError:
            status = GateStatusEnum.ERROR

        # ── Map verdict ──
        verdict_str = agent_response.get("compliance_verdict")
        verdict = None
        if verdict_str:
            try:
                verdict = ComplianceVerdictEnum(verdict_str)
            except ValueError:
                verdict = None

        # ── Escalation contact ──
        escalation = _get_escalation(
            request.jurisdiction,
            verdict.value if verdict else ""
        )

        # ── Warning ──
        warning = None
        if status in [GateStatusEnum.HOLD, GateStatusEnum.BLOCK]:
            warning = (
                "Insufficient regulatory context for definitive answer. "
                "Do not make compliance decisions based on this response alone."
            )

        latency = (time.time() - start_time) * 1000

        logger.info(
            f"POST /compliance complete | "
            f"verdict={verdict} | "
            f"status={status.value} | "
            f"latency={latency:.0f}ms"
        )

        return ComplianceResponse(
            scenario=request.scenario,
            query_used=query,
            compliance_verdict=verdict,
            answer=agent_response.get("answer", ""),
            status=status,
            reasoning_steps=reasoning_steps,
            citations=[],
            confidence=agent_response.get("confidence", 0.0),
            cost_usd=agent_response.get("cost_usd", 0.0),
            latency_ms=latency,
            gate_id=agent_response.get("gate_id", ""),
            escalation_contact=escalation,
            warning=warning,
        )

    except Exception as e:
        logger.error(f"POST /compliance error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Compliance check failed: {str(e)}"
        )