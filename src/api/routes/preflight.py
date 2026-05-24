"""
preflight.py — Preflight Check API Endpoint

WHAT:
    POST /preflight — structured preflight compliance check.
    Takes flight_type, aircraft_type, jurisdiction and returns
    a complete compliance analysis with citations.

WHY:
    Preflight is the most common aviation compliance query.
    A structured endpoint (vs generic /query) lets us:
    1. Auto-build the right query from structured inputs
    2. Extract key_requirements as a clean list for UI display
    3. Return jurisdiction-specific context (FAA vs DGCA)
    4. Validate inputs before hitting the expensive LLM pipeline

HOW:
    1. Validate PreflightRequest via Pydantic
    2. Build query string from structured fields
    3. Run AeroLexAgent
    4. Parse answer into key_requirements list
    5. Map agent response → PreflightResponse schema
    6. Return structured JSON

Official Docs:
    FastAPI routing: https://fastapi.tiangolo.com/tutorial/bigger-applications/
"""

import re
import time
from fastapi import APIRouter, Depends, HTTPException

from src.utils.logger import get_logger
from src.agents.agent_graph import AeroLexAgent
from src.api.dependencies import get_agent
from src.api.schemas import (
    PreflightRequest,
    PreflightResponse,
    CitationSchema,
    GateStatusEnum,
    ComplianceVerdictEnum,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/preflight", tags=["Preflight"])


# ── Query Builder ─────────────────────────────────────────────────────────

def _build_preflight_query(req: PreflightRequest) -> str:
    """
    Build natural language query from structured PreflightRequest.

    Converts structured inputs into the query format AeroLexAgent
    expects. This translation layer decouples API design from
    agent prompt design.

    Args:
        req: PreflightRequest with flight_type, aircraft_type, etc.

    Returns:
        Natural language query string

    Examples:
        VFR + general + FAA → "What are the preflight requirements
        for VFR general aviation flight under FAA Part 91 regulations?"
    """
    jurisdiction_map = {
        "FAA":  "FAA Part 91 regulations",
        "DGCA": "DGCA Civil Aviation Requirements",
    }
    reg_ref = jurisdiction_map.get(req.jurisdiction, req.jurisdiction)

    if req.specific_question:
        # Use specific question if provided
        query = (
            f"{req.specific_question} "
            f"(Context: {req.flight_type} flight, "
            f"{req.aircraft_type} aviation, {reg_ref})"
        )
    else:
        # Build generic preflight query
        query = (
            f"What are the preflight requirements for "
            f"{req.flight_type} {req.aircraft_type} aviation flight "
            f"under {reg_ref}?"
        )

    logger.debug(f"Built preflight query: '{query}'")
    return query


# ── Key Requirements Extractor ────────────────────────────────────────────

def _extract_key_requirements(answer: str) -> list[str]:
    """
    Extract key requirements as bullet points from LLM answer.

    Parses the answer text to find enumerated or bulleted requirements.
    Falls back to splitting by sentences if no list structure found.

    Args:
        answer: LLM-generated answer text

    Returns:
        List of requirement strings (max 8)
    """
    requirements = []

    # Look for bullet/numbered list patterns
    lines = answer.split('\n')
    for line in lines:
        line = line.strip()
        # Match: "- item", "* item", "1. item", "• item"
        if re.match(r'^[-*•]\s+.+', line) or re.match(r'^\d+\.\s+.+', line):
            # Clean up the bullet/number prefix
            clean = re.sub(r'^[-*•\d\.]+\s*', '', line).strip()
            if len(clean) > 10:  # Skip very short fragments
                requirements.append(clean)

    # Fallback: extract sentences with "shall", "must", "required"
    if not requirements:
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        for sent in sentences:
            if any(kw in sent.lower() for kw in ["shall", "must", "required", "pilot"]):
                clean = sent.strip()
                if 20 < len(clean) < 300:
                    requirements.append(clean)

    return requirements[:8]  # Cap at 8 requirements


# ── Citation Mapper ───────────────────────────────────────────────────────

def _map_citations(agent_response: dict) -> list[CitationSchema]:
    """
    Map agent response citations to CitationSchema objects.

    Agent response contains raw citation data from citation_builder.
    This function maps it to the API's Pydantic schema.

    Args:
        agent_response: Dict from AeroLexAgent.run()

    Returns:
        List of CitationSchema objects
    """
    # Agent response doesn't directly expose citations
    # They are embedded in the validation pipeline
    # For now return empty list — Phase 6b will add citation passthrough
    return []


# ── Escalation Contact ────────────────────────────────────────────────────

def _get_escalation_contact(jurisdiction: str) -> str:
    """Get appropriate escalation contact based on jurisdiction."""
    contacts = {
        "FAA":  "FAA Flight Standards District Office (FSDO) — https://www.faa.gov/contact",
        "DGCA": "DGCA Regional Office — https://www.dgca.gov.in/digigov-portal/",
    }
    return contacts.get(jurisdiction, "Relevant aviation authority")


# ── Endpoint ──────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=PreflightResponse,
    summary="Preflight Compliance Check",
    description="""
    Check preflight regulatory requirements for a specific flight type.

    Automatically queries the AeroLex knowledge base for the relevant
    regulatory authority (FAA Part 91 or DGCA CARs) and returns a
    structured compliance analysis with citations.

    **Flight Types:** VFR, IFR, SVFR
    **Aircraft Types:** general, transport, rotorcraft
    **Jurisdictions:** FAA, DGCA
    """,
    responses={
        200: {"description": "Preflight analysis returned"},
        422: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error"},
    }
)
async def preflight_check(
    request: PreflightRequest,
    agent: AeroLexAgent = Depends(get_agent),
) -> PreflightResponse:
    """
    POST /preflight — Preflight compliance check endpoint.

    Takes structured flight parameters and returns regulatory
    requirements with citations and compliance analysis.
    """
    start_time = time.time()
    logger.info(
        f"POST /preflight | "
        f"flight_type={request.flight_type} | "
        f"aircraft_type={request.aircraft_type} | "
        f"jurisdiction={request.jurisdiction}"
    )

    try:
        # ── Build query ──
        query = _build_preflight_query(request)

        # ── Run agent ──
        agent_response = agent.run(query)

        # ── Extract requirements ──
        key_requirements = _extract_key_requirements(
            agent_response.get("answer", "")
        )

        # ── Map citations ──
        citations = _map_citations(agent_response)

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

        # ── Warning for HOLD/BLOCK ──
        warning = None
        if status in [GateStatusEnum.HOLD, GateStatusEnum.BLOCK]:
            warning = (
                f"This query requires expert review. "
                f"Contact: {_get_escalation_contact(request.jurisdiction)}"
            )

        latency = (time.time() - start_time) * 1000

        logger.info(
            f"POST /preflight complete | "
            f"status={status.value} | "
            f"latency={latency:.0f}ms"
        )

        return PreflightResponse(
            flight_type=request.flight_type,
            aircraft_type=request.aircraft_type,
            jurisdiction=request.jurisdiction,
            query_used=query,
            answer=agent_response.get("answer", ""),
            status=status,
            compliance_verdict=verdict,
            key_requirements=key_requirements,
            citations=citations,
            confidence=agent_response.get("confidence", 0.0),
            cost_usd=agent_response.get("cost_usd", 0.0),
            latency_ms=latency,
            gate_id=agent_response.get("gate_id", ""),
            warning=warning,
        )

    except Exception as e:
        logger.error(f"POST /preflight error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Preflight check failed: {str(e)}"
        )