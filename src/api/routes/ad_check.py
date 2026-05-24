"""
ad_check.py — Airworthiness Directive Check Endpoint

WHAT:
    POST /ad-check — Airworthiness Directive lookup endpoint.
    Takes aircraft model and returns relevant AD information
    from the FAA_AD corpus.

WHY:
    ADs are mandatory safety compliance actions — every aircraft
    operator must comply with applicable ADs or ground the aircraft.
    A dedicated endpoint auto-filters to FAA_AD source, ensuring
    only directive content is searched (not advisory circulars
    or general regulations which could dilute results).

HOW:
    1. Validate ADCheckRequest
    2. Auto-build query with AD-specific framing
    3. Force source_filter="faa_ad" — always search AD corpus
    4. Run AeroLexAgent with LOOKUP strategy
    5. Return ADCheckResponse with citations

Official Docs:
    FAA AD database: https://rgl.faa.gov/
    FastAPI routing: https://fastapi.tiangolo.com/tutorial/bigger-applications/
"""

import time
from fastapi import APIRouter, Depends, HTTPException

from src.utils.logger import get_logger
from src.agents.agent_graph import AeroLexAgent
from src.api.dependencies import get_agent
from src.api.schemas import (
    ADCheckRequest,
    ADCheckResponse,
    CitationSchema,
    GateStatusEnum,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/ad-check", tags=["Airworthiness Directives"])


# ── Query Builder ─────────────────────────────────────────────────────────

def _build_ad_query(req: ADCheckRequest) -> str:
    """
    Build AD-specific query from structured request.

    Always frames query around airworthiness directives to ensure
    LOOKUP classification and FAA_AD source filtering.

    Args:
        req: ADCheckRequest with aircraft_model + optional query

    Returns:
        AD-framed query string
    """
    if req.query:
        query = (
            f"{req.query} "
            f"(Aircraft: {req.aircraft_model}, "
            f"Source: FAA Airworthiness Directives)"
        )
    else:
        query = (
            f"What airworthiness directives apply to the "
            f"{req.aircraft_model}? List applicable ADs with "
            f"compliance requirements and effective dates."
        )

    logger.debug(f"Built AD query: '{query[:100]}'")
    return query


# ── Endpoint ──────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=ADCheckResponse,
    summary="Airworthiness Directive Check",
    description="""
    Check applicable Airworthiness Directives for a specific aircraft model.

    Automatically searches the FAA Airworthiness Directive corpus
    and returns applicable ADs with compliance requirements.

    **Note:** ADs are mandatory — non-compliance grounds the aircraft.
    Always verify with the official FAA AD database:
    https://rgl.faa.gov/Regulatory_and_Guidance_Library/rgAD.nsf/

    **Example aircraft models:**
    - Boeing 737-800
    - Airbus A320
    - Cessna 172S
    """,
)
async def ad_check(
    request: ADCheckRequest,
    agent: AeroLexAgent = Depends(get_agent),
) -> ADCheckResponse:
    """POST /ad-check — Airworthiness Directive lookup endpoint."""
    start_time = time.time()
    logger.info(
        f"POST /ad-check | "
        f"aircraft='{request.aircraft_model}'"
    )

    try:
        # ── Build query ──
        query = _build_ad_query(request)

        # ── Run agent ──
        # Note: agent.run() uses auto_filter which may detect FAA_AD
        # In Phase 6 enhancement we can pass explicit source filter
        agent_response = agent.run(query)

        # ── Map status ──
        status_str = agent_response.get("status", "ERROR")
        try:
            status = GateStatusEnum(status_str)
        except ValueError:
            status = GateStatusEnum.ERROR

        # ── Warning for non-AUTO_APPROVE ──
        warning = None
        if status != GateStatusEnum.AUTO_APPROVE:
            warning = (
                f"AeroLex could not find sufficient AD information for "
                f"{request.aircraft_model} in current corpus. "
                f"Always verify with official FAA AD database: "
                f"https://rgl.faa.gov"
            )

        latency = (time.time() - start_time) * 1000

        logger.info(
            f"POST /ad-check complete | "
            f"status={status.value} | "
            f"latency={latency:.0f}ms"
        )

        return ADCheckResponse(
            aircraft_model=request.aircraft_model,
            query_used=query,
            answer=agent_response.get("answer", ""),
            status=status,
            citations=[],
            confidence=agent_response.get("confidence", 0.0),
            cost_usd=agent_response.get("cost_usd", 0.0),
            latency_ms=latency,
            gate_id=agent_response.get("gate_id", ""),
            warning=warning,
        )

    except Exception as e:
        logger.error(f"POST /ad-check error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"AD check failed: {str(e)}"
        )