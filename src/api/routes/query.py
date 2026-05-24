from fastapi import APIRouter, Depends
from src.agents.agent_graph import AeroLexAgent
from src.api.dependencies import get_agent
from src.api.schemas import QueryRequest, QueryResponse, GateStatusEnum, ComplianceVerdictEnum, CitationSchema
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])

@router.post("/", response_model=QueryResponse)
async def generic_query(
    request: QueryRequest,
    agent: AeroLexAgent = Depends(get_agent),
) -> QueryResponse:
    logger.info(f"POST /query | '{request.query[:60]}'")
    response = agent.run(request.query)
    
    try:
        status = GateStatusEnum(response.get("status", "ERROR"))
    except ValueError:
        status = GateStatusEnum.ERROR

    verdict = None
    if response.get("compliance_verdict"):
        try:
            verdict = ComplianceVerdictEnum(response["compliance_verdict"])
        except ValueError:
            pass

    return QueryResponse(
        query=request.query,
        answer=response.get("answer", ""),
        status=status,
        query_type=response.get("query_type", "unknown"),
        strategy=response.get("strategy", "unknown"),
        confidence=response.get("confidence", 0.0),
        cost_usd=response.get("cost_usd", 0.0),
        latency_ms=response.get("latency_ms", 0.0),
        gate_id=response.get("gate_id", ""),
        compliance_verdict=verdict,
        citations=[],
    )