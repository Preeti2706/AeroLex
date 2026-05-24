"""
schemas.py — Pydantic Request/Response Models for AeroLex API

WHAT:
    Defines all data contracts for the AeroLex FastAPI endpoints.
    Every request body and response body is typed here.

WHY:
    Pydantic schemas give us:
    1. Automatic request validation — wrong types = 422 error before
       our code even runs
    2. Auto-generated OpenAPI docs — FastAPI uses these to build
       /docs (Swagger UI) and /redoc automatically
    3. Type safety — IDE autocomplete + mypy catches bugs early
    4. Serialization — Pydantic handles dict ↔ model conversion

HOW:
    BaseModel for all schemas.
    Request schemas → validate incoming JSON.
    Response schemas → structure outgoing JSON.
    Field() → default values + validation + description for docs.

DESIGN:
    Three endpoint groups, each with Request + Response:
    1. /query        — generic query endpoint
    2. /preflight    — preflight check specific
    3. /compliance   — regulatory compliance check
    4. /ad-check     — airworthiness directive check

Official Docs:
    Pydantic v2: https://docs.pydantic.dev/latest/
    FastAPI schemas: https://fastapi.tiangolo.com/tutorial/body/
"""

from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────────────────

class QueryTypeEnum(str, Enum):
    LOOKUP     = "LOOKUP"
    COMPARISON = "COMPARISON"
    ADVISORY   = "ADVISORY"
    UNKNOWN    = "unknown"


class GateStatusEnum(str, Enum):
    AUTO_APPROVE = "AUTO_APPROVE"
    HOLD         = "HOLD"
    BLOCK        = "BLOCK"
    ERROR        = "ERROR"


class ComplianceVerdictEnum(str, Enum):
    COMPLIANT     = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    UNCLEAR       = "UNCLEAR"


# ── Generic Query ─────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    Generic query request — used by POST /query.
    Works for any aviation regulatory question.
    """
    query: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Aviation regulatory question",
        examples=["What does 14 CFR 91.103 say about preflight requirements?"]
    )
    source_filter: Optional[str] = Field(
        default=None,
        description="Optional source filter — 'ecfr', 'dgca', 'faa_ad', 'faa_ac', 'skybrary'",
        examples=["ecfr"]
    )
    collection: str = Field(
        default="aerolex_voyage",
        description="Qdrant collection to search"
    )


class CitationSchema(BaseModel):
    """Single citation from retrieved chunk."""
    source_num:      int
    regulation_ref:  str
    source:          str
    part_number:     str
    confidence:      float
    text_snippet:    str
    used_in_answer:  bool


class QueryResponse(BaseModel):
    """
    Generic query response — returned by POST /query.
    Complete pipeline output with full provenance.
    """
    query:               str
    answer:              str
    status:              GateStatusEnum
    query_type:          QueryTypeEnum
    strategy:            str
    confidence:          float
    cost_usd:            float
    latency_ms:          float
    gate_id:             str
    compliance_verdict:  Optional[ComplianceVerdictEnum] = None
    citations:           list[CitationSchema] = Field(default_factory=list)
    warning:             Optional[str] = None
    error:               Optional[str] = None


# ── Preflight Check ───────────────────────────────────────────────────────

class PreflightRequest(BaseModel):
    """
    Preflight check request — POST /preflight.
    Structured input for preflight compliance queries.
    """
    flight_type: str = Field(
        ...,
        description="Type of flight — 'VFR', 'IFR', 'SVFR'",
        examples=["VFR"]
    )
    aircraft_type: str = Field(
        ...,
        description="Aircraft category — 'general', 'transport', 'rotorcraft'",
        examples=["general"]
    )
    jurisdiction: str = Field(
        default="FAA",
        description="Regulatory jurisdiction — 'FAA', 'DGCA'",
        examples=["FAA"]
    )
    specific_question: Optional[str] = Field(
        default=None,
        description="Specific preflight question (optional)",
        examples=["What fuel reserves are required for VFR day flight?"]
    )


class PreflightResponse(BaseModel):
    """
    Preflight check response — POST /preflight.
    """
    flight_type:          str
    aircraft_type:        str
    jurisdiction:         str
    query_used:           str
    answer:               str
    status:               GateStatusEnum
    compliance_verdict:   Optional[ComplianceVerdictEnum] = None
    key_requirements:     list[str] = Field(default_factory=list)
    citations:            list[CitationSchema] = Field(default_factory=list)
    confidence:           float
    cost_usd:             float
    latency_ms:           float
    gate_id:              str
    warning:              Optional[str] = None


# ── Compliance Check ──────────────────────────────────────────────────────

class ComplianceRequest(BaseModel):
    """
    Compliance check request — POST /compliance.
    Structured input for regulatory compliance advisory.
    """
    scenario: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Describe the operational scenario to check for compliance",
        examples=["I want to depart with an inoperative altimeter on a VFR flight"]
    )
    regulation_part: Optional[str] = Field(
        default=None,
        description="Specific FAR part to check against — '91', '121', '135'",
        examples=["91"]
    )
    jurisdiction: str = Field(
        default="FAA",
        description="Regulatory jurisdiction",
        examples=["FAA"]
    )


class ComplianceResponse(BaseModel):
    """
    Compliance check response — POST /compliance.
    """
    scenario:             str
    query_used:           str
    compliance_verdict:   Optional[ComplianceVerdictEnum]
    answer:               str
    status:               GateStatusEnum
    reasoning_steps:      list[str] = Field(default_factory=list)
    citations:            list[CitationSchema] = Field(default_factory=list)
    confidence:           float
    cost_usd:             float
    latency_ms:           float
    gate_id:              str
    escalation_contact:   Optional[str] = None
    warning:              Optional[str] = None


# ── AD Check ─────────────────────────────────────────────────────────────

class ADCheckRequest(BaseModel):
    """
    Airworthiness Directive check — POST /ad-check.
    """
    aircraft_model: str = Field(
        ...,
        description="Aircraft make and model",
        examples=["Boeing 737-800"]
    )
    query: Optional[str] = Field(
        default=None,
        description="Specific AD question (optional)",
        examples=["What ADs apply to the CFM56-7B engine?"]
    )


class ADCheckResponse(BaseModel):
    """
    AD check response — POST /ad-check.
    """
    aircraft_model:    str
    query_used:        str
    answer:            str
    status:            GateStatusEnum
    citations:         list[CitationSchema] = Field(default_factory=list)
    confidence:        float
    cost_usd:          float
    latency_ms:        float
    gate_id:           str
    warning:           Optional[str] = None


# ── Health Check ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response — GET /health."""
    status:      str = "healthy"
    version:     str = "1.0.0"
    qdrant:      str = "connected"
    mlflow:      str = "connected"
    collections: list[str] = Field(default_factory=list)


# ── Error ─────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error response."""
    error:   str
    detail:  Optional[str] = None
    status:  int = 500