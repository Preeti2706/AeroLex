"""
main.py — AeroLex FastAPI Application Entry Point

WHAT:
    Creates and configures the FastAPI application.
    Registers all routers, middleware, and lifecycle events.
    Single entry point for the entire AeroLex API.

WHY:
    FastAPI main.py is the orchestrator — it wires together:
    - Lifespan (startup/shutdown)
    - Routers (preflight, compliance, ad_check)
    - Middleware (CORS, logging, timing)
    - Exception handlers (structured error responses)
    - Health check endpoint

HOW:
    1. Create FastAPI app with lifespan
    2. Add CORS middleware (allows Streamlit to call API)
    3. Register routers with /api/v1 prefix
    4. Add request timing middleware
    5. Add global exception handler
    6. Health check endpoint

CORS:
    Streamlit runs on port 8501.
    FastAPI runs on port 8000.
    Without CORS, browser blocks cross-origin requests.
    We allow localhost:8501 in development.

Official Docs:
    FastAPI: https://fastapi.tiangolo.com/
    CORS: https://fastapi.tiangolo.com/tutorial/cors/
    Middleware: https://fastapi.tiangolo.com/tutorial/middleware/
"""

import time
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# Existing include_router lines ke saath:
from src.api.routes.query import router as query_router

from src.utils.logger import get_logger
from src.api.dependencies import lifespan, get_qdrant_status, get_mlflow_status
from src.api.schemas import HealthResponse, ErrorResponse
from src.api.routes.preflight import router as preflight_router
from src.api.routes.compliance import router as compliance_router
from src.api.routes.ad_check import router as ad_router

logger = get_logger(__name__)


# ── FastAPI App ───────────────────────────────────────────────────────────

app = FastAPI(
    title="AeroLex API",
    description="""
## AeroLex — Aviation Regulatory Compliance Assistant

Production-grade RAG system for FAA and DGCA aviation regulations.
Powered by LangGraph agents, Qdrant vector store, and Claude.

### Endpoints
- **POST /api/v1/query** — Generic regulatory query
- **POST /api/v1/preflight** — Preflight compliance check
- **POST /api/v1/compliance** — Regulatory compliance advisory
- **POST /api/v1/ad-check** — Airworthiness Directive lookup
- **GET /health** — System health check

### Architecture
Query → LangGraph Agent → Hybrid Retrieval → Claude → HITL Gate → Response

### Author
Preeti | United Airlines | FAANG Portfolio Project
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── CORS Middleware ───────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",   # Streamlit dev
        "http://localhost:3000",   # React dev (future)
        "http://127.0.0.1:8501",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Timing Middleware ─────────────────────────────────────────────

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """
    Add X-Process-Time header to every response.

    Measures end-to-end API latency including:
    - Request parsing
    - Agent pipeline execution
    - Response serialization

    This is standard practice in production APIs —
    helps monitor latency trends in dashboards.
    """
    start = time.time()
    response = await call_next(request)
    process_time = (time.time() - start) * 1000
    response.headers["X-Process-Time"] = f"{process_time:.0f}ms"
    logger.debug(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} "
        f"[{process_time:.0f}ms]"
    )
    return response


# ── Global Exception Handler ──────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler — returns structured JSON error.

    Without this, unhandled exceptions return HTML error pages
    which are useless for API clients.

    Returns structured ErrorResponse with:
    - error: Exception type
    - detail: Exception message
    - status: HTTP status code
    """
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: "
        f"{type(exc).__name__}: {exc}\n"
        f"{traceback.format_exc()}"
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=type(exc).__name__,
            detail=str(exc),
            status=500,
        ).model_dump(),
    )


# ── Routers ───────────────────────────────────────────────────────────────

API_PREFIX = "/api/v1"

app.include_router(preflight_router, prefix=API_PREFIX)
app.include_router(compliance_router, prefix=API_PREFIX)
app.include_router(ad_router,         prefix=API_PREFIX)
app.include_router(query_router,      prefix=API_PREFIX)


# ── Health Check ──────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="System Health Check",
    description="Check AeroLex API health — Qdrant, MLflow, agent status.",
)
async def health_check() -> HealthResponse:
    """
    GET /health — system health check.

    Checks:
    - Qdrant connection + available collections
    - MLflow tracking connection
    - Overall API status

    Used by:
    - Docker health checks
    - Load balancer health probes
    - Monitoring dashboards
    """
    qdrant_info = get_qdrant_status()
    mlflow_status = get_mlflow_status()

    logger.info(
        f"Health check | "
        f"qdrant={qdrant_info['status']} | "
        f"mlflow={mlflow_status}"
    )

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        qdrant=qdrant_info["status"],
        mlflow=mlflow_status,
        collections=qdrant_info.get("collections", []),
    )


# ── Root ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"], summary="API Root")
async def root():
    """Root endpoint — API info and links."""
    return {
        "name":        "AeroLex API",
        "version":     "1.0.0",
        "description": "Aviation Regulatory Compliance Assistant",
        "docs":        "/docs",
        "redoc":       "/redoc",
        "health":      "/health",
        "endpoints": {
            "preflight":  "/api/v1/preflight",
            "compliance": "/api/v1/compliance",
            "ad_check":   "/api/v1/ad-check",
        }
    }


# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,       # Auto-reload on code changes (dev only)
        log_level="info",
    )