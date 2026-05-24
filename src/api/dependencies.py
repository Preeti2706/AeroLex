"""
dependencies.py — FastAPI Dependency Injection for AeroLex

WHAT:
    Provides shared, reusable dependencies for FastAPI endpoints.
    Main responsibility: create AeroLexAgent ONCE and share across
    all requests — not re-initialize per request.

WHY:
    AeroLexAgent initialization is expensive:
    - Loads VoyageReranker (API connection)
    - Initializes QdrantStore (connection)
    - Compiles LangGraph StateGraph
    This takes ~15 seconds on cold start.

    Without dependency injection:
        Every request → new AeroLexAgent() → 15 sec wait → bad UX
    With dependency injection:
        App startup → one AeroLexAgent() → shared across all requests
        Every request → reuse existing agent → instant

HOW:
    FastAPI's Depends() system:
    1. lifespan context manager → initialize agent at startup
    2. get_agent() → yield shared instance to each endpoint
    3. Endpoints declare Depends(get_agent) → FastAPI injects it

PATTERN:
    This is the Singleton pattern via FastAPI lifespan.
    Same pattern used in production ML systems:
    - Load model once at startup
    - Serve many requests from same model instance
    - Never reload model per request

Official Docs:
    FastAPI Dependencies: https://fastapi.tiangolo.com/tutorial/dependencies/
    Lifespan: https://fastapi.tiangolo.com/advanced/events/
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Depends

from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, AgentError
from src.agents.agent_graph import AeroLexAgent

logger = get_logger(__name__)

# ── Global Agent Instance ─────────────────────────────────────────────────
# Module-level variable — shared across all requests
# Initialized once at startup, never re-created
_agent: AeroLexAgent | None = None


# ── Lifespan Manager ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    FastAPI lifespan context manager.

    Code BEFORE yield → runs at startup (initialize resources)
    Code AFTER yield  → runs at shutdown (cleanup resources)

    This is the modern FastAPI replacement for @app.on_event("startup").

    Analogy: Airport ground crew before/after a flight.
    Before flight: fuel aircraft, load bags, run checks.
    After flight:  deplane passengers, clean aircraft, log flight.

    Args:
        app: FastAPI application instance

    Yields:
        None — just manages lifecycle
    """
    global _agent

    # ── STARTUP ──
    logger.info("AeroLex API starting up — initializing agent...")
    try:
        _agent = AeroLexAgent()
        logger.info("AeroLexAgent initialized successfully — ready for requests")
    except Exception as e:
        logger.error(f"Failed to initialize AeroLexAgent: {e}")
        raise RuntimeError(f"Startup failed: {e}")

    # ── STARTUP ──
    logger.info("AeroLex API starting up — initializing agent...")
    try:
        _agent = AeroLexAgent()
        logger.info("AeroLexAgent initialized successfully — ready for requests")
        
        # ← ADD THIS BLOCK HERE:
        logger.info("Warming up BM25 index...")
        try:
            from src.retrieval.hybrid_retriever import HybridRetriever
            _warmer = HybridRetriever(
                collection="aerolex_voyage",
                embedding_model="voyage"
            )
            _warmer._build_bm25_index()
            logger.info("BM25 index warmed up — ready for fast retrieval")
        except Exception as warmup_err:
            logger.warning(f"BM25 warmup failed (non-critical): {warmup_err}")
        # ← END ADD

    except Exception as e:
        logger.error(f"Failed to initialize AeroLexAgent: {e}")
        raise RuntimeError(f"Startup failed: {e}")

    yield  # ← API is live and serving requests here

    # ── SHUTDOWN ──
    logger.info("AeroLex API shutting down — cleaning up...")
    _agent = None
    logger.info("Cleanup complete")


# ── Dependency Functions ──────────────────────────────────────────────────

def get_agent() -> AeroLexAgent:
    """
    FastAPI dependency — provides shared AeroLexAgent instance.

    Called by FastAPI's Depends() system for every request.
    Returns the already-initialized global agent — no re-creation.

    Usage in endpoints:
        @router.post("/query")
        async def query(
            request: QueryRequest,
            agent: AeroLexAgent = Depends(get_agent)
        ):
            return agent.run(request.query)

    Raises:
        AgentError: If agent was never initialized (startup failed)
    """
    if _agent is None:
        raise AgentError(
            message="AeroLexAgent not initialized — startup may have failed",
            context="dependencies.get_agent"
        )
    return _agent


def get_settings_dep():
    """
    FastAPI dependency — provides application settings.

    Usage:
        @router.get("/health")
        async def health(settings = Depends(get_settings_dep)):
            return {"env": settings.APP_ENV}
    """
    from config.settings import settings
    return settings


def get_qdrant_status() -> dict:
    """
    Check Qdrant connection status.
    Used by health check endpoint.

    Returns:
        Dict with connection status and collection info
    """
    try:
        from src.retrieval.qdrant_store import QdrantStore
        store = QdrantStore(collection="aerolex_voyage")
        collections = store.client.get_collections().collections
        names = [c.name for c in collections]
        return {
            "status": "connected",
            "collections": names,
        }
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
        return {
            "status": "disconnected",
            "collections": [],
            "error": str(e),
        }


def get_mlflow_status() -> str:
    """
    Check MLflow connection status.
    Used by health check endpoint.

    Returns:
        'connected' or 'disconnected'
    """
    try:
        import mlflow
        mlflow.get_tracking_uri()
        return "connected"
    except Exception:
        return "disconnected"