"""
AeroLex — Custom Exception Classes + Global Exception Handler

Every module in AeroLex raises these specific exceptions instead of
generic Python exceptions. This gives us:
- Meaningful error messages
- Consistent error structure across the entire codebase
- Easy filtering in logs (search "IngestionError" to find all ingestion issues)
- Clean API error responses (FastAPI catches these and returns proper HTTP codes)

Usage:
    from src.utils.exception_handler import (
        AeroLexException,
        IngestionError,
        ParsingError,
        handle_exception
    )

    try:
        # your code
    except Exception as e:
        handle_exception(e, context="ecfr_ingestor.download()")
"""

import sys
import traceback
from src.utils.logger import get_logger

# Module logger
logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# BASE EXCEPTION
# ══════════════════════════════════════════════════════════════════════════════

class AeroLexException(Exception):
    """
    Base exception for all AeroLex errors.
    All custom exceptions inherit from this.
    This way you can catch ALL AeroLex errors with one except clause:
        except AeroLexException as e: ...
    """

    def __init__(self, message: str, context: str = "", original_error: Exception = None):
        """
        Args:
            message: Human-readable error description
            context: Where did this happen (e.g., "ecfr_ingestor.download()")
            original_error: The original exception that caused this (for chaining)
        """
        self.message = message
        self.context = context
        self.original_error = original_error

        # Build full error message
        full_message = f"{message}"
        if context:
            full_message += f" | Context: {context}"
        if original_error:
            full_message += f" | Caused by: {type(original_error).__name__}: {str(original_error)}"

        super().__init__(full_message)


# ══════════════════════════════════════════════════════════════════════════════
# SPECIFIC EXCEPTION CLASSES — One per module/layer
# ══════════════════════════════════════════════════════════════════════════════

class ConfigurationError(AeroLexException):
    """Raised when environment variables or config settings are missing/invalid."""
    pass


class IngestionError(AeroLexException):
    """Raised when data ingestion fails (API call, download, scraping)."""
    pass


class ParsingError(AeroLexException):
    """Raised when document parsing fails (PDF, XML, HTML)."""
    pass


class ChunkingError(AeroLexException):
    """Raised when text chunking fails."""
    pass


class EmbeddingError(AeroLexException):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(AeroLexException):
    """Raised when Qdrant vector store operations fail."""
    pass


class RetrievalError(AeroLexException):
    """Raised when document retrieval fails."""
    pass


class RAGError(AeroLexException):
    """Raised when RAG chain execution fails."""
    pass


class AgentError(AeroLexException):
    """Raised when LangGraph agent execution fails."""
    pass


class DatabaseError(AeroLexException):
    """Raised when SQLite database operations fail."""
    pass


class AlertError(AeroLexException):
    """Raised when email alert sending fails."""
    pass


class APIError(AeroLexException):
    """Raised when FastAPI endpoint encounters an error."""
    pass


class MonitoringError(AeroLexException):
    """Raised when MLflow or LangSmith tracking fails."""
    pass


class SchedulerError(AeroLexException):
    """Raised when APScheduler job fails."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL EXCEPTION HANDLER
# ══════════════════════════════════════════════════════════════════════════════

def handle_exception(
    error: Exception,
    context: str = "",
    reraise: bool = False,
    critical: bool = False
) -> None:
    """
    Central exception handler — call this in every except block.

    What it does:
    1. Logs the error with full traceback
    2. Logs context (where it happened)
    3. Optionally re-raises the exception
    4. Marks critical errors separately

    Args:
        error: The caught exception
        context: Where did this happen (module.function description)
        reraise: If True, re-raises the exception after logging
        critical: If True, logs as CRITICAL instead of ERROR

    Example:
        try:
            download_pdf(url)
        except Exception as e:
            handle_exception(e, context="faa_ad_ingestor.download_pdf()", reraise=True)
    """

    # Get full traceback as string
    tb = traceback.format_exc()

    # Build log message
    error_type = type(error).__name__
    error_msg = str(error)

    log_message = (
        f"\n{'='*60}\n"
        f"ERROR TYPE : {error_type}\n"
        f"MESSAGE    : {error_msg}\n"
        f"CONTEXT    : {context if context else 'Not specified'}\n"
        f"TRACEBACK  :\n{tb}"
        f"{'='*60}"
    )

    # Log at appropriate level
    if critical:
        logger.critical(log_message)
    else:
        logger.error(log_message)

    # Re-raise if requested
    if reraise:
        raise error


def setup_global_exception_handler() -> None:
    """
    Sets up a global uncaught exception handler.
    Any exception NOT caught by try/except blocks will be logged here
    instead of just printing to stderr and disappearing.

    Call this once in main entry points (FastAPI app, Streamlit app, scripts).
    """

    def global_handler(exc_type, exc_value, exc_traceback):
        """Handles uncaught exceptions at the top level."""

        # Don't intercept KeyboardInterrupt (Ctrl+C) — let it exit normally
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical(
            "UNCAUGHT EXCEPTION — This should never happen in production!",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

    # Replace Python's default exception handler
    sys.excepthook = global_handler
    logger.info("Global exception handler registered successfully")


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test 1: Custom exception
    print("\n--- Test 1: Custom Exception ---")
    try:
        raise IngestionError(
            message="Failed to download FAA AD PDF",
            context="faa_ad_ingestor.download_pdf()",
            original_error=ConnectionError("Connection timed out")
        )
    except AeroLexException as e:
        handle_exception(e, context="test block 1")

    # Test 2: Generic exception
    print("\n--- Test 2: Generic Exception ---")
    try:
        result = 1 / 0
    except Exception as e:
        handle_exception(e, context="test block 2 — division by zero")

    # Test 3: Global handler
    print("\n--- Test 3: Global Handler Setup ---")
    setup_global_exception_handler()

    print("\n✅ Exception handler working!")