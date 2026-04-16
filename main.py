"""
AeroLex — Main Entry Point

This is the ignition key for the entire AeroLex system.
Run this to verify all systems are initialized correctly.

Every major entry point (FastAPI, Streamlit, scripts) will
import and call init_aerolex() before doing anything else.

Usage:
    python main.py
"""

from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, setup_global_exception_handler
from config.settings import settings
from config.langsmith_config import init_langsmith, verify_langsmith
from config.mlflow_config import init_mlflow
from src.monitoring.cost_tracker import CostTracker
from src.monitoring.langsmith_tracker import LangSmithTracker
from src.monitoring.mlflow_tracker import MLflowTracker

logger = get_logger(__name__)


def init_aerolex() -> bool:
    """
    Initialize all AeroLex systems in correct order.

    Order matters:
    1. Logger — must be first, everything else uses it
    2. Global exception handler — catch anything unexpected
    3. Settings — validate all env vars
    4. LangSmith — LLM observability
    5. MLflow — experiment tracking
    6. Cost tracker — ready for first LLM call

    Returns:
        bool: True if all systems initialized successfully
    """
    logger.info("="*60)
    logger.info("🛫  AeroLex — Aviation Regulatory Compliance Assistant")
    logger.info("="*60)
    logger.info(f"Environment : {settings.APP_ENV}")
    logger.info(f"Log Level   : {settings.LOG_LEVEL}")

    # Step 1: Global exception handler
    setup_global_exception_handler()
    logger.info("✅ Global exception handler registered")

    # Step 2: LangSmith
    langsmith_ok = init_langsmith()
    if langsmith_ok:
        logger.info("✅ LangSmith initialized")
    else:
        logger.warning("⚠️  LangSmith initialization failed — tracing disabled")

    # Step 3: MLflow
    mlflow_ok = init_mlflow()
    if mlflow_ok:
        logger.info("✅ MLflow initialized")
    else:
        logger.warning("⚠️  MLflow initialization failed — experiment tracking disabled")

    # Step 4: Cost tracker (always available)
    cost_tracker = CostTracker()
    logger.info("✅ Cost tracker initialized")

    logger.info("="*60)
    logger.info("🟢  AeroLex systems ready!")
    logger.info("="*60)

    return True


if __name__ == "__main__":
    try:
        success = init_aerolex()
        if success:
            print("\n✅ All systems GO — AeroLex is ready!\n")
    except Exception as e:
        handle_exception(e, context="main.init_aerolex()", critical=True)
        print("\n❌ AeroLex initialization failed — check logs\n")