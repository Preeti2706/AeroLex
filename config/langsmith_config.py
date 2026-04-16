"""
AeroLex — LangSmith Configuration & Tracing Setup

LangSmith traces every LLM call automatically when LANGSMITH_TRACING=true.
What gets tracked per call:
- Full prompt sent to Claude/GPT
- Full response received
- Latency (milliseconds)
- Token usage (input + output)
- Cost (calculated automatically)
- RAG pipeline steps (retrieval → reranking → generation)
- LangGraph agent steps (classifier → planner → router → synthesizer)

Usage:
    from config.langsmith_config import init_langsmith, verify_langsmith
    init_langsmith()
"""

import os
from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, MonitoringError

load_dotenv()
logger = get_logger(__name__)

# ── LangSmith Settings ────────────────────────────────────────────────────────
LANGSMITH_API_KEY    = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT    = os.getenv("LANGSMITH_PROJECT", "aerolex")
LANGSMITH_TRACING    = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
LANGSMITH_ENDPOINT   = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")


def init_langsmith() -> bool:
    """
    Initialize LangSmith tracing.
    Sets required environment variables that LangChain reads automatically.

    Why env vars? LangChain checks these at import time — setting them here
    before any LangChain import ensures tracing is always on.

    Returns:
        bool: True if setup successful, False if API key missing
    """
    try:
        # Validate API key exists
        if not LANGSMITH_API_KEY:
            raise MonitoringError(
                message="LANGSMITH_API_KEY not found in .env",
                context="langsmith_config.init_langsmith()"
            )

        # Set environment variables — LangChain reads these automatically
        os.environ["LANGSMITH_API_KEY"]   = LANGSMITH_API_KEY
        os.environ["LANGSMITH_PROJECT"]   = LANGSMITH_PROJECT
        os.environ["LANGSMITH_TRACING"]   = str(LANGSMITH_TRACING).lower()
        os.environ["LANGSMITH_ENDPOINT"]  = LANGSMITH_ENDPOINT

        logger.info(f"LangSmith initialized — Project: {LANGSMITH_PROJECT}")
        logger.info(f"LangSmith tracing: {'ENABLED' if LANGSMITH_TRACING else 'DISABLED'}")
        logger.info(f"LangSmith endpoint: {LANGSMITH_ENDPOINT}")
        logger.info("View traces at: https://smith.langchain.com")

        return True

    except Exception as e:
        handle_exception(e, context="langsmith_config.init_langsmith()")
        return False


def verify_langsmith() -> bool:
    """
    Verify LangSmith connection by making a test API call.

    Returns:
        bool: True if connection successful
    """
    try:
        from langsmith import Client

        client = Client(
            api_url=LANGSMITH_ENDPOINT,
            api_key=LANGSMITH_API_KEY
        )

        # List projects — simple API call to verify connection
        projects = list(client.list_projects())
        project_names = [p.name for p in projects]

        logger.info(f"LangSmith connection verified!")
        logger.info(f"Available projects: {project_names}")

        if LANGSMITH_PROJECT in project_names:
            logger.info(f"Project '{LANGSMITH_PROJECT}' found ✅")
        else:
            logger.warning(f"Project '{LANGSMITH_PROJECT}' not found — will be created on first trace")

        return True

    except Exception as e:
        handle_exception(e, context="langsmith_config.verify_langsmith()")
        return False


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Testing LangSmith Config ---\n")

    # Step 1: Initialize
    success = init_langsmith()
    if not success:
        print("❌ LangSmith initialization failed — check .env")
        exit(1)

    # Step 2: Verify connection
    print("\n--- Verifying LangSmith Connection ---\n")
    verified = verify_langsmith()

    if verified:
        print(f"\n✅ LangSmith ready!")
        print(f"   Project  : {LANGSMITH_PROJECT}")
        print(f"   Tracing  : {'ENABLED' if LANGSMITH_TRACING else 'DISABLED'}")
        print(f"   Dashboard: https://smith.langchain.com/o/your-org/projects/p/{LANGSMITH_PROJECT}")
    else:
        print("\n❌ LangSmith verification failed — check API key")