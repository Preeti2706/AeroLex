"""
AeroLex — MLflow Configuration & Experiment Tracking Setup

MLflow tracks:
- Chunking experiments (recursive vs semantic vs hierarchical)
- Embedding experiments (OpenAI vs sentence-transformers)
- Retrieval experiments (dense vs hybrid vs hybrid+reranker)
- RAG evaluation metrics (faithfulness, answer relevance, context recall)
- LLM comparison (Claude vs GPT — latency, cost, quality)

Usage:
    from config.mlflow_config import init_mlflow, get_experiment_id
    init_mlflow()
"""

import os
import mlflow
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, MonitoringError
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)

# ── MLflow Settings ───────────────────────────────────────────────────────────

# Tracking URI — where MLflow stores data
# Local: "http://localhost:5000" (MLflow server)
# Remote: can point to AWS S3, Azure, GCS in production
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# ── Experiment Names — one per major phase ────────────────────────────────────
# Why separate experiments? So you can filter and compare within a phase cleanly
EXPERIMENTS = {
    "chunking":   "aerolex-chunking-experiments",
    "embedding":  "aerolex-embedding-experiments",
    "retrieval":  "aerolex-retrieval-experiments",
    "rag":        "aerolex-rag-evaluation",
    "llm":        "aerolex-llm-comparison",  # Claude vs GPT
    "ingestion":  "aerolex-ingestion-runs",
}


def init_mlflow() -> bool:
    """
    Initialize MLflow connection and create all experiments.

    Returns:
        bool: True if successful, False if MLflow server not reachable
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")

        # Create all experiments if they don't exist
        for key, experiment_name in EXPERIMENTS.items():
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    mlflow.create_experiment(experiment_name)
                    logger.info(f"MLflow experiment created: {experiment_name}")
                else:
                    logger.info(f"MLflow experiment exists: {experiment_name}")
            except Exception as e:
                logger.warning(f"Could not create experiment {experiment_name}: {e}")

        logger.info("MLflow initialization complete")
        return True

    except Exception as e:
        handle_exception(e, context="mlflow_config.init_mlflow()")
        return False


def get_experiment_name(phase: str) -> str:
    """
    Get experiment name for a given phase.

    Args:
        phase: One of 'chunking', 'embedding', 'retrieval', 'rag', 'llm', 'ingestion'

    Returns:
        str: MLflow experiment name
    """
    if phase not in EXPERIMENTS:
        raise MonitoringError(
            message=f"Unknown experiment phase: {phase}",
            context="mlflow_config.get_experiment_name()"
        )
    return EXPERIMENTS[phase]


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Testing MLflow Config ---")
    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    print("\n⚠️  MLflow server must be running for full test.")
    print("Start it with: mlflow ui --port 5000")
    print("Then open: http://localhost:5000")
    print("\nExperiments that will be created:")
    for key, name in EXPERIMENTS.items():
        print(f"  {key:12} → {name}")