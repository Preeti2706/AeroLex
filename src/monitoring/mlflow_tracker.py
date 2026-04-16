"""
AeroLex — MLflow Experiment Tracker

Provides a clean interface for logging experiments to MLflow.

What gets tracked:
- Chunking experiments: strategy, chunk_size, overlap, RAGAS scores
- Embedding experiments: model, dimensions, RAGAS scores
- Retrieval experiments: method, top_k, precision, recall
- RAG evaluation: faithfulness, answer_relevance, context_recall
- LLM comparison: Claude vs GPT — latency, cost, quality scores

Why wrap MLflow?
- Adds AeroLex-specific context to every run
- Handles MLflow server unavailability gracefully
- Single place to change tracking behavior
- Clean API: log_experiment() instead of mlflow.log_params() everywhere

Usage:
    from src.monitoring.mlflow_tracker import MLflowTracker
    tracker = MLflowTracker()

    with tracker.start_run(experiment="chunking", run_name="recursive_512"):
        tracker.log_params({"strategy": "recursive", "chunk_size": 512})
        tracker.log_metrics({"faithfulness": 0.87, "latency_ms": 1200})
"""

import os
import mlflow
from contextlib import contextmanager
from typing import Optional
from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, MonitoringError
from config.settings import settings
from config.mlflow_config import init_mlflow, get_experiment_name

load_dotenv()
logger = get_logger(__name__)


class MLflowTracker:
    """
    AeroLex wrapper around MLflow for experiment tracking.

    Tracks all experiments across phases:
    - Phase 2: Chunking + Embedding experiments
    - Phase 3: Retrieval experiments
    - Phase 4: RAG evaluation
    - Phase 5: LLM comparison (Claude vs GPT)
    """

    def __init__(self):
        """Initialize MLflow connection."""
        self.enabled = False
        self.tracking_uri = settings.MLFLOW_TRACKING_URI
        self._init()

    def _init(self) -> None:
        """Connect to MLflow server."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)

            # Test connection by listing experiments
            experiments = mlflow.search_experiments()
            self.enabled = True
            logger.info(
                f"MLflow tracker initialized | "
                f"URI: {self.tracking_uri} | "
                f"Experiments found: {len(experiments)}"
            )

        except Exception as e:
            logger.warning(
                f"MLflow server not reachable at {self.tracking_uri} — "
                f"tracking disabled. Start with: mlflow ui --port 5000"
            )
            self.enabled = False

    @contextmanager
    def start_run(
        self,
        experiment: str,
        run_name: str,
        tags: Optional[dict] = None
    ):
        """
        Context manager for MLflow runs.

        Why context manager?
        - Automatically ends the run even if an exception occurs
        - Clean syntax: 'with tracker.start_run(...):'
        - No need to manually call mlflow.end_run()

        Args:
            experiment: Phase key ('chunking', 'embedding', 'retrieval', 'rag', 'llm')
            run_name: Descriptive name for this run
            tags: Optional dict of tags

        Usage:
            with tracker.start_run("chunking", "recursive_512") as run:
                tracker.log_params({"chunk_size": 512})
                tracker.log_metrics({"faithfulness": 0.87})
        """
        if not self.enabled:
            logger.warning("MLflow disabled — yielding dummy context")
            yield None
            return

        try:
            # Set experiment
            experiment_name = get_experiment_name(experiment)
            mlflow.set_experiment(experiment_name)

            # Default tags for every AeroLex run
            default_tags = {
                "project": "aerolex",
                "env": settings.APP_ENV,
                "phase": experiment,
            }
            if tags:
                default_tags.update(tags)

            # Start run
            with mlflow.start_run(run_name=run_name, tags=default_tags) as run:
                logger.info(
                    f"MLflow run started | "
                    f"Experiment: {experiment_name} | "
                    f"Run: {run_name} | "
                    f"Run ID: {run.info.run_id}"
                )
                yield run
                logger.info(f"MLflow run completed | Run ID: {run.info.run_id}")

        except Exception as e:
            handle_exception(e, context=f"MLflowTracker.start_run({experiment}, {run_name})")
            yield None

    def log_params(self, params: dict) -> None:
        """
        Log parameters for current run.
        Parameters = settings/config (chunk_size, model_name, strategy)
        These don't change during a run.

        Args:
            params: Dict of parameter name → value
        """
        if not self.enabled:
            return
        try:
            mlflow.log_params(params)
            logger.debug(f"MLflow params logged: {params}")
        except Exception as e:
            logger.warning(f"MLflow log_params failed (non-critical): {e}")

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """
        Log metrics for current run.
        Metrics = results/scores (faithfulness, latency, cost)
        These can change over steps.

        Args:
            metrics: Dict of metric name → float value
            step: Optional step number (for tracking metrics over time)
        """
        if not self.enabled:
            return
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"MLflow metrics logged: {metrics}")
        except Exception as e:
            logger.warning(f"MLflow log_metrics failed (non-critical): {e}")

    def log_artifact(self, local_path: str) -> None:
        """
        Log a file as artifact (e.g., evaluation report, config file).

        Args:
            local_path: Path to file to upload
        """
        if not self.enabled:
            return
        try:
            mlflow.log_artifact(local_path)
            logger.info(f"MLflow artifact logged: {local_path}")
        except Exception as e:
            logger.warning(f"MLflow log_artifact failed (non-critical): {e}")

    def log_llm_comparison(
        self,
        query: str,
        claude_result: dict,
        openai_result: dict
    ) -> None:
        """
        Log a side-by-side Claude vs OpenAI comparison experiment.

        Args:
            query: The query that was tested
            claude_result: Dict with keys: model, latency_ms, cost_usd, tokens, score
            openai_result: Dict with keys: model, latency_ms, cost_usd, tokens, score
        """
        if not self.enabled:
            logger.warning("MLflow disabled — skipping LLM comparison log")
            return

        try:
            experiment_name = get_experiment_name("llm")
            mlflow.set_experiment(experiment_name)

            # Log Claude run
            with mlflow.start_run(
                run_name=f"claude_{claude_result.get('model', 'unknown')}",
                tags={"provider": "anthropic", "project": "aerolex"}
            ):
                mlflow.log_param("model", claude_result.get("model"))
                mlflow.log_param("query_preview", query[:100])
                mlflow.log_metrics({
                    "latency_ms":   claude_result.get("latency_ms", 0),
                    "cost_usd":     claude_result.get("cost_usd", 0),
                    "total_tokens": claude_result.get("tokens", 0),
                    "quality_score":claude_result.get("score", 0),
                })

            # Log OpenAI run
            with mlflow.start_run(
                run_name=f"openai_{openai_result.get('model', 'unknown')}",
                tags={"provider": "openai", "project": "aerolex"}
            ):
                mlflow.log_param("model", openai_result.get("model"))
                mlflow.log_param("query_preview", query[:100])
                mlflow.log_metrics({
                    "latency_ms":   openai_result.get("latency_ms", 0),
                    "cost_usd":     openai_result.get("cost_usd", 0),
                    "total_tokens": openai_result.get("tokens", 0),
                    "quality_score":openai_result.get("score", 0),
                })

            logger.info(
                f"LLM comparison logged | "
                f"Claude: {claude_result.get('model')} vs "
                f"OpenAI: {openai_result.get('model')}"
            )

        except Exception as e:
            handle_exception(e, context="MLflowTracker.log_llm_comparison()")

    def get_best_run(self, experiment: str, metric: str, mode: str = "max") -> Optional[dict]:
        """
        Get the best run for an experiment based on a metric.
        Useful for picking the best chunking/embedding strategy.

        Args:
            experiment: Phase key ('chunking', 'embedding', etc.)
            metric: Metric name to optimize (e.g., 'faithfulness')
            mode: 'max' to maximize, 'min' to minimize

        Returns:
            dict: Best run info or None
        """
        if not self.enabled:
            return None

        try:
            experiment_name = get_experiment_name(experiment)
            order = "DESC" if mode == "max" else "ASC"

            runs = mlflow.search_runs(
                experiment_names=[experiment_name],
                order_by=[f"metrics.{metric} {order}"],
                max_results=1
            )

            if runs.empty:
                logger.info(f"No runs found for experiment: {experiment_name}")
                return None

            best = runs.iloc[0].to_dict()
            logger.info(
                f"Best run for {experiment_name} by {metric}: "
                f"{best.get(f'metrics.{metric}', 'N/A')}"
            )
            return best

        except Exception as e:
            handle_exception(e, context="MLflowTracker.get_best_run()")
            return None


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Testing MLflow Tracker ---\n")
    print("⚠️  Make sure MLflow server is running: mlflow ui --port 5000\n")

    # Initialize experiments first
    init_mlflow()

    tracker = MLflowTracker()

    if not tracker.enabled:
        print("❌ MLflow server not running — start it first!")
        exit(1)

    # Test 1: Log a chunking experiment
    print("Test 1: Logging chunking experiment...")
    with tracker.start_run("chunking", "recursive_chunk_512"):
        tracker.log_params({
            "strategy":   "recursive",
            "chunk_size": 512,
            "overlap":    50,
            "model":      "text-embedding-3-small"
        })
        tracker.log_metrics({
            "faithfulness":       0.87,
            "answer_relevance":   0.91,
            "context_recall":     0.83,
            "latency_ms":         1200,
            "cost_usd":           0.0023
        })
    print("  ✅ Chunking experiment logged")

    # Test 2: Log LLM comparison
    print("\nTest 2: Logging Claude vs GPT comparison...")
    tracker.log_llm_comparison(
        query="What are B787 APU compliance requirements?",
        claude_result={
            "model":      "claude-sonnet-4-20250514",
            "latency_ms": 1800,
            "cost_usd":   0.0105,
            "tokens":     1900,
            "score":      0.92
        },
        openai_result={
            "model":      "gpt-4o-mini",
            "latency_ms": 950,
            "cost_usd":   0.000465,
            "tokens":     1900,
            "score":      0.88
        }
    )
    print("  ✅ LLM comparison logged")

    # Test 3: Get best run
    print("\nTest 3: Getting best chunking run...")
    best = tracker.get_best_run("chunking", "faithfulness", mode="max")
    if best:
        print(f"  Best faithfulness: {best.get('metrics.faithfulness', 'N/A')}")

    print(f"\n✅ MLflow tracker working!")
    print(f"View experiments at: http://localhost:5000")