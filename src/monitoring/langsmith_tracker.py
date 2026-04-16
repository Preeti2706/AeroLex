"""
AeroLex — LangSmith Tracker

Wraps LangSmith's client to provide:
- Custom run logging with AeroLex-specific metadata
- RAG pipeline step tracking
- LLM call metadata (model, tokens, cost, latency)
- HITL feedback logging
- Easy enable/disable for testing

Why wrap LangSmith instead of using it directly?
- Adds AeroLex-specific metadata to every trace
- Single place to change tracking behavior
- Can mock this in tests without hitting LangSmith API
- Adds error handling so tracking failures never break the main app

Usage:
    from src.monitoring.langsmith_tracker import LangSmithTracker
    tracker = LangSmithTracker()
    tracker.log_rag_query(
        query="What are B787 APU requirements?",
        retrieved_chunks=chunks,
        answer="...",
        model="claude-sonnet-4-20250514",
        latency_ms=1200,
        tokens_used=1900
    )
"""

import os
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, MonitoringError
from config.settings import settings

load_dotenv()
logger = get_logger(__name__)


class LangSmithTracker:
    """
    AeroLex wrapper around LangSmith for structured LLM observability.

    Tracks:
    - RAG queries (query → retrieval → generation)
    - LLM calls (model, tokens, cost, latency)
    - Agent steps (classifier → planner → router → synthesizer)
    - HITL feedback (accurate / partial / wrong)
    """

    def __init__(self):
        """Initialize LangSmith client."""
        self.enabled = settings.LANGSMITH_TRACING == "true"
        self.project = settings.LANGSMITH_PROJECT
        self.client = None

        if self.enabled:
            self._init_client()

    def _init_client(self) -> None:
        """Initialize LangSmith client with error handling."""
        try:
            from langsmith import Client

            self.client = Client(
                api_url=settings.LANGSMITH_ENDPOINT,
                api_key=settings.LANGSMITH_API_KEY
            )

            # Set env vars for automatic LangChain tracing
            os.environ["LANGSMITH_TRACING"]    = "true"
            os.environ["LANGSMITH_PROJECT"]    = self.project
            os.environ["LANGSMITH_API_KEY"]    = settings.LANGSMITH_API_KEY
            os.environ["LANGSMITH_ENDPOINT"]   = settings.LANGSMITH_ENDPOINT

            logger.info(f"LangSmith tracker initialized — Project: {self.project}")

        except Exception as e:
            handle_exception(e, context="LangSmithTracker._init_client()")
            self.enabled = False
            logger.warning("LangSmith tracking DISABLED due to initialization error")

    def log_rag_query(
        self,
        query: str,
        retrieved_chunks: list,
        answer: str,
        model: str,
        latency_ms: float,
        tokens_used: int,
        cost_usd: float = 0.0,
        confidence_score: float = 0.0,
        metadata: Optional[dict] = None
    ) -> Optional[str]:
        """
        Log a complete RAG query to LangSmith.

        Args:
            query: User's input query
            retrieved_chunks: List of retrieved document chunks
            answer: Generated answer
            model: LLM model used
            latency_ms: Total pipeline latency
            tokens_used: Total tokens consumed
            cost_usd: Total cost in USD
            confidence_score: RAG confidence score (0-1)
            metadata: Additional metadata dict

        Returns:
            str: LangSmith run ID if successful, None otherwise
        """
        if not self.enabled or not self.client:
            logger.debug("LangSmith tracking disabled — skipping log_rag_query")
            return None

        try:
            run_metadata = {
                "query": query,
                "model": model,
                "num_chunks_retrieved": len(retrieved_chunks),
                "latency_ms": round(latency_ms, 2),
                "tokens_used": tokens_used,
                "cost_usd": round(cost_usd, 6),
                "confidence_score": round(confidence_score, 4),
                "timestamp": datetime.now().isoformat(),
                "phase": "rag_pipeline",
                "project": self.project,
                **(metadata or {})
            }

            # Create a run in LangSmith
            run = self.client.create_run(
                name="rag_query",
                run_type="chain",
                inputs={"query": query, "num_chunks": len(retrieved_chunks)},
                outputs={"answer": answer, "confidence": confidence_score},
                extra={"metadata": run_metadata},
                project_name=self.project,
                tags=["rag", model, settings.APP_ENV]
            )

            logger.info(
                f"LangSmith RAG trace logged | "
                f"Model: {model} | Chunks: {len(retrieved_chunks)} | "
                f"Latency: {latency_ms:.0f}ms | Cost: ${cost_usd:.6f}"
            )

            return str(run.id) if run else None

        except Exception as e:
            # IMPORTANT: Tracking failures should NEVER break main app
            # Just log warning and continue
            logger.warning(f"LangSmith logging failed (non-critical): {e}")
            return None

    def log_llm_call(
        self,
        prompt: str,
        response: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        context: str = ""
    ) -> Optional[str]:
        """
        Log a single LLM API call to LangSmith.

        Args:
            prompt: The prompt sent to the LLM
            response: The response received
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count
            latency_ms: Response latency
            context: Where this call happened

        Returns:
            str: LangSmith run ID if successful
        """
        if not self.enabled or not self.client:
            return None

        try:
            run = self.client.create_run(
                name="llm_call",
                run_type="llm",
                inputs={"prompt": prompt, "model": model},
                outputs={"response": response},
                extra={
                    "metadata": {
                        "model": model,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                        "latency_ms": round(latency_ms, 2),
                        "context": context,
                        "timestamp": datetime.now().isoformat()
                    }
                },
                project_name=self.project,
                tags=["llm_call", model, settings.APP_ENV]
            )

            logger.info(
                f"LangSmith LLM trace logged | "
                f"Model: {model} | "
                f"Tokens: {input_tokens}+{output_tokens} | "
                f"Latency: {latency_ms:.0f}ms"
            )

            return str(run.id) if run else None

        except Exception as e:
            logger.warning(f"LangSmith LLM logging failed (non-critical): {e}")
            return None

    def log_feedback(
        self,
        run_id: str,
        feedback: str,
        comment: str = ""
    ) -> None:
        """
        Log HITL feedback for a specific run.

        Args:
            run_id: LangSmith run ID to attach feedback to
            feedback: One of 'accurate', 'partial', 'wrong'
            comment: Optional free-text correction
        """
        if not self.enabled or not self.client:
            return

        try:
            # Map feedback to numeric score
            score_map = {
                "accurate": 1.0,
                "partial":  0.5,
                "wrong":    0.0
            }
            score = score_map.get(feedback, 0.5)

            self.client.create_feedback(
                run_id=run_id,
                key="user_feedback",
                score=score,
                comment=comment or feedback
            )

            logger.info(f"HITL feedback logged | Run: {run_id} | Score: {score} | Feedback: {feedback}")

        except Exception as e:
            logger.warning(f"LangSmith feedback logging failed (non-critical): {e}")

    def get_project_stats(self) -> Optional[dict]:
        """
        Get basic stats for the current project from LangSmith.

        Returns:
            dict: Project stats or None if unavailable
        """
        if not self.enabled or not self.client:
            return None

        try:
            runs = list(self.client.list_runs(
                project_name=self.project,
                limit=100
            ))

            if not runs:
                logger.info("No runs found in LangSmith project yet")
                return {"total_runs": 0}

            stats = {
                "total_runs": len(runs),
                "project": self.project,
                "latest_run": runs[0].start_time.isoformat() if runs else None
            }

            logger.info(f"LangSmith project stats: {stats}")
            return stats

        except Exception as e:
            logger.warning(f"Could not fetch LangSmith stats: {e}")
            return None


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Testing LangSmith Tracker ---\n")

    tracker = LangSmithTracker()

    # Test 1: Log a simulated RAG query
    print("Test 1: Logging simulated RAG query...")
    run_id = tracker.log_rag_query(
        query="What are the compliance requirements for B787 APU operations?",
        retrieved_chunks=["chunk1", "chunk2", "chunk3"],
        answer="Based on FAA 14 CFR Part 121...",
        model="claude-sonnet-4-20250514",
        latency_ms=1800,
        tokens_used=1900,
        cost_usd=0.0105,
        confidence_score=0.92
    )
    print(f"  Run ID: {run_id}")

    # Test 2: Log a simulated LLM call
    print("\nTest 2: Logging simulated LLM call...")
    llm_run_id = tracker.log_llm_call(
        prompt="Summarize the following regulation...",
        response="The regulation states...",
        model="claude-sonnet-4-20250514",
        input_tokens=500,
        output_tokens=200,
        latency_ms=900,
        context="rag_chain.generate_answer()"
    )
    print(f"  Run ID: {llm_run_id}")

    # Test 3: Get project stats
    print("\nTest 3: Getting project stats...")
    stats = tracker.get_project_stats()
    print(f"  Stats: {stats}")

    print("\n✅ LangSmith tracker working!")
    print("View traces at: https://smith.langchain.com")