"""
AeroLex — Cost & Latency Tracker

Tracks per-call and cumulative costs for Claude and OpenAI API calls.
Also tracks latency (response time) for every LLM call.

Why track cost separately from LangSmith?
- LangSmith shows cost per trace but not cumulative project cost
- We need custom alerts when cost exceeds thresholds
- We need Claude vs GPT cost comparison for experiment decisions

Pricing (as of 2025 — update if prices change):
    Claude claude-sonnet-4-20250514:
        Input:  $3.00 per 1M tokens
        Output: $15.00 per 1M tokens

    Claude claude-haiku-4-5-20251001:
        Input:  $0.80 per 1M tokens
        Output: $4.00 per 1M tokens

    GPT-4o:
        Input:  $2.50 per 1M tokens
        Output: $10.00 per 1M tokens

    GPT-4o-mini:
        Input:  $0.15 per 1M tokens
        Output: $0.60 per 1M tokens

    text-embedding-3-small:
        Input:  $0.02 per 1M tokens

Usage:
    from src.monitoring.cost_tracker import CostTracker
    tracker = CostTracker()
    tracker.log_llm_call(
        model="claude-sonnet-4-20250514",
        input_tokens=500,
        output_tokens=200,
        latency_ms=1200,
        context="rag_chain.query()"
    )
    tracker.print_summary()
"""

import time
import json
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, MonitoringError

logger = get_logger(__name__)

# ── Pricing Table (per 1M tokens in USD) ─────────────────────────────────────
PRICING = {
    # Claude models
    "claude-sonnet-4-20250514": {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5-20251001":        {"input": 0.80,  "output": 4.00},
    "claude-opus-4-20250514":   {"input": 15.00, "output": 75.00},

    # OpenAI models
    "gpt-4o":                   {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":              {"input": 0.15,  "output": 0.60},

    # Embedding models
    "text-embedding-3-small":   {"input": 0.02,  "output": 0.0},
    "text-embedding-3-large":   {"input": 0.13,  "output": 0.0},
}

# ── Alert Thresholds ──────────────────────────────────────────────────────────
ALERT_THRESHOLDS = {
    "single_call_cost_usd":   0.10,   # Alert if single call > $0.10
    "session_cost_usd":       1.00,   # Alert if session total > $1.00
    "latency_ms":             5000,   # Alert if response > 5 seconds
}


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class LLMCallRecord:
    """Stores data for a single LLM API call."""
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    latency_ms: float
    context: str


@dataclass
class SessionSummary:
    """Cumulative stats for the current session."""
    total_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    calls_by_model: dict = field(default_factory=dict)
    cost_by_model: dict = field(default_factory=dict)
    alerts_fired: int = 0


# ── Cost Tracker ──────────────────────────────────────────────────────────────

class CostTracker:
    """
    Tracks cost and latency for all LLM API calls in a session.

    Why a class instead of functions?
    - Maintains state (session totals) across multiple calls
    - Can be instantiated once and shared across modules
    - Easy to extend with database persistence later
    """

    def __init__(self, save_to_file: bool = True):
        """
        Args:
            save_to_file: If True, saves all records to a JSON log file
        """
        self.session = SessionSummary()
        self.records: list[LLMCallRecord] = []
        self.save_to_file = save_to_file

        # Log file for cost records
        self.log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "logs"
        )
        self.cost_log_file = os.path.join(
            self.log_dir,
            f"cost_tracker_{datetime.now().strftime('%Y%m%d')}.json"
        )

        logger.info("CostTracker initialized")

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> tuple[float, float, float]:
        """
        Calculate cost for a given model and token counts.

        Args:
            model: Model name (must be in PRICING table)
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens

        Returns:
            tuple: (input_cost, output_cost, total_cost) in USD
        """
        if model not in PRICING:
            logger.warning(f"Model '{model}' not in pricing table — cost will be 0")
            return 0.0, 0.0, 0.0

        rates = PRICING[model]

        # Cost = (tokens / 1,000,000) * rate_per_million
        input_cost  = (input_tokens  / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]
        total_cost  = input_cost + output_cost

        return input_cost, output_cost, total_cost

    def log_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        context: str = ""
    ) -> LLMCallRecord:
        """
        Log a completed LLM API call with cost and latency.

        Args:
            model: Model used (e.g., "claude-sonnet-4-20250514")
            input_tokens: Input token count
            output_tokens: Output token count
            latency_ms: Response time in milliseconds
            context: Where this call happened

        Returns:
            LLMCallRecord: The logged record
        """
        try:
            input_cost, output_cost, total_cost = self.calculate_cost(
                model, input_tokens, output_tokens
            )

            record = LLMCallRecord(
                timestamp=datetime.now().isoformat(),
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                input_cost_usd=round(input_cost, 8),
                output_cost_usd=round(output_cost, 8),
                total_cost_usd=round(total_cost, 8),
                latency_ms=round(latency_ms, 2),
                context=context
            )

            # Update session totals
            self.records.append(record)
            self.session.total_calls += 1
            self.session.total_tokens += record.total_tokens
            self.session.total_cost_usd += total_cost
            self.session.total_latency_ms += latency_ms

            # Update per-model tracking
            if model not in self.session.calls_by_model:
                self.session.calls_by_model[model] = 0
                self.session.cost_by_model[model] = 0.0
            self.session.calls_by_model[model] += 1
            self.session.cost_by_model[model] += total_cost

            # Log the call
            logger.info(
                f"LLM Call | {model} | "
                f"Tokens: {input_tokens}in + {output_tokens}out = {record.total_tokens} | "
                f"Cost: ${total_cost:.6f} | "
                f"Latency: {latency_ms:.0f}ms | "
                f"Context: {context}"
            )

            # Check alerts
            self._check_alerts(record)

            # Save to file
            if self.save_to_file:
                self._save_record(record)

            return record

        except Exception as e:
            handle_exception(e, context="cost_tracker.log_llm_call()")

    def _check_alerts(self, record: LLMCallRecord) -> None:
        """Check if any alert thresholds are breached."""

        # Single call cost alert
        if record.total_cost_usd > ALERT_THRESHOLDS["single_call_cost_usd"]:
            logger.warning(
                f"🚨 COST ALERT: Single call cost ${record.total_cost_usd:.4f} "
                f"exceeds threshold ${ALERT_THRESHOLDS['single_call_cost_usd']} | "
                f"Model: {record.model} | Context: {record.context}"
            )
            self.session.alerts_fired += 1

        # Session total cost alert
        if self.session.total_cost_usd > ALERT_THRESHOLDS["session_cost_usd"]:
            logger.warning(
                f"🚨 SESSION COST ALERT: Total session cost "
                f"${self.session.total_cost_usd:.4f} "
                f"exceeds threshold ${ALERT_THRESHOLDS['session_cost_usd']}"
            )
            self.session.alerts_fired += 1

        # Latency alert
        if record.latency_ms > ALERT_THRESHOLDS["latency_ms"]:
            logger.warning(
                f"🚨 LATENCY ALERT: Response time {record.latency_ms:.0f}ms "
                f"exceeds threshold {ALERT_THRESHOLDS['latency_ms']}ms | "
                f"Model: {record.model} | Context: {record.context}"
            )
            self.session.alerts_fired += 1

    def _save_record(self, record: LLMCallRecord) -> None:
        """Append record to JSON log file."""
        try:
            # Load existing records
            existing = []
            if os.path.exists(self.cost_log_file):
                with open(self.cost_log_file, "r") as f:
                    existing = json.load(f)

            existing.append(asdict(record))

            with open(self.cost_log_file, "w") as f:
                json.dump(existing, f, indent=2)

        except Exception as e:
            logger.warning(f"Could not save cost record to file: {e}")

    def print_summary(self) -> None:
        """Print a formatted session cost summary."""
        avg_latency = (
            self.session.total_latency_ms / self.session.total_calls
            if self.session.total_calls > 0 else 0
        )

        print("\n" + "="*60)
        print("📊 AEROLEX — SESSION COST SUMMARY")
        print("="*60)
        print(f"  Total Calls     : {self.session.total_calls}")
        print(f"  Total Tokens    : {self.session.total_tokens:,}")
        print(f"  Total Cost      : ${self.session.total_cost_usd:.6f}")
        print(f"  Avg Latency     : {avg_latency:.0f}ms")
        print(f"  Alerts Fired    : {self.session.alerts_fired}")
        print("\n  Cost by Model:")
        for model, cost in self.session.cost_by_model.items():
            calls = self.session.calls_by_model[model]
            print(f"    {model}: ${cost:.6f} ({calls} calls)")
        print("="*60)


# ── Timing Decorator ──────────────────────────────────────────────────────────

def track_latency(func):
    """
    Decorator to measure and log function execution time.

    Usage:
        @track_latency
        def my_function():
            ...

    Why a decorator? Clean way to add timing to any function
    without changing its code — just add @track_latency above it.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        latency_ms = (time.time() - start) * 1000
        logger.info(f"⏱️  {func.__name__} completed in {latency_ms:.2f}ms")
        return result
    return wrapper


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Testing Cost Tracker ---\n")

    tracker = CostTracker(save_to_file=True)

    # Simulate Claude Sonnet call
    tracker.log_llm_call(
        model="claude-sonnet-4-20250514",
        input_tokens=1500,
        output_tokens=400,
        latency_ms=1800,
        context="test_rag_chain.query()"
    )

    # Simulate GPT-4o-mini call
    tracker.log_llm_call(
        model="gpt-4o-mini",
        input_tokens=1500,
        output_tokens=400,
        latency_ms=950,
        context="test_rag_chain.query()"
    )

    # Simulate Claude Haiku call
    tracker.log_llm_call(
        model="claude-haiku-4-5-20251001",
        input_tokens=1500,
        output_tokens=400,
        latency_ms=600,
        context="test_rag_chain.query()"
    )

    # Simulate high latency call — should trigger alert
    tracker.log_llm_call(
        model="gpt-4o",
        input_tokens=3000,
        output_tokens=800,
        latency_ms=6000,
        context="test_complex_query()"
    )

    tracker.print_summary()
    print(f"\n✅ Cost tracker working! Log saved to: {tracker.cost_log_file}")