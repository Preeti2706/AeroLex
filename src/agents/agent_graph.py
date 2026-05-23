"""
agent_graph.py — LangGraph StateGraph Orchestrator

WHAT:
    Wires all Phase 4 + Phase 5 components into a single
    LangGraph StateGraph — the complete AeroLex agent.

WHY:
    Individual components work in isolation but need an
    orchestrator that:
    - Manages shared state across all nodes
    - Routes conditionally based on intermediate results
    - Handles errors at each step gracefully
    - Provides checkpointing for long-running queries
    - Creates a clean entry point for FastAPI (Phase 6)

HOW:
    LangGraph StateGraph — directed graph where:
    - Nodes = processing functions (classify, plan, route, synthesize, validate, gate)
    - Edges = data flow between nodes
    - State = shared TypedDict passed between all nodes
    - Conditional edges = smart routing based on state

GRAPH STRUCTURE:
    START
      ↓
    classify_node       ← QueryClassifier
      ↓
    plan_node           ← RetrievalPlanner
      ↓
    route_node          ← RetrievalRouter
      ↓
    synthesize_node     ← AnswerSynthesizer
      ↓
    validate_node       ← answer_validator
      ↓
    gate_node           ← HITLGate
      ↓
    END

CONDITIONAL ROUTING:
    After gate_node:
    - AUTO_APPROVE → END (serve answer)
    - HOLD         → END (serve hold message)
    - BLOCK        → END (serve block message)

STATE DESIGN:
    Why TypedDict shared state over direct function calls?
    - LangGraph checkpoints state at each node — resumable
    - LangSmith traces each node independently — debuggable
    - Human-in-the-loop: pause after any node, inject human input
    - Parallel execution: independent nodes can run concurrently

Official Docs:
    LangGraph: https://langchain-ai.github.io/langgraph/concepts/
    StateGraph: https://langchain-ai.github.io/langgraph/reference/graphs/
"""

import time
from typing import TypedDict, Optional, Any
from dataclasses import dataclass

import mlflow
from langgraph.graph import StateGraph, END, START

from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, AgentError
from src.agents.query_classifier import QueryClassifier, ClassificationResult, QueryType
from src.agents.planner import RetrievalPlanner, RetrievalPlan
from src.agents.router import RetrievalRouter, RouterResult
from src.agents.synthesizer import AnswerSynthesizer, SynthesisResult
from src.rag.citation_builder import build_citations, CitedResponse
from src.rag.answer_validator import validate_answer, ValidationResult
from src.alerts.hitl_gate import HITLGate, GateDecision, GateStatus
from config.settings import settings

logger = get_logger(__name__)


# ── Agent State ──────────────────────────────────────────────────────────────

class AeroLexState(TypedDict):
    """
    Shared state passed between all LangGraph nodes.

    Why TypedDict?
    - Type safety — each field has explicit type
    - LangGraph serializes this for checkpointing
    - Every node reads + writes to this single object
    - No hidden state — full observability

    Design principle: Accumulate, never overwrite.
    Each node ADDS its output to state.
    Previous node outputs remain available downstream.
    """
    # ── Input ──
    query:          str

    # ── Node outputs ── (None until node runs)
    classification: Optional[ClassificationResult]
    plan:           Optional[RetrievalPlan]
    router_result:  Optional[RouterResult]
    synthesis:      Optional[SynthesisResult]
    cited:          Optional[CitedResponse]
    validation:     Optional[ValidationResult]
    gate_decision:  Optional[GateDecision]

    # ── Final output ──
    final_answer:   Optional[str]
    final_status:   Optional[str]   # AUTO_APPROVE / HOLD / BLOCK

    # ── Metadata ──
    total_latency_ms: Optional[float]
    total_cost_usd:   Optional[float]
    error:            Optional[str]
    start_time:       Optional[float]


# ── Node Functions ───────────────────────────────────────────────────────────

def classify_node(state: AeroLexState) -> AeroLexState:
    """
    Node 1: Classify query into LOOKUP / COMPARISON / ADVISORY.
    """
    logger.info(f"[Node: classify] Query: '{state['query'][:60]}'")

    classifier     = QueryClassifier()
    classification = classifier.classify(state["query"])

    logger.info(
        f"[Node: classify] → {classification.query_type.value} | "
        f"confidence={classification.confidence:.2f}"
    )

    return {**state, "classification": classification}


def plan_node(state: AeroLexState) -> AeroLexState:
    """
    Node 2: Build retrieval plan from classification.
    """
    logger.info(
        f"[Node: plan] Type: {state['classification'].query_type.value}"
    )

    planner = RetrievalPlanner()
    plan    = planner.plan(state["classification"])

    logger.info(
        f"[Node: plan] → Strategy: {plan.strategy.value} | "
        f"top_k={plan.total_top_k} | "
        f"hops={len(plan.hop_queries)}"
    )

    return {**state, "plan": plan}


def route_node(state: AeroLexState) -> AeroLexState:
    """
    Node 3: Execute retrieval plan — fetch chunks.
    """
    logger.info(
        f"[Node: route] Strategy: {state['plan'].strategy.value}"
    )

    router        = RetrievalRouter()
    router_result = router.route(state["plan"])

    logger.info(
        f"[Node: route] → Chunks: {router_result.final_count} | "
        f"Latency: {router_result.latency_ms:.0f}ms"
    )

    return {**state, "router_result": router_result}


def synthesize_node(state: AeroLexState) -> AeroLexState:
    """
    Node 4: Generate answer from retrieved chunks.
    """
    logger.info(
        f"[Node: synthesize] Chunks: {state['router_result'].final_count}"
    )

    synthesizer = AnswerSynthesizer()
    synthesis   = synthesizer.synthesize(
        router_result=state["router_result"],
        classification=state["classification"],
    )

    logger.info(
        f"[Node: synthesize] → Cost: ${synthesis.cost_usd:.6f} | "
        f"Confidence: {synthesis.confidence:.3f}"
    )

    return {**state, "synthesis": synthesis}


def validate_node(state: AeroLexState) -> AeroLexState:
    """
    Node 5: Build citations + validate answer quality.

    Bridges Phase 4 (RAG chain) and Phase 5 (agents).
    Creates a RAGResponse-compatible object from SynthesisResult
    so citation_builder and answer_validator can process it.
    """
    logger.info("[Node: validate] Building citations + validating")

    synthesis = state["synthesis"]

    # Build a mock RAGResponse from SynthesisResult
    # citation_builder expects RAGResponse structure
    from src.rag.rag_chain import RAGResponse, RetrievedChunk

    # Convert router chunks to RetrievedChunk objects
    retrieved_chunks = []
    for i, chunk in enumerate(state["router_result"].chunks, 1):
        retrieved_chunks.append(RetrievedChunk(
            text=chunk.get("text", ""),
            source=chunk.get("source", "unknown"),
            doc_type=chunk.get("doc_type", "unknown"),
            part_number=chunk.get("part_number", "unknown"),
            chunk_id=chunk.get("chunk_id", "unknown"),
            similarity_score=chunk.get(
                "rerank_score", chunk.get("weighted_score", 0.0)
            ),
            source_num=i,
        ))

    rag_response = RAGResponse(
        answer=synthesis.answer,
        sources=retrieved_chunks,
        confidence=synthesis.confidence,
        model_used="claude-sonnet-4-5",
        latency_ms=synthesis.latency_ms,
        input_tokens=synthesis.input_tokens,
        output_tokens=synthesis.output_tokens,
        cost_usd=synthesis.cost_usd,
        query=synthesis.query,
    )

    # Build citations
    cited = build_citations(rag_response)

    # Validate
    validation = validate_answer(cited)

    logger.info(
        f"[Node: validate] → Band: {validation.confidence_band.value} | "
        f"Score: {validation.groundedness_score:.4f} | "
        f"HITL: {validation.needs_human_review}"
    )

    return {**state, "cited": cited, "validation": validation}


def gate_node(state: AeroLexState) -> AeroLexState:
    """
    Node 6: HITL gate — final go/no-go decision.
    """
    logger.info(
        f"[Node: gate] Band: {state['validation'].confidence_band.value}"
    )

    gate     = HITLGate()
    decision = gate.evaluate(
        validation=state["validation"],
        query=state["query"],
        answer=state["synthesis"].answer,
    )

    # Build final answer based on gate decision
    if decision.status == GateStatus.AUTO_APPROVE:
        final_answer = state["synthesis"].answer
    else:
        final_answer = decision.user_message

    # Calculate totals
    total_latency = (time.time() - state["start_time"]) * 1000
    total_cost    = state["synthesis"].cost_usd

    logger.info(
        f"[Node: gate] → Status: {decision.status.value} | "
        f"Total latency: {total_latency:.0f}ms | "
        f"Total cost: ${total_cost:.6f}"
    )

    return {
        **state,
        "gate_decision":    decision,
        "final_answer":     final_answer,
        "final_status":     decision.status.value,
        "total_latency_ms": total_latency,
        "total_cost_usd":   total_cost,
    }


# ── Graph Builder ────────────────────────────────────────────────────────────

def build_aerolex_graph() -> StateGraph:
    """
    Build and compile the AeroLex LangGraph StateGraph.

    Graph structure:
        START → classify → plan → route → synthesize → validate → gate → END

    Returns:
        Compiled StateGraph ready for invocation
    """
    graph = StateGraph(AeroLexState)

    # ── Add nodes ──
    graph.add_node("classify",   classify_node)
    graph.add_node("plan",       plan_node)
    graph.add_node("route",      route_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("validate",   validate_node)
    graph.add_node("gate",       gate_node)

    # ── Add edges ──
    graph.add_edge(START,        "classify")
    graph.add_edge("classify",   "plan")
    graph.add_edge("plan",       "route")
    graph.add_edge("route",      "synthesize")
    graph.add_edge("synthesize", "validate")
    graph.add_edge("validate",   "gate")
    graph.add_edge("gate",       END)

    return graph.compile()


# ── AeroLex Agent ────────────────────────────────────────────────────────────

class AeroLexAgent:
    """
    Complete AeroLex Agent — single entry point for all queries.

    Wraps the LangGraph StateGraph with:
    - MLflow run tracking
    - Error handling
    - Clean response formatting

    Usage:
        agent = AeroLexAgent()
        response = agent.run("What does 91.103 say about preflight?")
        print(response["answer"])
        print(response["status"])   # AUTO_APPROVE / HOLD / BLOCK
        print(response["cost_usd"])
    """

    def __init__(self):
        self.graph = build_aerolex_graph()
        logger.info("AeroLexAgent initialized — graph compiled")

    def run(self, query: str) -> dict[str, Any]:
        """
        Run the complete AeroLex pipeline for a query.

        Args:
            query: User's aviation regulatory question

        Returns:
            Dict with answer, status, cost, latency, metadata
        """
        logger.info(f"AeroLexAgent.run() | Query: '{query[:80]}'")

        with mlflow.start_run(run_name="aerolex_agent"):
            mlflow.log_param("query", query[:250])

            try:
                # Initial state
                initial_state: AeroLexState = {
                    "query":          query,
                    "classification": None,
                    "plan":           None,
                    "router_result":  None,
                    "synthesis":      None,
                    "cited":          None,
                    "validation":     None,
                    "gate_decision":  None,
                    "final_answer":   None,
                    "final_status":   None,
                    "total_latency_ms": None,
                    "total_cost_usd":   None,
                    "error":          None,
                    "start_time":     time.time(),
                }

                # Run graph
                final_state = self.graph.invoke(initial_state)

                # Log to MLflow
                mlflow.log_metric(
                    "total_latency_ms",
                    final_state.get("total_latency_ms", 0)
                )
                mlflow.log_metric(
                    "total_cost_usd",
                    final_state.get("total_cost_usd", 0)
                )
                mlflow.log_param(
                    "final_status",
                    final_state.get("final_status", "unknown")
                )
                mlflow.log_param(
                    "query_type",
                    final_state["classification"].query_type.value
                    if final_state.get("classification") else "unknown"
                )

                return {
                    "query":        query,
                    "answer":       final_state.get("final_answer", ""),
                    "status":       final_state.get("final_status", "UNKNOWN"),
                    "query_type":   final_state["classification"].query_type.value
                                    if final_state.get("classification") else "unknown",
                    "strategy":     final_state["plan"].strategy.value
                                    if final_state.get("plan") else "unknown",
                    "confidence":   final_state["validation"].groundedness_score
                                    if final_state.get("validation") else 0.0,
                    "cost_usd":     final_state.get("total_cost_usd", 0.0),
                    "latency_ms":   final_state.get("total_latency_ms", 0.0),
                    "gate_id":      final_state["gate_decision"].gate_id
                                    if final_state.get("gate_decision") else "",
                    "compliance_verdict": final_state["synthesis"].compliance_verdict
                                          if final_state.get("synthesis") else None,
                }

            except Exception as e:
                logger.error(f"AeroLexAgent error: {e}")
                mlflow.log_param("error", str(e)[:250])
                return {
                    "query":   query,
                    "answer":  "AeroLex encountered an error processing your query. Please try again.",
                    "status":  "ERROR",
                    "error":   str(e),
                    "cost_usd": 0.0,
                    "latency_ms": 0.0,
                }


def format_agent_response(response: dict) -> str:
    """Format AeroLexAgent response for CLI output."""

    status_icons = {
        "AUTO_APPROVE": "✅ AUTO_APPROVE",
        "HOLD":         "🟡 HOLD",
        "BLOCK":        "🔴 BLOCK",
        "ERROR":        "❌ ERROR",
    }

    verdict_icons = {
        "COMPLIANT":     "✅ COMPLIANT",
        "NON_COMPLIANT": "❌ NON-COMPLIANT",
        "UNCLEAR":       "🟡 UNCLEAR",
        None:            "N/A",
    }

    lines = [
        f"\n{'═'*65}",
        f"AEROLEX AGENT RESPONSE",
        f"{'═'*65}",
        f"Query      : {response['query']}",
        f"Status     : {status_icons.get(response['status'], response['status'])}",
        f"Type       : {response.get('query_type', 'unknown').upper()}",
        f"Strategy   : {response.get('strategy', 'unknown')}",
        f"Confidence : {response.get('confidence', 0.0):.4f}",
        f"Cost       : ${response.get('cost_usd', 0.0):.6f}",
        f"Latency    : {response.get('latency_ms', 0.0):.0f}ms",
        f"Gate ID    : {response.get('gate_id', 'N/A')}",
        f"Verdict    : {verdict_icons.get(response.get('compliance_verdict'), 'N/A')}",
        f"\nANSWER:",
        f"{'─'*65}",
        response.get("answer", ""),
        f"{'═'*65}",
    ]

    return "\n".join(lines)


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== AeroLex Agent — Full Pipeline Test ===\n")

    agent = AeroLexAgent()

    test_queries = [
        "What does 14 CFR 91.103 say about preflight requirements?",
        "How do FAA and DGCA preflight rules differ?",
    ]

    for query in test_queries:
        print(f"\nRunning: {query}")
        response = agent.run(query)
        print(format_agent_response(response))