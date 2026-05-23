"""
synthesizer.py — Multi-Strategy Answer Synthesizer

WHAT:
    Takes RouterResult (retrieved chunks) and generates a final
    answer using Claude. Strategy depends on query type:
    - LOOKUP:     Direct regulatory answer with citations
    - COMPARISON: Structured comparison table + analysis
    - ADVISORY:   Step-by-step compliance reasoning chain

WHY:
    Different query types need different answer formats:
    - LOOKUP answer = "§ 91.103 says X [Source 1]"
    - COMPARISON answer = "FAA requires X, DGCA requires Y, key diff is Z"
    - ADVISORY answer = "Step 1: Check MEL... Step 2: Check § 91.213..."

    A single generic prompt produces mediocre answers for all types.
    Query-type-specific prompts produce expert-level answers.

HOW:
    1. Receive RouterResult + ClassificationResult
    2. Select prompt template based on query_type
    3. Build context from retrieved chunks
    4. Call Claude with specialized prompt
    5. Return SynthesisResult with answer + citations + metadata

PROMPT STRATEGIES:
    LOOKUP:     "You are an aviation regulatory expert. Answer directly
                from the retrieved regulatory text. Cite [Source N]."

    COMPARISON: "You are comparing two regulatory frameworks.
                Structure your answer as: FAA Position | DGCA Position
                | Key Differences | Practical Implications"

    ADVISORY:   "You are an aviation compliance officer.
                Think step by step: 1) Identify applicable regulations
                2) Apply to scenario 3) Give compliance verdict"

MATH:
    Token budget per strategy:
    LOOKUP:     ~500 context + ~300 answer = ~800 total
    COMPARISON: ~1000 context + ~500 answer = ~1500 total
    ADVISORY:   ~1500 context + ~800 answer = ~2300 total

Official Docs:
    Anthropic: https://docs.anthropic.com/en/api/messages
    Chain-of-thought: https://arxiv.org/abs/2201.11903
"""

import time
from dataclasses import dataclass, field
from typing import Optional
import anthropic
import mlflow

from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, AgentError
from src.agents.query_classifier import ClassificationResult, QueryType
from src.agents.router import RouterResult
from config.settings import settings

logger = get_logger(__name__)


# ── Synthesis Result ─────────────────────────────────────────────────────────

@dataclass
class SynthesisResult:
    """
    Complete synthesis output — final answer + metadata.

    Fields:
        query:           Original user query
        answer:          Claude-generated answer
        query_type:      LOOKUP / COMPARISON / ADVISORY
        strategy_used:   Which prompt strategy was applied
        sources_cited:   List of source references used
        chunks_used:     Number of chunks in context
        input_tokens:    Claude API input tokens
        output_tokens:   Claude API output tokens
        cost_usd:        API call cost
        latency_ms:      Synthesis latency
        confidence:      Avg rerank score of used chunks
        compliance_verdict: For ADVISORY — COMPLIANT/NON_COMPLIANT/UNCLEAR
    """
    query:              str
    answer:             str
    query_type:         QueryType
    strategy_used:      str
    sources_cited:      list[str]
    chunks_used:        int
    input_tokens:       int
    output_tokens:      int
    cost_usd:           float
    latency_ms:         float
    confidence:         float
    compliance_verdict: Optional[str] = None
    warnings:           list[str] = field(default_factory=list)


# ── Context Builder ──────────────────────────────────────────────────────────

def _build_context(chunks: list[dict], max_chunks: int = 8) -> tuple[str, list[str]]:
    """
    Build formatted context string from retrieved chunks.

    Returns:
        Tuple of (context_string, list_of_source_refs)
    """
    context_parts = []
    source_refs   = []

    for i, chunk in enumerate(chunks[:max_chunks], 1):
        source      = chunk.get("source", "unknown")
        part_number = chunk.get("part_number", "unknown")
        chunk_id    = chunk.get("chunk_id", "unknown")
        text        = chunk.get("text", "")
        score       = chunk.get("rerank_score", chunk.get("weighted_score", 0.0))

        # Extract section from chunk_id
        section = "unknown"
        if "Section_" in chunk_id:
            section = chunk_id.split("Section_")[1].split("_hier")[0]

        ref = f"14 CFR § {section}" if source.lower() == "ecfr" else f"{source} {section}"
        source_refs.append(ref)

        context_parts.append(
            f"[Source {i}]\n"
            f"Regulation: {ref} | Type: {source} | Part: {part_number}\n"
            f"Relevance Score: {score:.4f}\n"
            f"Text: {text}\n"
        )

    return "\n---\n".join(context_parts), source_refs


# ── Prompt Templates ─────────────────────────────────────────────────────────

def _build_lookup_prompt(query: str, context: str) -> str:
    """
    LOOKUP prompt — direct regulatory answer with citations.

    Design:
    - Aviation regulatory expert persona
    - Strict grounding — only answer from context
    - [Source N] citation format for traceability
    - Honest "insufficient context" fallback
    """
    return f"""You are AeroLex — an expert Aviation Regulatory Compliance Assistant specializing in FAA and DGCA regulations.

Your task: Answer the regulatory question DIRECTLY and PRECISELY from the provided context.

RULES:
1. Answer ONLY from the regulatory context below — never hallucinate
2. Cite sources inline as [Source N] after each claim
3. Use exact regulatory language — section numbers, part references
4. If context is insufficient, say exactly: "The regulatory context does not contain sufficient information to answer this question. Relevant sections may include: [suggest likely sections]"
5. Keep answer concise — one paragraph maximum for simple lookups

REGULATORY CONTEXT:
{context}

QUESTION: {query}

REGULATORY ANSWER:"""


def _build_comparison_prompt(query: str, context: str) -> str:
    """
    COMPARISON prompt — structured multi-source analysis.

    Design:
    - Comparative analysis framework
    - Structured output: FAA | DGCA | Differences | Implications
    - Side-by-side regulatory language
    - Practical implications for pilots/operators
    """
    return f"""You are AeroLex — an expert Aviation Regulatory Compliance Assistant with deep knowledge of both FAA (US) and DGCA (India) regulations.

Your task: Compare the regulatory positions from multiple sources and provide a structured analysis.

RULES:
1. Structure your answer with these sections:
   **FAA Position**: What FAA regulations say [cite Sources]
   **DGCA Position**: What DGCA regulations say [cite Sources]
   **Key Differences**: Specific differences between the frameworks
   **Practical Implications**: What this means for operators/pilots
2. Cite each claim with [Source N]
3. If one framework is missing from context, state: "DGCA/FAA context not available in current corpus"
4. Be precise — regulatory differences have compliance implications

REGULATORY CONTEXT:
{context}

COMPARISON QUESTION: {query}

STRUCTURED COMPARISON:"""


def _build_advisory_prompt(query: str, context: str) -> str:
    """
    ADVISORY prompt — step-by-step compliance reasoning.

    Design:
    - Aviation compliance officer persona
    - Chain-of-thought reasoning — explicit steps
    - Compliance verdict at end — COMPLIANT/NON_COMPLIANT/UNCLEAR
    - Safety-first framing — when in doubt, don't fly
    - HITL escalation recommendation for unclear cases

    Why chain-of-thought for ADVISORY?
    Math: P(correct_verdict | step_by_step_reasoning) >
          P(correct_verdict | direct_answer)
    Complex compliance requires intermediate reasoning steps.
    """
    return f"""You are AeroLex — an Aviation Regulatory Compliance Officer with authority to provide compliance assessments under FAA and DGCA regulations.

Your task: Provide a step-by-step compliance analysis for the scenario described.

RULES:
1. Think step by step — show your regulatory reasoning explicitly
2. Structure your answer as:
   **Step 1 — Identify Applicable Regulations**: Which regulations apply? [cite Sources]
   **Step 2 — Analyze Requirements**: What do the regulations require?
   **Step 3 — Apply to Scenario**: How do requirements apply to this specific situation?
   **Step 4 — Compliance Verdict**: COMPLIANT / NON-COMPLIANT / UNCLEAR
   **Step 5 — Recommended Action**: What should the pilot/operator do?
3. For UNCLEAR verdicts: "Recommend consultation with FAA Flight Standards District Office (FSDO) or DGCA before flight"
4. Aviation safety principle: When in doubt, do not fly
5. Cite all regulatory claims with [Source N]

REGULATORY CONTEXT:
{context}

COMPLIANCE QUESTION: {query}

STEP-BY-STEP COMPLIANCE ANALYSIS:"""


# ── Verdict Extractor ────────────────────────────────────────────────────────

def _extract_compliance_verdict(answer: str) -> Optional[str]:
    """
    Extract compliance verdict from ADVISORY answer.

    Looks for COMPLIANT / NON-COMPLIANT / UNCLEAR in answer text.

    Args:
        answer: Claude-generated advisory answer

    Returns:
        "COMPLIANT" / "NON_COMPLIANT" / "UNCLEAR" / None
    """
    answer_upper = answer.upper()

    if "NON-COMPLIANT" in answer_upper or "NON_COMPLIANT" in answer_upper:
        return "NON_COMPLIANT"
    elif "COMPLIANT" in answer_upper:
        return "COMPLIANT"
    elif "UNCLEAR" in answer_upper:
        return "UNCLEAR"
    return None


# ── LLM Caller ───────────────────────────────────────────────────────────────

def _call_claude(
    prompt: str,
    max_tokens: int = 1024,
) -> tuple[str, int, int, float]:
    """
    Call Claude API for synthesis.

    Args:
        prompt:     Complete synthesis prompt
        max_tokens: Max response tokens

    Returns:
        Tuple of (answer, input_tokens, output_tokens, cost_usd)
    """
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )

    answer       = response.content[0].text.strip()
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    # claude-sonnet-4-5 pricing
    cost_usd = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000

    logger.info(
        f"Claude synthesis | "
        f"input={input_tokens} | "
        f"output={output_tokens} | "
        f"cost=${cost_usd:.6f}"
    )

    return answer, input_tokens, output_tokens, cost_usd


# ── Main Synthesizer ─────────────────────────────────────────────────────────

class AnswerSynthesizer:
    """
    Query-type-aware answer synthesizer.

    Routes to specialized prompt templates based on query type.
    Produces expert-level answers for LOOKUP, COMPARISON, ADVISORY.

    Usage:
        synthesizer = AnswerSynthesizer()
        result = synthesizer.synthesize(
            router_result=router_result,
            classification=classification
        )
        print(result.answer)
        print(result.compliance_verdict)  # For ADVISORY
    """

    # Max tokens per query type
    MAX_TOKENS = {
        QueryType.LOOKUP:     512,
        QueryType.COMPARISON: 800,
        QueryType.ADVISORY:   1024,
    }

    def __init__(self):
        logger.info("AnswerSynthesizer initialized")

    def synthesize(
        self,
        router_result: RouterResult,
        classification: ClassificationResult,
    ) -> SynthesisResult:
        """
        Synthesize final answer from retrieved chunks.

        Args:
            router_result:   Retrieved chunks from RetrievalRouter
            classification:  Query classification from QueryClassifier

        Returns:
            SynthesisResult with answer + full metadata

        Raises:
            AgentError: If synthesis fails
        """
        try:
            start_time = time.time()
            query      = classification.query
            query_type = classification.query_type

            logger.info(
                f"AnswerSynthesizer.synthesize() | "
                f"Type: {query_type.value} | "
                f"Chunks: {len(router_result.chunks)} | "
                f"Query: '{query[:60]}'"
            )

            # ── Build context ──
            context, source_refs = _build_context(router_result.chunks)

            if not router_result.chunks:
                logger.warning("No chunks available — returning insufficient context")
                return SynthesisResult(
                    query=query,
                    answer=(
                        "AeroLex could not retrieve sufficient regulatory context "
                        "to answer this question. Please try rephrasing your query "
                        "or consult official FAA/DGCA sources directly."
                    ),
                    query_type=query_type,
                    strategy_used="fallback_no_context",
                    sources_cited=[],
                    chunks_used=0,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    latency_ms=0.0,
                    confidence=0.0,
                    warnings=["No chunks retrieved — answer not grounded"],
                )

            # ── Select prompt + max_tokens ──
            max_tokens = self.MAX_TOKENS.get(query_type, 512)

            if query_type == QueryType.LOOKUP:
                prompt   = _build_lookup_prompt(query, context)
                strategy = "lookup_direct"

            elif query_type == QueryType.COMPARISON:
                prompt   = _build_comparison_prompt(query, context)
                strategy = "comparison_structured"

            else:  # ADVISORY
                prompt   = _build_advisory_prompt(query, context)
                strategy = "advisory_chain_of_thought"

            logger.info(
                f"Strategy: {strategy} | "
                f"Prompt length: {len(prompt)} chars | "
                f"Max tokens: {max_tokens}"
            )

            # ── Call Claude ──
            with mlflow.start_run(run_name=f"synthesizer_{query_type.value}", nested=True):
                mlflow.log_param("query_type", query_type.value)
                mlflow.log_param("strategy", strategy)
                mlflow.log_param("chunks_used", len(router_result.chunks))

                answer, input_tokens, output_tokens, cost_usd = _call_claude(
                    prompt=prompt,
                    max_tokens=max_tokens,
                )

                mlflow.log_metric("input_tokens", input_tokens)
                mlflow.log_metric("output_tokens", output_tokens)
                mlflow.log_metric("cost_usd", cost_usd)

            # ── Confidence ──
            scores = [
                c.get("rerank_score", c.get("weighted_score", 0.0))
                for c in router_result.chunks
            ]
            confidence = sum(scores) / len(scores) if scores else 0.0

            # ── Extract compliance verdict (ADVISORY only) ──
            verdict = None
            if query_type == QueryType.ADVISORY:
                verdict = _extract_compliance_verdict(answer)
                if verdict:
                    logger.info(f"Compliance verdict: {verdict}")

            latency_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Synthesis complete | "
                f"Latency: {latency_ms:.0f}ms | "
                f"Confidence: {confidence:.3f} | "
                f"Verdict: {verdict}"
            )

            return SynthesisResult(
                query=query,
                answer=answer,
                query_type=query_type,
                strategy_used=strategy,
                sources_cited=source_refs,
                chunks_used=len(router_result.chunks),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                confidence=confidence,
                compliance_verdict=verdict,
            )

        except Exception as e:
            handle_exception(
                e,
                context="AnswerSynthesizer.synthesize",
                raise_as=AgentError
            )


def format_synthesis(result: SynthesisResult) -> str:
    """Format SynthesisResult for CLI output."""

    type_icons = {
        QueryType.LOOKUP:     "🔍",
        QueryType.COMPARISON: "⚖️",
        QueryType.ADVISORY:   "⚠️",
    }

    verdict_icons = {
        "COMPLIANT":     "✅ COMPLIANT",
        "NON_COMPLIANT": "❌ NON-COMPLIANT",
        "UNCLEAR":       "🟡 UNCLEAR — Consult FSDO",
        None:            "",
    }

    lines = [
        f"\n{'═'*65}",
        f"SYNTHESIS RESULT — {type_icons.get(result.query_type, '')} {result.query_type.value}",
        f"{'═'*65}",
        f"Query     : {result.query}",
        f"Strategy  : {result.strategy_used}",
        f"Chunks    : {result.chunks_used}",
        f"Confidence: {result.confidence:.3f}",
        f"Cost      : ${result.cost_usd:.6f}",
        f"Latency   : {result.latency_ms:.0f}ms",
    ]

    if result.compliance_verdict:
        lines.append(
            f"Verdict   : {verdict_icons.get(result.compliance_verdict, result.compliance_verdict)}"
        )

    lines += [
        f"\nANSWER:",
        f"{'─'*65}",
        result.answer,
        f"{'─'*65}",
        f"Sources: {', '.join(result.sources_cited[:5])}",
        f"{'═'*65}",
    ]

    return "\n".join(lines)


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== AeroLex Answer Synthesizer — Test ===\n")

    from src.agents.query_classifier import QueryClassifier
    from src.agents.planner import RetrievalPlanner
    from src.agents.router import RetrievalRouter

    classifier  = QueryClassifier()
    planner     = RetrievalPlanner()
    router      = RetrievalRouter()
    synthesizer = AnswerSynthesizer()

    test_queries = [
        "What does 14 CFR 91.103 say about preflight requirements?",
        "How do FAA and DGCA preflight rules differ?",
    ]

    for query in test_queries:
        print(f"\nProcessing: {query}")
        cl     = classifier.classify(query)
        pl     = planner.plan(cl)
        rt     = router.route(pl)
        result = synthesizer.synthesize(rt, cl)
        print(format_synthesis(result))