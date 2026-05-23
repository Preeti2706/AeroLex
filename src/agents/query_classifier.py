"""
query_classifier.py — Aviation Query Intent Classifier

WHAT:
    Classifies incoming user queries into one of three types:
    LOOKUP, COMPARISON, or ADVISORY. This drives the entire
    LangGraph agent routing decision.

WHY:
    Not all aviation queries need the same pipeline:
    - "What does 91.103 say?" → simple RAG lookup
    - "FAA vs DGCA preflight rules?" → multi-source comparison
    - "Is my flight plan legal?" → multi-hop reasoning

    Sending every query through the most expensive pipeline
    wastes compute and increases latency. Classification first
    = intelligent resource allocation.

HOW:
    Two-stage classification:
    1. Rule-based fast path — regex + keyword matching (< 1ms)
    2. LLM-based slow path — Claude classifies ambiguous queries

    Rule-based runs first — only escalate to LLM if rules
    cannot confidently classify. This is the "cheap gate first"
    pattern used in production ML systems.

QUERY TYPES:
    LOOKUP:     Single regulation fetch
                Keywords: "what does", "define", "explain", section numbers
                Example: "What does 14 CFR 91.103 say?"

    COMPARISON: Multi-source retrieval + synthesis
                Keywords: "vs", "compare", "difference", "FAA vs DGCA"
                Example: "How do FAA and DGCA preflight rules differ?"

    ADVISORY:   Multi-hop reasoning across regulations
                Keywords: "legal", "compliant", "can I", "is it allowed"
                Example: "Is my VFR flight plan legal under Part 91?"

MATH:
    Rule confidence threshold: 0.85
    Below threshold → escalate to LLM classifier
    LLM returns structured JSON with type + confidence + reasoning

Official Docs:
    LangGraph: https://langchain-ai.github.io/langgraph/
    Intent Classification: https://arxiv.org/abs/2305.14325
"""

import re
import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import anthropic

from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, AgentError
from config.settings import settings

logger = get_logger(__name__)


# ── Query Type Enum ──────────────────────────────────────────────────────────

class QueryType(str, Enum):
    """
    Three query types driving LangGraph routing.

    LOOKUP:     Single source, single regulation, direct answer
    COMPARISON: Multiple sources, synthesis required
    ADVISORY:   Multi-hop reasoning, compliance check
    """
    LOOKUP     = "LOOKUP"
    COMPARISON = "COMPARISON"
    ADVISORY   = "ADVISORY"


# ── Classification Result ────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """
    Complete classification output for one query.

    Fields:
        query:        Original user query
        query_type:   LOOKUP / COMPARISON / ADVISORY
        confidence:   0.0 - 1.0 classification confidence
        method:       "rule_based" or "llm"
        reasoning:    Why this classification was chosen
        sources_hint: Which sources to search (from query intent)
        complexity:   "simple" / "medium" / "complex"
    """
    query:        str
    query_type:   QueryType
    confidence:   float
    method:       str
    reasoning:    str
    sources_hint: list[str]
    complexity:   str


# ── Rule-Based Patterns ──────────────────────────────────────────────────────

# LOOKUP patterns — single regulation fetch
LOOKUP_PATTERNS = [
    r'\bwhat (does|is|are)\b',
    r'\bdefine\b',
    r'\bexplain\b',
    r'\bdescribe\b',
    r'\blist\b',
    r'\b\d+\.\d+\b',          # Section numbers like 91.103
    r'\bpart \d+\b',           # Part 91, Part 121
    r'\bsection\b',
    r'\bregulation\b',
    r'\bshow me\b',
    r'\btell me about\b',
]

# COMPARISON patterns — multi-source synthesis
COMPARISON_PATTERNS = [
    r'\bvs\b',
    r'\bversus\b',
    r'\bcompare\b',
    r'\bdifference\b',
    r'\bdifferent\b',
    r'\bsimilar\b',
    r'\bboth\b',
    r'\bfaa.*dgca\b',
    r'\bdgca.*faa\b',
    r'\bindia.*us\b',
    r'\bus.*india\b',
    r'\bcontrast\b',
    r'\bhow does.*differ\b',
]

# ADVISORY patterns — multi-hop compliance reasoning
ADVISORY_PATTERNS = [
    r'\bis it (legal|allowed|permitted|compliant|valid)\b',
    r'\bcan i\b',
    r'\bam i (allowed|permitted|compliant)\b',
    r'\bdo i (need|have to|must)\b',
    r'\bshould i\b',
    r'\bwould.*violate\b',
    r'\bcomplian\w+\b',
    r'\blegal\w*\b',
    r'\bpermit\w*\b',
    r'\ballow\w*\b',
    r'\bmy (flight|plan|aircraft|operation)\b',
    r'\bwhat (should|must|do) (i|we|pilot)\b',
]

# Source hint keywords
SOURCE_HINTS = {
    "ecfr":     ["14 cfr", "far", "federal", "ecfr", "part 91", "part 121", "part 135"],
    "dgca":     ["dgca", "india", "indian", "car ", "civil aviation requirement"],
    "faa_ad":   ["airworthiness directive", "ad ", "faa ad", "mandatory action"],
    "faa_ac":   ["advisory circular", "ac ", "faa ac", "guidance"],
    "skybrary": ["safety", "accident", "incident", "skybrary"],
}


# ── Rule-Based Classifier ────────────────────────────────────────────────────

def _rule_based_classify(query: str) -> tuple[Optional[QueryType], float, str]:
    """
    Fast rule-based classification using regex patterns.

    Returns:
        Tuple of (QueryType or None, confidence, reasoning)
        None = rules cannot confidently classify — escalate to LLM
    """
    query_lower = query.lower()

    # Count pattern matches for each type
    comparison_hits = sum(
        1 for p in COMPARISON_PATTERNS
        if re.search(p, query_lower)
    )
    advisory_hits = sum(
        1 for p in ADVISORY_PATTERNS
        if re.search(p, query_lower)
    )
    lookup_hits = sum(
        1 for p in LOOKUP_PATTERNS
        if re.search(p, query_lower)
    )

    logger.debug(
        f"Rule hits — LOOKUP: {lookup_hits} | "
        f"COMPARISON: {comparison_hits} | "
        f"ADVISORY: {advisory_hits}"
    )

    # COMPARISON takes priority — explicit comparison keywords are strong signal
    if comparison_hits >= 1:
        confidence = min(0.70 + (comparison_hits * 0.05), 0.95)
        return (
            QueryType.COMPARISON,
            confidence,
            f"Comparison keywords detected ({comparison_hits} matches)"
        )

    # ADVISORY — compliance/legality keywords
    if advisory_hits >= 1:
        confidence = min(0.70 + (advisory_hits * 0.05), 0.95)
        return (
            QueryType.ADVISORY,
            confidence,
            f"Advisory/compliance keywords detected ({advisory_hits} matches)"
        )

    # LOOKUP — factual/definition queries
    if lookup_hits >= 2:
        confidence = min(0.65 + (lookup_hits * 0.05), 0.90)
        return (
            QueryType.LOOKUP,
            confidence,
            f"Lookup keywords detected ({lookup_hits} matches)"
        )

    # Cannot confidently classify — escalate to LLM
    return None, 0.0, "Insufficient rule signal — escalating to LLM"


def _extract_source_hints(query: str) -> list[str]:
    """
    Extract which regulatory sources are likely relevant.

    Args:
        query: User query string

    Returns:
        List of source names — ["ecfr", "dgca", etc.]
    """
    query_lower = query.lower()
    hints = []

    for source, keywords in SOURCE_HINTS.items():
        if any(kw in query_lower for kw in keywords):
            hints.append(source)

    # Default to eCFR if no specific source detected
    if not hints:
        hints = ["ecfr"]

    return hints


def _classify_complexity(query_type: QueryType, query: str) -> str:
    """
    Classify query complexity for downstream resource allocation.

    Args:
        query_type: Classified query type
        query:      User query

    Returns:
        "simple" / "medium" / "complex"
    """
    word_count = len(query.split())

    if query_type == QueryType.LOOKUP:
        return "simple" if word_count < 15 else "medium"
    elif query_type == QueryType.COMPARISON:
        return "medium" if word_count < 20 else "complex"
    else:  # ADVISORY
        return "complex"


# ── LLM Classifier ───────────────────────────────────────────────────────────

def _llm_classify(query: str) -> tuple[QueryType, float, str]:
    """
    LLM-based classification for ambiguous queries.

    Called only when rule-based classifier confidence < 0.85.
    Uses Claude with structured JSON output.

    Args:
        query: User query string

    Returns:
        Tuple of (QueryType, confidence, reasoning)
    """
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    prompt = f"""You are classifying aviation regulatory queries for a RAG system.

Classify this query into exactly one of three types:

LOOKUP: User wants to know what a specific regulation says.
        Single source, factual retrieval.
        Examples: "What does 91.103 say?", "Define MEL", "List Part 91 requirements"

COMPARISON: User wants to compare regulations from multiple sources.
            Requires multi-source retrieval and synthesis.
            Examples: "FAA vs DGCA preflight rules", "How does Part 91 differ from Part 135?"

ADVISORY: User wants compliance advice or legality check.
          Requires multi-hop reasoning across regulations.
          Examples: "Is my VFR flight plan legal?", "Can I fly without an MEL?", "Am I compliant?"

QUERY TO CLASSIFY: "{query}"

Return ONLY a JSON object:
{{
  "query_type": "LOOKUP" or "COMPARISON" or "ADVISORY",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<one sentence explaining classification>"
}}

Return ONLY the JSON. No preamble."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text.strip()
    clean = text.replace("```json", "").replace("```", "").strip()
    data = json.loads(clean)

    query_type = QueryType(data["query_type"])
    confidence = float(data["confidence"])
    reasoning  = data["reasoning"]

    logger.info(
        f"LLM classification: {query_type.value} | "
        f"Confidence: {confidence:.2f} | "
        f"Reasoning: {reasoning}"
    )

    return query_type, confidence, reasoning


# ── Main Classifier Class ────────────────────────────────────────────────────

class QueryClassifier:
    """
    Two-stage query classifier — rule-based fast path + LLM slow path.

    Design pattern: "Cheap gate first"
    1. Try rule-based (< 1ms, free)
    2. If confidence < threshold → LLM (100-500ms, costs tokens)

    This pattern appears in production ML systems everywhere:
    Gmail spam filter: rule-based first, ML model for uncertain cases
    Content moderation: keyword blocklist first, model for edge cases
    AeroLex: regex patterns first, Claude for ambiguous queries

    Usage:
        classifier = QueryClassifier()
        result = classifier.classify("What does 91.103 say?")
        print(result.query_type)   # LOOKUP
        print(result.confidence)   # 0.85
        print(result.sources_hint) # ["ecfr"]
    """

    # Confidence threshold below which LLM is called
    RULE_CONFIDENCE_THRESHOLD = 0.85

    def __init__(self):
        logger.info("QueryClassifier initialized | Threshold: 0.85")

    def classify(self, query: str) -> ClassificationResult:
        """
        Classify a user query into LOOKUP / COMPARISON / ADVISORY.

        Two-stage pipeline:
        1. Rule-based fast path
        2. LLM slow path (if rules insufficient)

        Args:
            query: Raw user query string

        Returns:
            ClassificationResult with full classification metadata

        Raises:
            AgentError: If classification fails completely
        """
        try:
            logger.info(f"Classifying query: '{query[:80]}'")

            # ── Stage 1: Rule-based ──
            rule_type, rule_conf, rule_reasoning = _rule_based_classify(query)

            if rule_type and rule_conf >= self.RULE_CONFIDENCE_THRESHOLD:
                # Rules confident enough — skip LLM
                logger.info(
                    f"Rule-based classification: {rule_type.value} | "
                    f"Confidence: {rule_conf:.2f}"
                )
                query_type = rule_type
                confidence = rule_conf
                reasoning  = rule_reasoning
                method     = "rule_based"

            else:
                # ── Stage 2: LLM classifier ──
                logger.info(
                    f"Rule confidence {rule_conf:.2f} < {self.RULE_CONFIDENCE_THRESHOLD} "
                    f"— escalating to LLM"
                )
                query_type, confidence, reasoning = _llm_classify(query)
                method = "llm"

            # Extract source hints
            sources_hint = _extract_source_hints(query)

            # Classify complexity
            complexity = _classify_complexity(query_type, query)

            result = ClassificationResult(
                query=query,
                query_type=query_type,
                confidence=confidence,
                method=method,
                reasoning=reasoning,
                sources_hint=sources_hint,
                complexity=complexity,
            )

            logger.info(
                f"Classification complete | "
                f"Type: {query_type.value} | "
                f"Confidence: {confidence:.2f} | "
                f"Method: {method} | "
                f"Complexity: {complexity} | "
                f"Sources: {sources_hint}"
            )

            return result

        except Exception as e:
            handle_exception(
                e,
                context="QueryClassifier.classify",
                raise_as=AgentError
            )

    def classify_batch(self, queries: list[str]) -> list[ClassificationResult]:
        """
        Classify multiple queries.

        Args:
            queries: List of user queries

        Returns:
            List of ClassificationResult
        """
        results = []
        for query in queries:
            result = self.classify(query)
            results.append(result)
        return results


def format_classification(result: ClassificationResult) -> str:
    """Format ClassificationResult for CLI output."""
    type_icons = {
        QueryType.LOOKUP:     "🔍 LOOKUP",
        QueryType.COMPARISON: "⚖️  COMPARISON",
        QueryType.ADVISORY:   "⚠️  ADVISORY",
    }
    method_icons = {
        "rule_based": "⚡ rule_based",
        "llm":        "🤖 llm",
    }
    complexity_icons = {
        "simple":  "🟢 simple",
        "medium":  "🟡 medium",
        "complex": "🔴 complex",
    }

    return (
        f"\n{'─'*60}\n"
        f"Query     : {result.query}\n"
        f"Type      : {type_icons[result.query_type]}\n"
        f"Confidence: {result.confidence:.2f}\n"
        f"Method    : {method_icons[result.method]}\n"
        f"Complexity: {complexity_icons[result.complexity]}\n"
        f"Sources   : {result.sources_hint}\n"
        f"Reasoning : {result.reasoning}\n"
        f"{'─'*60}"
    )


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== AeroLex Query Classifier — Test ===\n")

    classifier = QueryClassifier()

    test_queries = [
        # LOOKUP
        "What does 14 CFR 91.103 say about preflight requirements?",
        "Define MEL in aviation context",
        "List all Part 91 VFR weather minimums",
        # COMPARISON
        "How do FAA and DGCA preflight rules differ?",
        "Compare Part 91 vs Part 135 fuel requirements",
        "FAA vs DGCA — what are the differences in maintenance requirements?",
        # ADVISORY
        "Is my VFR flight plan legal under Part 91?",
        "Can I fly without an MEL if the equipment is inoperative?",
        "Am I compliant if I depart with a broken altimeter?",
        # Ambiguous — will go to LLM
        "Tell me about preflight",
        "Aviation safety regulations",
    ]

    for query in test_queries:
        result = classifier.classify(query)
        print(format_classification(result))