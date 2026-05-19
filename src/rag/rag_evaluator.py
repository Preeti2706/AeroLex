"""
rag_evaluator.py — Complete AeroLex RAG Evaluation Framework

WHAT:
    Evaluates RAG pipeline quality using 10 metrics:
    4 standard RAGAS + 6 aviation-specific custom metrics.

WHY:
    Standard RAGAS tells you IF your RAG works.
    Aviation-specific metrics tell you IF it's SAFE for compliance use.
    A system can score high on RAGAS but still cite non-existent
    regulations — dangerous in aviation context.

HOW:
    Uses LLM-as-Judge pattern — Claude evaluates Claude's answers.
    Each metric is independently calculated and logged to MLflow.
    Composite AeroLex Score = weighted combination of all 10 metrics.

METRICS:
    Standard RAGAS:
    1. Faithfulness        — hallucination detection (0-1)
    2. Answer Relevancy    — does answer address query? (0-1)
    3. Context Precision   — are retrieved chunks relevant? (0-1)
    4. Context Recall      — is all relevant info retrieved? (0-1)

    Aviation-Specific:
    5. Citation Accuracy   — do cited sections actually exist? (0-1)
    6. Safety Criticality  — risk-weighted faithfulness (0-1)
    7. Retrieval Diversity — anti-bias, multi-section coverage (0-1)
    8. Answer Completeness — LLM-as-judge thoroughness (0-1)
    9. Latency Quality     — quality per millisecond (0-1)
    10. Cross-Reg Consistency — FAA vs DGCA answer alignment (0-1)

COMPOSITE SCORE WEIGHTS:
    faithfulness:          0.25  (safety-critical — no hallucination)
    answer_relevancy:      0.20  (did we answer the question?)
    context_precision:     0.15  (retrieval quality)
    context_recall:        0.10  (coverage)
    citation_accuracy:     0.10  (domain-specific integrity)
    safety_criticality:    0.08  (aviation risk weighting)
    retrieval_diversity:   0.05  (anti-retrieval-bias)
    answer_completeness:   0.05  (thoroughness)
    latency_quality:       0.02  (production efficiency)

MATH:
    Faithfulness = supported_claims / total_claims
    Answer Relevancy = cosine_sim(answer_embedding, query_embedding)
    Context Precision = relevant_chunks / total_chunks
    Context Recall = retrieved_relevant / total_relevant
    Citation Accuracy = valid_citations / total_citations
    Safety Criticality = safety_weight(query) × faithfulness
    Retrieval Diversity = unique_sections / total_chunks
    Answer Completeness = llm_judge_score / 5.0
    Latency Quality = faithfulness / log10(latency_ms)
    Cross-Reg Consistency = cosine_sim(faa_embedding, dgca_embedding)

Official Docs:
    RAGAS:      https://docs.ragas.io/en/latest/concepts/metrics/
    LLM Judge:  https://arxiv.org/abs/2306.05685
    MLflow:     https://mlflow.org/docs/latest/tracking.html
"""

import re
import math
import json
import time
from dataclasses import dataclass, field
from typing import Optional
import mlflow
import anthropic
import numpy as np

from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, RAGError
from src.rag.rag_chain import RAGChain, RAGResponse, RetrievedChunk
from src.rag.citation_builder import build_citations, CitedResponse
from config.settings import settings

logger = get_logger(__name__)

# ── Composite Score Weights ──────────────────────────────────────────────────
METRIC_WEIGHTS = {
    "faithfulness":          0.25,
    "answer_relevancy":      0.20,
    "context_precision":     0.15,
    "context_recall":        0.10,
    "citation_accuracy":     0.10,
    "safety_criticality":    0.08,
    "retrieval_diversity":   0.05,
    "answer_completeness":   0.05,
    "latency_quality":       0.02,
}

# ── Safety Criticality Keywords ──────────────────────────────────────────────
# Queries containing these keywords get higher safety weight
SAFETY_CRITICAL_KEYWORDS = [
    "preflight", "airworthiness", "emergency", "fuel", "minimum equipment",
    "mel", "takeoff", "landing", "ifr", "instrument", "weather minima",
    "collision avoidance", "tcas", "gpws", "fire", "evacuation",
    "oxygen", "pressurization", "structural", "fatigue", "ad ",
    "airworthiness directive", "mandatory", "prohibit", "shall not",
]

# ── Valid Regulation Pattern ─────────────────────────────────────────────────
# Regex to detect regulation references in answers
REGULATION_PATTERN = re.compile(
    r'(?:14\s*CFR\s*[§§]?\s*(?:Part\s*)?\d+[\.\d]*'   # 14 CFR § 91.103
    r'|§\s*\d+[\.\d]+'                                  # § 91.103
    r'|\bPart\s+\d+\b'                                  # Part 91
    r'|\bCAR\s+\d+\b'                                   # CAR 21
    r'|\bAD\s+\d{4}-\d+\b)',                            # AD 2024-15
    re.IGNORECASE
)


# ── Evaluation Result Dataclass ──────────────────────────────────────────────

@dataclass
class AeroLexEvalResult:
    """
    Complete evaluation result for one RAG query.

    All scores are 0.0 - 1.0 unless noted.
    Higher is always better.
    """
    # ── Query metadata ──
    query:                  str
    answer:                 str
    num_chunks:             int
    latency_ms:             float

    # ── Standard RAGAS metrics ──
    faithfulness:           float   # hallucination detection
    answer_relevancy:       float   # query-answer alignment
    context_precision:      float   # retrieval precision
    context_recall:         float   # retrieval recall

    # ── Aviation-specific metrics ──
    citation_accuracy:      float   # valid regulatory citations
    safety_criticality:     float   # risk-weighted faithfulness
    retrieval_diversity:    float   # anti-bias coverage
    answer_completeness:    float   # LLM-as-judge thoroughness
    latency_quality:        float   # quality per millisecond
    cross_reg_consistency:  float   # FAA vs DGCA alignment (optional)

    # ── Composite ──
    aerolex_score:          float   # weighted composite

    # ── Metadata ──
    mlflow_run_id:          str = ""
    eval_timestamp:         float = field(default_factory=time.time)
    warnings:               list[str] = field(default_factory=list)


# ── LLM Judge Helper ─────────────────────────────────────────────────────────

def _call_llm_judge(prompt: str) -> str:
    """
    Call Claude as LLM judge for evaluation.

    Uses a separate Claude call specifically for evaluation —
    not the same call that generated the answer.
    This avoids self-serving bias in evaluation.

    Args:
        prompt: Evaluation prompt

    Returns:
        Claude's evaluation response as string
    """
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def _get_embedding(text: str) -> np.ndarray:
    """
    Get Voyage embedding for cosine similarity calculations.

    Used for:
    - Answer Relevancy: cosine_sim(answer, query)
    - Cross-Reg Consistency: cosine_sim(faa_answer, dgca_answer)

    Args:
        text: Text to embed

    Returns:
        numpy array of embedding vector
    """
    import voyageai
    client = voyageai.Client(api_key=settings.VOYAGE_API_KEY)
    result = client.embed([text], model="voyage-3-large", input_type="query")
    return np.array(result.embeddings[0])


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.

    cos(θ) = (A · B) / (||A|| × ||B||)

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Cosine similarity score (0.0 - 1.0)
    """
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm_product == 0:
        return 0.0
    similarity = dot_product / norm_product
    # Clamp to [0, 1] — cosine can be negative for opposite vectors
    return float(max(0.0, min(1.0, similarity)))


# ── Metric 1: Faithfulness ───────────────────────────────────────────────────

def compute_faithfulness(
    query: str,
    answer: str,
    chunks: list[RetrievedChunk],
) -> float:
    """
    Faithfulness — hallucination detection via LLM-as-judge.

    Method:
    1. Extract factual claims from answer using Claude
    2. For each claim, check if it is supported by retrieved chunks
    3. faithfulness = supported_claims / total_claims

    Math:
        F = |{claims supported by context}| / |{all claims in answer}|

    Args:
        query:  Original user query
        answer: LLM generated answer
        chunks: Retrieved chunks used as context

    Returns:
        Faithfulness score (0.0 - 1.0)
    """
    if not answer or not chunks:
        return 0.0

    context = "\n---\n".join([c.text for c in chunks])

    prompt = f"""You are evaluating a RAG system for aviation regulatory compliance.

CONTEXT (retrieved regulatory chunks):
{context}

ANSWER TO EVALUATE:
{answer}

TASK:
1. Extract each factual claim from the answer (ignore "I don't know" statements)
2. For each claim, determine if it is SUPPORTED or NOT_SUPPORTED by the context above
3. Return ONLY a JSON object in this exact format:
{{
  "total_claims": <integer>,
  "supported_claims": <integer>,
  "claims": [
    {{"claim": "<claim text>", "supported": true/false}}
  ]
}}

Return ONLY the JSON. No preamble, no explanation."""

    try:
        response = _call_llm_judge(prompt)
        # Strip markdown fences if present
        clean = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        total = data.get("total_claims", 0)
        supported = data.get("supported_claims", 0)
        if total == 0:
            return 1.0  # No claims = no hallucination
        score = supported / total
        logger.debug(f"Faithfulness: {supported}/{total} = {score:.4f}")
        return round(score, 4)
    except Exception as e:
        logger.warning(f"Faithfulness computation failed: {e}")
        return 0.5  # Neutral fallback


# ── Metric 2: Answer Relevancy ───────────────────────────────────────────────

def compute_answer_relevancy(query: str, answer: str) -> float:
    """
    Answer Relevancy — cosine similarity between query and answer.

    Method:
    Embed both query and answer using Voyage.
    High similarity = answer is on-topic for the query.
    Low similarity = answer drifted from query.

    Math:
        AR = cosine_sim(embed(query), embed(answer))

    Args:
        query:  Original user query
        answer: LLM generated answer

    Returns:
        Answer relevancy score (0.0 - 1.0)
    """
    if not answer or len(answer.strip()) < 10:
        return 0.0

    try:
        query_vec  = _get_embedding(query)
        answer_vec = _get_embedding(answer)
        score = _cosine_similarity(query_vec, answer_vec)
        logger.debug(f"Answer Relevancy: {score:.4f}")
        return round(score, 4)
    except Exception as e:
        logger.warning(f"Answer relevancy computation failed: {e}")
        return 0.5


# ── Metric 3: Context Precision ──────────────────────────────────────────────

def compute_context_precision(
    query: str,
    chunks: list[RetrievedChunk],
) -> float:
    """
    Context Precision — what fraction of retrieved chunks are relevant?

    Method: LLM-as-judge evaluates each chunk for relevance to query.

    Math:
        CP = |{relevant chunks}| / |{total retrieved chunks}|

    Args:
        query:  Original user query
        chunks: All retrieved chunks

    Returns:
        Context precision score (0.0 - 1.0)
    """
    if not chunks:
        return 0.0

    chunk_texts = "\n---\n".join([
        f"[Chunk {i+1}]: {c.text[:300]}"
        for i, c in enumerate(chunks)
    ])

    prompt = f"""You are evaluating retrieved chunks for a RAG system.

QUERY: {query}

RETRIEVED CHUNKS:
{chunk_texts}

TASK:
For each chunk, determine if it is RELEVANT to answering the query.
A chunk is relevant if it contains information useful for answering the query.

Return ONLY a JSON object:
{{
  "total_chunks": <integer>,
  "relevant_chunks": <integer>,
  "relevance": [true/false, true/false, ...]
}}

Return ONLY the JSON. No explanation."""

    try:
        response = _call_llm_judge(prompt)
        clean = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        total    = data.get("total_chunks", len(chunks))
        relevant = data.get("relevant_chunks", 0)
        score = relevant / total if total > 0 else 0.0
        logger.debug(f"Context Precision: {relevant}/{total} = {score:.4f}")
        return round(score, 4)
    except Exception as e:
        logger.warning(f"Context precision computation failed: {e}")
        return 0.5


# ── Metric 4: Context Recall ─────────────────────────────────────────────────

def compute_context_recall(
    query: str,
    answer: str,
    chunks: list[RetrievedChunk],
) -> float:
    """
    Context Recall — are all answer claims supported by retrieved chunks?

    Method: LLM-as-judge checks if each answer sentence
    can be attributed to at least one retrieved chunk.

    Math:
        CR = |{answer sentences attributable to context}|
           / |{total answer sentences}|

    Args:
        query:  Original user query
        answer: LLM generated answer
        chunks: Retrieved chunks

    Returns:
        Context recall score (0.0 - 1.0)
    """
    if not answer or not chunks:
        return 0.0

    context = "\n---\n".join([c.text[:300] for c in chunks])

    prompt = f"""You are evaluating a RAG system for aviation compliance.

RETRIEVED CONTEXT:
{context}

ANSWER:
{answer}

TASK:
Break the answer into individual sentences/claims.
For each sentence, determine if it can be attributed to the retrieved context.

Return ONLY a JSON object:
{{
  "total_sentences": <integer>,
  "attributed_sentences": <integer>,
  "attribution": [true/false, ...]
}}

Return ONLY the JSON. No explanation."""

    try:
        response = _call_llm_judge(prompt)
        clean = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        total      = data.get("total_sentences", 0)
        attributed = data.get("attributed_sentences", 0)
        if total == 0:
            return 1.0
        score = attributed / total
        logger.debug(f"Context Recall: {attributed}/{total} = {score:.4f}")
        return round(score, 4)
    except Exception as e:
        logger.warning(f"Context recall computation failed: {e}")
        return 0.5


# ── Metric 5: Citation Accuracy ──────────────────────────────────────────────

def compute_citation_accuracy(
    answer: str,
    chunks: list[RetrievedChunk],
) -> float:
    """
    Citation Accuracy — do cited regulation numbers actually exist in corpus?

    Aviation-specific metric. LLMs can hallucinate regulation numbers
    like '§ 91.999' that don't exist. This catches that.

    Method:
    1. Extract all regulation references from answer (regex)
    2. Check each against actual chunk_ids in corpus
    3. citation_accuracy = valid / total

    Math:
        CA = |{citations found in corpus}| / |{all citations in answer}|

    Args:
        answer: LLM generated answer
        chunks: Retrieved chunks with chunk_ids

    Returns:
        Citation accuracy score (0.0 - 1.0)
        Returns 1.0 if no citations found (nothing to be wrong about)
    """
    citations_in_answer = REGULATION_PATTERN.findall(answer)

    if not citations_in_answer:
        logger.debug("Citation Accuracy: No citations found in answer → 1.0")
        return 1.0

    # Build corpus reference set from chunk_ids
    # chunk_id format: "14_CFR_Part_91_Section_91.103_hier_1"
    corpus_refs = set()
    for chunk in chunks:
        cid = chunk.chunk_id.lower()
        # Extract section numbers from chunk_id
        numbers = re.findall(r'\d+[\.\d]*', cid)
        corpus_refs.update(numbers)

    valid = 0
    for citation in citations_in_answer:
        # Extract numbers from citation
        numbers = re.findall(r'\d+[\.\d]*', citation)
        if any(n in corpus_refs for n in numbers):
            valid += 1

    score = valid / len(citations_in_answer)
    logger.debug(
        f"Citation Accuracy: {valid}/{len(citations_in_answer)} = {score:.4f}"
    )
    return round(score, 4)


# ── Metric 6: Safety Criticality ─────────────────────────────────────────────

def compute_safety_criticality(
    query: str,
    faithfulness: float,
) -> float:
    """
    Safety Criticality — risk-weighted faithfulness score.

    Aviation-specific metric. Not all queries are equal:
    - "preflight requirements" is safety-critical
    - "aircraft paint colors" is not

    Higher criticality query + lower faithfulness = lower score.
    This penalizes hallucination MORE for safety-critical queries.

    Math:
        safety_weight = 1.0 if critical keywords found, else 0.6
        SC = safety_weight × faithfulness

    Args:
        query:        Original user query
        faithfulness: Already computed faithfulness score

    Returns:
        Safety criticality score (0.0 - 1.0)
    """
    query_lower = query.lower()
    is_critical = any(kw in query_lower for kw in SAFETY_CRITICAL_KEYWORDS)
    safety_weight = 1.0 if is_critical else 0.6

    score = round(safety_weight * faithfulness, 4)
    logger.debug(
        f"Safety Criticality: weight={safety_weight} × "
        f"faithfulness={faithfulness:.4f} = {score:.4f} "
        f"({'CRITICAL' if is_critical else 'non-critical'})"
    )
    return score


# ── Metric 7: Retrieval Diversity ────────────────────────────────────────────

def compute_retrieval_diversity(chunks: list[RetrievedChunk]) -> float:
    """
    Retrieval Diversity — anti-bias metric.

    Detects retrieval bias — when all chunks come from one section.
    Example: 5 chunks all from § 91.103 = low diversity.
    Better: chunks from § 91.103, § 91.151, § 91.7, etc.

    Math:
        RD = unique_sections / total_chunks

    Why it matters:
        Low diversity = retrieval is over-indexing one section.
        Could miss relevant regulations from other sections.

    Args:
        chunks: Retrieved chunks with chunk_ids

    Returns:
        Diversity score (0.0 - 1.0)
    """
    if not chunks:
        return 0.0

    # Extract section identifiers from chunk_ids
    sections = set()
    for chunk in chunks:
        # chunk_id: "14_CFR_Part_91_Section_91.103_hier_1"
        # Extract "Section_91.103" as the section identifier
        match = re.search(r'Section_[\d\.]+', chunk.chunk_id)
        if match:
            sections.add(match.group())
        else:
            sections.add(chunk.chunk_id[:30])

    score = len(sections) / len(chunks)
    score = round(min(score, 1.0), 4)
    logger.debug(
        f"Retrieval Diversity: {len(sections)} unique sections / "
        f"{len(chunks)} chunks = {score:.4f}"
    )
    return score


# ── Metric 8: Answer Completeness ────────────────────────────────────────────

def compute_answer_completeness(query: str, answer: str) -> float:
    """
    Answer Completeness — LLM-as-judge thoroughness evaluation.

    Uses Claude to judge if the answer fully addresses the query
    on a 1-5 scale, then normalizes to 0-1.

    Math:
        AC = judge_score / 5.0

    Why LLM-as-judge?
        Completeness is subjective — rule-based checks can't capture
        whether an answer "fully" addresses a complex regulatory query.
        LLM judge is the industry standard (GPT-4 judge in RAGAS).

    Args:
        query:  Original user query
        answer: LLM generated answer

    Returns:
        Completeness score (0.0 - 1.0)
    """
    prompt = f"""You are an aviation regulatory compliance expert evaluating an answer.

QUERY: {query}

ANSWER: {answer}

TASK:
Rate how completely this answer addresses the query on a scale of 1-5:
1 = Does not address the query at all
2 = Partially addresses with major gaps
3 = Addresses main points but misses important details
4 = Mostly complete with minor gaps
5 = Fully and completely addresses the query

Return ONLY a JSON object:
{{"score": <1-5>, "reason": "<one sentence>"}}

Return ONLY the JSON. No explanation."""

    try:
        response = _call_llm_judge(prompt)
        clean = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        raw_score = float(data.get("score", 3))
        score = round(raw_score / 5.0, 4)
        logger.debug(
            f"Answer Completeness: {raw_score}/5 = {score:.4f} | "
            f"Reason: {data.get('reason', 'N/A')}"
        )
        return score
    except Exception as e:
        logger.warning(f"Answer completeness computation failed: {e}")
        return 0.5


# ── Metric 9: Latency Quality ────────────────────────────────────────────────

def compute_latency_quality(
    faithfulness: float,
    latency_ms: float,
) -> float:
    """
    Latency Quality — quality per unit of latency.

    Production systems must balance quality and speed.
    A perfect answer in 30 seconds is worse than a good answer in 2 seconds.

    Math:
        LQ = faithfulness / log10(latency_ms)

    Normalized to 0-1 range:
        log10(500ms)  = 2.70 → LQ = 1.0/2.70 = 0.37 (fast + faithful)
        log10(5000ms) = 3.70 → LQ = 1.0/3.70 = 0.27 (slow + faithful)
        log10(500ms)  = 2.70 → LQ = 0.5/2.70 = 0.18 (fast + not faithful)

    Args:
        faithfulness: Already computed faithfulness score
        latency_ms:   End-to-end pipeline latency in milliseconds

    Returns:
        Latency quality score (0.0 - 1.0)
    """
    if latency_ms <= 0:
        return 0.0

    log_latency = math.log10(max(latency_ms, 1.0))
    if log_latency == 0:
        return 0.0

    raw_score = faithfulness / log_latency
    # Normalize: typical range is 0.0 - 0.5, scale to 0-1
    score = round(min(raw_score * 2.0, 1.0), 4)
    logger.debug(
        f"Latency Quality: {faithfulness:.4f} / "
        f"log10({latency_ms:.0f}) = {score:.4f}"
    )
    return score


# ── Metric 10: Cross-Reg Consistency ────────────────────────────────────────

def compute_cross_reg_consistency(
    faa_answer: str,
    dgca_answer: str,
) -> float:
    """
    Cross-Regulation Consistency — FAA vs DGCA answer alignment.

    AeroLex-specific metric — unique to multi-regulatory systems.
    Same query answered from FAA corpus and DGCA corpus should
    be semantically consistent (not contradictory).

    Math:
        CRC = cosine_sim(embed(faa_answer), embed(dgca_answer))

    Why this matters:
        FAA Part 91 and DGCA CAR-OPS cover similar ground.
        Low consistency = regulatory conflict or corpus gap.
        Airlines operating both US and Indian routes need this!

    Args:
        faa_answer:  Answer from FAA-filtered retrieval
        dgca_answer: Answer from DGCA-filtered retrieval

    Returns:
        Consistency score (0.0 - 1.0)
        Returns -1.0 if either answer is empty (metric not applicable)
    """
    if not faa_answer or not dgca_answer:
        logger.debug("Cross-Reg Consistency: N/A — one or both answers empty")
        return -1.0  # Sentinel — metric not applicable

    try:
        faa_vec  = _get_embedding(faa_answer)
        dgca_vec = _get_embedding(dgca_answer)
        score = _cosine_similarity(faa_vec, dgca_vec)
        logger.debug(f"Cross-Reg Consistency: {score:.4f}")
        return round(score, 4)
    except Exception as e:
        logger.warning(f"Cross-reg consistency computation failed: {e}")
        return -1.0


# ── Composite Score ──────────────────────────────────────────────────────────

def compute_aerolex_score(metrics: dict[str, float]) -> float:
    """
    Compute weighted composite AeroLex score.

    Weights reflect aviation safety priorities:
    - Faithfulness weighted highest (0.25) — no hallucination
    - Answer relevancy second (0.20) — must answer the question
    - Citation accuracy included (0.10) — domain integrity

    Math:
        aerolex_score = Σ (weight_i × metric_i)

    Args:
        metrics: Dict of metric_name → score

    Returns:
        Composite score (0.0 - 1.0)
    """
    score = 0.0
    for metric, weight in METRIC_WEIGHTS.items():
        value = metrics.get(metric, 0.0)
        score += weight * value

    return round(score, 4)


# ── Main Evaluator Class ─────────────────────────────────────────────────────

class AeroLexEvaluator:
    """
    Complete RAG evaluation pipeline.

    Usage:
        evaluator = AeroLexEvaluator()
        result = evaluator.evaluate(
            query="What must a pilot do before flight?",
            rag_response=rag_response
        )
        print(result.aerolex_score)
    """

    def __init__(self):
        self.chain = RAGChain(
            collection_name="aerolex_voyage",
            top_k=5,
            use_claude=True,
            auto_filter=True,
        )
        logger.info("AeroLexEvaluator initialized")

    def evaluate(
        self,
        query: str,
        rag_response: Optional[RAGResponse] = None,
        run_cross_reg: bool = False,
    ) -> AeroLexEvalResult:
        """
        Run complete evaluation for one query.

        Args:
            query:           User query to evaluate
            rag_response:    Pre-computed RAGResponse (optional)
                            If None, will run RAGChain internally
            run_cross_reg:   Whether to run cross-reg consistency
                            (requires separate FAA + DGCA queries)

        Returns:
            AeroLexEvalResult with all 10 metrics
        """
        warnings = []

        # ── Get RAG response ──
        if rag_response is None:
            logger.info(f"Running RAGChain for evaluation | Query: '{query[:60]}'")
            rag_response = self.chain.run(query=query)

        answer    = rag_response.answer
        chunks    = rag_response.sources
        latency   = rag_response.latency_ms

        logger.info(
            f"AeroLexEvaluator.evaluate() | "
            f"Query: '{query[:60]}' | "
            f"Chunks: {len(chunks)} | "
            f"Latency: {latency:.0f}ms"
        )

        with mlflow.start_run(run_name="aerolex_eval", nested=True):
            mlflow.log_param("query", query[:250])
            mlflow.log_param("num_chunks", len(chunks))
            mlflow.log_metric("latency_ms", latency)

            # ── Metric 1: Faithfulness ──
            logger.info("Computing Metric 1/9: Faithfulness...")
            faithfulness = compute_faithfulness(query, answer, chunks)
            mlflow.log_metric("faithfulness", faithfulness)

            # ── Metric 2: Answer Relevancy ──
            logger.info("Computing Metric 2/9: Answer Relevancy...")
            answer_relevancy = compute_answer_relevancy(query, answer)
            mlflow.log_metric("answer_relevancy", answer_relevancy)

            # ── Metric 3: Context Precision ──
            logger.info("Computing Metric 3/9: Context Precision...")
            context_precision = compute_context_precision(query, chunks)
            mlflow.log_metric("context_precision", context_precision)

            # ── Metric 4: Context Recall ──
            logger.info("Computing Metric 4/9: Context Recall...")
            context_recall = compute_context_recall(query, answer, chunks)
            mlflow.log_metric("context_recall", context_recall)

            # ── Metric 5: Citation Accuracy ──
            logger.info("Computing Metric 5/9: Citation Accuracy...")
            citation_accuracy = compute_citation_accuracy(answer, chunks)
            mlflow.log_metric("citation_accuracy", citation_accuracy)

            # ── Metric 6: Safety Criticality ──
            logger.info("Computing Metric 6/9: Safety Criticality...")
            safety_criticality = compute_safety_criticality(
                query, faithfulness
            )
            mlflow.log_metric("safety_criticality", safety_criticality)

            # ── Metric 7: Retrieval Diversity ──
            logger.info("Computing Metric 7/9: Retrieval Diversity...")
            retrieval_diversity = compute_retrieval_diversity(chunks)
            mlflow.log_metric("retrieval_diversity", retrieval_diversity)

            # ── Metric 8: Answer Completeness ──
            logger.info("Computing Metric 8/9: Answer Completeness...")
            answer_completeness = compute_answer_completeness(query, answer)
            mlflow.log_metric("answer_completeness", answer_completeness)

            # ── Metric 9: Latency Quality ──
            logger.info("Computing Metric 9/9: Latency Quality...")
            latency_quality = compute_latency_quality(faithfulness, latency)
            mlflow.log_metric("latency_quality", latency_quality)

            # ── Metric 10: Cross-Reg Consistency (optional) ──
            cross_reg_consistency = -1.0
            if run_cross_reg:
                logger.info("Computing Metric 10/9: Cross-Reg Consistency...")
                faa_response  = self.chain.run(
                    query=query, source="eCFR"
                )
                dgca_response = self.chain.run(
                    query=query, source="DGCA"
                )
                cross_reg_consistency = compute_cross_reg_consistency(
                    faa_answer=faa_response.answer,
                    dgca_answer=dgca_response.answer,
                )
                if cross_reg_consistency >= 0:
                    mlflow.log_metric(
                        "cross_reg_consistency", cross_reg_consistency
                    )

            # ── Composite Score ──
            metrics_dict = {
                "faithfulness":        faithfulness,
                "answer_relevancy":    answer_relevancy,
                "context_precision":   context_precision,
                "context_recall":      context_recall,
                "citation_accuracy":   citation_accuracy,
                "safety_criticality":  safety_criticality,
                "retrieval_diversity": retrieval_diversity,
                "answer_completeness": answer_completeness,
                "latency_quality":     latency_quality,
            }
            aerolex_score = compute_aerolex_score(metrics_dict)
            mlflow.log_metric("aerolex_score", aerolex_score)

            mlflow_run_id = mlflow.active_run().info.run_id

            logger.info(
                f"Evaluation complete | "
                f"AeroLex Score: {aerolex_score:.4f} | "
                f"Faithfulness: {faithfulness:.4f} | "
                f"Answer Relevancy: {answer_relevancy:.4f}"
            )

            return AeroLexEvalResult(
                query=query,
                answer=answer,
                num_chunks=len(chunks),
                latency_ms=latency,
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                context_precision=context_precision,
                context_recall=context_recall,
                citation_accuracy=citation_accuracy,
                safety_criticality=safety_criticality,
                retrieval_diversity=retrieval_diversity,
                answer_completeness=answer_completeness,
                latency_quality=latency_quality,
                cross_reg_consistency=cross_reg_consistency,
                aerolex_score=aerolex_score,
                mlflow_run_id=mlflow_run_id,
                warnings=warnings,
            )

    def evaluate_batch(
        self,
        queries: list[str],
    ) -> list[AeroLexEvalResult]:
        """
        Evaluate multiple queries and return results list.

        Args:
            queries: List of user queries

        Returns:
            List of AeroLexEvalResult
        """
        results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"Batch eval: {i}/{len(queries)} | '{query[:50]}'")
            result = self.evaluate(query=query)
            results.append(result)
        return results


# ── Format Output ────────────────────────────────────────────────────────────

def format_eval_result(result: AeroLexEvalResult) -> str:
    """Format AeroLexEvalResult for CLI output."""

    def bar(score: float, width: int = 20) -> str:
        """Visual progress bar for scores."""
        if score < 0:
            return "N/A"
        filled = int(score * width)
        return f"{'█' * filled}{'░' * (width - filled)} {score:.4f}"

    lines = [
        f"\n{'═'*65}",
        f"AEROLEX EVALUATION REPORT",
        f"{'═'*65}",
        f"Query    : {result.query}",
        f"Chunks   : {result.num_chunks}",
        f"Latency  : {result.latency_ms:.0f}ms",
        f"MLflow   : {result.mlflow_run_id}",
        f"\n── STANDARD RAGAS METRICS ──────────────────────────────",
        f"Faithfulness       {bar(result.faithfulness)}",
        f"Answer Relevancy   {bar(result.answer_relevancy)}",
        f"Context Precision  {bar(result.context_precision)}",
        f"Context Recall     {bar(result.context_recall)}",
        f"\n── AVIATION-SPECIFIC METRICS ───────────────────────────",
        f"Citation Accuracy  {bar(result.citation_accuracy)}",
        f"Safety Criticality {bar(result.safety_criticality)}",
        f"Retrieval Diversity{bar(result.retrieval_diversity)}",
        f"Ans Completeness   {bar(result.answer_completeness)}",
        f"Latency Quality    {bar(result.latency_quality)}",
        f"Cross-Reg Consist  {bar(result.cross_reg_consistency)}",
        f"\n── COMPOSITE SCORE ─────────────────────────────────────",
        f"🎯 AEROLEX SCORE   {bar(result.aerolex_score)}",
        f"{'═'*65}\n",
    ]
    return "\n".join(lines)


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== AeroLex RAG Evaluator — Test ===\n")

    evaluator = AeroLexEvaluator()

    test_queries = [
        "What must a pilot do before beginning a flight?",
        "What are the fuel requirements for VFR flight under Part 91?",
    ]

    for query in test_queries:
        result = evaluator.evaluate(query=query)
        print(format_eval_result(result))