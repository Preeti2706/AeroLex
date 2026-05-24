# ✈️ AeroLex — Aviation Regulatory Compliance Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1-orange)
![Qdrant](https://img.shields.io/badge/Qdrant-1.9-red?logo=qdrant)
![Claude](https://img.shields.io/badge/Claude-Sonnet_4.5-purple)
![MLflow](https://img.shields.io/badge/MLflow-2.13-blue?logo=mlflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)

**Production-grade RAG system for FAA and DGCA aviation regulations**

[Live Demo](#demo) • [Architecture](#architecture) • [Quick Start](#quick-start) • [API Docs](#api-reference) • [Evaluation](#evaluation)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Evaluation Results](#evaluation)
- [Phase Breakdown](#phase-breakdown)
- [Key Design Decisions](#key-design-decisions)

---

## 🎯 Overview

AeroLex is a **production-grade Aviation Regulatory Compliance Assistant** built as a portfolio project. It answers complex aviation regulatory queries by retrieving and reasoning over real FAA and DGCA regulatory documents.

### What AeroLex Can Do

| Query Type | Example | Pipeline |
|------------|---------|----------|
| **LOOKUP** | "What does 14 CFR 91.103 say?" | Single-source hybrid retrieval |
| **COMPARISON** | "FAA vs DGCA preflight rules?" | Parallel multi-source retrieval |
| **ADVISORY** | "Is my flight plan legal?" | Multi-hop chain-of-thought reasoning |

### Why AeroLex is Different

- 🔒 **Zero hallucination policy** — HITL gate blocks low-confidence answers
- 📊 **10-metric evaluation** — 4 standard RAGAS + 6 aviation-specific metrics
- 🔍 **Hybrid retrieval** — BM25 + Dense + RRF + Voyage reranking
- 🤖 **LangGraph agents** — intelligent query routing, not fixed pipelines
- 📈 **Full observability** — MLflow + LangSmith tracking for every query

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  FastAPI REST API                    │
│         POST /query | /preflight | /compliance       │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              LangGraph StateGraph                    │
│                                                      │
│  classify → plan → route → synthesize → validate → gate │
│                                                      │
│  Node 1: QueryClassifier  (LOOKUP/COMPARISON/ADVISORY)│
│  Node 2: RetrievalPlanner (STANDARD/MULTI_SOURCE/    │
│                             MULTI_HOP)               │
│  Node 3: RetrievalRouter  (executes plan)            │
│  Node 4: AnswerSynthesizer (Claude + specialized     │
│                              prompts)                │
│  Node 5: CitationBuilder + AnswerValidator           │
│  Node 6: HITLGate (AUTO_APPROVE/HOLD/BLOCK)          │
└─────────────────────┬───────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌──────────┐ ┌─────────┐ ┌────────┐
    │  Qdrant  │ │ Claude  │ │ Voyage │
    │ aerolex_ │ │ Sonnet  │ │rerank-2│
    │  voyage  │ │  4.5    │ │        │
    └──────────┘ └─────────┘ └────────┘
```

### Retrieval Pipeline

```
Query
  │
  ├─── BM25 (keyword search, in-memory, 624 docs)
  │         k1=1.5, b=0.75 — saturation + length normalization
  │
  ├─── Dense (Voyage voyage-3-large, 1024 dims)
  │         HNSW m=16, ef_construct=100
  │
  └─── RRF Fusion (k=60, rank-based — no score scale issues)
            │
            ▼
       Voyage rerank-2 (top-20 → top-5)
            │
            ▼
       Claude Synthesis (query-type-aware prompt)
            │
            ▼
       HITL Gate (groundedness × confidence × authority)
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **LLM** | Claude Sonnet 4.5 | Best reasoning for regulatory text |
| **Embeddings** | Voyage voyage-3-large (1024d) | Outperforms OpenAI on domain text |
| **Reranker** | Voyage rerank-2 | Same distribution as embeddings |
| **Vector DB** | Qdrant (4 collections) | Native hybrid search + filtering |
| **Sparse Search** | BM25 (in-memory) | Exact regulation number matching |
| **Agents** | LangGraph StateGraph | Checkpointing + HITL support |
| **API** | FastAPI + Pydantic | Auto OpenAPI docs + validation |
| **UI** | Streamlit | Rapid demo UI in pure Python |
| **Tracking** | MLflow + LangSmith | Full experiment + trace tracking |
| **Ingestion** | eCFR XML + Playwright + BeautifulSoup4 | Multi-source regulatory corpus |

---

## 📁 Project Structure

```
AeroLex/
├── src/
│   ├── agents/                 # Phase 5 — LangGraph Agents
│   │   ├── query_classifier.py # LOOKUP/COMPARISON/ADVISORY classification
│   │   ├── planner.py          # Retrieval strategy planning
│   │   ├── router.py           # STANDARD/MULTI_SOURCE/MULTI_HOP execution
│   │   ├── synthesizer.py      # Query-type-aware answer generation
│   │   └── agent_graph.py      # LangGraph StateGraph orchestrator
│   │
│   ├── api/                    # Phase 6 — FastAPI
│   │   ├── main.py             # App entry point + CORS + middleware
│   │   ├── schemas.py          # Pydantic request/response models
│   │   ├── dependencies.py     # Dependency injection + lifespan
│   │   └── routes/
│   │       ├── query.py        # POST /api/v1/query
│   │       ├── preflight.py    # POST /api/v1/preflight
│   │       ├── compliance.py   # POST /api/v1/compliance
│   │       └── ad_check.py     # POST /api/v1/ad-check
│   │
│   ├── rag/                    # Phase 4 — RAG Chain
│   │   ├── rag_chain.py        # Core RAG orchestrator
│   │   ├── citation_builder.py # Structured citation extraction
│   │   ├── answer_validator.py # Groundedness scoring
│   │   ├── rag_evaluator.py    # 10-metric evaluation framework
│   │   └── hitl_gate.py        # AUTO_APPROVE/HOLD/BLOCK routing
│   │
│   ├── retrieval/              # Phase 3 — Vector Search
│   │   ├── qdrant_store.py     # Qdrant client wrapper
│   │   ├── dense_retriever.py  # Voyage embedding search
│   │   ├── hybrid_retriever.py # BM25 + Dense + RRF fusion
│   │   ├── reranker.py         # Voyage + CrossEncoder reranking
│   │   └── metadata_filter.py  # Pre-filtering by source/part
│   │
│   ├── embeddings/             # Phase 2 — Embedding Models
│   │   ├── voyage_embedder.py
│   │   ├── openai_embedder.py
│   │   ├── e5_embedder.py
│   │   └── local_embedder.py   # BGE-M3
│   │
│   ├── chunking/               # Phase 2 — Text Chunking
│   │   ├── recursive_chunker.py
│   │   ├── semantic_chunker.py
│   │   └── hierarchical_chunker.py
│   │
│   ├── ingestion/              # Phase 1 — Data Ingestion
│   │   ├── ecfr_ingestor.py    # FAA eCFR XML parser
│   │   ├── faa_ad_ingestor.py  # Airworthiness Directives
│   │   ├── dgca_ingestor.py    # DGCA CARs (Playwright)
│   │   ├── faa_ac_ingestor.py  # Advisory Circulars CSV
│   │   └── skybrary_ingestor.py # SKYbrary BS4
│   │
│   ├── alerts/
│   │   └── hitl_gate.py        # Human-in-the-loop gate
│   │
│   ├── monitoring/             # Phase 0 — LLMOps
│   │   ├── cost_tracker.py
│   │   ├── langsmith_tracker.py
│   │   └── mlflow_tracker.py
│   │
│   ├── ui/
│   │   └── streamlit_app.py    # Phase 6 — Demo UI
│   │
│   └── utils/
│       ├── logger.py
│       └── exception_handler.py
│
├── config/
│   ├── settings.py             # Pydantic settings + .env loading
│   └── prompts.py              # Prompt templates
│
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- API Keys: Anthropic, OpenAI, Voyage AI, LangSmith

### Option 1: Docker (Recommended)

```bash
# 1. Clone repo
git clone https://github.com/Preeti2706/AeroLex.git
cd AeroLex

# 2. Setup environment
cp .env.example .env
# Edit .env with your API keys

# 3. Start everything
docker-compose up -d

# 4. Verify
curl http://localhost:8000/health

# Access:
# API:      http://localhost:8000/docs
# UI:       http://localhost:8501
# MLflow:   http://localhost:5000
# Qdrant:   http://localhost:6333/dashboard
```

### Option 2: Local Development

```bash
# 1. Clone + setup
git clone https://github.com/Preeti2706/AeroLex.git
cd AeroLex
python -m venv aerolex
aerolex\Scripts\activate        # Windows
# source aerolex/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Environment
cp .env.example .env
# Edit .env with your API keys

# 4. Start services
# Terminal 1: Qdrant
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# Terminal 2: MLflow
mlflow ui --port 5000

# Terminal 3: FastAPI
python -m src.api.main

# Terminal 4: Streamlit
streamlit run src/ui/streamlit_app.py --server.port 8502
```

---

## 📡 API Reference

### POST /api/v1/query
Generic aviation regulatory query.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does 14 CFR 91.103 say about preflight?"}'
```

**Response:**
```json
{
  "query": "What does 14 CFR 91.103 say about preflight?",
  "answer": "Per 14 CFR § 91.103, each pilot in command shall...",
  "status": "AUTO_APPROVE",
  "query_type": "LOOKUP",
  "strategy": "STANDARD",
  "confidence": 0.778,
  "cost_usd": 0.004836,
  "latency_ms": 26568,
  "gate_id": "DD2D40B0"
}
```

### POST /api/v1/preflight
Structured preflight compliance check.

```bash
curl -X POST http://localhost:8000/api/v1/preflight \
  -H "Content-Type: application/json" \
  -d '{
    "flight_type": "VFR",
    "aircraft_type": "general",
    "jurisdiction": "FAA"
  }'
```

### POST /api/v1/compliance
Step-by-step regulatory compliance advisory.

```bash
curl -X POST http://localhost:8000/api/v1/compliance \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": "Departing with inoperative altimeter on VFR flight",
    "regulation_part": "91",
    "jurisdiction": "FAA"
  }'
```

**Response includes:**
```json
{
  "compliance_verdict": "NON_COMPLIANT",
  "reasoning_steps": [
    "Step 1 — Applicable regulations: § 91.205, § 91.213...",
    "Step 2 — Requirements: altimeter is required equipment...",
    "Step 3 — Applied to scenario: departure not permitted..."
  ]
}
```

### GET /health
```bash
curl http://localhost:8000/health
# {"status":"healthy","qdrant":"connected","mlflow":"connected","collections":["aerolex_voyage",...]}
```

---

## 📊 Evaluation Results

AeroLex uses a **10-metric evaluation framework** — 4 standard RAGAS + 6 aviation-specific.

### Results on Preflight Query

| Metric | Score | Type |
|--------|-------|------|
| Faithfulness | 0.857 | Standard RAGAS |
| Answer Relevancy | 0.675 | Standard RAGAS |
| Context Precision | 0.800 | Standard RAGAS |
| Context Recall | 0.889 | Standard RAGAS |
| Citation Accuracy | 1.000 | ✨ Aviation-specific |
| Safety Criticality | 0.514 | ✨ Aviation-specific |
| Retrieval Diversity | 0.600 | ✨ Aviation-specific |
| Answer Completeness | 0.800 | ✨ Aviation-specific |
| Latency Quality | 0.423 | ✨ Aviation-specific |
| **AeroLex Score** | **0.778** | **Composite** |

### Aviation-Specific Metrics Explained

**Citation Accuracy** — Detects hallucinated regulation numbers (e.g., § 91.999 doesn't exist). Standard RAGAS cannot catch this.

**Safety Criticality** — Risk-weighted faithfulness. A hallucinated fuel requirement scores higher penalty than a hallucinated paint color. `SC = safety_weight × faithfulness` where `safety_weight = 1.0` for safety-critical queries.

**Retrieval Diversity** — Detects retrieval bias. If all 5 chunks come from § 91.103, the system is over-indexing one section.

**Cross-Reg Consistency** — FAA vs DGCA answer alignment. Unique to multi-regulatory systems.

### Roadmap to 0.90+ Score

| Change | Expected Gain | New Score |
|--------|--------------|-----------|
| Corpus expansion (§ 91.151, § 91.155) | +0.074 | 0.852 |
| HyDE + query expansion | +0.056 | 0.908 |
| MMR diversity | +0.015 | 0.923 |
| BM25 cache + Redis | +0.007 | 0.930 |

---

## 📦 Phase Breakdown

| Phase | What | Key Files |
|-------|------|-----------|
| **Phase 0** | LLMOps — Logger, MLflow, LangSmith, Cost Tracker | `src/monitoring/` |
| **Phase 1** | Data Ingestion — eCFR XML, FAA ADs, DGCA CARs, SKYbrary | `src/ingestion/` |
| **Phase 2** | Chunking (3 strategies) + Embeddings (4 models) | `src/chunking/`, `src/embeddings/` |
| **Phase 3** | Qdrant (4 collections) + Hybrid BM25+RRF + Voyage Reranker | `src/retrieval/` |
| **Phase 4** | RAG Chain + Citations + Validator + HITL Gate + RAGAS | `src/rag/` |
| **Phase 5** | LangGraph Agents — Classifier + Planner + Router + Synthesizer | `src/agents/` |
| **Phase 6** | FastAPI REST API + Streamlit Demo UI | `src/api/`, `src/ui/` |
| **Phase 7** | Docker + docker-compose + README | Root |

---

## 🧠 Key Design Decisions

### 1. Why 4 Qdrant Collections?
Each embedding model lives in a different vector space — cross-model cosine similarity is mathematically meaningless. One collection per model ensures query-document space alignment. Also enables A/B testing in MLflow.

### 2. Why BM25 + Dense + RRF?
BM25 excels at exact regulation number matching (e.g., "91.103"). Dense excels at semantic queries ("preflight duties"). RRF combines rankings using a universal currency — rank — avoiding the incompatible scale problem of score-based fusion.

### 3. Why LangGraph Over Simple Function Calls?
TypedDict shared state enables: checkpointing (resumable on failure), human-in-the-loop interrupts after any node, native LangSmith tracing, and future parallel node execution.

### 4. Why Safety Criticality Metric?
Standard RAGAS treats all errors equally. A hallucinated fuel requirement is more dangerous than a hallucinated paint color. Safety Criticality = `safety_weight × faithfulness`, where weight reflects query risk level — the same mental model as FMEA in aviation engineering.

### 5. Why HITL Gate?
Aviation compliance is safety-critical. The HITL gate ensures no answer with groundedness < 0.50 reaches users — they receive a "under expert review" message with FSDO contact instead.

---

## 👩‍💻 Author

**Preeti** 

- 📧 Built as a portfolio project
- ✈️ Domain expertise: Data Science

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ using LangGraph + Qdrant + Claude + FastAPI + Streamlit
</div>