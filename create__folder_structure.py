# ============================================
# AeroLex — Project Structure Setup Script
# Run once from project root to create all
# folders and placeholder files
# ============================================

import os
import sys

# ── Root of the project ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Complete folder + file structure ─────────────────────────────────────────
# Format: "path/to/file.ext" creates file
#         "path/to/folder/"  creates empty folder (with .gitkeep)
STRUCTURE = [

    # ── Configuration ────────────────────────────────────────────────────────
    "config/__init__.py",
    "config/settings.py",          # Centralized app settings via pydantic
    "config/logging_config.py",    # Logging configuration (handlers, formatters)
    "config/mlflow_config.py",     # MLflow experiment config
    "config/langsmith_config.py",  # LangSmith tracing config
    "config/prompts.py",           # All LLM prompt templates in one place

    # ── Source — Utilities (shared across all modules) ───────────────────────
    "src/__init__.py",
    "src/utils/__init__.py",
    "src/utils/logger.py",         # Centralized structured logger
    "src/utils/exception_handler.py",  # Custom exception classes + global handler
    "src/utils/file_utils.py",     # File read/write helpers
    "src/utils/hash_utils.py",     # Hash-based change detection (avoid re-ingestion)
    "src/utils/time_utils.py",     # Timing decorators for latency tracking

    # ── Source — Data Ingestion ──────────────────────────────────────────────
    "src/ingestion/__init__.py",
    "src/ingestion/ecfr_ingestor.py",      # eCFR XML API downloader
    "src/ingestion/faa_ad_ingestor.py",    # FAA Airworthiness Directives
    "src/ingestion/dgca_ingestor.py",      # DGCA CARs via Playwright
    "src/ingestion/faa_ac_ingestor.py",    # FAA Advisory Circulars
    "src/ingestion/skybrary_ingestor.py",  # SKYbrary scraper
    "src/ingestion/base_ingestor.py",      # Abstract base class for all ingestors

    # ── Source — Document Parsing ────────────────────────────────────────────
    "src/parsing/__init__.py",
    "src/parsing/pdf_parser.py",       # pdfplumber based PDF parser
    "src/parsing/xml_parser.py",       # lxml based XML parser for eCFR
    "src/parsing/html_parser.py",      # BeautifulSoup HTML parser
    "src/parsing/base_parser.py",      # Abstract base class for parsers

    # ── Source — Chunking ────────────────────────────────────────────────────
    "src/chunking/__init__.py",
    "src/chunking/recursive_chunker.py",    # Strategy 1: Recursive character splitting
    "src/chunking/semantic_chunker.py",     # Strategy 2: Semantic chunking
    "src/chunking/hierarchical_chunker.py", # Strategy 3: Regulation-aware hierarchical
    "src/chunking/chunking_evaluator.py",   # RAGAS evaluation for chunking strategies

    # ── Source — Embeddings ──────────────────────────────────────────────────
    "src/embeddings/__init__.py",
    "src/embeddings/openai_embedder.py",       # OpenAI text-embedding-3-small
    "src/embeddings/local_embedder.py",        # sentence-transformers BGE-M3
    "src/embeddings/embedding_evaluator.py",   # Compare embedding strategies

    # ── Source — Vector Store & Retrieval ───────────────────────────────────
    "src/retrieval/__init__.py",
    "src/retrieval/qdrant_store.py",       # Qdrant vector DB client
    "src/retrieval/dense_retriever.py",    # Dense vector search
    "src/retrieval/hybrid_retriever.py",   # Dense + BM25 sparse hybrid
    "src/retrieval/reranker.py",           # Cross-encoder re-ranking
    "src/retrieval/metadata_filter.py",    # Metadata filtering logic

    # ── Source — RAG Pipeline ────────────────────────────────────────────────
    "src/rag/__init__.py",
    "src/rag/rag_chain.py",            # Core RAG chain (LangChain)
    "src/rag/citation_builder.py",     # Citation assembler from retrieved chunks
    "src/rag/answer_validator.py",     # Confidence scoring + validation
    "src/rag/rag_evaluator.py",        # RAGAS metrics evaluation

    # ── Source — Agentic Orchestration (LangGraph) ──────────────────────────
    "src/agents/__init__.py",
    "src/agents/query_classifier.py",  # Classify query type
    "src/agents/planner.py",           # Multi-hop query planner
    "src/agents/router.py",            # Route to correct data source
    "src/agents/synthesizer.py",       # Synthesize multi-source answers
    "src/agents/agent_graph.py",       # LangGraph graph definition

    # ── Source — Monitoring & Observability ─────────────────────────────────
    "src/monitoring/__init__.py",
    "src/monitoring/langsmith_tracker.py",  # LangSmith trace management
    "src/monitoring/mlflow_tracker.py",     # MLflow experiment/run tracking
    "src/monitoring/metrics_collector.py",  # Custom metrics: latency, tokens, cost
    "src/monitoring/alert_manager.py",      # Alert rules + threshold checks
    "src/monitoring/cost_tracker.py",       # Per-token cost calculation (Claude + OpenAI)
    "src/monitoring/dashboard.py",          # Monitoring summary dashboard

    # ── Source — Database ────────────────────────────────────────────────────
    "src/database/__init__.py",
    "src/database/sqlite_store.py",    # SQLite metadata store
    "src/database/models.py",          # SQLite table schemas
    "src/database/migrations.py",      # DB migration scripts

    # ── Source — Alerts & Notifications ─────────────────────────────────────
    "src/alerts/__init__.py",
    "src/alerts/email_alerts.py",      # smtplib email sender
    "src/alerts/alert_templates.py",   # Email templates for AD alerts
    "src/alerts/hitl_gate.py",         # Human-in-the-loop confidence gate

    # ── Source — Scheduler ───────────────────────────────────────────────────
    "src/scheduler/__init__.py",
    "src/scheduler/refresh_scheduler.py",  # APScheduler jobs
    "src/scheduler/job_registry.py",       # All scheduled job definitions

    # ── Source — API ─────────────────────────────────────────────────────────
    "src/api/__init__.py",
    "src/api/main.py",                 # FastAPI app entry point
    "src/api/routes/compliance.py",    # POST /query/compliance
    "src/api/routes/ad_check.py",      # POST /query/ad-check
    "src/api/routes/preflight.py",     # POST /query/preflight
    "src/api/routes/__init__.py",
    "src/api/schemas.py",              # Pydantic request/response schemas
    "src/api/dependencies.py",         # FastAPI dependency injection
    "src/api/middleware.py",           # Request logging, error handling middleware

    # ── Source — UI ──────────────────────────────────────────────────────────
    "src/ui/__init__.py",
    "src/ui/streamlit_app.py",         # Streamlit demo UI
    "src/ui/components.py",            # Reusable UI components
    "src/ui/feedback_handler.py",      # HITL feedback widget logic

    # ── Experiments ──────────────────────────────────────────────────────────
    "experiments/chunking/",           # Chunking experiment results
    "experiments/embeddings/",         # Embedding experiment results
    "experiments/retrieval/",          # Retrieval experiment results
    "experiments/rag/",                # RAG evaluation results

    # ── Notebooks ────────────────────────────────────────────────────────────
    "notebooks/01_data_exploration.ipynb",
    "notebooks/02_chunking_experiments.ipynb",
    "notebooks/03_embedding_experiments.ipynb",
    "notebooks/04_retrieval_experiments.ipynb",
    "notebooks/05_rag_evaluation.ipynb",

    # ── Tests ────────────────────────────────────────────────────────────────
    "tests/__init__.py",
    "tests/test_ingestion.py",
    "tests/test_parsing.py",
    "tests/test_chunking.py",
    "tests/test_embeddings.py",
    "tests/test_retrieval.py",
    "tests/test_rag.py",
    "tests/test_agents.py",
    "tests/test_api.py",
    "tests/test_monitoring.py",
    "tests/conftest.py",               # Pytest fixtures

    # ── Docker ───────────────────────────────────────────────────────────────
    "docker/Dockerfile.api",
    "docker/Dockerfile.scheduler",
    "docker/Dockerfile.ui",
    "docker/docker-compose.yml",
    "docker/docker-compose.dev.yml",

    # ── CI/CD ────────────────────────────────────────────────────────────────
    ".github/workflows/ci.yml",        # GitHub Actions CI pipeline
    ".github/workflows/deploy.yml",    # GitHub Actions deploy pipeline

    # ── Data Folders ─────────────────────────────────────────────────────────
    "data/raw/ecfr/",
    "data/raw/faa_ads/",
    "data/raw/dgca/",
    "data/raw/faa_acs/",
    "data/raw/skybrary/",
    "data/processed/",
    "data/embeddings/",

    # ── Logs ─────────────────────────────────────────────────────────────────
    "logs/",

    # ── Reports ──────────────────────────────────────────────────────────────
    "reports/ragas/",
    "reports/mlflow/",
    "reports/alerts/",

    # ── Scripts ──────────────────────────────────────────────────────────────
    "scripts/setup_qdrant.py",         # One-time Qdrant collection setup
    "scripts/seed_data.py",            # Seed test data
    "scripts/run_experiments.py",      # Run all chunking/embedding experiments
    "scripts/evaluate_rag.py",         # Full RAGAS evaluation runner

]

# ── Helper functions ──────────────────────────────────────────────────────────

def create_file(filepath: str) -> None:
    """Create an empty file with a placeholder comment if it's a .py file."""
    full_path = os.path.join(ROOT, filepath)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    if os.path.exists(full_path):
        print(f"  [EXISTS]  {filepath}")
        return

    with open(full_path, "w", encoding="utf-8") as f:
        if filepath.endswith(".py"):
            # Add module docstring as placeholder
            module_name = os.path.basename(filepath).replace(".py", "")
            f.write(f'"""\nAeroLex — {module_name}\nTODO: Implement this module\n"""\n')
        elif filepath.endswith(".yml") or filepath.endswith(".yaml"):
            f.write("# TODO: Configure this file\n")
        elif filepath.endswith(".ipynb"):
            # Minimal valid notebook
            f.write('{"cells":[],"metadata":{},"nbformat":4,"nbformat_minor":5}')
        else:
            f.write("")

    print(f"  [CREATED] {filepath}")


def create_folder(folderpath: str) -> None:
    """Create a folder with a .gitkeep so Git tracks it."""
    full_path = os.path.join(ROOT, folderpath)
    os.makedirs(full_path, exist_ok=True)

    gitkeep = os.path.join(full_path, ".gitkeep")
    if not os.path.exists(gitkeep):
        with open(gitkeep, "w") as f:
            f.write("")

    print(f"  [FOLDER]  {folderpath}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n🚀 AeroLex — Setting up project structure...\n")

    for item in STRUCTURE:
        if item.endswith("/"):
            # It's a folder
            create_folder(item)
        else:
            # It's a file
            create_file(item)

    print("\n✅ Project structure created successfully!")
    print(f"📁 Root: {ROOT}")
    print("\nNext step: Implement src/utils/logger.py\n")


if __name__ == "__main__":
    main()