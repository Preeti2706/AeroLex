"""
AeroLex — Centralized Application Settings

Uses Pydantic BaseSettings for:
- Automatic .env file loading
- Type validation at startup
- Clear error messages for missing required fields
- Single source of truth for all configuration

Usage:
    from config.settings import settings
    print(settings.ANTHROPIC_API_KEY)
    print(settings.MLFLOW_TRACKING_URI)
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AeroLexSettings(BaseSettings):
    """
    All AeroLex configuration in one place.
    Pydantic automatically reads from .env file.
    If a required field is missing, app crashes at startup with clear message.
    This is GOOD — fail fast, fail loud.
    """

    # ── LLM API Keys ─────────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = Field(..., description="Claude API key")
    OPENAI_API_KEY: str    = Field(..., description="OpenAI API key")

    # ── LangSmith ────────────────────────────────────────────────────────────
    LANGSMITH_API_KEY: str          = Field(..., description="LangSmith API key")
    LANGSMITH_PROJECT: str          = Field(default="aerolex")
    LANGSMITH_TRACING: str          = Field(default="true")
    LANGSMITH_ENDPOINT: str         = Field(default="https://api.smith.langchain.com")

    # ── MLflow ───────────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str        = Field(default="http://localhost:5000")

    # ── App Config ───────────────────────────────────────────────────────────
    APP_ENV: str                    = Field(default="development")
    LOG_LEVEL: str                  = Field(default="DEBUG")

    # ── LLM Model Selection ──────────────────────────────────────────────────
    # Primary models — can be changed here to switch everywhere
    CLAUDE_PRIMARY_MODEL: str       = Field(default="claude-sonnet-4-20250514")
    CLAUDE_FAST_MODEL: str          = Field(default="claude-haiku-4-5-20251001")
    OPENAI_PRIMARY_MODEL: str       = Field(default="gpt-4o-mini")
    OPENAI_STRONG_MODEL: str        = Field(default="gpt-4o")
    EMBEDDING_MODEL_OPENAI: str     = Field(default="text-embedding-3-small")
    EMBEDDING_MODEL_LOCAL: str      = Field(default="BAAI/bge-m3")

    # ── Qdrant Vector DB ─────────────────────────────────────────────────────
    QDRANT_HOST: str                = Field(default="localhost")
    QDRANT_PORT: int                = Field(default=6333)

    # ── Cost Alert Thresholds ────────────────────────────────────────────────
    COST_ALERT_SINGLE_CALL_USD: float   = Field(default=0.10)
    COST_ALERT_SESSION_USD: float       = Field(default=1.00)
    LATENCY_ALERT_MS: int               = Field(default=5000)

    # ── Email Alerts ─────────────────────────────────────────────────────────
    ALERT_EMAIL_RECIPIENT: str      = Field(default="srivastavashantam@gmail.com")
    GMAIL_SENDER_EMAIL: Optional[str]   = Field(default=None)
    GMAIL_APP_PASSWORD: Optional[str]   = Field(default=None)

    # ── RAG Config ───────────────────────────────────────────────────────────
    RAG_TOP_K: int                  = Field(default=5)
    RAG_CONFIDENCE_THRESHOLD: float = Field(default=0.85)

    # ── Validators ───────────────────────────────────────────────────────────
    @field_validator("APP_ENV")
    @classmethod
    def validate_app_env(cls, v):
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"APP_ENV must be one of {allowed}, got: {v}")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v):
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}, got: {v}")
        return v.upper()

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore"  # Ignore extra fields in .env
    }


# ── Singleton instance ────────────────────────────────────────────────────────
# Import this everywhere — don't instantiate AeroLexSettings directly
# Why singleton? So settings are loaded once and reused — not reloaded per import
try:
    settings = AeroLexSettings()
    logger.info(f"Settings loaded — ENV: {settings.APP_ENV} | LOG_LEVEL: {settings.LOG_LEVEL}")
except Exception as e:
    logger.critical(f"Failed to load settings: {e}")
    raise


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Testing Settings ---\n")
    print(f"APP_ENV              : {settings.APP_ENV}")
    print(f"LOG_LEVEL            : {settings.LOG_LEVEL}")
    print(f"ANTHROPIC_API_KEY    : {settings.ANTHROPIC_API_KEY[:8]}...")
    print(f"OPENAI_API_KEY       : {settings.OPENAI_API_KEY[:8]}...")
    print(f"LANGSMITH_PROJECT    : {settings.LANGSMITH_PROJECT}")
    print(f"MLFLOW_TRACKING_URI  : {settings.MLFLOW_TRACKING_URI}")
    print(f"CLAUDE_PRIMARY_MODEL : {settings.CLAUDE_PRIMARY_MODEL}")
    print(f"OPENAI_PRIMARY_MODEL : {settings.OPENAI_PRIMARY_MODEL}")
    print(f"QDRANT_HOST          : {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    print(f"RAG_TOP_K            : {settings.RAG_TOP_K}")
    print(f"\n✅ Settings working!")