"""Configuration settings using Pydantic Settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project paths (computed at import time)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HELP_CSV = DATA_DIR / "help.csv"
QUESTIONS_CSV = DATA_DIR / "questions.csv"
CONVERSATIONS_DIR = DATA_DIR / "conversations"
SCENARIOS_JSON = CONVERSATIONS_DIR / "scenarios.json"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = Field(..., description="OpenAI API key")

    # LangFuse observability
    langfuse_public_key: str = Field(..., description="LangFuse public key")
    langfuse_secret_key: str = Field(..., description="LangFuse secret key")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="LangFuse host URL",
    )

    # ChromaDB
    chroma_persist_dir: Path = Field(
        default=DATA_DIR / "db",
        description="ChromaDB persistence directory",
    )
    chroma_collection_name: str = Field(
        default="help_center",
        description="ChromaDB collection name",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    log_format: str = Field(
        default="console",
        description="Log format: 'console' (dev) or 'json' (prod)",
    )

    # RAG settings
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI LLM model",
    )
    retrieval_top_k: int = Field(
        default=5,
        description="Number of chunks to retrieve",
    )

    # Memory settings
    max_conversation_turns: int = Field(
        default=10,
        description="Maximum conversation turns to keep in context",
    )


settings = Settings()
