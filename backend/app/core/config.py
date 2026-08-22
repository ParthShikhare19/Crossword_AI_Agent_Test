"""
Application configuration for the Aster & Row support agent.

This module centralizes environment variables and application-level
configuration. Keeping configuration in one place prevents secrets and
runtime settings from being hard-coded throughout the codebase.

Environment variables are loaded from a local .env file during development.
The .env file itself must never be committed to version control.
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve the backend directory from this file so paths work regardless
# of the directory from which the application is started.
BASE_DIR = Path(__file__).resolve().parents[2]

# Load environment variables from backend/.env when running locally.
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    """
    Runtime configuration for the support agent.

    Pydantic validates configuration values when the application starts,
    allowing configuration errors to be detected early instead of failing
    later during an API request.
    """

    # Groq credentials and model configuration.
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    groq_model: str = Field(
    default="openai/gpt-oss-20b",
    validation_alias="GROQ_MODEL",
)

    # Embedding model used to convert knowledge-base chunks into vectors.
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )

    # Storage location for the generated FAISS index and associated metadata.
    faiss_index_path: str = Field(
        default="storage/faiss",
        alias="FAISS_INDEX_PATH",
    )

    # Application behavior.
    debug: bool = Field(default=False, alias="DEBUG")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


# A single settings instance is shared by the application.
# This avoids repeatedly loading and parsing environment variables.
settings = Settings()