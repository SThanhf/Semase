# src/settings.py
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()



def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and (val is None or str(val).strip() == ""):
        raise RuntimeError(
            f"Missing required environment variable: {name}\n"
            f"Please set it in your environment or .env file."
        )
    return str(val) if val is not None else ""


def _env_int(name: str, default: int | None = None, *, required: bool = False) -> int:
    raw = os.getenv(name, None)
    if raw is None or raw.strip() == "":
        if required and default is None:
            raise RuntimeError(
                f"Missing required environment variable: {name}\n"
                f"Please set it in your environment or .env file."
            )
        if default is None:
            raise RuntimeError(f"Missing int env var {name} and no default provided.")
        return int(default)
    try:
        return int(raw)
    except ValueError as e:
        raise RuntimeError(f"Environment variable {name} must be an integer, got: {raw}") from e


@dataclass(frozen=True)
class Settings:
    # Azure AI Search
    search_endpoint: str
    search_key: str
    search_document_index_name: str
    search_chunk_index_name: str

    search_indexer_name: str
    search_skillset_name: str
    search_data_source_name: str
    search_data_source_connection: str
    search_data_container_name: str

    # Azure OpenAI
    ai_endpoint: str
    ai_key: str
    ai_api_version: str

    # Embeddings
    search_embedding_model: str
    search_embedding_dim: int

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            # Search
            search_endpoint=_env("AZURE_SEARCH_ENDPOINT", required=True),
            search_key=_env("AZURE_SEARCH_KEY", required=True),
            search_document_index_name=_env("AZURE_SEARCH_DOCUMENT_INDEX", "documents"),
            search_chunk_index_name=_env("AZURE_SEARCH_CHUNK_INDEX", "chunks"),
            search_indexer_name=_env("AZURE_SEARCH_INDEXER", "documents-indexer"),
            search_skillset_name=_env("AZURE_SEARCH_SKILLSET", "documents-skillset"),
            search_data_source_name=_env("AZURE_SEARCH_DATASOURCE", "documents-datasource"),
            search_data_source_connection=_env("AZURE_SEARCH_BLOB_CONNECTION_STRING", required=True),
            search_data_container_name=_env("AZURE_SEARCH_BLOB_CONTAINER", required=True),
            # Azure OpenAI
            ai_endpoint=_env("AZURE_OPENAI_ENDPOINT", required=True),
            ai_key=_env("AZURE_OPENAI_KEY", required=True),
            ai_api_version=_env("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            # Embedding deployment/model
            search_embedding_model=_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", required=True),
            search_embedding_dim=_env_int("AZURE_OPENAI_EMBEDDING_DIM", 1536),
        )


SETTINGS = Settings.from_env()
