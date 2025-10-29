"""Configuration settings for the local RAG Streamlit app."""
from __future__ import annotations

import os
from functools import lru_cache

QDRANT_URL_DOCKER = "http://qdrant:6333"
QDRANT_URL_LOCAL = "http://localhost:6333"

COLLECTION_NAME = "rag_chunks"

OLLAMA_URL_DOCKER = "http://ollama:11434"
OLLAMA_URL_LOCAL = "http://localhost:11434"

PAGE_INDEX_DIR = "data/pages"


@lru_cache(maxsize=1)
def get_runtime_mode() -> str:
    """Return the runtime mode, defaulting to local."""
    return os.environ.get("RUNTIME_MODE", "local").strip().lower()


def get_qdrant_url() -> str:
    """Return the Qdrant URL based on runtime mode."""
    if get_runtime_mode() == "docker":
        return QDRANT_URL_DOCKER
    return QDRANT_URL_LOCAL


def get_ollama_url() -> str:
    """Return the Ollama URL based on runtime mode."""
    if get_runtime_mode() == "docker":
        return OLLAMA_URL_DOCKER
    return OLLAMA_URL_LOCAL


def get_page_index_dir() -> str:
    """Return the directory containing page or image assets."""
    return PAGE_INDEX_DIR


__all__ = [
    "COLLECTION_NAME",
    "QDRANT_URL_DOCKER",
    "QDRANT_URL_LOCAL",
    "OLLAMA_URL_DOCKER",
    "OLLAMA_URL_LOCAL",
    "PAGE_INDEX_DIR",
    "get_runtime_mode",
    "get_qdrant_url",
    "get_ollama_url",
    "get_page_index_dir",
]
