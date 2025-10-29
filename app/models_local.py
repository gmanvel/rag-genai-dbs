"""Local model wrappers for embeddings, language generation, and vision summaries."""
from __future__ import annotations

import json
from typing import List

import requests
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

from . import config


class _EmbeddingSingleton:
    """Singleton wrapper so we only create the embedding model once."""

    _instance: BaseEmbedding | None = None

    @classmethod
    def instance(cls) -> BaseEmbedding:
        if cls._instance is None:
            cls._instance = OllamaEmbedding(
                model="nomic-embed-text",
                base_url=config.get_ollama_url(),
            )
        return cls._instance


def embed_text(text: str) -> List[float]:
    """Generate an embedding vector for the given text using local Ollama."""
    embedding_model = _EmbeddingSingleton.instance()
    return embedding_model.get_text_embedding(text)


def _ollama_generate(model: str, prompt: str) -> str:
    """Call the local Ollama service to generate text."""
    response = requests.post(
        f"{config.get_ollama_url().rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    # Ollama returns a JSON object with the final response under "response".
    if isinstance(data, dict) and "response" in data:
        return data["response"].strip()
    return json.dumps(data)


def generate_llm_answer(prompt: str) -> str:
    """Generate an answer from the local Mistral model via Ollama."""
    return _ollama_generate(model="mistral", prompt=prompt)


def summarize_image_with_qwen(image_path: str, query: str) -> str:
    """Stubbed deterministic summary emulating a local Qwen VLM call."""
    # This is a deterministic stub. In a real setup this would call a local Qwen pipeline.
    return (
        f"Summary for {image_path} given query '{query}': "
        "Highlights likely relevant visual cues and annotations."
    )


__all__ = [
    "embed_text",
    "generate_llm_answer",
    "summarize_image_with_qwen",
]
