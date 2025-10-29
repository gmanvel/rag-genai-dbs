"""Prompt assembly and answer generation."""
from __future__ import annotations

from typing import List

from .models_local import generate_llm_answer
from .utils import truncate_text


def _format_text_evidence(text_evidence: List[dict]) -> str:
    if not text_evidence:
        return "- No relevant text chunks found."
    lines = []
    for chunk in text_evidence:
        preview = truncate_text(chunk.get("text", ""), width=400)
        source = chunk.get("source_doc_id", "unknown")
        lines.append(f"- Source {source}: {preview}")
    return "\n".join(lines)


def _format_vision_evidence(vision_evidence: List[dict]) -> str:
    if not vision_evidence:
        return "- No relevant pages or images found."
    lines = []
    for entry in vision_evidence:
        path = entry.get("page_image_path", "unknown")
        summary = truncate_text(entry.get("summary", ""), width=400)
        lines.append(f"- {path}: {summary}")
    return "\n".join(lines)


def generate_answer(question: str, text_evidence: List[dict], vision_evidence: List[dict]) -> str:
    """Combine evidence and ask the local LLM for an answer."""
    prompt = (
        "You are a helpful assistant that answers questions using provided evidence.\n"
        f"Question: {question}\n\n"
        "Text Evidence:\n"
        f"{_format_text_evidence(text_evidence)}\n\n"
        "Vision Evidence:\n"
        f"{_format_vision_evidence(vision_evidence)}\n\n"
        "Provide a concise answer that cites relevant evidence in natural language."
    )
    return generate_llm_answer(prompt)


__all__ = ["generate_answer"]
