"""Utility helpers for formatting and truncating evidence."""
from __future__ import annotations

from textwrap import shorten
from typing import Iterable, List


def truncate_text(text: str, width: int = 200) -> str:
    """Return a human-friendly truncated string."""
    return shorten(text, width=width, placeholder="…")


def format_text_evidence(chunks: Iterable[dict]) -> List[str]:
    """Format text evidence chunks for display."""
    formatted: List[str] = []
    for chunk in chunks:
        preview = truncate_text(chunk.get("text", ""))
        source = chunk.get("source_doc_id", "unknown")
        score = chunk.get("score")
        if score is not None:
            formatted.append(f"{preview}\nSource: {source} (score: {score:.3f})")
        else:
            formatted.append(f"{preview}\nSource: {source}")
    return formatted


def format_vision_evidence(entries: Iterable[dict]) -> List[str]:
    """Format vision evidence summaries for display."""
    formatted: List[str] = []
    for entry in entries:
        path = entry.get("page_image_path", "unknown")
        summary = truncate_text(entry.get("summary", ""))
        formatted.append(f"{path}: {summary}")
    return formatted


__all__ = [
    "truncate_text",
    "format_text_evidence",
    "format_vision_evidence",
]
