"""Vision retrieval branch using an in-memory page index."""
from __future__ import annotations

import logging
from typing import List

from .models_local import summarize_image_with_qwen
from .utils import truncate_text

logger = logging.getLogger(__name__)

PAGE_INDEX = [
    {
        "page_image_path": "data/pages/page_01.png",
        "page_text_hint": "Introduction and executive summary about renewable energy credits.",
    },
    {
        "page_image_path": "data/pages/page_12.png",
        "page_text_hint": "Renewal terms and contract obligations for service level agreements.",
    },
    {
        "page_image_path": "data/pages/page_18.png",
        "page_text_hint": "Financial projections and bar charts comparing quarterly revenue.",
    },
]


def _score_page(query_tokens: List[str], hint: str) -> int:
    score = 0
    hint_lower = hint.lower()
    for token in query_tokens:
        if token and token.lower() in hint_lower:
            score += 1
    return score


def run_vision_branch(query: str, max_results: int = 3) -> List[dict]:
    """Select relevant pages and summarize them with the local Qwen VLM stub."""
    query_tokens = [token.strip() for token in query.split()]
    scored_pages = []
    for page in PAGE_INDEX:
        score = _score_page(query_tokens, page.get("page_text_hint", ""))
        if score > 0:
            scored_pages.append((score, page))

    scored_pages.sort(key=lambda item: item[0], reverse=True)
    selected_pages = [page for _, page in scored_pages[:max_results]]

    summaries: List[dict] = []
    for page in selected_pages:
        image_path = page.get("page_image_path", "")
        hint = page.get("page_text_hint", "")
        try:
            summary = summarize_image_with_qwen(image_path=image_path, query=query)
        except Exception as exc:  # pragma: no cover - defensive programming
            logger.exception("Vision branch summary failed", exc_info=exc)
            summary = f"Failed to summarize {image_path}: {exc}"
        summaries.append(
            {
                "page_image_path": image_path,
                "summary": summary,
                "hint": truncate_text(hint),
            }
        )
    return summaries


__all__ = ["PAGE_INDEX", "run_vision_branch"]
