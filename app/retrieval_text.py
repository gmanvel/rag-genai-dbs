"""Text retrieval branch using Qdrant."""
from __future__ import annotations

import atexit
import logging
from typing import List

from qdrant_client import QdrantClient
from . import config
from .models_local import embed_text
from .utils import truncate_text

logger = logging.getLogger(__name__)

_client = QdrantClient(url=config.get_qdrant_url())


def _cleanup_client() -> None:
    """Close the Qdrant client on shutdown."""
    close_method = getattr(_client, "close", None)
    if callable(close_method):
        try:
            close_method()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.debug("Error closing Qdrant client: %s", exc)


atexit.register(_cleanup_client)


def run_text_branch(query: str, limit: int = 3) -> List[dict]:
    """Retrieve similar text chunks from Qdrant."""
    try:
        embedding = embed_text(query)
        results = _client.search(
            collection_name=config.COLLECTION_NAME,
            query_vector=embedding,
            limit=limit,
            with_payload=True,
        )
    except Exception as exc:
        logger.exception("Text retrieval branch failed", exc_info=exc)
        raise

    formatted: List[dict] = []
    for point in results:
        payload = point.payload or {}
        text = payload.get("text", "")
        source_doc_id = payload.get("source_doc_id", "unknown")
        formatted.append(
            {
                "text": text,
                "source_doc_id": source_doc_id,
                "score": point.score,
                "preview": truncate_text(text),
            }
        )
    return formatted


__all__ = ["run_text_branch"]
