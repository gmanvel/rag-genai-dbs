"""
Text RAG branch: Retrieves relevant text chunks from Qdrant.
Uses the exact approach from local_rag_playground.ipynb notebook.
"""

from typing import List, Dict, Any
from qdrant_client.models import ScoredPoint

from config import (
    TEXT_COLLECTION_NAME,
    TEXT_RETRIEVAL_LIMIT,
    MIN_TEXT_SCORE,
)
from models_local import get_text_embedding_model, get_qdrant_client


def retrieve_text_chunks(query: str, limit: int = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant text chunks from Qdrant based on the user query.

    This implements the exact approach from the local_rag_playground.ipynb notebook:
    1. Embed the query using nomic-embed-text via OllamaEmbedding
    2. Search the 'video_chunks' collection in Qdrant
    3. Return structured results with text, source_doc_id, and score

    Args:
        query: User's question/query text
        limit: Maximum number of chunks to retrieve (defaults to config setting)

    Returns:
        List of dictionaries with keys:
        - text: The chunk text content
        - source_doc_id: Source document identifier
        - score: Similarity score (cosine distance)
        - chunk_id: Unique chunk identifier

    Raises:
        Exception: If embedding or retrieval fails
    """
    if limit is None:
        limit = TEXT_RETRIEVAL_LIMIT

    try:
        # Step 1: Get embedding model and Qdrant client
        embedding_model = get_text_embedding_model()
        qdrant_client = get_qdrant_client()

        print(f"[TEXT RETRIEVAL] Embedding query: '{query[:50]}...'")

        # Step 2: Embed the query using get_query_embedding (exact notebook method)
        query_vector = embedding_model.get_query_embedding(query)

        print(f"[TEXT RETRIEVAL] Query vector dimension: {len(query_vector)}")

        # Step 3: Search Qdrant (exact notebook approach)
        search_results = qdrant_client.search(
            collection_name=TEXT_COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,  # Include payload (text, source_doc_id)
        )

        print(f"[TEXT RETRIEVAL] Found {len(search_results)} results")

        # Step 4: Format results to match expected structure
        formatted_results = []
        for result in search_results:
            # Filter by minimum score if configured
            if result.score < MIN_TEXT_SCORE:
                continue

            formatted_results.append({
                "text": result.payload.get("text", ""),
                "source_doc_id": result.payload.get("source_doc_id", "unknown"),
                "score": float(result.score),
                "chunk_id": str(result.id),  # UUID from notebook
            })

        print(f"[TEXT RETRIEVAL] Returning {len(formatted_results)} chunks after filtering")

        return formatted_results

    except Exception as e:
        print(f"[ERROR] Text retrieval failed: {e}")
        raise


def check_text_collection_exists() -> bool:
    """
    Check if the text collection exists in Qdrant.
    Useful for validation at startup.

    Returns:
        True if collection exists, False otherwise
    """
    try:
        qdrant_client = get_qdrant_client()
        collections = qdrant_client.get_collections()

        collection_names = [c.name for c in collections.collections]
        exists = TEXT_COLLECTION_NAME in collection_names

        if exists:
            print(f"[TEXT RETRIEVAL] Collection '{TEXT_COLLECTION_NAME}' found in Qdrant")
        else:
            print(f"[WARNING] Collection '{TEXT_COLLECTION_NAME}' NOT found in Qdrant")
            print(f"[WARNING] Available collections: {collection_names}")

        return exists

    except Exception as e:
        print(f"[ERROR] Failed to check text collection: {e}")
        return False


def get_text_collection_info() -> Dict[str, Any]:
    """
    Get information about the text collection (vector count, config, etc.).
    Useful for debugging and UI display.

    Returns:
        Dictionary with collection information
    """
    try:
        qdrant_client = get_qdrant_client()

        collection_info = qdrant_client.get_collection(TEXT_COLLECTION_NAME)

        return {
            "name": TEXT_COLLECTION_NAME,
            "vectors_count": collection_info.vectors_count,
            "points_count": collection_info.points_count,
            "status": collection_info.status,
        }

    except Exception as e:
        print(f"[ERROR] Failed to get text collection info: {e}")
        return {
            "name": TEXT_COLLECTION_NAME,
            "error": str(e),
        }


def test_text_retrieval() -> bool:
    """
    Test text retrieval with a simple query.
    Useful for health checks and debugging.

    Returns:
        True if retrieval works, False otherwise
    """
    try:
        test_query = "What is this document about?"
        results = retrieve_text_chunks(test_query, limit=1)

        success = len(results) > 0
        if success:
            print(f"[TEXT RETRIEVAL TEST] SUCCESS - Retrieved {len(results)} results")
        else:
            print("[TEXT RETRIEVAL TEST] WARNING - No results returned")

        return success

    except Exception as e:
        print(f"[TEXT RETRIEVAL TEST] FAILED - {e}")
        return False
