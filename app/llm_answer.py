"""
LLM answer generation: Merges context from both branches and calls Mistral.
Generates the final answer using local Mistral model via Ollama.
"""

from typing import List, Dict, Any

from config import (
    MAX_TEXT_CONTEXT_LENGTH,
    MAX_VISION_CONTEXT_LENGTH,
)
from models_local import call_mistral_llm
from utils import build_rag_prompt


def generate_answer(
    user_question: str,
    text_chunks: List[Dict[str, Any]],
    vision_summaries: List[Dict[str, Any]],
    max_tokens: int = 1000,
) -> str:
    """
    Generate the final answer by merging context from both RAG branches.

    Workflow:
    1. Combine text chunks and vision summaries into a single context
    2. Build a prompt for Mistral that includes:
       - User's question
       - Text evidence from text retrieval branch
       - Visual evidence from vision retrieval branch
    3. Call local Mistral via Ollama
    4. Return the generated answer

    Args:
        user_question: The original user question
        text_chunks: Retrieved text chunks from text RAG branch
        vision_summaries: Page summaries from vision RAG branch
        max_tokens: Maximum tokens to generate

    Returns:
        Generated answer string from Mistral

    Raises:
        Exception: If LLM call fails
    """
    try:
        print("[ANSWER GENERATION] Building prompt from merged context...")

        # Build the complete prompt using utility function
        prompt = build_rag_prompt(
            user_question=user_question,
            text_chunks=text_chunks,
            vision_summaries=vision_summaries,
            max_text_length=MAX_TEXT_CONTEXT_LENGTH,
            max_vision_length=MAX_VISION_CONTEXT_LENGTH,
        )

        print(f"[ANSWER GENERATION] Prompt length: {len(prompt)} characters")
        print(f"[ANSWER GENERATION] Calling Mistral LLM (max_tokens={max_tokens})...")

        # Call Mistral via Ollama
        answer = call_mistral_llm(prompt=prompt, max_tokens=max_tokens)

        print(f"[ANSWER GENERATION] Generated answer length: {len(answer)} characters")

        return answer.strip()

    except Exception as e:
        print(f"[ERROR] Answer generation failed: {e}")
        raise


def generate_answer_text_only(
    user_question: str,
    text_chunks: List[Dict[str, Any]],
    max_tokens: int = 1000,
) -> str:
    """
    Generate answer using only text context (fallback when vision fails).

    Args:
        user_question: The original user question
        text_chunks: Retrieved text chunks from text RAG branch
        max_tokens: Maximum tokens to generate

    Returns:
        Generated answer string from Mistral
    """
    return generate_answer(
        user_question=user_question,
        text_chunks=text_chunks,
        vision_summaries=[],
        max_tokens=max_tokens,
    )


def generate_answer_vision_only(
    user_question: str,
    vision_summaries: List[Dict[str, Any]],
    max_tokens: int = 1000,
) -> str:
    """
    Generate answer using only vision context (fallback when text fails).

    Args:
        user_question: The original user question
        vision_summaries: Page summaries from vision RAG branch
        max_tokens: Maximum tokens to generate

    Returns:
        Generated answer string from Mistral
    """
    return generate_answer(
        user_question=user_question,
        text_chunks=[],
        vision_summaries=vision_summaries,
        max_tokens=max_tokens,
    )


def validate_context(
    text_chunks: List[Dict[str, Any]],
    vision_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Validate that the context from both branches is sufficient for answer generation.

    Args:
        text_chunks: Retrieved text chunks
        vision_summaries: Page summaries

    Returns:
        Dictionary with validation results:
        - has_text: bool
        - has_vision: bool
        - has_any: bool
        - text_count: int
        - vision_count: int
        - warnings: list of warning messages
    """
    has_text = len(text_chunks) > 0
    has_vision = len(vision_summaries) > 0

    warnings = []

    if not has_text and not has_vision:
        warnings.append("No context available from either branch")
    elif not has_text:
        warnings.append("No text context available - answer will be based only on visual sources")
    elif not has_vision:
        warnings.append("No visual context available - answer will be based only on text sources")

    return {
        "has_text": has_text,
        "has_vision": has_vision,
        "has_any": has_text or has_vision,
        "text_count": len(text_chunks),
        "vision_count": len(vision_summaries),
        "warnings": warnings,
    }


def test_answer_generation() -> bool:
    """
    Test answer generation with mock data.
    Useful for health checks and debugging.

    Returns:
        True if answer generation works, False otherwise
    """
    try:
        mock_question = "What is the main topic?"

        mock_text_chunks = [
            {
                "text": "This is a test document about machine learning.",
                "source_doc_id": "test_doc",
                "score": 0.9,
            }
        ]

        mock_vision_summaries = [
            {
                "summary": "A diagram showing neural network architecture.",
                "page_index": 0,
                "score": 0.85,
            }
        ]

        answer = generate_answer(
            user_question=mock_question,
            text_chunks=mock_text_chunks,
            vision_summaries=mock_vision_summaries,
            max_tokens=100,
        )

        success = len(answer) > 0
        if success:
            print(f"[ANSWER GENERATION TEST] SUCCESS - Generated {len(answer)} characters")
        else:
            print("[ANSWER GENERATION TEST] WARNING - Empty answer generated")

        return success

    except Exception as e:
        print(f"[ANSWER GENERATION TEST] FAILED - {e}")
        return False
