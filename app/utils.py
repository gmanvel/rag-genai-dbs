"""
Utility functions for the RAG Q&A application.
Includes text processing, prompt building, evidence formatting, and ColPali helpers.
"""

import torch
from typing import List, Dict, Any
from PIL import Image


# ============================================================================
# TEXT PROCESSING
# ============================================================================

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length, adding a suffix if truncated.

    Args:
        text: Input text to truncate
        max_length: Maximum character length
        suffix: Suffix to add if truncated (default: "...")

    Returns:
        Truncated text with suffix if needed
    """
    if len(text) <= max_length:
        return text

    # Truncate and add suffix
    return text[:max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """
    Clean and normalize text for display or processing.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = " ".join(text.split())

    # Remove null bytes that might cause issues
    text = text.replace("\x00", "")

    return text.strip()


# ============================================================================
# COLPALI MEAN POOLING (exact notebook implementation)
# ============================================================================

def mean_pool_colpali_output(model_output: torch.Tensor) -> torch.Tensor:
    """
    Mean pool ColPali output to get a single vector per image.

    ColPali natively produces [num_regions, emb_dim] embeddings (e.g., ~1000 vectors of 128-dim).
    For page-level retrieval, we collapse to a single vector using mean pooling.

    Args:
        model_output: Tensor of shape [num_regions, emb_dim] or [batch, num_regions, emb_dim]

    Returns:
        Tensor of shape [emb_dim] or [batch, emb_dim] after mean pooling
    """
    if model_output.dim() == 3:
        # Batch mode: [batch, num_regions, emb_dim] -> [batch, emb_dim]
        return model_output.mean(dim=1)
    elif model_output.dim() == 2:
        # Single mode: [num_regions, emb_dim] -> [emb_dim]
        return model_output.mean(dim=0)
    else:
        raise ValueError(f"Unexpected ColPali output shape: {model_output.shape}")


def embed_query_colpali(query_text: str, model, processor, device) -> List[float]:
    """
    Embed a text query using ColPali model with mean pooling.
    This is the exact approach from the notebook.

    Args:
        query_text: User's query text
        model: ColPali model
        processor: ColPali processor
        device: Torch device (cuda/mps/cpu)

    Returns:
        List of floats representing the query embedding (128-dim)
    """
    # Process query text
    query_inputs = processor.process_queries([query_text])
    query_inputs = {k: v.to(device) for k, v in query_inputs.items()}

    with torch.no_grad():
        output = model(**query_inputs)

    # ColPali query embeddings are [batch, seq_len, 128]
    # Mean pool across sequence dimension
    if output.dim() == 3:
        output = output[0]  # Take first (and only) batch item: [seq_len, 128]

    # Mean pool to get [128]
    pooled = mean_pool_colpali_output(output)

    # Convert to float32 list
    return pooled.to(torch.float32).cpu().tolist()


# ============================================================================
# PROMPT BUILDING
# ============================================================================

def build_rag_prompt(
    user_question: str,
    text_chunks: List[Dict[str, Any]],
    vision_summaries: List[Dict[str, Any]],
    max_text_length: int = 2000,
    max_vision_length: int = 1500,
) -> str:
    """
    Build the final prompt for the LLM by combining user question with context from both branches.

    Args:
        user_question: User's original question
        text_chunks: List of retrieved text chunks with 'text' and 'source_doc_id' keys
        vision_summaries: List of vision summaries with 'summary' and 'page_index' keys
        max_text_length: Max chars per text chunk
        max_vision_length: Max chars per vision summary

    Returns:
        Complete prompt string for the LLM
    """
    prompt_parts = []

    # System instruction
    prompt_parts.append(
        "You are a helpful AI assistant answering questions based on provided context. "
        "Use the context below to answer the user's question accurately and concisely. "
        "If the context doesn't contain enough information, say so clearly."
    )
    prompt_parts.append("")

    # Text context section
    if text_chunks:
        prompt_parts.append("=== TEXT CONTEXT ===")
        for idx, chunk in enumerate(text_chunks, 1):
            text = truncate_text(chunk["text"], max_text_length)
            source = chunk.get("source_doc_id", "unknown")
            prompt_parts.append(f"\n[Text Source {idx}] (from {source}):")
            prompt_parts.append(text)
        prompt_parts.append("")

    # Vision context section
    if vision_summaries:
        prompt_parts.append("=== VISUAL CONTEXT ===")
        for idx, summary in enumerate(vision_summaries, 1):
            summary_text = truncate_text(summary["summary"], max_vision_length)
            page_num = summary.get("page_index", -1) + 1  # Convert to 1-based
            prompt_parts.append(f"\n[Visual Source {idx}] (from page {page_num}):")
            prompt_parts.append(summary_text)
        prompt_parts.append("")

    # User question
    prompt_parts.append("=== QUESTION ===")
    prompt_parts.append(user_question)
    prompt_parts.append("")

    # Request for answer
    prompt_parts.append("=== YOUR ANSWER ===")
    prompt_parts.append("Based on the context above, provide a clear and accurate answer:")

    return "\n".join(prompt_parts)


# ============================================================================
# EVIDENCE FORMATTING
# ============================================================================

def format_text_evidence(text_chunks: List[Dict[str, Any]], max_display_length: int = 300) -> str:
    """
    Format text evidence for display in the UI.

    Args:
        text_chunks: List of text chunks with 'text', 'source_doc_id', and 'score'
        max_display_length: Maximum characters to show per chunk

    Returns:
        Formatted markdown string for display
    """
    if not text_chunks:
        return "_No text evidence found._"

    lines = []
    for idx, chunk in enumerate(text_chunks, 1):
        text = truncate_text(chunk["text"], max_display_length)
        source = chunk.get("source_doc_id", "unknown")
        score = chunk.get("score", 0.0)

        lines.append(f"**{idx}. {source}** (similarity: {score:.3f})")
        lines.append(f"> {text}")
        lines.append("")

    return "\n".join(lines)


def format_vision_evidence(vision_summaries: List[Dict[str, Any]], max_display_length: int = 400) -> str:
    """
    Format vision evidence for display in the UI.

    Args:
        vision_summaries: List of summaries with 'summary', 'page_index', 'page_image_path', and 'score'
        max_display_length: Maximum characters to show per summary

    Returns:
        Formatted markdown string for display
    """
    if not vision_summaries:
        return "_No visual evidence found._"

    lines = []
    for idx, summary in enumerate(vision_summaries, 1):
        summary_text = truncate_text(summary["summary"], max_display_length)
        page_num = summary.get("page_index", -1) + 1  # Convert to 1-based
        score = summary.get("score", 0.0)

        lines.append(f"**{idx}. Page {page_num}** (similarity: {score:.3f})")
        lines.append(f"> {summary_text}")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_retrieval_results(results: List[Dict], source_type: str = "unknown") -> bool:
    """
    Validate that retrieval results have the expected structure.

    Args:
        results: List of retrieval results
        source_type: Type of results for error messages ("text" or "vision")

    Returns:
        True if valid, raises ValueError otherwise
    """
    if not isinstance(results, list):
        raise ValueError(f"{source_type} results must be a list, got {type(results)}")

    for idx, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"{source_type} result {idx} must be a dict, got {type(result)}")

    return True


# ============================================================================
# IMAGE UTILITIES
# ============================================================================

def load_and_validate_image(image_path: str) -> Image.Image:
    """
    Load an image from path and validate it's a valid PIL image.

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image object

    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image is invalid
    """
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found: {image_path}")
    except Exception as e:
        raise ValueError(f"Invalid image at {image_path}: {e}")


def get_image_dimensions(image_path: str) -> tuple:
    """
    Get image dimensions without loading the full image into memory.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (width, height)
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        print(f"[WARNING] Could not get dimensions for {image_path}: {e}")
        return (0, 0)
