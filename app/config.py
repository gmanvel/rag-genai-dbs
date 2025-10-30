"""
Centralized configuration for the local RAG Q&A application.
All settings for Qdrant, Ollama, models, and runtime behavior.
"""

import os
from pathlib import Path

# ============================================================================
# QDRANT CONFIGURATION
# ============================================================================

# Qdrant URL - defaults to Docker service name, fallback to localhost
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

# Collection names (must match what was created in preprocessing notebooks)
TEXT_COLLECTION_NAME = "video_chunks"  # Text chunks with semantic splitting
VISION_COLLECTION_NAME = "pdf_pages"   # PDF pages as images with ColPali embeddings

# Vector dimensions (must match embedding models)
TEXT_VECTOR_DIM = 768   # nomic-embed-text dimension
VISION_VECTOR_DIM = 128  # ColPali mean-pooled dimension

# ============================================================================
# OLLAMA CONFIGURATION
# ============================================================================

# Ollama URL - for embeddings and LLM inference
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

# Model names
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"  # For text embeddings
OLLAMA_LLM_MODEL = "mistral"                 # For final answer generation

# ============================================================================
# VISION MODEL CONFIGURATION
# ============================================================================

# ColPali model for vision retrieval embeddings
COLPALI_MODEL_NAME = "vidore/colpali-v1.2"

# Qwen Vision-Language Model for page summarization
QWEN_VLM_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# Vision model inference settings
VLM_MAX_NEW_TOKENS = 500  # Maximum tokens for VLM summaries
VLM_BATCH_SIZE = 2        # Process 2 pages at a time (memory management)

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

# Number of results to retrieve from each branch
TEXT_RETRIEVAL_LIMIT = 3    # Top-3 text chunks
VISION_RETRIEVAL_LIMIT = 3  # Top-3 pages

# Score thresholds (optional filtering)
MIN_TEXT_SCORE = 0.0   # Minimum cosine similarity for text results
MIN_VISION_SCORE = 0.0  # Minimum cosine similarity for vision results

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

import torch

def get_device():
    """
    Determine the best available device for PyTorch models.
    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# Preferred dtype based on device
def get_preferred_dtype():
    """Get the best dtype for the current device."""
    if DEVICE.type in {"cuda", "mps"}:
        return torch.bfloat16
    else:
        return torch.float32

PREFERRED_DTYPE = get_preferred_dtype()

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Base directories (adjust based on your setup)
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
PDF_PAGES_DIR = DATA_DIR / "pdf_pages"  # Where page images are stored

# ============================================================================
# PROMPT CONFIGURATION
# ============================================================================

# Maximum length for context sections in the final prompt
MAX_TEXT_CONTEXT_LENGTH = 2000   # Characters per text chunk
MAX_VISION_CONTEXT_LENGTH = 1500  # Characters per vision summary
MAX_TOTAL_CONTEXT_LENGTH = 8000   # Total context characters

# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================

# Streamlit settings
STREAMLIT_PORT = 8501
STREAMLIT_ADDRESS = "0.0.0.0"

# Concurrency settings
PARALLEL_EXECUTOR_WORKERS = 2  # One for text branch, one for vision branch

# Error handling
ALLOW_PARTIAL_RESULTS = True  # Show results even if one branch fails

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """
    Validate configuration settings at startup.
    Raises ValueError if critical settings are invalid.
    """
    if TEXT_VECTOR_DIM != 768:
        raise ValueError(f"TEXT_VECTOR_DIM must be 768 for nomic-embed-text, got {TEXT_VECTOR_DIM}")

    if VISION_VECTOR_DIM != 128:
        raise ValueError(f"VISION_VECTOR_DIM must be 128 for ColPali mean-pooled, got {VISION_VECTOR_DIM}")

    if TEXT_RETRIEVAL_LIMIT < 1 or VISION_RETRIEVAL_LIMIT < 1:
        raise ValueError("Retrieval limits must be at least 1")

    print(f"[CONFIG] Device: {DEVICE}")
    print(f"[CONFIG] Dtype: {PREFERRED_DTYPE}")
    print(f"[CONFIG] Qdrant URL: {QDRANT_URL}")
    print(f"[CONFIG] Ollama URL: {OLLAMA_URL}")
    print(f"[CONFIG] Text Collection: {TEXT_COLLECTION_NAME}")
    print(f"[CONFIG] Vision Collection: {VISION_COLLECTION_NAME}")
