"""
Local model initialization and inference wrappers.
All models are loaded once at module import and cached for reuse.
Handles: Ollama embeddings, ColPali vision embeddings, Qwen VLM, Mistral LLM.
"""

import atexit
import torch
from typing import Optional
from PIL import Image

# Import configuration
from config import (
    OLLAMA_URL,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_LLM_MODEL,
    COLPALI_MODEL_NAME,
    QWEN_VLM_MODEL_NAME,
    DEVICE,
    PREFERRED_DTYPE,
)

# ============================================================================
# GLOBAL MODEL INSTANCES (loaded once, reused across requests)
# ============================================================================

_text_embedding_model = None
_colpali_model = None
_colpali_processor = None
_qwen_vlm_model = None
_qwen_vlm_processor = None
_qdrant_client = None

# ============================================================================
# TEXT EMBEDDING MODEL (Ollama - nomic-embed-text)
# ============================================================================

def get_text_embedding_model():
    """
    Returns the OllamaEmbedding model for text embeddings.
    Uses nomic-embed-text model via local Ollama.
    Loaded once and cached.
    """
    global _text_embedding_model

    if _text_embedding_model is None:
        print(f"[MODELS] Loading text embedding model: {OLLAMA_EMBEDDING_MODEL}")
        from llama_index.embeddings.ollama import OllamaEmbedding

        _text_embedding_model = OllamaEmbedding(
            model_name=OLLAMA_EMBEDDING_MODEL,
            base_url=OLLAMA_URL,
        )
        print(f"[MODELS] Text embedding model loaded successfully")

    return _text_embedding_model


# ============================================================================
# COLPALI VISION EMBEDDING MODEL
# ============================================================================

def get_colpali_models():
    """
    Returns (model, processor) tuple for ColPali vision embeddings.
    Used for retrieving relevant pages from Qdrant.
    Loaded once and cached.
    """
    global _colpali_model, _colpali_processor

    if _colpali_model is None or _colpali_processor is None:
        print(f"[MODELS] Loading ColPali model: {COLPALI_MODEL_NAME}")
        from colpali_engine.models import ColPali, ColPaliProcessor

        _colpali_processor = ColPaliProcessor.from_pretrained(COLPALI_MODEL_NAME)

        _colpali_model = ColPali.from_pretrained(
            COLPALI_MODEL_NAME,
            torch_dtype=PREFERRED_DTYPE,
        ).to(DEVICE)

        _colpali_model.eval()  # Set to evaluation mode

        print(f"[MODELS] ColPali model loaded successfully on {DEVICE}")

    return _colpali_model, _colpali_processor


# ============================================================================
# QWEN VISION-LANGUAGE MODEL
# ============================================================================

def get_qwen_vlm_models():
    """
    Returns (model, processor) tuple for Qwen2-VL vision-language model.
    Used for generating summaries from retrieved page images.
    Loaded once and cached.
    """
    global _qwen_vlm_model, _qwen_vlm_processor

    if _qwen_vlm_model is None or _qwen_vlm_processor is None:
        print(f"[MODELS] Loading Qwen VLM model: {QWEN_VLM_MODEL_NAME}")
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        _qwen_vlm_processor = AutoProcessor.from_pretrained(QWEN_VLM_MODEL_NAME)

        _qwen_vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_VLM_MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        ).to(DEVICE)

        _qwen_vlm_model.eval()  # Set to evaluation mode

        print(f"[MODELS] Qwen VLM model loaded successfully on {DEVICE}")

    return _qwen_vlm_model, _qwen_vlm_processor


# ============================================================================
# MISTRAL LLM (via Ollama)
# ============================================================================

def get_mistral_ollama_url():
    """
    Returns the Ollama URL for calling Mistral LLM.
    No model loading needed - Ollama handles this server-side.
    """
    return OLLAMA_URL


def call_mistral_llm(prompt: str, max_tokens: int = 1000) -> str:
    """
    Calls the local Mistral LLM via Ollama to generate a response.

    Args:
        prompt: The input prompt to send to Mistral
        max_tokens: Maximum number of tokens to generate

    Returns:
        Generated text response from Mistral
    """
    import requests

    url = f"{OLLAMA_URL}/api/generate"

    payload = {
        "model": OLLAMA_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        return result.get("response", "")

    except Exception as e:
        print(f"[ERROR] Failed to call Mistral LLM: {e}")
        raise


# ============================================================================
# QDRANT CLIENT
# ============================================================================

def get_qdrant_client():
    """
    Returns a persistent Qdrant client instance.
    Loaded once and cached.
    """
    global _qdrant_client

    if _qdrant_client is None:
        print("[MODELS] Initializing Qdrant client")
        from qdrant_client import QdrantClient
        from config import QDRANT_URL

        _qdrant_client = QdrantClient(url=QDRANT_URL)
        print(f"[MODELS] Qdrant client connected to {QDRANT_URL}")

    return _qdrant_client


# ============================================================================
# CLEANUP ON EXIT
# ============================================================================

def cleanup_resources():
    """
    Cleanup function to close connections and free memory on exit.
    Registered with atexit to run automatically when the process terminates.
    """
    global _qdrant_client, _text_embedding_model
    global _colpali_model, _colpali_processor
    global _qwen_vlm_model, _qwen_vlm_processor

    print("[MODELS] Cleaning up resources...")

    # Close Qdrant client
    if _qdrant_client is not None:
        try:
            _qdrant_client.close()
            print("[MODELS] Qdrant client closed")
        except Exception as e:
            print(f"[MODELS] Error closing Qdrant client: {e}")

    # Clear model references
    _text_embedding_model = None
    _colpali_model = None
    _colpali_processor = None
    _qwen_vlm_model = None
    _qwen_vlm_processor = None

    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[MODELS] CUDA cache cleared")
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("[MODELS] MPS cache cleared")

    print("[MODELS] Cleanup complete")


# Register cleanup function
atexit.register(cleanup_resources)


# ============================================================================
# HEALTH CHECK
# ============================================================================

def check_models_available() -> dict:
    """
    Check if all required models and services are available.
    Useful for debugging and health checks.

    Returns:
        Dictionary with availability status for each component
    """
    status = {
        "text_embedding": False,
        "colpali": False,
        "qwen_vlm": False,
        "mistral_llm": False,
        "qdrant": False,
    }

    # Check text embedding
    try:
        model = get_text_embedding_model()
        status["text_embedding"] = model is not None
    except Exception as e:
        print(f"[HEALTH] Text embedding unavailable: {e}")

    # Check ColPali
    try:
        model, proc = get_colpali_models()
        status["colpali"] = (model is not None and proc is not None)
    except Exception as e:
        print(f"[HEALTH] ColPali unavailable: {e}")

    # Check Qwen VLM
    try:
        model, proc = get_qwen_vlm_models()
        status["qwen_vlm"] = (model is not None and proc is not None)
    except Exception as e:
        print(f"[HEALTH] Qwen VLM unavailable: {e}")

    # Check Mistral via Ollama
    try:
        import requests
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        status["mistral_llm"] = response.status_code == 200
    except Exception as e:
        print(f"[HEALTH] Mistral/Ollama unavailable: {e}")

    # Check Qdrant
    try:
        client = get_qdrant_client()
        status["qdrant"] = client is not None
    except Exception as e:
        print(f"[HEALTH] Qdrant unavailable: {e}")

    return status
