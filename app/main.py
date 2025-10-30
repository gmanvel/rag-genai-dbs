"""
Streamlit main application for local RAG Q&A.
Parallel execution of text and vision retrieval branches.
"""

import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Tuple, List
import time

# Import configuration
from config import (
    validate_config,
    PARALLEL_EXECUTOR_WORKERS,
    ALLOW_PARTIAL_RESULTS,
)

# Import retrieval modules
from retrieval_text import (
    retrieve_text_chunks,
    check_text_collection_exists,
    get_text_collection_info,
)
from retrieval_vision import (
    retrieve_vision_pages,
    check_vision_collection_exists,
    get_vision_collection_info,
)

# Import answer generation
from llm_answer import generate_answer, validate_context

# Import utilities
from utils import format_text_evidence, format_vision_evidence


# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Local RAG Q&A",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# INITIALIZATION AND HEALTH CHECKS
# ============================================================================

@st.cache_resource
def initialize_app():
    """
    Initialize the application on first load.
    Validates config and checks that collections exist.
    """
    print("[APP] Initializing application...")

    # Validate configuration
    try:
        validate_config()
    except Exception as e:
        st.error(f"Configuration validation failed: {e}")
        return False

    # Check collections exist
    text_exists = check_text_collection_exists()
    vision_exists = check_vision_collection_exists()

    if not text_exists or not vision_exists:
        st.warning(
            "⚠️ Some collections are missing. "
            "Please run the preprocessing notebooks first."
        )

    print("[APP] Initialization complete")
    return True


# Initialize on app start
initialization_success = initialize_app()


# ============================================================================
# PARALLEL RETRIEVAL EXECUTION
# ============================================================================

def execute_text_branch(query: str) -> Tuple[List[Dict], str]:
    """
    Execute text retrieval branch.

    Returns:
        Tuple of (results, error_message)
    """
    try:
        results = retrieve_text_chunks(query)
        return results, None
    except Exception as e:
        error_msg = f"Text retrieval failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return [], error_msg


def execute_vision_branch(query: str) -> Tuple[List[Dict], str]:
    """
    Execute vision retrieval branch.

    Returns:
        Tuple of (results, error_message)
    """
    try:
        results = retrieve_vision_pages(query)
        return results, None
    except Exception as e:
        error_msg = f"Vision retrieval failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return [], error_msg


def execute_parallel_retrieval(query: str) -> Dict[str, Any]:
    """
    Execute both retrieval branches in parallel using ThreadPoolExecutor.

    Args:
        query: User's question

    Returns:
        Dictionary with keys:
        - text_chunks: List of text results
        - vision_summaries: List of vision results
        - text_error: Error message from text branch (if any)
        - vision_error: Error message from vision branch (if any)
        - duration: Total execution time in seconds
    """
    start_time = time.time()

    results = {
        "text_chunks": [],
        "vision_summaries": [],
        "text_error": None,
        "vision_error": None,
        "duration": 0.0,
    }

    # Execute both branches in parallel
    with ThreadPoolExecutor(max_workers=PARALLEL_EXECUTOR_WORKERS) as executor:
        # Submit both tasks
        text_future = executor.submit(execute_text_branch, query)
        vision_future = executor.submit(execute_vision_branch, query)

        # Wait for both to complete
        for future in as_completed([text_future, vision_future]):
            if future == text_future:
                text_results, text_error = future.result()
                results["text_chunks"] = text_results
                results["text_error"] = text_error
                print(f"[APP] Text branch completed: {len(text_results)} results")

            elif future == vision_future:
                vision_results, vision_error = future.result()
                results["vision_summaries"] = vision_results
                results["vision_error"] = vision_error
                print(f"[APP] Vision branch completed: {len(vision_results)} results")

    results["duration"] = time.time() - start_time
    print(f"[APP] Parallel retrieval completed in {results['duration']:.2f}s")

    return results


# ============================================================================
# MAIN UI
# ============================================================================

def main():
    """Main Streamlit application."""

    # Title and description
    st.title("🔍 Local RAG Q&A System")
    st.markdown(
        "Ask questions about your documents. "
        "The system searches both **text content** and **visual pages** in parallel, "
        "then generates a comprehensive answer using local AI models."
    )
    st.markdown("---")

    # Sidebar with info and settings
    with st.sidebar:
        st.header("ℹ️ System Info")

        if st.button("Check Collections"):
            text_info = get_text_collection_info()
            vision_info = get_vision_collection_info()

            st.subheader("Text Collection")
            if "error" in text_info:
                st.error(f"Error: {text_info['error']}")
            else:
                st.write(f"**Name:** {text_info['name']}")
                st.write(f"**Vectors:** {text_info['vectors_count']}")
                st.write(f"**Status:** {text_info['status']}")

            st.subheader("Vision Collection")
            if "error" in vision_info:
                st.error(f"Error: {vision_info['error']}")
            else:
                st.write(f"**Name:** {vision_info['name']}")
                st.write(f"**Vectors:** {vision_info['vectors_count']}")
                st.write(f"**Status:** {vision_info['status']}")

        st.markdown("---")
        st.markdown(
            "**100% Local Execution**\n\n"
            "- No cloud API calls\n"
            "- No API keys required\n"
            "- All models run locally\n"
            "- Data never leaves your machine"
        )

    # Main query input
    user_question = st.text_input(
        "Ask a question:",
        placeholder="e.g., What are the main topics covered in the documents?",
        help="Enter your question and press Enter",
    )

    # Process query when submitted
    if user_question:
        st.markdown("---")

        # Show loading state
        with st.spinner("🔄 Processing your question..."):

            # Execute parallel retrieval
            retrieval_results = execute_parallel_retrieval(user_question)

            text_chunks = retrieval_results["text_chunks"]
            vision_summaries = retrieval_results["vision_summaries"]
            duration = retrieval_results["duration"]

            # Show any errors
            if retrieval_results["text_error"]:
                st.warning(f"⚠️ {retrieval_results['text_error']}")

            if retrieval_results["vision_error"]:
                st.warning(f"⚠️ {retrieval_results['vision_error']}")

            # Validate context
            validation = validate_context(text_chunks, vision_summaries)

            if not validation["has_any"]:
                st.error(
                    "❌ No relevant context found. "
                    "Please try rephrasing your question or check that data is loaded."
                )
                return

            # Show validation warnings
            for warning in validation["warnings"]:
                if ALLOW_PARTIAL_RESULTS:
                    st.info(f"ℹ️ {warning}")
                else:
                    st.error(f"❌ {warning}")
                    return

            # Generate answer
            try:
                with st.spinner("🤖 Generating answer..."):
                    answer = generate_answer(
                        user_question=user_question,
                        text_chunks=text_chunks,
                        vision_summaries=vision_summaries,
                    )

                # Display answer
                st.success("✅ Answer generated successfully!")
                st.markdown(f"**⏱️ Total time:** {duration:.2f} seconds")
                st.markdown("---")

                # Answer section
                st.subheader("💡 Answer")
                st.markdown(answer)

                st.markdown("---")

                # Evidence section
                st.subheader("📚 Supporting Evidence")

                # Create two columns for text and vision evidence
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**📄 Text Sources**")
                    if text_chunks:
                        text_evidence = format_text_evidence(text_chunks)
                        st.markdown(text_evidence)
                    else:
                        st.markdown("_No text sources used_")

                with col2:
                    st.markdown("**🖼️ Visual Sources**")
                    if vision_summaries:
                        vision_evidence = format_vision_evidence(vision_summaries)
                        st.markdown(vision_evidence)
                    else:
                        st.markdown("_No visual sources used_")

                # Optional: Show detailed evidence in expander
                with st.expander("🔍 View Detailed Evidence"):
                    st.markdown("**Text Chunks:**")
                    st.json(text_chunks)

                    st.markdown("**Vision Summaries:**")
                    st.json(vision_summaries)

            except Exception as e:
                st.error(f"❌ Failed to generate answer: {str(e)}")
                print(f"[ERROR] Answer generation failed: {e}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if initialization_success:
        main()
    else:
        st.error(
            "❌ Application initialization failed. "
            "Please check the logs and ensure all dependencies are running."
        )
