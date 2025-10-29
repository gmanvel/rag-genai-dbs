"""Streamlit entry point for the local RAG application."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import streamlit as st

from app.llm_answer import generate_answer
from app.retrieval_text import run_text_branch
from app.retrieval_vision import run_vision_branch
from app.utils import format_text_evidence, format_vision_evidence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Local RAG Q&A", layout="wide")
st.title("Local RAG Q&A Playground")
st.write(
    "Ask a question to retrieve evidence from semantic text chunks and page summaries."
)


def _execute_branches(question: str) -> tuple[List[dict], List[dict]]:
    """Run retrieval branches in parallel, handling errors gracefully."""
    text_results: List[dict] = []
    vision_results: List[dict] = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_branch = {
            executor.submit(run_text_branch, question): "text",
            executor.submit(run_vision_branch, question): "vision",
        }
        for future in as_completed(future_to_branch):
            branch_name = future_to_branch[future]
            try:
                result = future.result()
                if branch_name == "text":
                    text_results = result or []
                else:
                    vision_results = result or []
            except Exception as exc:
                logger.exception("%s branch failed", branch_name, exc_info=exc)
                st.warning(f"{branch_name.title()} branch failed: {exc}")
    return text_results, vision_results


question = st.text_input("Question")
submit = st.button("Submit")

if submit and question:
    with st.spinner("Retrieving evidence and generating answer..."):
        text_evidence, vision_evidence = _execute_branches(question)
        try:
            answer = generate_answer(question, text_evidence, vision_evidence)
        except Exception as exc:
            logger.exception("Answer generation failed", exc_info=exc)
            st.error(f"Failed to generate answer: {exc}")
            answer = ""

    if answer:
        st.subheader("Answer")
        st.write(answer)

    st.subheader("Evidence: Text Chunks")
    formatted_text = format_text_evidence(text_evidence)
    if formatted_text:
        for entry in formatted_text:
            st.markdown(f"- {entry.replace(chr(10), '<br>')}", unsafe_allow_html=True)
    else:
        st.write("No text evidence available.")

    st.subheader("Evidence: Pages / Images")
    formatted_pages = format_vision_evidence(vision_evidence)
    if formatted_pages:
        for entry in formatted_pages:
            st.markdown(f"- {entry}")
    else:
        st.write("No page or image evidence available.")
else:
    st.info("Enter a question and click Submit to run the local RAG pipeline.")
