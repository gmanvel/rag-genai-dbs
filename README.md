# Local Streamlit RAG Q&A Stack

## Overview
This project provides a local-only retrieval augmented generation (RAG) playground built with Streamlit. A user enters a question, the app gathers relevant semantic text chunks from Qdrant and page/image summaries from a (stubbed) Qwen vision-language model, and then a local Mistral model (served by Ollama) produces the final answer. The UI displays both the answer and the supporting evidence.

## Prerequisites: Prepare data first
Before running the app, you **must** ingest your documents using a separate Jupyter notebook (not included here). That notebook should:

1. Chunk your source documents semantically.
2. Generate embeddings for each chunk using the `nomic-embed-text` model served by Ollama.
3. Write the embeddings and payloads into the Qdrant collection named `rag_chunks`.
4. Produce or export page images/screenshots used by the vision branch and place them under `data/pages`.

## Run the stack
1. Build and start all services:
   ```bash
   docker compose up --build
   ```
2. Open your browser to [http://localhost:8501](http://localhost:8501) to access the Streamlit interface.

The Compose stack launches:
- `app`: Streamlit UI and retrieval logic.
- `qdrant`: Vector database storing semantic chunks.
- `ollama`: Hosts both the embedding model (`nomic-embed-text`) and the answer model (`mistral`).
- `vlm`: Placeholder container where a local Qwen VLM pipeline can be deployed.

## How a query is answered
1. When a user submits a question, the app starts a `ThreadPoolExecutor` with two workers.
2. **Text branch** embeds the question with Ollama, searches the `rag_chunks` collection in Qdrant, and returns the top matching chunks.
3. **Vision branch** scores candidate pages/images from an in-memory index, and for the most relevant ones calls a deterministic stub that imitates a Qwen VLM summary.
4. Once both branches finish, their evidence is merged into a single prompt that is sent to the local Mistral model via Ollama.
5. The generated answer is displayed together with evidence sections:
   - **Evidence: Text Chunks** lists the retrieved excerpts and their `source_doc_id` values.
   - **Evidence: Pages / Images** lists page/image references and summaries.

## Local only
All model interactions stay on your machine:
- Ollama serves both embeddings and the Mistral text generation model.
- Qdrant runs locally with no authentication.
- The Qwen vision-language model is expected to run locally; the current implementation ships with a deterministic stub so the app runs without external dependencies.
- No remote or cloud APIs are invoked by the application.

## Cleanup
The application registers `atexit` handlers to close the Qdrant client, ensuring connections are cleaned up when the Streamlit app shuts down.
