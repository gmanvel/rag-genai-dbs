# Local RAG Q&A System

A **100% local** Retrieval-Augmented Generation (RAG) system that answers questions by searching both text content and visual document pages. No cloud services, no API keys, everything runs on your machine.

## 🎯 Overview

This application implements a dual-branch RAG pipeline:

- **Text Branch**: Retrieves relevant text chunks using semantic embeddings
- **Vision Branch**: Retrieves relevant page images and generates visual summaries
- **Parallel Execution**: Both branches run simultaneously for maximum speed
- **Local LLM**: Mistral generates the final answer by merging both contexts

### Key Features

- 🔒 **100% Local** - No internet required, no data leaves your machine
- 🚀 **Parallel Retrieval** - Text and vision branches execute simultaneously
- 🧠 **Multi-Modal** - Combines text and visual information
- 📊 **Evidence-Based** - Shows sources for every answer
- 🐳 **Docker-Ready** - Simple deployment with Docker Compose

## 🏗️ Architecture

```
User Question
     │
     ├──────────────────┬──────────────────┐
     │                  │                  │
Text Branch      Vision Branch             │
     │                  │                  │
  Embed with      Embed with               │
nomic-embed-text   ColPali                 │
     │                  │                  │
  Search            Search                 │
 Qdrant            Qdrant                  │
(text_chunks)    (pdf_pages)               │
     │                  │                  │
  Return            Return                 │
Text Chunks     Page Images                │
     │                  │                  │
     │           Qwen2-VL                  │
     │          Summarizes                 │
     │                  │                  │
     └──────────────────┴──────────────────┘
                  │
            Merge Context
                  │
           Mistral LLM
                  │
         Final Answer + Evidence
```

## 📋 Prerequisites

Before running this application, you **MUST** complete the preprocessing steps:

### 1. Text Data Preprocessing

Run the `local_rag_playground.ipynb` notebook to:
- Load your text documents (e.g., transcripts, articles)
- Split text using semantic chunking (SemanticSplitterNodeParser)
- Embed chunks with `nomic-embed-text` via Ollama
- Store embeddings in Qdrant collection `video_chunks`

**Key steps from notebook**:
```python
# 1. Semantic splitting
splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=70,
    embed_model=OllamaEmbedding(model_name="nomic-embed-text"),
)

# 2. Embed and store in Qdrant
qdrant_client.upsert(
    collection_name="video_chunks",
    points=[...],  # Chunks with 768-dim vectors
)
```

### 2. Vision Data Preprocessing

Run the `single_pdf_colpali_qdrant.ipynb` notebook to:
- Convert PDF pages to PNG images
- Embed pages using ColPali (mean-pooled to 128-dim vectors)
- Store embeddings in Qdrant collection `pdf_pages`
- Save page images to `./data/pdf_pages/`

**Key steps from notebook**:
```python
# 1. Convert PDF to images
images = convert_from_path(pdf_path)
# Save to ./data/pdf_pages/page_0001.png, etc.

# 2. Embed with ColPali and mean-pool
page_embeddings = embed_pages_mean_pooled(pages, model, processor)

# 3. Store in Qdrant
qdrant_client.upsert(
    collection_name="pdf_pages",
    points=[...],  # Pages with 128-dim vectors
)
```

### 3. Install Ollama and Pull Models

```bash
# Install Ollama (if not using Docker)
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull nomic-embed-text
ollama pull mistral
```

## 🚀 Quick Start with Docker

### Step 1: Clone and Prepare Data

```bash
# Ensure you have:
# 1. Qdrant collections populated (from notebooks)
# 2. Page images in ./data/pdf_pages/
# 3. Directory structure:
#    rag-genai-dbs-codex/
#    ├── app/
#    ├── data/
#    │   └── pdf_pages/
#    ├── docker-compose.yml
#    ├── Dockerfile
#    └── requirements.txt
```

### Step 2: Start Services

```bash
# Start all services (Qdrant, Ollama, App)
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f app
```

### Step 3: Pull Ollama Models

```bash
# Connect to Ollama container
docker exec -it rag-ollama bash

# Inside container, pull models
ollama pull nomic-embed-text
ollama pull mistral

# Exit container
exit
```

### Step 4: Access the UI

Open your browser to:
```
http://localhost:8501
```

You should see the Streamlit Q&A interface!

## 🎮 How to Use

### Asking Questions

1. **Enter your question** in the text input box
2. **Press Enter** to submit
3. **Wait** for parallel retrieval (text + vision branches)
4. **Review** the answer and supporting evidence

### Example Questions

```
What are the main topics covered?
Explain the key findings from page 3.
What does the chart on page 5 show?
Summarize the conclusion.
```

### Understanding Results

The UI displays:

1. **Answer Section**: Final answer generated by Mistral
2. **Text Sources**: Chunks retrieved from text documents
3. **Visual Sources**: Page summaries from vision analysis
4. **Metadata**: Similarity scores, source IDs, page numbers

## 🔧 Configuration

Edit `app/config.py` to customize:

```python
# Collection names
TEXT_COLLECTION_NAME = "video_chunks"
VISION_COLLECTION_NAME = "pdf_pages"

# Retrieval limits
TEXT_RETRIEVAL_LIMIT = 3
VISION_RETRIEVAL_LIMIT = 3

# Model names
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "mistral"
QWEN_VLM_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
```

## 🐳 Docker Services

### App Service
- **Image**: Custom-built from Dockerfile
- **Port**: 8501 (Streamlit UI)
- **Memory**: 4-8 GB
- **Volumes**: `./data` (read-only)

### Qdrant Service
- **Image**: `qdrant/qdrant:v1.7.4`
- **Ports**: 6333 (HTTP), 6334 (gRPC)
- **Volume**: `qdrant_data` (persists collections)

### Ollama Service
- **Image**: `ollama/ollama:latest`
- **Port**: 11434
- **Volume**: `ollama_models` (persists downloaded models)
- **Note**: Uncomment GPU section for NVIDIA GPU support

## 🔍 Data Flow

When you ask a question, here's what happens:

### 1. Parallel Retrieval (Both Branches)

**Text Branch**:
```
Query → Embed (nomic-embed-text) → Search Qdrant (video_chunks) → Return top-3 chunks
```

**Vision Branch**:
```
Query → Embed (ColPali) → Search Qdrant (pdf_pages) → Return top-3 pages
        ↓
   For each page: Load image → Qwen2-VL → Generate summary
```

**Concurrency**: Both branches run in parallel using `ThreadPoolExecutor`

### 2. Context Merging

```python
merged_context = {
    "text_chunks": [...],      # Text content with source_doc_id
    "vision_summaries": [...],  # Page summaries with page_index
}
```

### 3. Answer Generation

```python
prompt = build_rag_prompt(
    user_question=question,
    text_chunks=text_chunks,
    vision_summaries=vision_summaries,
)

answer = mistral_llm.generate(prompt)
```

### 4. Response Display

```
┌─────────────────────────────────────┐
│ Answer                              │
│ (Generated by Mistral)              │
├─────────────────────────────────────┤
│ Supporting Evidence                 │
│                                     │
│ Text Sources:                       │
│ - Chunk 1 (source_doc_id, score)   │
│ - Chunk 2 (source_doc_id, score)   │
│                                     │
│ Visual Sources:                     │
│ - Page 1 (summary, score)          │
│ - Page 2 (summary, score)          │
└─────────────────────────────────────┘
```

## 🛠️ Development

### Running Without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure services are running
# - Qdrant at http://localhost:6333
# - Ollama at http://localhost:11434

# Set environment variables
export QDRANT_URL=http://localhost:6333
export OLLAMA_URL=http://localhost:11434

# Run Streamlit
streamlit run app/main.py
```

### Testing Components

```python
# Test text retrieval
from app.retrieval_text import test_text_retrieval
test_text_retrieval()

# Test vision retrieval
from app.retrieval_vision import test_vision_retrieval
test_vision_retrieval()

# Test answer generation
from app.llm_answer import test_answer_generation
test_answer_generation()
```

## 📦 Extending the System

### Swapping the VLM

To use a different vision-language model:

1. Edit `app/config.py`:
```python
QWEN_VLM_MODEL_NAME = "your-model-name"
```

2. Update `app/models_local.py` `get_qwen_vlm_models()` function

### Using Different Collections

To query a different Qdrant collection:

1. Update `app/config.py`:
```python
TEXT_COLLECTION_NAME = "my_custom_collection"
```

2. Ensure the collection exists and has compatible vector dimensions

### Changing the LLM

To use a different model for answer generation:

1. Edit `app/config.py`:
```python
OLLAMA_LLM_MODEL = "llama2"  # or any other Ollama model
```

2. Pull the model in Ollama:
```bash
ollama pull llama2
```

## ⚠️ Troubleshooting

### "Collection not found" Error

**Cause**: Preprocessing notebooks not run or collections not created

**Solution**:
1. Run `local_rag_playground.ipynb` to create `video_chunks`
2. Run `single_pdf_colpali_qdrant.ipynb` to create `pdf_pages`
3. Verify collections exist:
```bash
curl http://localhost:6333/collections
```

### "Model not found" Error in Ollama

**Cause**: Models not pulled

**Solution**:
```bash
docker exec -it rag-ollama ollama pull nomic-embed-text
docker exec -it rag-ollama ollama pull mistral
```

### "Image not found" Error

**Cause**: Page images missing or path incorrect

**Solution**:
1. Ensure images exist in `./data/pdf_pages/`
2. Check image paths in Qdrant payloads
3. Verify volume mount in `docker-compose.yml`

### Out of Memory Errors

**Cause**: Vision models (ColPali, Qwen2-VL) require significant memory

**Solution**:
1. Increase Docker memory limits in `docker-compose.yml`
2. Use smaller batch sizes in `app/config.py`:
```python
VLM_BATCH_SIZE = 1  # Process 1 page at a time
```
3. Reduce retrieval limits:
```python
TEXT_RETRIEVAL_LIMIT = 2
VISION_RETRIEVAL_LIMIT = 2
```

### Slow Response Times

**Cause**: Vision models run on CPU, VLM inference is slow

**Solution**:
1. Use GPU if available (uncomment GPU section in `docker-compose.yml`)
2. Reduce number of pages retrieved
3. Cache VLM results for common queries

## 🧹 Cleanup

The application automatically cleans up resources on shutdown:

```python
# Registered in models_local.py
atexit.register(cleanup_resources)

# Cleanup includes:
# - Close Qdrant client
# - Clear model references
# - Empty GPU cache (CUDA/MPS)
```

To manually stop and remove containers:

```bash
# Stop services
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v
```

## 📊 System Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 16 GB
- **Storage**: 20 GB (for models and data)
- **OS**: Linux, macOS, or Windows with WSL2

### Recommended Requirements
- **CPU**: 8+ cores
- **RAM**: 32 GB
- **GPU**: NVIDIA GPU with 8+ GB VRAM (for faster inference)
- **Storage**: 50 GB SSD

## 🔒 Security & Privacy

- ✅ **No cloud API calls** - Everything runs locally
- ✅ **No API keys required** - No external authentication
- ✅ **Data stays local** - Documents never leave your machine
- ✅ **No telemetry** - No usage tracking or analytics
- ✅ **Offline capable** - Works without internet connection

## 📄 License

This project uses the exact RAG approaches from the preprocessing notebooks in this repository.

## 🙏 Acknowledgments

- **Ollama**: Local LLM serving
- **Qdrant**: Vector database
- **LlamaIndex**: Semantic chunking and embeddings
- **ColPali**: Vision embeddings
- **Qwen**: Vision-language model
- **Mistral**: Answer generation LLM

---

**Remember**: This system is 100% local. No external API keys. No remote calls. Everything runs on your hardware.
