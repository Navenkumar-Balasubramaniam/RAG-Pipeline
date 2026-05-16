# Local RAG Document Q&A

A fully local, privacy-preserving Retrieval-Augmented Generation (RAG)
pipeline. Upload a PDF, ask questions about it, get grounded answers — with
**nothing ever leaving your machine**. Text and scanned (image-based) PDFs
are both supported.

## Why this project

Most RAG demos send your documents to a cloud LLM API. This one runs the
entire stack locally:

- **PDF parsing** with PyMuPDF, automatic fallback to **OCR** (OpenCV
  preprocessing + PaddleOCR) for scanned documents
- **Semantic chunking** and local **embeddings** (BAAI/bge-small-en-v1.5)
- **Hybrid retrieval**: dense vectors + BM25 keyword search, fused
- **Cross-encoder reranking** for answer precision
- **Local LLM**: Mistral 7B via Ollama — no API keys, no data exfiltration
- **Persistent vector store** with a document library (survives restarts)
- **Streamlit UI**: upload, chat, manage documents

## Architecture

```
PDF ──► pdf_loader ──► chunker ──► embedder ──► vector_store (persisted)
                                                      │
question ──► llm (rewrite) ──► retriever ──► reranker ─┘──► llm (answer)
              (Mistral)        (vector+BM25)  (cross-enc)    (Mistral)
```

Each stage is an independent, tested module. `rag_pipeline.py` orchestrates
them; `streamlit_app.py` is the UI.

## Quick start

See [SETUP.md](SETUP.md) for full prerequisites (conda, Ollama, the WSL
`libgomp1` system dependency).

```bash
# 1. Create the environment
conda env create -f environment.yaml
conda activate rag-pipeline

# 2. Pull the local LLM (one-time, ~4 GB)
ollama pull mistral:7b-instruct

# 3. Run the app
streamlit run streamlit_app.py
```

Then open http://localhost:8501, upload a PDF, and start asking questions.

## Configuration

All tunables live in `config.py` and can be overridden via a `.env` file
(see `.env.example`). Key settings:

| Setting | Default | Purpose |
|---|---|---|
| `RAG_LLM_MODEL` | `mistral:7b-instruct` | Ollama model (swappable) |
| `RAG_TOP_K_RETRIEVAL` | `5` | Chunks retrieved before rerank |
| `RAG_TOP_N_RERANK` | `3` | Chunks kept after rerank |
| `RAG_ENABLE_QUERY_REWRITING` | `true` | LLM query expansion (slower on CPU) |
| `RAG_LOG_LEVEL` | `INFO` | Logging verbosity |

## Testing

```bash
pytest                # fast tests (mocked, no models) — seconds
pytest -m slow        # integration tests (real models + Mistral) — minutes
pytest --cov=src      # with coverage report
```

~40 tests across all modules: unit tests with mocking for logic,
integration tests for the real model pipeline.

## Tech stack

Python 3.11 · LlamaIndex · PyMuPDF · OpenCV · PaddleOCR ·
sentence-transformers · Ollama (Mistral 7B) · Streamlit · pytest

## Project structure

```
config.py            Central Pydantic settings
streamlit_app.py     Web UI entry point
src/
  logger.py          Loguru-based central logging
  pdf_loader.py      Text extraction + OCR fallback
  chunker.py         Semantic chunking
  embedder.py        Local embeddings
  vector_store.py    Persistence + document registry
  retriever.py       Hybrid (vector + BM25) retrieval
  reranker.py        Cross-encoder reranking
  rag_pipeline.py    Orchestrator
tests/               Pytest suite (one file per module)
```

## License

MIT
