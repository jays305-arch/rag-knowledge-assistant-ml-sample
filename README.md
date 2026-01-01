# LLM Document Intelligence Demo

This repository demonstrates a lightweight, explainable RAG (Retrieval-Augmented Generation) pipeline.

Overview

- Documents are embedded with `sentence-transformers`.
- Embeddings are indexed with `faiss`.
- Queries retrieve relevant context and optionally generate a grounded answer via OpenAI.

Quickstart

1. Create a Python virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Place documents in `data/` (supports `.txt`, `.md`, `.pdf`).

3. Ingest documents and build the FAISS index:

```bash
python src/ingest.py --data-dir data --index-path artifacts/faiss.index --meta-path artifacts/meta.json
```

4. Query the index interactively:

```bash
python src/query.py --index-path artifacts/faiss.index --meta-path artifacts/meta.json --openai
```

Notes

- To enable LLM grounding via OpenAI, set `OPENAI_API_KEY` in your environment before running `src/query.py --openai`.
- The example uses `sentence-transformers` for embeddings and `faiss-cpu` for vector search.

Files added

- `src/ingest.py`: ingest documents and build FAISS index
- `src/query.py`: retrieve nearest chunks and optionally call OpenAI for a grounded answer
- `requirements.txt`: Python dependencies
- `notebooks/demo.ipynb`: short demo instructions

If you'd like, I can also:
- Add a small sample dataset into `data/` for a runnable demo
- Add tests or a simple CLI wrapper
# rag-knowledge-assistant-ml-sample
RAG Knowledge Assistant pipeline that allows users to query a document corpus using an LLM, with embeddings-based retrieval and transparent evaluation
