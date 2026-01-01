# LLM Document Intelligence Demo

This repository demonstrates a lightweight, explainable RAG (Retrieval-Augmented Generation) pipeline.

## Overview

- Documents are embedded with `sentence-transformers`.
- Embeddings are indexed with `faiss` (or a fallback ANN index).
- Queries retrieve relevant context and optionally generate a grounded answer via OpenAI.

## Quickstart

1. Create a Python virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Place documents in the `data/` directory (supports `.txt`, `.md`, `.pdf`).

3. Ingest documents and build the FAISS index:

```bash
python src/ingest.py --data-dir data --index-path artifacts/faiss.index --meta-path artifacts/meta.json
```

4. Query the index interactively:

```bash
python src/query.py --index-path artifacts/faiss.index --meta-path artifacts/meta.json --openai
```

## Notes

- To enable LLM grounding via OpenAI, set `OPENAI_API_KEY` in your environment before running `src/query.py --openai`.
- The example uses `sentence-transformers` for embeddings and `faiss-cpu` for vector search by default.

## Files of interest

- `src/ingest.py`: ingest documents and build FAISS index
- `src/query.py`: retrieve nearest chunks and optionally call OpenAI for a grounded answer
- `streamlit_app.py`: Streamlit demo application
- `requirements.txt`: Python dependencies

## Streamlit demo

- App file: `streamlit_app.py` (top-level)

Run locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Streamlit Cloud note: `faiss-cpu` can fail to install on some cloud runners. For a successful Cloud deploy you can either:

- Remove `faiss-cpu` from `requirements.txt` (the app will lazy-load faiss and show an error if not available), or
- Deploy using an environment that supports conda and install `faiss-cpu` from `conda-forge`, or
- Build a custom Docker image that pre-installs `faiss-cpu` and use that for deployment.

If you plan to run locally and want FAISS installed, the recommended way is to use `conda`/`mamba`:

```bash
conda install -c conda-forge faiss-cpu
# or
mamba install -c conda-forge faiss-cpu
```

Alternatively (less reliable across platforms) try pip:

```bash
python -m pip install faiss-cpu
```

## Policy-grade refusal pattern

This demo enforces a simple, auditable refusal pattern for situations where the
provided sources do not contain sufficient information. The assistant is
instructed to begin any such response with the exact prefix `REFUSAL:` followed
by a concise reason and optionally a short suggested action (for example,
"provide more documents" or "specify a timeframe"). See `src/prompt_template.py`.

---

RAG Knowledge Assistant pipeline that allows users to query a document corpus using an LLM, with embeddings-based retrieval and transparent evaluation.
