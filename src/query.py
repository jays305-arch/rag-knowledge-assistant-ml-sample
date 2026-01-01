import os
import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except Exception:
    faiss = None

try:
    import openai
    from openai import OpenAI
except Exception:
    openai = None
    OpenAI = None

from src.prompt_template import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    build_user_prompt,
    format_refusal,
    REFUSAL_PREFIX,
)


def generate_grounded_response(openai_client: OpenAI, retrieved_text: str, question: str):
    """Generate a grounded response using an OpenAI-compatible client.

    This helper mirrors a modern OpenAI client call shape (`client.chat.completions.create`).
    """
    prompt = USER_PROMPT_TEMPLATE.format(
        retrieved_context=retrieved_text,
        user_question=question,
    )

    response = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,  # low temp for factual grounding
    )
    return response.choices[0].message.content


def load_index(index_path: Path):
    if faiss is None:
        raise RuntimeError("faiss is required (install faiss-cpu)")
    return faiss.read_index(str(index_path))


def main(index_path, meta_path, model_name, top_k, openai_completion):
    index_path = Path(index_path)
    meta_path = Path(meta_path)
    if not index_path.exists() or not meta_path.exists():
        print("Index or metadata not found. Run ingest first.")
        return

    model = SentenceTransformer(model_name)
    index = load_index(index_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadatas = json.load(f)

    query = input("Enter your question: ")
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")

    D, I = index.search(q_emb, top_k)
    I = I[0]

    results = []
    for idx in I:
        if idx < 0 or idx >= len(metadatas):
            continue
        results.append(metadatas[idx])

    print("Retrieved sources:")
    for r in results:
        print('-', r.get('source'), 'chunk', r.get('chunk_index'))

    # Optionally generate a grounded answer using OpenAI if available
    if openai_completion and openai is not None and os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Use chunk text saved in metadata (fallback to reading file if missing)
        def _read_file_fallback(path: Path) -> str:
            try:
                return path.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                return ""

        context_excerpts = []
        for r in results:
            src = r.get('source', '<unknown>')
            text = r.get('text')
            if not text:
                # fallback: read entire file (best-effort)
                text = _read_file_fallback(Path(src))
            context_excerpts.append(f"Source: {src}\n{text}")

        retrieved_context = "\n---\n".join(context_excerpts)
        user_prompt = build_user_prompt(retrieved_context=retrieved_context, user_question=query)

        # Auto-refusal for sensitive domains (legal / medical / policy) when not present in context
        sensitive_keywords = [
            "legal", "law", "legal advice", "attorney", "court", "litigation",
            "medical", "medicine", "doctor", "diagnosis", "treatment", "clinic",
            "policy", "regulation", "regulatory", "compliance", "policy guidance",
        ]

        def _is_sensitive_question(q: str) -> bool:
            ql = q.lower()
            return any(kw in ql for kw in sensitive_keywords)

        def _context_contains_evidence(ctx_texts) -> bool:
            joined = "\n".join(ctx_texts).lower()
            return any(kw in joined for kw in sensitive_keywords)

        if _is_sensitive_question(query) and not _context_contains_evidence(context_excerpts):
            reason = (
                "Question requests legal/medical/policy advice but the provided sources do not "
                "explicitly contain such information."
            )
            suggestion = "Provide authoritative documents or consult a qualified professional."
            refusal_msg = format_refusal(reason, suggestion)
            print("\nGrounded answer:")
            print(refusal_msg)
            return

        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=400,
                temperature=0.0,
            )
            answer = resp['choices'][0]['message']['content']
            print("\nGrounded answer:")
            print(answer)
        except Exception as e:
            print("OpenAI request failed:", e)
    else:
        print("\nTo generate a grounded LLM answer, set OPENAI_API_KEY and run with --openai")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", default="artifacts/faiss.index")
    parser.add_argument("--meta-path", default="artifacts/meta.json")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--openai", dest="openai_completion", action="store_true")
    args = parser.parse_args()
    main(args.index_path, args.meta_path, args.model, args.top_k, args.openai_completion)
