import os
from pathlib import Path

import streamlit as st

try:
    import faiss
except Exception:
    faiss = None

# Delay importing heavy/optional dependencies (sentence_transformers, faiss, src.query)
# until runtime to avoid import errors during app startup on Streamlit Cloud.


st.set_page_config(page_title="RAG Demo", layout="wide")

st.title("Retrieval-Augmented Generation — Demo")

col1, col2 = st.columns([2, 1])

with col2:
    index_path = st.text_input("FAISS index path", value="artifacts/faiss.index")
    meta_path = st.text_input("Metadata JSON path", value="artifacts/meta.json")
    model_name = st.text_input("Embedding model", value="all-MiniLM-L6-v2")
    top_k = st.slider("Top K", 1, 10, 5)
    use_openai = st.checkbox("Enable OpenAI grounded answer (requires OPENAI_API_KEY)")

with col1:
    query = st.text_input("Enter your question")
    submit = st.button("Search")

if submit:
    if faiss is None:
        st.error("faiss is not available in this environment. Install faiss-cpu.")
    else:
        idx_path = Path(index_path)
        meta_p = Path(meta_path)
        if not idx_path.exists() or not meta_p.exists():
            st.error("Index or metadata file not found. Run ingest first.")
        else:
            with st.spinner("Loading model and index..."):
                try:
                    from sentence_transformers import SentenceTransformer
                    from src.query import load_index, generate_grounded_response
                except Exception as e:
                    st.error(
                        "Missing or failed-to-import dependency: %s. "
                        "Ensure requirements are installed in the deployment." % e
                    )
                    st.stop()

                model = SentenceTransformer(model_name)
                index = load_index(idx_path)

            # load metadata
            import json

            with open(meta_p, "r", encoding="utf-8") as f:
                metadatas = json.load(f)

            q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
            D, I = index.search(q_emb, top_k)
            I = I[0]

            results = []
            for idx in I:
                if idx < 0 or idx >= len(metadatas):
                    continue
                results.append(metadatas[idx])

            st.subheader("Retrieved sources")
            for r in results:
                st.markdown(f"- **{r.get('source','<unknown>')}** — chunk {r.get('chunk_index')}")
                excerpt = r.get('text', '')[:500]
                st.code(excerpt)

            if use_openai:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    st.warning("OPENAI_API_KEY is not set in the environment.")
                else:
                    # build concatenated retrieved_context
                    context_excerpts = []
                    for r in results:
                        src = r.get("source", "<unknown>")
                        text = r.get("text", "")
                        context_excerpts.append(f"Source: {src}\n{text}")
                    retrieved_context = "\n---\n".join(context_excerpts)

                    with st.spinner("Generating grounded answer..."):
                        try:
                            # generate_grounded_response expects an OpenAI-style client; try to instantiate if available
                            try:
                                from openai import OpenAI

                                client = OpenAI(api_key=api_key)
                                answer = generate_grounded_response(client, retrieved_context, query)
                            except Exception:
                                # fallback to module-style
                                import openai as _openai

                                _openai.api_key = api_key
                                resp = _openai.ChatCompletion.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": "Use only provided sources."},
                                        {"role": "user", "content": retrieved_context + "\n\nQuestion:\n" + query},
                                    ],
                                    max_tokens=400,
                                    temperature=0.0,
                                )
                                answer = resp["choices"][0]["message"]["content"]

                            st.subheader("Grounded answer")
                            st.write(answer)
                        except Exception as e:
                            st.error(f"OpenAI request failed: {e}")
