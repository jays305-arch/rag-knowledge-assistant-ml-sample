import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except Exception as e:
    faiss = None

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


def read_file(path: Path) -> str:
    if path.suffix.lower() in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".pdf":
        if PdfReader is None:
            raise RuntimeError("PyPDF2 is required to read PDFs")
        text = []
        reader = PdfReader(str(path))
        for p in reader.pages:
            text.append(p.extract_text() or "")
        return "\n".join(text)
    return ""


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        yield text[start:end]
        start = max(0, end - overlap)


def main(data_dir, index_path, meta_path, model_name):
    data_dir = Path(data_dir)
    files = [p for p in data_dir.rglob("*") if p.suffix.lower() in (".txt", ".md", ".pdf")]
    if not files:
        print("No documents found in", data_dir)
        return

    model = SentenceTransformer(model_name)

    docs = []
    metadatas = []
    for f in files:
        try:
            text = read_file(f)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue
        for i, chunk in enumerate(chunk_text(text)):
            chunk = chunk.strip()
            if not chunk:
                continue
            docs.append(chunk)
            metadatas.append({"source": str(f), "chunk_index": i, "text": chunk})

    embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.array(embeddings).astype("float32")

    if faiss is None:
        raise RuntimeError("faiss is required (install faiss-cpu)")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    index_dir = Path(index_path)
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

    print(f"Wrote index to {index_path} and metadata to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Directory with documents (.txt, .md, .pdf)")
    parser.add_argument("--index-path", default="artifacts/faiss.index", help="Path to write FAISS index")
    parser.add_argument("--meta-path", default="artifacts/meta.json", help="Path to write metadata JSON")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    args = parser.parse_args()
    main(args.data_dir, args.index_path, args.meta_path, args.model)
