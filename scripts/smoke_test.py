import subprocess
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except Exception:
    faiss = None

ARTIFACT_INDEX = Path("artifacts/faiss.index")
ARTIFACT_META = Path("artifacts/meta.json")

print("Running ingest...")
subprocess.run(["python", "src/ingest.py", "--data-dir", "data", "--index-path", str(ARTIFACT_INDEX), "--meta-path", str(ARTIFACT_META)] , check=True)

if not ARTIFACT_INDEX.exists() or not ARTIFACT_META.exists():
    raise SystemExit("Index or metadata not found after ingest")

print("Loading metadata...")
with open(ARTIFACT_META, "r", encoding="utf-8") as f:
    metas = json.load(f)

print(f"Loaded {len(metas)} chunks from metadata")

model = SentenceTransformer("all-MiniLM-L6-v2")
query = "How do we reduce hallucinations?"
q_emb = model.encode([query], convert_to_numpy=True).astype("float32")

if faiss is None:
    raise SystemExit("faiss is not available in this environment")

index = faiss.read_index(str(ARTIFACT_INDEX))
D, I = index.search(q_emb, 3)

print("Top matches:")
for idx in I[0]:
    if idx < 0 or idx >= len(metas):
        continue
    m = metas[idx]
    print('-', m.get('source'), 'chunk', m.get('chunk_index'))
    print('  excerpt:', (m.get('text') or '')[:200])

print("Smoke test complete")
