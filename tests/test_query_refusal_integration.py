import os
import subprocess
import sys
from pathlib import Path

# ensure repo root is on sys.path so `src` imports work in tests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_query_auto_refusal_integration():
    # Ensure the artifacts exist
    assert os.path.exists("artifacts/faiss.index")
    assert os.path.exists("artifacts/meta.json")

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "testing"
    env["PYTHONPATH"] = "."

    proc = subprocess.run(
        ["python3", "src/query.py", "--index-path", "artifacts/faiss.index", "--meta-path", "artifacts/meta.json", "--openai", "--top-k", "2"],
        input="Is this legal advice?\n",
        text=True,
        capture_output=True,
        env=env,
    )
    out = proc.stdout + proc.stderr
    assert "REFUSAL:" in out
