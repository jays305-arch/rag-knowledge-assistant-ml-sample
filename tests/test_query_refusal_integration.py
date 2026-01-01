import os
import subprocess
import sys


def test_query_auto_refusal_integration():
    # Ensure the artifacts exist
    assert os.path.exists("artifacts/faiss.index")
    assert os.path.exists("artifacts/meta.json")

    cmd = (
        "PYTHONPATH=. OPENAI_API_KEY=testing "
        "bash -lc 'echo "Is this legal advice?" | python3 src/query.py --index-path artifacts/faiss.index --meta-path artifacts/meta.json --openai --top-k 2'"
    )
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = proc.stdout + proc.stderr
    assert "REFUSAL:" in out
