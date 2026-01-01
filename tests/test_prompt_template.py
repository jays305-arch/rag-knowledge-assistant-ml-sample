import sys
from pathlib import Path

# ensure repo root is on sys.path so `src` imports work in tests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import prompt_template as pt


def test_format_refusal_basic():
    msg = pt.format_refusal("Insufficient info", "Provide more docs")
    assert msg.startswith(pt.REFUSAL_PREFIX)
    assert "Insufficient info" in msg
    assert "Provide more docs" in msg


def test_system_prompt_contains_refusal_instruction():
    assert "REFUSAL:" in pt.SYSTEM_PROMPT
