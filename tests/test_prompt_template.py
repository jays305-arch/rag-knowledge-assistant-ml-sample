from src import prompt_template as pt


def test_format_refusal_basic():
    msg = pt.format_refusal("Insufficient info", "Provide more docs")
    assert msg.startswith(pt.REFUSAL_PREFIX)
    assert "Insufficient info" in msg
    assert "Provide more docs" in msg


def test_system_prompt_contains_refusal_instruction():
    assert "REFUSAL:" in pt.SYSTEM_PROMPT
