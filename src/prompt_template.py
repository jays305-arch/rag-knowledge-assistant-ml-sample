"""Prompt template helpers for grounded LLM answers.

Provides a small helper to format a deterministic, source-citing prompt
suitable for use with OpenAI or other chat LLMs.

Usage:
    from src.prompt_template import build_prompt
    prompt = build_prompt(query, contexts)

`contexts` should be an iterable of dicts with keys: `source` and `text`.
"""

from typing import Iterable, Dict, List


DEFAULT_INSTRUCTIONS = (
    "You are a helpful, factual assistant. Answer the question using only the information "
    "explicitly provided in the sources below. Do not hallucinate or invent facts. "
    "When you refer to information from a source, cite the source filename in brackets. "
    "If the answer is not contained in the sources, respond: 'I don't know; the sources do not contain this information.'"
)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def build_prompt(
    query: str,
    contexts: Iterable[Dict[str, str]],
    max_context_chars: int = 1500,
    instructions: str = DEFAULT_INSTRUCTIONS,
) -> str:
    """Build a prompt string that includes instructions, context excerpts, and the user query.

    Args:
        query: The user question.
        contexts: Iterable of dicts with `source` and `text` keys.
        max_context_chars: Max characters to include per context excerpt.
        instructions: Instructional header for the LLM.

    Returns:
        A single string prompt ready to send to a chat/completion API.
    """
    parts: List[str] = [instructions, "\n\nSources:"]

    for ctx in contexts:
        src = ctx.get("source", "<unknown>")
        text = ctx.get("text", "")
        excerpt = _truncate(text.replace('\n', ' '), max_context_chars)
        parts.append(f"Source: {src}\n{excerpt}\n---")

    parts.append("\nQuestion:\n" + query)
    parts.append("\nAnswer:")

    return "\n".join(parts)


# User-provided system and user prompt templates (enterprise-friendly guardrails)
SYSTEM_PROMPT = """
You are an AI assistant designed to answer questions using ONLY the provided source material.

GUARDRAILS:
- Do NOT invent facts.
- Do NOT rely on general knowledge outside the provided context.
- If the answer cannot be found in the sources, say:
    "The provided documents do not contain sufficient information to answer this question."
- When refusing due to insufficient source material, ALWAYS begin the response with the exact
- prefix `REFUSAL:` followed by a concise reason and, where appropriate, a short suggestion
- for how the user can obtain the required information (for example: "provide more documents",
- "specify a timeframe", or "ask a domain expert").
- Cite relevant source excerpts when possible.
- Maintain a neutral, professional tone suitable for enterprise and government use.
 - If the user asks for legal, medical, or policy advice that is not explicitly contained in the
     provided sources, REFUSE using the `REFUSAL:` pattern and explain that qualified professional
     advice is required (for example: a lawyer, clinician, or policy analyst). Suggest an action
     such as providing authoritative documents, consulting a qualified expert, or narrowing the
     question to material present in the sources.
"""
USER_PROMPT_TEMPLATE = """
CONTEXT:
{retrieved_context}

QUESTION:
{user_question}

INSTRUCTIONS:
- Base your answer strictly on the CONTEXT above.
- If multiple sources are relevant, synthesize them clearly.
- If the context is insufficient, explicitly state that limitation.
- Do not speculate or provide personal opinions.

ANSWER:
"""


def build_user_prompt(retrieved_context: str, user_question: str) -> str:
    """Return a user prompt formatted with `USER_PROMPT_TEMPLATE`.

    Args:
        retrieved_context: Concatenated context excerpts to place into the template.
        user_question: The user's question.

    Returns:
        Formatted user message string.
    """
    return USER_PROMPT_TEMPLATE.format(retrieved_context=retrieved_context, user_question=user_question)


REFUSAL_PREFIX = "REFUSAL:"


def format_refusal(reason: str, suggestion: str = None) -> str:
    """Format a canonical refusal message following the `REFUSAL:` pattern.

    Args:
        reason: Short explanation why the sources are insufficient.
        suggestion: Optional action the user can take to obtain an answer.

    Returns:
        Standardized refusal string.
    """
    parts = [f"{REFUSAL_PREFIX} {reason.strip()}"]
    if suggestion:
        parts.append(f"Suggested action: {suggestion.strip()}")
    return " -- ".join(parts)


if __name__ == "__main__":
    # quick local demo
    q = "What reduces hallucinations in LLMs?"
    ctxt = [
        {"source": "doc1.txt", "text": "Grounding LLMs with retrieved evidence reduces hallucination."},
        {"source": "doc2.txt", "text": "Ensemble verification and retrieval-augmented generation improve accuracy."},
    ]
    print(build_prompt(q, ctxt))
