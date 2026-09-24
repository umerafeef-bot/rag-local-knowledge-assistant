"""Context assembly and the prompt template.

Pure functions with no database, no model and no HTTP, so these are the cheapest
tests in the suite and they guard the part that decides whether an answer is
grounded.
"""

from __future__ import annotations

import pytest

from app.services.prompting import (
    CONTEXT_HEADER,
    INSUFFICIENT_CONTEXT_ANSWER,
    QUESTION_HEADER,
    SYSTEM_PROMPT,
    build_context,
    build_prompt,
    estimate_prompt_tokens,
    format_block,
    prompt_fits_window,
)
from app.services.retrieval import RetrievedChunk


def chunk(
    *,
    text: str,
    page: int = 1,
    chunk_index: int = 0,
    filename: str = "notes.pdf",
    document_id: str = "doc1",
    similarity: float = 0.5,
) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=document_id,
        filename=filename,
        page=page,
        chunk_index=chunk_index,
        similarity=similarity,
        text=text,
    )


# --- formatting ------------------------------------------------------------
def test_a_block_is_numbered_and_labelled_with_its_origin() -> None:
    block = format_block(2, chunk(text="Entropy is a measure of disorder.", page=7))

    assert block.startswith("[2] ")
    assert "notes.pdf" in block
    assert "page 7" in block
    assert block.endswith("Entropy is a measure of disorder.")


# --- budget ----------------------------------------------------------------
def test_context_keeps_every_hit_when_it_fits() -> None:
    hits = [chunk(text="a" * 50, page=1), chunk(text="b" * 50, page=2, chunk_index=1)]

    context = build_context(hits, max_chars=2000)

    assert len(context.used) == 2
    assert context.truncated is False
    assert context.chars == len(context.text)
    assert "[1]" in context.text and "[2]" in context.text


def test_context_never_exceeds_the_budget() -> None:
    hits = [chunk(text="x" * 400, page=page, chunk_index=page) for page in range(1, 6)]

    context = build_context(hits, max_chars=600)

    assert context.chars <= 600
    assert context.truncated is True
    assert len(context.used) < len(hits)


def test_dropping_keeps_the_highest_ranked_hits() -> None:
    """Rank order is preserved: the best hit is never traded for a worse one."""
    hits = [
        chunk(text="best " * 60, page=1, chunk_index=0),
        chunk(text="second " * 60, page=2, chunk_index=1),
        chunk(text="third " * 60, page=3, chunk_index=2),
    ]

    context = build_context(hits, max_chars=400)

    assert context.used[0].text.startswith("best")
    assert [hit.chunk_index for hit in context.used] == list(
        range(len(context.used))
    )
    assert context.used[0].page == 1


def test_a_single_oversized_chunk_is_cut_not_dropped() -> None:
    """One huge chunk must still produce an answer, just a shortened one."""
    context = build_context([chunk(text="y" * 5000, page=4)], max_chars=300)

    assert context.chars == 300
    assert context.truncated is True
    assert len(context.used) == 1
    assert context.used[0].page == 4


def test_no_hits_yields_an_empty_context() -> None:
    context = build_context([], max_chars=1000)

    assert context.text == ""
    assert context.used == ()
    assert context.chars == 0
    assert context.truncated is False


def test_numbering_is_sequential_and_starts_at_one() -> None:
    hits = [chunk(text=f"chunk {index}", page=index + 1, chunk_index=index) for index in range(3)]

    context = build_context(hits, max_chars=2000)

    assert context.text.count("[1]") == 1
    assert context.text.count("[2]") == 1
    assert context.text.count("[3]") == 1
    assert context.text.index("[1]") < context.text.index("[2]") < context.text.index("[3]")


@pytest.mark.parametrize("max_chars", [0, -1])
def test_a_non_positive_budget_is_a_programming_error(max_chars: int) -> None:
    with pytest.raises(ValueError):
        build_context([chunk(text="hello")], max_chars=max_chars)


# --- prompt ----------------------------------------------------------------
def test_the_system_prompt_forbids_outside_knowledge() -> None:
    """The grounding rule is a product requirement, so it is asserted."""
    assert "ONLY" in SYSTEM_PROMPT
    assert "context" in SYSTEM_PROMPT.lower()
    assert "Never use outside knowledge" in SYSTEM_PROMPT
    assert "never guess" in SYSTEM_PROMPT.lower()
    assert INSUFFICIENT_CONTEXT_ANSWER in SYSTEM_PROMPT


def test_the_system_prompt_asks_for_citations() -> None:
    assert "[1]" in SYSTEM_PROMPT
    assert "cite" in SYSTEM_PROMPT.lower()


def test_the_prompt_contains_the_context_the_question_and_the_rule() -> None:
    context = build_context([chunk(text="Gravity pulls masses together.", page=3)], max_chars=1000)

    prompt = build_prompt("Why do things fall?", context.text)

    assert CONTEXT_HEADER in prompt
    assert "Gravity pulls masses together." in prompt
    assert QUESTION_HEADER in prompt
    assert "Why do things fall?" in prompt
    assert INSUFFICIENT_CONTEXT_ANSWER in prompt
    assert "only the context" in prompt.lower()


def test_the_prompt_puts_context_before_the_question() -> None:
    """Instruction-following is better when the context precedes the question."""
    prompt = build_prompt("Question here?", "Block text here.")

    assert prompt.index("Block text here.") < prompt.index("Question here?")


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_a_blank_question_is_rejected(question: str) -> None:
    with pytest.raises(ValueError):
        build_prompt(question, "context")


# --- window estimate -------------------------------------------------------
def test_token_estimate_grows_with_the_prompt() -> None:
    assert estimate_prompt_tokens("a" * 400) > estimate_prompt_tokens("a" * 40)


def test_a_small_prompt_fits_a_small_window() -> None:
    assert prompt_fits_window("short prompt", context_window=2048) is True


def test_an_enormous_prompt_does_not_fit() -> None:
    assert prompt_fits_window("a" * 40000, context_window=2048) is False
