"""Chunker behaviour: boundaries, overlap, empty input, no text loss."""

from __future__ import annotations

import pytest

from app.utils.chunking import Chunk, chunk_pages, chunk_text, split_recursive


def sentences(count: int) -> str:
    return " ".join(
        f"Sentence {index} examines retrieval quality of grounded answers in detail."
        for index in range(count)
    )


def test_blank_text_produces_no_chunks() -> None:
    assert chunk_text("", 100, 20) == []
    assert chunk_text("   \n\n\t  ", 100, 20) == []


def test_short_text_becomes_a_single_chunk() -> None:
    text = "Gravity is a force."
    assert chunk_text(text, 100, 20) == [text]


def test_text_exactly_at_the_boundary_is_one_chunk() -> None:
    text = "x" * 80
    assert chunk_text(text, 80, 20) == [text]


def test_long_text_produces_several_chunks_within_budget() -> None:
    chunks = chunk_text(sentences(20), 200, 40)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_consecutive_chunks_overlap() -> None:
    """The last word of a chunk must reappear in the next one."""
    chunks = chunk_text(sentences(20), 200, 40)
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.split()[-1] in current


def test_no_sentence_is_lost() -> None:
    chunks = chunk_text(sentences(12), 150, 30)
    for index in range(12):
        needle = f"Sentence {index} examines"
        assert any(needle in chunk for chunk in chunks), f"lost {needle!r}"


def test_a_token_longer_than_the_budget_is_hard_split() -> None:
    token = "a" * 500
    chunks = chunk_text(token, 100, 20)
    assert "".join(chunks) == token
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_split_recursive_keeps_every_character() -> None:
    text = sentences(6)
    pieces = split_recursive(text, 120)
    assert "".join(piece.replace(" ", "") for piece in pieces) == text.replace(" ", "")


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_text("text", 0, 0)
    with pytest.raises(ValueError):
        chunk_text("text", 100, 100)


def test_chunk_metadata_tracks_page_and_index() -> None:
    pages = [sentences(6), "", sentences(6)]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=40)

    assert chunks, "expected chunks"
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.page for chunk in chunks} == {1, 3}
    # Indices keep increasing across pages, and the empty page contributes none.
    page_one = [chunk for chunk in chunks if chunk.page == 1]
    page_three = [chunk for chunk in chunks if chunk.page == 3]
    assert max(chunk.index for chunk in page_one) < min(chunk.index for chunk in page_three)


def test_chunk_rejects_invalid_metadata() -> None:
    with pytest.raises(ValueError):
        Chunk(text="hello", page=0, index=0)
    with pytest.raises(ValueError):
        Chunk(text="   ", page=1, index=0)
    with pytest.raises(ValueError):
        Chunk(text="hello", page=1, index=-1)
