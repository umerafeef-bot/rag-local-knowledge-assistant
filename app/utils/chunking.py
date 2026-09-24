"""Recursive character chunking with overlap.

Deliberately hand-written rather than delegated to a framework, because the
retrieval quality of the whole app depends on it and every line should be
explainable.

Two stages:

1. **Recursive splitting.** Try to break the text on the most semantically
   meaningful boundary first (blank line -> newline -> sentence -> space) and
   fall back to a hard character cut when a single token is longer than the
   budget. This guarantees every piece is at most ``chunk_size`` characters and
   that no text is dropped.
2. **Overlap stitching.** Prepend the tail of the previous chunk so a sentence
   straddling a boundary is retrievable from either side, trimming that tail if
   the pair would exceed the budget.

Chunking is per page so every chunk can carry an exact page number, which is
what makes the cited sources trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

# Ordered from the most to the least meaningful boundary.
SEPARATORS: Final[tuple[str, ...]] = ("\n\n", "\n", ". ", " ", "")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit of text plus the metadata used for citations."""

    text: str
    page: int
    index: int

    def __post_init__(self) -> None:
        if self.page < 1:
            raise ValueError("page must be 1-based and >= 1")
        if self.index < 0:
            raise ValueError("index must be >= 0")
        if not self.text.strip():
            raise ValueError("chunk text must not be blank")


def _hard_split(text: str, size: int) -> list[str]:
    return [text[offset : offset + size] for offset in range(0, len(text), size)]


def split_recursive(text: str, size: int, separators: Sequence[str] = SEPARATORS) -> list[str]:
    """Split ``text`` into pieces of at most ``size`` characters.

    Pieces are as close to ``size`` as the separators allow, and concatenating
    the pieces (ignoring the whitespace trimmed at boundaries) reproduces the
    original text.
    """
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= size:
        return [stripped]
    if not separators:
        return _hard_split(stripped, size)

    separator, remaining = separators[0], separators[1:]
    if separator == "" or separator not in text:
        return split_recursive(stripped, size, remaining)

    # Keep the separator attached to the piece before it so nothing is lost.
    parts = text.split(separator)
    segments = [part + separator for part in parts[:-1]] + [parts[-1]]

    pieces: list[str] = []
    buffer = ""
    for segment in segments:
        if not segment.strip():
            continue
        if len(segment) > size:
            if buffer.strip():
                pieces.append(buffer.strip())
                buffer = ""
            pieces.extend(split_recursive(segment, size, remaining))
        elif len(buffer) + len(segment) <= size:
            buffer += segment
        else:
            pieces.append(buffer.strip())
            buffer = segment
    if buffer.strip():
        pieces.append(buffer.strip())
    return [piece for piece in pieces if piece]


def _overlap_tail(chunk: str, overlap: int) -> str:
    """Return the trailing ``overlap`` characters, aligned to a word boundary."""
    if overlap <= 0:
        return ""
    tail = chunk[-overlap:]
    if " " in tail:
        # Drop the leading partial word so the tail starts cleanly.
        tail = tail.split(" ", 1)[1] if tail.split(" ", 1)[0] else tail.lstrip()
    return tail.strip()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split one page's text into overlapping chunks of at most ``size``."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    pieces = split_recursive(text, size, SEPARATORS)
    chunks: list[str] = []
    for piece in pieces:
        if not chunks:
            chunks.append(piece)
            continue
        tail = _overlap_tail(chunks[-1], overlap)
        if tail:
            budget = size - len(piece) - 1
            tail = tail[-budget:] if budget > 0 else ""
            piece = f"{tail} {piece}".strip() if tail else piece
        chunks.append(piece)
    return [chunk for chunk in chunks if chunk.strip()]


def chunk_pages(
    pages: Sequence[str],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Chunk each page independently, returning page-tagged chunks."""
    chunks: list[Chunk] = []
    index = 0
    for page_number, page_text in enumerate(pages, start=1):
        for piece in chunk_text(page_text, chunk_size, chunk_overlap):
            chunks.append(Chunk(text=piece, page=page_number, index=index))
            index += 1
    return chunks
