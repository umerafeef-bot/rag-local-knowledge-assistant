"""Cleaning of raw per-page PDF text before chunking.

Raw ``pdfplumber`` output is faithful to the layout, which means wrapped lines,
hyphenated line breaks, boilerplate running headers/footers and stray page
numbers. Embedding that noise directly costs retrieval quality, so each page
goes through a fixed, documented sequence of steps:

    NFKC normalisation -> repeated-line removal -> de-hyphenation
    -> page-number removal -> wrapped-line merging -> blank-line collapsing

Each step is conservative: it never reorders text and never invents words.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Final, Sequence

# Characters flagged by most PDF extractors that carry the same meaning as
# their ASCII counterparts.
_PUNCTUATION_MAP: Final[dict[str, str]] = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2026": "...",
    "\u00a0": " ",
    "\u2009": " ",
    "\u200a": " ",
    "\u202f": " ",
    "\ufeff": "",
}

_PUNCTUATION_TABLE = str.maketrans(_PUNCTUATION_MAP)
_TRAILING_SPACE = re.compile(r"[ \t]+$")
_HYPHEN_LINE_BREAK = re.compile(r"(\w)-\n(\w)")
_PAGE_NUMBER_LINE = re.compile(
    r"^(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?$",
    re.IGNORECASE,
)
_BLANK_RUN = re.compile(r"\n{3,}")
_SENTENCE_TERMINATORS: Final[tuple[str, ...]] = (".", "!", "?", ":", ";", ")", "]")


def normalize_text(text: str) -> str:
    """NFKC-normalise and flatten typographically-pretty characters.

    NFKC also expands ligatures (``\ufb01`` -> ``fi``) and full-width forms,
    so no manual ligature table is needed.
    """
    normalized = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION_TABLE)
    lines = [_TRAILING_SPACE.sub("", line).strip("\r") for line in normalized.splitlines()]
    return "\n".join(lines)


def dehyphenate(text: str) -> str:
    """Rejoin words split across a line break by a hyphen."""
    return _HYPHEN_LINE_BREAK.sub(r"\1\2", text)


def drop_page_number_lines(text: str) -> str:
    """Remove lines that are nothing but a page number or ``Page N of M``."""
    kept = [line for line in text.splitlines() if not _PAGE_NUMBER_LINE.match(line.strip())]
    return "\n".join(kept)


def collapse_blank_lines(text: str) -> str:
    """Collapse runs of blank lines so paragraphs stay distinguishable."""
    return _BLANK_RUN.sub("\n\n", text).strip()


def strip_repeated_lines(
    pages: Sequence[str],
    *,
    min_pages: int = 3,
    ratio: float = 0.6,
    max_line_length: int = 120,
) -> list[str]:
    """Drop running headers/footers.

    A line that appears on at least ``ratio`` of the pages (and on at least two
    pages) is boilerplate, not content: textbooks repeat the chapter title and
    the book name on every page. ``max_line_length`` protects long, genuinely
    repeated body lines from being removed.
    """
    if len(pages) < min_pages:
        return list(pages)

    occurrences: Counter[str] = Counter()
    for page in pages:
        distinct = {
            line.strip()
            for line in page.splitlines()
            if line.strip() and len(line.strip()) <= max_line_length
        }
        occurrences.update(distinct)

    threshold = max(2, math.ceil(ratio * len(pages)))
    boilerplate = {line for line, count in occurrences.items() if count >= threshold}
    if not boilerplate:
        return list(pages)

    return [
        "\n".join(line for line in page.splitlines() if line.strip() not in boilerplate)
        for page in pages
    ]


def _continues(previous: str, current: str) -> bool:
    """Decide whether ``current`` is a wrapped continuation of ``previous``.

    The rule is intentionally narrow: a line that ends mid-sentence (lowercase
    letter or comma) continues only into a line that starts lowercase. That
    keeps headings, list items and new sentences intact.
    """
    previous = previous.rstrip()
    if not previous or previous.endswith(_SENTENCE_TERMINATORS):
        return False
    if not (previous[-1].islower() or previous[-1] == ","):
        return False
    return current[:1].islower()


def merge_wrapped_lines(text: str) -> str:
    """Join lines that were wrapped only because of the page width."""
    merged: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            merged.append("")
            continue
        if merged and merged[-1] and _continues(merged[-1], stripped):
            merged[-1] = f"{merged[-1]} {stripped}"
        else:
            merged.append(stripped)
    return "\n".join(merged)


def clean_page_text(text: str) -> str:
    """Apply the per-page cleaning pipeline."""
    cleaned = dehyphenate(normalize_text(text))
    cleaned = drop_page_number_lines(cleaned)
    cleaned = merge_wrapped_lines(cleaned)
    return collapse_blank_lines(cleaned)


def clean_pages(
    pages: Sequence[str],
    *,
    min_pages: int = 3,
    header_footer_ratio: float = 0.6,
) -> list[str]:
    """Clean every page, removing cross-page boilerplate first."""
    normalized = [normalize_text(page) for page in pages]
    without_boilerplate = strip_repeated_lines(
        normalized, min_pages=min_pages, ratio=header_footer_ratio
    )
    return [clean_page_text(page) for page in without_boilerplate]


_WHITESPACE_RUN = re.compile(r"\s+")


def make_snippet(text: str, limit: int) -> str:
    """Flatten ``text`` into one line of at most ``limit`` characters.

    Used for the excerpt shown next to a cited source. Chunk text is a block of
    wrapped lines; a citation panel wants a single readable line, so the line
    breaks are collapsed before the cut. The cut lands on a word boundary and
    is marked with an ellipsis, so a truncated excerpt never looks like the
    document's own punctuation.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    flattened = _WHITESPACE_RUN.sub(" ", text).strip()
    if len(flattened) <= limit:
        return flattened

    window = flattened[:limit].rstrip()
    if " " in window:
        window = window[: window.rfind(" ")].rstrip()
    return f"{window}..."
