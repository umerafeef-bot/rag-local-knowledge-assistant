"""Cleaning pipeline: normalisation, de-hyphenation, boilerplate removal."""

from __future__ import annotations

from app.utils.text_cleaning import (
    clean_page_text,
    clean_pages,
    collapse_blank_lines,
    dehyphenate,
    drop_page_number_lines,
    merge_wrapped_lines,
    normalize_text,
    strip_repeated_lines,
)


def test_nfkc_expands_ligatures_and_fancy_punctuation() -> None:
    assert normalize_text("e\ufb03cient") == "efficient"
    assert normalize_text("\u201cquoted\u201d") == '"quoted"'
    assert normalize_text("en\u2013dash\u2014em") == "en-dash-em"
    assert normalize_text("non\u00a0breaking") == "non breaking"


def test_trailing_whitespace_is_stripped_per_line() -> None:
    # A trailing newline carries no content and is dropped; blank lines that
    # separate content are preserved.
    assert normalize_text("one   \ntwo\t\n") == "one\ntwo"
    assert normalize_text("one  \n\ntwo   ") == "one\n\ntwo"


def test_dehyphenation_joins_split_words() -> None:
    assert dehyphenate("exam-\nple") == "example"
    assert dehyphenate("multi-\nline is fine") == "multiline is fine"


def test_page_number_lines_are_removed() -> None:
    cleaned = drop_page_number_lines("Intro\n42\nPage 7 of 220\nChapter 12")
    assert cleaned.splitlines() == ["Intro", "Chapter 12"]


def test_wrapped_lines_are_merged_but_sentences_are_not() -> None:
    text = "the quick brown\nfox jumps\n\nA new sentence starts here.\nAnother one follows."
    assert merge_wrapped_lines(text) == (
        "the quick brown fox jumps\n\nA new sentence starts here.\nAnother one follows."
    )


def test_numbered_headings_are_not_merged() -> None:
    text = "Chapter 3\nthis paragraph should stay separate"
    assert merge_wrapped_lines(text) == text


def test_repeated_headers_and_footers_are_dropped() -> None:
    pages = [
        f"Advanced Algorithms\nbody of page {number}\nCopyright 2024"
        for number in range(1, 5)
    ]
    cleaned = strip_repeated_lines(pages)
    assert all("Advanced Algorithms" not in page for page in cleaned)
    assert all("Copyright 2024" not in page for page in cleaned)
    assert "body of page 3" in cleaned[2]


def test_short_documents_keep_their_lines() -> None:
    """With fewer pages than ``min_pages`` there is no evidence of boilerplate."""
    pages = ["Title\nbody", "Title\nbody two"]
    assert strip_repeated_lines(pages, min_pages=3) == pages


def test_unique_lines_survive_boilerplate_removal() -> None:
    pages = ["Header\nunique one", "Header\nunique two", "Header\nunique three"]
    cleaned = strip_repeated_lines(pages)
    assert cleaned == ["unique one", "unique two", "unique three"]


def test_blank_line_runs_are_collapsed() -> None:
    assert collapse_blank_lines("a\n\n\n\n\nb") == "a\n\nb"


def test_clean_page_text_applies_the_whole_chain() -> None:
    raw = "The momentum of the system is\n exam-\nined below.\n\n\n12"
    assert clean_page_text(raw) == "The momentum of the system is examined below."


def test_clean_pages_removes_cross_page_boilerplate_only() -> None:
    pages = [
        "Running Header\nalpha unique text on page one here",
        "Running Header\nbeta unique text on page two here",
        "Running Header\ngamma unique text on page three here",
    ]
    cleaned = clean_pages(pages)
    assert all("Running Header" not in page for page in cleaned)
    assert [page.split()[0] for page in cleaned] == ["alpha", "beta", "gamma"]
