"""PDF extraction and its failure modes, against real PDF files on disk."""

from __future__ import annotations

import io
from pathlib import Path

import pdfplumber
import pytest
from pypdf import PdfReader

from app.core.errors import ErrorCode, InvalidDocumentError
from app.services.pdf_extraction import PdfExtractor
from tests.pdf_factory import build_encrypted_pdf, build_pdf, write_pdf

PAGES = [
    "Chapter one introduces momentum and gravity.",
    "Chapter two explains entropy and gradient descent.",
    "Chapter three measures retrieval latency.",
]


def extractor(*, max_pages: int = 20, min_chars: int = 20) -> PdfExtractor:
    return PdfExtractor(max_pages=max_pages, min_chars=min_chars)


def test_generated_fixture_is_a_real_pdf(tmp_path: Path) -> None:
    """Guards the fixture itself: if this fails, every other PDF test is suspect."""
    path = tmp_path / "fixture.pdf"
    write_pdf(path, PAGES)

    reader = PdfReader(str(path))
    assert len(reader.pages) == len(PAGES)
    with pdfplumber.open(path) as pdf:
        assert "momentum" in (pdf.pages[0].extract_text() or "")


def test_extraction_returns_one_entry_per_page(tmp_path: Path) -> None:
    path = tmp_path / "book.pdf"
    write_pdf(path, PAGES)

    extracted = extractor().extract(path)

    assert extracted.page_count == 3
    assert [page.number for page in extracted.pages] == [1, 2, 3]
    assert "momentum" in extracted.pages[0].text
    assert "retrieval" in extracted.pages[2].text
    assert extracted.char_count == sum(len(page.text) for page in extracted.pages)


def test_corrupted_pdf_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4\nthis body is not a pdf at all\n%%EOF\n")

    with pytest.raises(InvalidDocumentError) as error:
        extractor().extract(path)
    assert error.value.code is ErrorCode.PDF_CORRUPTED
    assert error.value.status_code == 422


def test_random_bytes_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "garbage.pdf"
    path.write_bytes(b"not a pdf")

    with pytest.raises(InvalidDocumentError) as error:
        extractor().extract(path)
    assert error.value.code is ErrorCode.PDF_CORRUPTED


def test_encrypted_pdf_is_rejected_with_an_actionable_message(tmp_path: Path) -> None:
    path = tmp_path / "locked.pdf"
    path.write_bytes(build_encrypted_pdf(PAGES, password="secret"))

    with pytest.raises(InvalidDocumentError) as error:
        extractor().extract(path)
    assert error.value.code is ErrorCode.PDF_ENCRYPTED
    assert "password" in error.value.message.lower()


def test_pdf_without_pages_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.pdf"
    path.write_bytes(build_pdf([]))

    with pytest.raises(InvalidDocumentError) as error:
        extractor().extract(path)
    assert error.value.code in {ErrorCode.PDF_EMPTY, ErrorCode.PDF_CORRUPTED}


def test_image_only_pdf_reports_missing_text(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    write_pdf(path, ["", "", ""])  # pages with no text at all

    with pytest.raises(InvalidDocumentError) as error:
        extractor().extract(path)
    assert error.value.code is ErrorCode.NO_EXTRACTABLE_TEXT
    assert "scan" in error.value.message.lower()
    assert error.value.details == {"page_count": 3, "char_count": 0}


def test_page_limit_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "long.pdf"
    write_pdf(path, PAGES)

    with pytest.raises(InvalidDocumentError) as error:
        extractor(max_pages=2).extract(path)
    assert error.value.code is ErrorCode.TOO_MANY_PAGES
    assert error.value.details == {"page_count": 3, "max_pages": 2}


def test_minimum_character_threshold_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "thin.pdf"
    write_pdf(path, ["hi"])

    with pytest.raises(InvalidDocumentError) as error:
        extractor(min_chars=1000).extract(path)
    assert error.value.code is ErrorCode.NO_EXTRACTABLE_TEXT


def test_extraction_works_from_a_stream_too(tmp_path: Path) -> None:
    """Sanity check that the fixture is not accidentally path-only."""
    payload = build_pdf(PAGES)
    with pdfplumber.open(io.BytesIO(payload)) as pdf:
        assert len(pdf.pages) == 3
