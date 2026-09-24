"""Per-page PDF text extraction with explicit failure modes.

Two libraries, each doing what it is best at:

* ``pypdf`` answers structural questions cheaply -- is the file parseable, is
  it encrypted, how many pages does it have -- so we can reject a bad file
  before paying for text extraction.
* ``pdfplumber`` (pdfminer.six underneath) does the actual layout-aware text
  extraction.

Every failure is mapped to a distinct :class:`ErrorCode` so the API can tell
the user something actionable instead of "invalid file".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

from app.core.errors import ErrorCode, InvalidDocumentError

logger = logging.getLogger(__name__)

_CORRUPTED_MESSAGE = (
    "The PDF could not be parsed. It may be corrupted, truncated or not a real PDF."
)


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Text of one page. ``number`` is 1-based, matching what a reader sees."""

    number: int
    text: str


@dataclass(frozen=True, slots=True)
class ExtractedPdf:
    pages: tuple[ExtractedPage, ...]
    char_count: int

    @property
    def page_count(self) -> int:
        return len(self.pages)


class PdfExtractor:
    """Turns a staged PDF file into cleaned-by-nobody, raw per-page text."""

    def __init__(self, *, max_pages: int, min_chars: int) -> None:
        self._max_pages = max_pages
        self._min_chars = min_chars

    def extract(self, path: Path) -> ExtractedPdf:
        """Extract text from every page or raise a mapped :class:`AppError`."""
        self._validate_structure(path)
        pages = self._extract_pages(path)
        char_count = sum(len(page.text) for page in pages)

        if char_count < self._min_chars:
            raise InvalidDocumentError(
                "No extractable text was found. This PDF is most likely a scan "
                "or a set of images, and OCR is not supported.",
                code=ErrorCode.NO_EXTRACTABLE_TEXT,
                details={"page_count": len(pages), "char_count": char_count},
            )
        return ExtractedPdf(pages=tuple(pages), char_count=char_count)

    # --- structure ---------------------------------------------------------
    def _validate_structure(self, path: Path) -> None:
        try:
            reader = PdfReader(str(path), strict=False)
        except Exception as exc:  # pypdf raises several distinct parse errors
            logger.warning("Rejected unparseable PDF during structural check")
            raise InvalidDocumentError(
                _CORRUPTED_MESSAGE, code=ErrorCode.PDF_CORRUPTED
            ) from exc

        if reader.is_encrypted:
            # PDFs are also "encrypted" when only permissions are restricted and
            # the user password is empty; those are legitimately readable, so
            # try the empty password before rejecting.
            if not reader.decrypt(""):
                raise InvalidDocumentError(
                    "This PDF is password-protected. Remove the password and upload it again.",
                    code=ErrorCode.PDF_ENCRYPTED,
                )

        try:
            page_count = len(reader.pages)
        except Exception as exc:
            raise InvalidDocumentError(
                _CORRUPTED_MESSAGE, code=ErrorCode.PDF_CORRUPTED
            ) from exc

        if page_count == 0:
            raise InvalidDocumentError(
                "The PDF contains no pages.", code=ErrorCode.PDF_EMPTY
            )
        if page_count > self._max_pages:
            raise InvalidDocumentError(
                f"This PDF has {page_count} pages, above the configured limit of "
                f"{self._max_pages} pages.",
                code=ErrorCode.TOO_MANY_PAGES,
                details={"page_count": page_count, "max_pages": self._max_pages},
            )

    # --- text --------------------------------------------------------------
    def _extract_pages(self, path: Path) -> list[ExtractedPage]:
        try:
            pages: list[ExtractedPage] = []
            with pdfplumber.open(path) as pdf:
                for number, page in enumerate(pdf.pages, start=1):
                    try:
                        text = page.extract_text() or ""
                    finally:
                        # Frees the per-page object graph; matters for the
                        # hundreds-of-pages case.
                        page.close()
                    pages.append(ExtractedPage(number=number, text=text))
            return pages
        except InvalidDocumentError:
            raise
        except Exception as exc:
            logger.warning("Rejected PDF during text extraction")
            raise InvalidDocumentError(
                _CORRUPTED_MESSAGE, code=ErrorCode.PDF_CORRUPTED
            ) from exc
