"""Validation of untrusted uploads.

Three independent checks, none of which trusts the client:

1. the declared filename (extension only -- the name itself is never used for
   anything but a display label),
2. the size cap, enforced *while streaming* so an oversized file is never
   buffered into memory,
3. the ``%PDF-`` signature, which is what actually decides whether the bytes
   are a PDF.

The SHA-256 is computed during the same single pass over the stream.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from fastapi import UploadFile

from app.core.errors import (
    ErrorCode,
    FileTooLargeError,
    InvalidDocumentError,
    UnsupportedFileTypeError,
    ValidationError,
)

PDF_EXTENSION: Final[str] = ".pdf"
PDF_MAGIC: Final[bytes] = b"%PDF-"
# The PDF specification tolerates up to 1024 bytes of junk before the header.
HEADER_SCAN_BYTES: Final[int] = 1024
READ_BLOCK_BYTES: Final[int] = 1024 * 1024
MAX_FILENAME_LENGTH: Final[int] = 180
DEFAULT_FILENAME: Final[str] = "document.pdf"

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE_RUN = re.compile(r"\s+")


def sanitize_filename(raw: str | None) -> str:
    """Reduce an attacker-controlled filename to a safe display label.

    Strips directory components for both separators (``../etc/passwd`` and
    ``..\\windows\\system32``), removes control characters, collapses
    whitespace and caps the length. The result is only ever shown to the user
    and stored in SQLite -- it never touches the filesystem.
    """
    candidate = (raw or "").replace("\\", "/").split("/")[-1]
    candidate = _CONTROL_CHARS.sub("", candidate)
    candidate = _WHITESPACE_RUN.sub(" ", candidate).strip().strip(".")
    if not candidate:
        return DEFAULT_FILENAME

    stem, suffix = os.path.splitext(candidate)
    if len(candidate) > MAX_FILENAME_LENGTH:
        keep = MAX_FILENAME_LENGTH - len(suffix)
        candidate = f"{stem[:keep]}{suffix}"
    return candidate or DEFAULT_FILENAME


def validate_extension(filename: str) -> str:
    """Reject anything that is not ``.pdf``. Returns the lower-cased suffix."""
    suffix = Path(filename).suffix.lower()
    if suffix != PDF_EXTENSION:
        shown = suffix or "no extension"
        raise UnsupportedFileTypeError(f"Only PDF files are supported (received {shown}).")
    return suffix


def looks_like_pdf(header: bytes) -> bool:
    """True if the PDF signature appears within the first 1024 bytes."""
    return PDF_MAGIC in header[:HEADER_SCAN_BYTES]


@dataclass(frozen=True, slots=True)
class StagedUpload:
    """A verified upload sitting in the staging directory.

    The caller owns ``path`` and must call :meth:`discard` when finished.
    """

    path: Path
    filename: str
    content_type: str
    size_bytes: int
    sha256: str

    def discard(self) -> None:
        """Delete the staged file. Idempotent."""
        self.path.unlink(missing_ok=True)


def stage_upload(upload: UploadFile, temp_dir: Path, max_bytes: int) -> StagedUpload:
    """Stream ``upload`` to a private staging file, enforcing every check.

    Raises a mapped :class:`~app.core.errors.AppError` subclass and removes the
    temporary file on any failure, so a rejected upload leaves nothing behind.
    """
    filename = sanitize_filename(upload.filename)
    validate_extension(filename)
    content_type = (upload.content_type or "application/pdf").split(";")[0].strip()
    if not content_type:
        content_type = "application/pdf"

    temp_dir.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=temp_dir, prefix="upload-", suffix=".part")
    digest = hashlib.sha256()
    header = b""
    size = 0

    try:
        with os.fdopen(handle, "wb") as sink:
            while True:
                block = upload.file.read(READ_BLOCK_BYTES)
                if not block:
                    break
                size += len(block)
                if size > max_bytes:
                    raise FileTooLargeError(max_bytes)
                if len(header) < HEADER_SCAN_BYTES:
                    header += block[: HEADER_SCAN_BYTES - len(header)]
                digest.update(block)
                sink.write(block)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise

    staged = StagedUpload(
        path=Path(temp_name),
        filename=filename,
        content_type=content_type,
        size_bytes=size,
        sha256=digest.hexdigest(),
    )

    if size == 0:
        staged.discard()
        raise ValidationError("The uploaded file is empty.")
    if not looks_like_pdf(header):
        staged.discard()
        raise InvalidDocumentError(
            "The file does not look like a PDF (the '%PDF' signature is missing).",
            code=ErrorCode.NOT_A_PDF,
        )
    return staged
