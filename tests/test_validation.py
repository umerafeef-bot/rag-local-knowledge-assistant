"""Upload validation: filenames, extensions, magic bytes, size cap, hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.core.errors import (
    ErrorCode,
    FileTooLargeError,
    InvalidDocumentError,
    UnsupportedFileTypeError,
    ValidationError,
)
from app.utils.validation import (
    looks_like_pdf,
    sanitize_filename,
    stage_upload,
    validate_extension,
)
from tests.fakes import make_upload
from tests.pdf_factory import build_pdf

VALID_PDF = build_pdf(["A page of text used by the validation tests."])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("..\\..\\windows\\system32\\evil.pdf", "evil.pdf"),
        ("/absolute/path/notes.pdf", "notes.pdf"),
        ("  spaced   name.pdf  ", "spaced name.pdf"),
        ("weird\x00name.pdf", "weirdname.pdf"),
        ("", "document.pdf"),
        (None, "document.pdf"),
        ("...", "document.pdf"),
        ("folder/", "document.pdf"),
    ],
)
def test_sanitize_filename(raw: str | None, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_caps_length_but_keeps_the_extension() -> None:
    sanitized = sanitize_filename("a" * 400 + ".pdf")
    assert len(sanitized) == 180
    assert sanitized.endswith(".pdf")


def test_validate_extension_accepts_only_pdf() -> None:
    assert validate_extension("report.PDF") == ".pdf"
    with pytest.raises(UnsupportedFileTypeError) as error:
        validate_extension("notes.txt")
    assert error.value.code is ErrorCode.UNSUPPORTED_FILE_TYPE
    assert error.value.status_code == 415


def test_extension_is_required() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        validate_extension("noextension")


def test_magic_bytes_are_detected_within_the_header_window() -> None:
    assert looks_like_pdf(b"%PDF-1.7\n") is True
    assert looks_like_pdf(b"garbage") is False
    assert looks_like_pdf(b"x" * 1000 + b"%PDF-1.4") is True
    assert looks_like_pdf(b"x" * 2000 + b"%PDF-1.4") is False


def test_stage_upload_streams_and_hashes(tmp_path: Path) -> None:
    staged = stage_upload(make_upload(VALID_PDF), tmp_path, max_bytes=10_000)

    assert staged.filename == "document.pdf"
    assert staged.content_type == "application/pdf"
    assert staged.size_bytes == len(VALID_PDF)
    assert staged.sha256 == hashlib.sha256(VALID_PDF).hexdigest()
    # The user's filename is never used on disk.
    assert staged.path.name.startswith("upload-")
    assert staged.path.read_bytes() == VALID_PDF

    staged.discard()
    assert not staged.path.exists()


def test_stage_upload_rejects_an_oversized_file(tmp_path: Path) -> None:
    payload = b"%PDF-1.4\n" + b"x" * 2048
    with pytest.raises(FileTooLargeError) as error:
        stage_upload(make_upload(payload), tmp_path, max_bytes=1024)
    assert error.value.code is ErrorCode.FILE_TOO_LARGE
    assert error.value.status_code == 413
    assert list(tmp_path.iterdir()) == []


def test_stage_upload_allows_a_file_exactly_at_the_limit(tmp_path: Path) -> None:
    payload = b"%PDF-1.4\n" + b"x" * 24  # 32 bytes
    staged = stage_upload(make_upload(payload), tmp_path, max_bytes=len(payload))
    assert staged.size_bytes == len(payload)


def test_stage_upload_rejects_an_empty_file(tmp_path: Path) -> None:
    with pytest.raises(ValidationError) as error:
        stage_upload(make_upload(b""), tmp_path, max_bytes=1024)
    assert error.value.status_code == 400
    assert list(tmp_path.iterdir()) == []


def test_stage_upload_rejects_a_mislabelled_file(tmp_path: Path) -> None:
    """A .pdf name is not evidence: the bytes decide."""
    with pytest.raises(InvalidDocumentError) as error:
        stage_upload(make_upload(b"hello there"), tmp_path, max_bytes=1024)
    assert error.value.code is ErrorCode.NOT_A_PDF
    assert list(tmp_path.iterdir()) == []


def test_stage_upload_rejects_the_wrong_extension_before_reading(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        stage_upload(make_upload(VALID_PDF, filename="notes.txt"), tmp_path, max_bytes=1024)
    assert list(tmp_path.iterdir()) == []


def test_stage_upload_uses_a_sanitised_display_name(tmp_path: Path) -> None:
    staged = stage_upload(
        make_upload(VALID_PDF, filename="../../etc/passwd.pdf"), tmp_path, max_bytes=10_000
    )
    assert staged.filename == "passwd.pdf"
