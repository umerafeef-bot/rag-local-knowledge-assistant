"""Application error types and stable error codes.

Every failure the API can produce has a machine-readable ``ErrorCode`` and an
HTTP status. Messages here are deliberately user-facing: they explain what the
caller can do next and never expose stack traces, SQL, filesystem paths or
library internals.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class ErrorCode(str, Enum):
    """Stable identifiers. Safe to match on from a client."""

    VALIDATION_ERROR = "validation_error"
    UNSUPPORTED_FILE_TYPE = "unsupported_file_type"
    FILE_TOO_LARGE = "file_too_large"
    NOT_A_PDF = "not_a_pdf"
    PDF_ENCRYPTED = "pdf_encrypted"
    PDF_CORRUPTED = "pdf_corrupted"
    PDF_EMPTY = "pdf_empty"
    NO_EXTRACTABLE_TEXT = "no_extractable_text"
    TOO_MANY_PAGES = "too_many_pages"
    DUPLICATE_DOCUMENT = "duplicate_document"
    DOCUMENT_NOT_FOUND = "document_not_found"
    STORAGE_ERROR = "storage_error"
    NO_DOCUMENTS = "no_documents"
    LLM_UNAVAILABLE = "llm_unavailable"
    HTTP_ERROR = "http_error"
    INTERNAL_ERROR = "internal_error"


class AppError(Exception):
    """Base class for every expected failure.

    ``status_code``, ``code`` and ``message`` are class-level defaults that an
    instance may override.
    """

    status_code: int = 500
    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    message: str = "An unexpected internal error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: ErrorCode | None = None,
        status_code: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.message = message or type(self).message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.details: dict[str, Any] | None = dict(details) if details else None
        super().__init__(self.message)


class ValidationError(AppError):
    status_code = 400
    code = ErrorCode.VALIDATION_ERROR
    message = "The request could not be validated."


class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = ErrorCode.UNSUPPORTED_FILE_TYPE
    message = "Only PDF files are supported."


class FileTooLargeError(AppError):
    status_code = 413
    code = ErrorCode.FILE_TOO_LARGE
    message = "The uploaded file is too large."

    def __init__(self, limit_bytes: int, message: str | None = None) -> None:
        limit_mb = max(1, round(limit_bytes / (1024 * 1024)))
        super().__init__(
            message or f"The file exceeds the {limit_mb} MB upload limit.",
            details={"limit_bytes": limit_bytes},
        )


class InvalidDocumentError(AppError):
    """A file that is a PDF by signature but cannot be turned into text."""

    status_code = 422
    code = ErrorCode.PDF_CORRUPTED
    message = "The PDF could not be read."


class DuplicateDocumentError(AppError):
    status_code = 409
    code = ErrorCode.DUPLICATE_DOCUMENT
    message = "This document has already been uploaded."

    def __init__(self, document_id: str, filename: str) -> None:
        super().__init__(
            f"This file is already in the library as '{filename}'.",
            details={"existing_document_id": document_id, "filename": filename},
        )


class NotFoundError(AppError):
    status_code = 404
    code = ErrorCode.DOCUMENT_NOT_FOUND
    message = "The requested document was not found."


class StorageError(AppError):
    status_code = 500
    code = ErrorCode.STORAGE_ERROR
    message = "A storage operation failed. No data was changed."


class NoDocumentsError(AppError):
    """Asked a question before anything was indexed.

    A conflict rather than a validation error: the request itself is fine, the
    library is simply empty. Distinct enough from "no relevant chunk" to
    deserve its own code -- the client should offer an upload, not a rephrase.
    """

    status_code = 409
    code = ErrorCode.NO_DOCUMENTS
    message = "No documents have been indexed yet, so there is nothing to answer from."


class ServiceUnavailableError(AppError):
    """A local dependency is down: the LLM server, or a model that is not pulled.

    Always a 503: the document library keeps working, so this is a transient
    capability failure rather than a client mistake.
    """

    status_code = 503
    code = ErrorCode.LLM_UNAVAILABLE
    message = "A required local service is unavailable."
