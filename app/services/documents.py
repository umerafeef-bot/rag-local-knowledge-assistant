"""Document lifecycle: ingest a PDF, or delete one completely.

Ingestion is deliberately **all-or-nothing**. A request either ends with a
searchable document (row in SQLite + chunks in ChromaDB + file in
``data/uploads``) or with nothing at all -- the file is only promoted to its
final name once every earlier step succeeded, and any later failure is
compensated.

Deletion removes the three copies in a fixed order: vectors, then the file,
then the metadata row. Vector removal failing aborts the whole operation so a
document is never left listed but unsearchable; the metadata row goes last so a
partially deleted document never looks healthy.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.exc import IntegrityError

from app.core.config import Settings
from app.core.errors import (
    DuplicateDocumentError,
    ErrorCode,
    InvalidDocumentError,
    NotFoundError,
    StorageError,
)
from app.db.models import Document
from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_repository import VectorRepository
from app.services.embedding import EmbeddingService
from app.services.pdf_extraction import PdfExtractor
from app.utils.chunking import chunk_pages
from app.utils.text_cleaning import clean_pages
from app.utils.validation import stage_upload

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionTimings:
    """Wall-clock seconds per stage. Reported by the API and the benchmark."""

    upload_seconds: float
    extract_seconds: float
    chunk_seconds: float
    embed_seconds: float
    store_seconds: float
    total_seconds: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document: Document
    timings: IngestionTimings


@dataclass(frozen=True, slots=True)
class DeletionResult:
    document_id: str
    deleted_vectors: int
    file_removed: bool


class DocumentService:
    def __init__(
        self,
        *,
        settings: Settings,
        documents: DocumentRepository,
        vectors: VectorRepository,
        embeddings: EmbeddingService,
        extractor: PdfExtractor,
    ) -> None:
        self._settings = settings
        self._documents = documents
        self._vectors = vectors
        self._embeddings = embeddings
        self._extractor = extractor

    # --- ingestion ---------------------------------------------------------
    def ingest(self, upload: UploadFile) -> IngestionResult:
        started = time.perf_counter()
        staged = stage_upload(
            upload,
            temp_dir=self._settings.temp_dir,
            max_bytes=self._settings.max_upload_bytes,
        )
        upload_seconds = time.perf_counter() - started

        try:
            duplicate = self._documents.find_by_sha256(staged.sha256)
            if duplicate is not None:
                raise DuplicateDocumentError(duplicate.id, duplicate.filename)

            extract_started = time.perf_counter()
            extracted = self._extractor.extract(staged.path)
            extract_seconds = time.perf_counter() - extract_started

            chunk_started = time.perf_counter()
            cleaned_pages = clean_pages([page.text for page in extracted.pages])
            chunks = chunk_pages(
                cleaned_pages,
                chunk_size=self._settings.chunk_size,
                chunk_overlap=self._settings.chunk_overlap,
            )
            chunk_seconds = time.perf_counter() - chunk_started
            if not chunks:
                raise InvalidDocumentError(
                    "No usable text remained after cleaning, so there is nothing to index.",
                    code=ErrorCode.NO_EXTRACTABLE_TEXT,
                )

            embed_started = time.perf_counter()
            embeddings = self._embeddings.embed_documents([chunk.text for chunk in chunks])
            embed_seconds = time.perf_counter() - embed_started

            store_started = time.perf_counter()
            document_id = uuid.uuid4().hex
            final_path = self._settings.upload_dir / f"{document_id}.pdf"
            try:
                # Atomic within data_dir: staging and uploads share a volume.
                os.replace(staged.path, final_path)
            except OSError as exc:
                raise StorageError(
                    "The uploaded file could not be stored. Check free disk space and permissions."
                ) from exc

            try:
                self._vectors.add_chunks(
                    document_id=document_id, chunks=chunks, embeddings=embeddings
                )
                document = self._documents.create(
                    document_id=document_id,
                    filename=staged.filename,
                    content_type=staged.content_type,
                    size_bytes=staged.size_bytes,
                    sha256=staged.sha256,
                    page_count=extracted.page_count,
                    chunk_count=len(chunks),
                    char_count=extracted.char_count,
                    embedding_model=self._embeddings.model_name,
                )
            except IntegrityError as exc:
                # Concurrent upload of the same bytes: the unique sha256 index
                # is the real arbiter, not the earlier check.
                self._compensate(document_id, final_path)
                existing = self._documents.find_by_sha256(staged.sha256)
                if existing is not None:
                    raise DuplicateDocumentError(existing.id, existing.filename) from exc
                raise StorageError("The document metadata could not be saved.") from exc
            except Exception as exc:
                self._compensate(document_id, final_path)
                raise StorageError(
                    "Indexing failed and was rolled back; no document was added."
                ) from exc
            store_seconds = time.perf_counter() - store_started

            timings = IngestionTimings(
                upload_seconds=round(upload_seconds, 4),
                extract_seconds=round(extract_seconds, 4),
                chunk_seconds=round(chunk_seconds, 4),
                embed_seconds=round(embed_seconds, 4),
                store_seconds=round(store_seconds, 4),
                total_seconds=round(time.perf_counter() - started, 4),
            )
            logger.info(
                "Ingested document %s: %d pages, %d chunks, %.2fs",
                document_id,
                extracted.page_count,
                len(chunks),
                timings.total_seconds,
            )
            return IngestionResult(document=document, timings=timings)
        finally:
            # No-op when the file was already promoted; removes it on failure.
            staged.discard()

    def _compensate(self, document_id: str, stored_path: Path) -> None:
        """Best-effort rollback of everything written for a failed ingestion."""
        try:
            self._vectors.delete_document(document_id)
        except Exception:
            logger.exception("Rollback: could not remove vectors for %s", document_id)
        stored_path.unlink(missing_ok=True)

    # --- deletion ----------------------------------------------------------
    def delete(self, document_id: str) -> DeletionResult:
        document = self._documents.get(document_id)
        if document is None:
            raise NotFoundError(f"No document exists with id '{document_id}'.")

        try:
            deleted_vectors = self._vectors.delete_document(document_id)
        except Exception as exc:
            raise StorageError(
                "The document's vectors could not be removed, so nothing was deleted. Try again."
            ) from exc

        file_removed = True
        stored_path = self._settings.upload_dir / f"{document_id}.pdf"
        try:
            stored_path.unlink(missing_ok=True)
        except OSError:
            # Typically a PDF viewer holding the file open on Windows. The
            # document is still removed from the library; the stray file is
            # reported so it can be cleaned up.
            file_removed = False
            logger.warning("Could not remove stored file for document %s", document_id)

        self._documents.delete(document_id)
        logger.info("Deleted document %s (%d chunks removed)", document_id, deleted_vectors)
        return DeletionResult(
            document_id=document_id,
            deleted_vectors=deleted_vectors,
            file_removed=file_removed,
        )

    # --- reads -------------------------------------------------------------
    def get(self, document_id: str) -> Document:
        document = self._documents.get(document_id)
        if document is None:
            raise NotFoundError(f"No document exists with id '{document_id}'.")
        return document

    def list(self, *, limit: int, offset: int) -> tuple[list[Document], int]:
        return self._documents.list(limit=limit, offset=offset)
