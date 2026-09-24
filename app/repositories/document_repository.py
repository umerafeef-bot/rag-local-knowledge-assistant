"""Document metadata persistence.

One session per operation: the work units here are single statements, so a
long-lived session would only add coordination problems.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Document


class DocumentRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create(
        self,
        *,
        document_id: str,
        filename: str,
        content_type: str,
        size_bytes: int,
        sha256: str,
        page_count: int,
        chunk_count: int,
        char_count: int,
        embedding_model: str,
    ) -> Document:
        document = Document(
            id=document_id,
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            sha256=sha256,
            page_count=page_count,
            chunk_count=chunk_count,
            char_count=char_count,
            embedding_model=embedding_model,
            created_at=datetime.now(timezone.utc),
        )
        with self._session_factory() as session:
            session.add(document)
            session.commit()
        return document

    def get(self, document_id: str) -> Document | None:
        with self._session_factory() as session:
            return session.get(Document, document_id)

    def find_by_sha256(self, sha256: str) -> Document | None:
        """Duplicate detection: the same bytes are the same document."""
        with self._session_factory() as session:
            return session.scalars(select(Document).where(Document.sha256 == sha256)).first()

    def count(self) -> int:
        """How many documents are indexed. Used to refuse chat on an empty library."""
        with self._session_factory() as session:
            return int(session.scalar(select(func.count()).select_from(Document)) or 0)

    def get_many(self, document_ids: Iterable[str]) -> dict[str, Document]:
        """Fetch several documents in one query.

        The vector store only knows document ids, so turning a page of search
        hits into citable filenames would otherwise be one query per hit.
        """
        wanted = list(document_ids)
        if not wanted:
            return {}
        with self._session_factory() as session:
            rows: Sequence[Document] = (
                session.scalars(select(Document).where(Document.id.in_(wanted))).all()
            )
        return {document.id: document for document in rows}

    def list(self, *, limit: int, offset: int) -> tuple[list[Document], int]:
        """Newest first, with a deterministic tie-break for stable paging."""
        with self._session_factory() as session:
            total = session.scalar(select(func.count()).select_from(Document)) or 0
            rows: Sequence[Document] = (
                session.scalars(
                    select(Document)
                    .order_by(Document.created_at.desc(), Document.id)
                    .limit(limit)
                    .offset(offset)
                )
                .all()
            )
        return list(rows), int(total)

    def delete(self, document_id: str) -> bool:
        """Delete metadata. Returns False when the document did not exist."""
        with self._session_factory() as session:
            document = session.get(Document, document_id)
            if document is None:
                return False
            session.delete(document)
            session.commit()
        return True
