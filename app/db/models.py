"""ORM models.

Document text is deliberately *not* stored here: the SQLite database holds
metadata only, while chunk text lives in the local ChromaDB store. That keeps
the database small and keeps a single source of truth for vectors.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Document(Base):
    """One accepted PDF and the numbers produced by the ingestion pipeline."""

    __tablename__ = "documents"

    # Random UUID4 hex. The file on disk is named "<id>.pdf" -- the user's
    # original filename is kept only as a display label.
    id: Mapped[str] = mapped_column(String(32), primary_key=True)

    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)

    # SHA-256 of the raw bytes, used to reject duplicate uploads.
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)

    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Which embedding model produced the vectors. If EMBEDDING_MODEL changes,
    # stored vectors are no longer comparable and the document must be
    # re-ingested -- the README documents this.
    embedding_model: Mapped[str] = mapped_column(String(200), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    def __repr__(self) -> str:
        return (
            f"<Document id={self.id!r} pages={self.page_count} "
            f"chunks={self.chunk_count} bytes={self.size_bytes}>"
        )
