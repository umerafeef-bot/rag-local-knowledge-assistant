"""Document metadata repository: create, lookup, paging, delete, uniqueness."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Document
from app.repositories.document_repository import DocumentRepository


def create(repo: DocumentRepository, index: int) -> Document:
    return repo.create(
        document_id=f"{index:032x}",
        filename=f"doc-{index}.pdf",
        content_type="application/pdf",
        size_bytes=1_000 + index,
        sha256=f"{index:064x}",
        page_count=index,
        chunk_count=index * 2,
        char_count=index * 100,
        embedding_model="test-model",
    )


def test_create_and_get(session_factory: sessionmaker[Session]) -> None:
    repo = DocumentRepository(session_factory)
    created = create(repo, 1)

    fetched = repo.get(created.id)
    assert fetched is not None
    assert fetched.filename == "doc-1.pdf"
    assert fetched.chunk_count == 2
    assert fetched.created_at is not None


def test_get_returns_none_for_unknown_ids(session_factory: sessionmaker[Session]) -> None:
    repo = DocumentRepository(session_factory)
    assert repo.get("does-not-exist") is None


def test_find_by_sha256(session_factory: sessionmaker[Session]) -> None:
    repo = DocumentRepository(session_factory)
    created = create(repo, 2)

    assert repo.find_by_sha256(created.sha256) is not None
    assert repo.find_by_sha256("f" * 64) is None


def test_sha256_is_unique_at_the_database_level(session_factory: sessionmaker[Session]) -> None:
    """The unique index is the real duplicate guard, not the earlier lookup."""
    repo = DocumentRepository(session_factory)
    create(repo, 3)

    with pytest.raises(IntegrityError):
        repo.create(
            document_id="c" * 32,
            filename="copy.pdf",
            content_type="application/pdf",
            size_bytes=10,
            sha256=f"{3:064x}",
            page_count=1,
            chunk_count=1,
            char_count=10,
            embedding_model="test-model",
        )


def test_list_paginates(session_factory: sessionmaker[Session]) -> None:
    repo = DocumentRepository(session_factory)
    created = [create(repo, index) for index in range(1, 4)]

    first_page, total = repo.list(limit=2, offset=0)
    second_page, _ = repo.list(limit=2, offset=2)

    assert total == 3
    assert len(first_page) == 2
    assert len(second_page) == 1
    assert {document.id for document in first_page + second_page} == {
        document.id for document in created
    }


def test_list_returns_newest_first(session_factory: sessionmaker[Session]) -> None:
    now = datetime.now(timezone.utc)
    with session_factory() as session:
        session.add_all(
            [
                Document(
                    id="a" * 32,
                    filename="old.pdf",
                    content_type="application/pdf",
                    sha256="1" * 64,
                    size_bytes=1,
                    page_count=1,
                    chunk_count=1,
                    char_count=1,
                    embedding_model="test-model",
                    created_at=now - timedelta(days=1),
                ),
                Document(
                    id="b" * 32,
                    filename="new.pdf",
                    content_type="application/pdf",
                    sha256="2" * 64,
                    size_bytes=1,
                    page_count=1,
                    chunk_count=1,
                    char_count=1,
                    embedding_model="test-model",
                    created_at=now,
                ),
            ]
        )
        session.commit()

    documents, total = DocumentRepository(session_factory).list(limit=10, offset=0)
    assert total == 2
    assert [document.filename for document in documents] == ["new.pdf", "old.pdf"]


def test_delete(session_factory: sessionmaker[Session]) -> None:
    repo = DocumentRepository(session_factory)
    created = create(repo, 4)

    assert repo.delete(created.id) is True
    assert repo.get(created.id) is None
    assert repo.delete(created.id) is False
