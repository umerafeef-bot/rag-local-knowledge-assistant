"""Shared fixtures.

Every test gets its own temporary data directory, its own SQLite file and its
own ChromaDB store, so tests never see each other's state and never touch the
developer's real ``data/`` folder.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.db.base import create_database, init_schema
from app.main import create_app
from tests.fakes import FakeEmbedding, FakeLLMProvider, make_upload


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Small, fast configuration: 1 MB cap, 200-character chunks.

    The retrieval and generation knobs are pinned here so a test never depends
    on a developer's ``.env``: a different similarity floor or context budget
    would change which chunks are retrieved and whether the context is cut.
    """
    return Settings(
        data_dir=tmp_path / "data",
        max_upload_mb=1,
        max_pdf_pages=50,
        min_extractable_chars=20,
        chunk_size=200,
        chunk_overlap=40,
        top_k=3,
        retrieval_min_similarity=0.2,
        max_context_chars=2000,
        source_snippet_chars=80,
        chat_temperature=0.0,
        chat_max_tokens=128,
        llm_context_window=2048,
        ollama_model="fake-model",
        _env_file=None,  # ignore any developer .env
    )


@pytest.fixture
def embeddings() -> FakeEmbedding:
    return FakeEmbedding()


@pytest.fixture
def llm() -> FakeLLMProvider:
    return FakeLLMProvider()


@pytest.fixture
def app(settings: Settings, embeddings: FakeEmbedding, llm: FakeLLMProvider):
    return create_app(settings=settings, embeddings=embeddings, llm=llm)


@pytest.fixture
def client(app):  # type: ignore[no-untyped-def]
    # The context manager runs the lifespan, which creates the schema.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def session_factory(settings: Settings):
    engine, factory = create_database(settings)
    settings.ensure_directories()
    init_schema(engine)
    yield factory
    engine.dispose()


@pytest.fixture
def upload_factory():  # type: ignore[no-untyped-def]
    """Expose the UploadFile builder to tests that want it as a fixture."""
    return make_upload
