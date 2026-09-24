"""GET /health and the shared response envelope."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.repositories.vector_repository import VectorStoreHealth
from tests.fakes import FakeEmbedding, FakeLLMProvider


def build_client(
    settings: Settings,
    *,
    llm=None,
    embeddings=None,
) -> TestClient:  # type: ignore[no-untyped-def]
    app = create_app(
        settings=settings,
        embeddings=embeddings or FakeEmbedding(),
        llm=llm or FakeLLMProvider(),
    )
    return TestClient(app)


def test_health_reports_every_component(settings: Settings) -> None:
    with build_client(settings) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["error"] is None

    data = body["data"]
    assert data["status"] == "ok"
    assert data["version"] == settings.app_version
    assert data["api"]["ok"] is True
    assert data["database"]["ok"] is True
    assert data["vector_store"] == {"ok": True, "chunks_indexed": 0, "detail": None}
    assert data["embeddings"]["loaded"] is True
    assert data["embeddings"]["model"] == FakeEmbedding.model_name
    assert data["llm"]["provider"] == "fake"
    assert data["llm"]["reachable"] is True
    assert data["llm"]["model_available"] is True


def test_health_is_degraded_when_ollama_is_down(settings: Settings) -> None:
    llm = FakeLLMProvider(reachable=False, model_available=False)
    with build_client(settings, llm=llm) as client:
        response = client.get("/health")

    # The app stays usable for uploads and search, so this is not an HTTP error.
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "degraded"
    assert data["llm"]["ok"] is False
    assert data["llm"]["reachable"] is False
    assert data["llm"]["detail"] is not None


def test_health_reports_a_database_failure(settings: Settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with build_client(settings) as client:
        def explode() -> None:
            raise RuntimeError("database is gone")

        monkeypatch.setattr(client.app.state.container, "check_database", explode)
        response = client.get("/health")

    data = response.json()["data"]
    assert data["status"] == "degraded"
    assert data["database"]["ok"] is False
    assert data["database"]["detail"] is not None
    # The internal error message must not leak through.
    assert "database is gone" not in response.text


def test_health_reports_a_vector_store_failure(settings: Settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with build_client(settings) as client:
        monkeypatch.setattr(
            client.app.state.container.vectors,
            "health",
            lambda: VectorStoreHealth(ok=False, detail="The vector store could not be read."),
        )
        response = client.get("/health")

    data = response.json()["data"]
    assert data["status"] == "degraded"
    assert data["vector_store"]["ok"] is False
    assert data["vector_store"]["chunks_indexed"] is None


def test_service_descriptor_is_served_at_info(settings: Settings) -> None:
    with build_client(settings) as client:
        response = client.get("/info")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["name"] == settings.app_name
    assert data["docs_url"] == "/docs"


def test_unknown_routes_use_the_error_envelope(settings: Settings) -> None:
    with build_client(settings) as client:
        response = client.get("/nope")

    assert response.status_code == 404
    body = response.json()
    assert body["success"] is False
    assert body["data"] is None
    assert body["error"]["code"] == "http_error"


def test_openapi_schema_builds(settings: Settings) -> None:
    """Guards the generic Envelope[...] response models."""
    with build_client(settings) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert {"/health", "/documents", "/documents/upload"} <= set(paths)
