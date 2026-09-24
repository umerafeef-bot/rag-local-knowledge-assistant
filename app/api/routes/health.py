"""GET /health -- one report covering every local dependency."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.api.deps import ContainerDep
from app.api.schemas import (
    DatabaseStatus,
    EmbeddingStatus,
    Envelope,
    HealthData,
    LLMStatus,
    ServiceStatus,
    VectorStoreStatus,
)
from app.container import Container

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


def _database_status(container: Container) -> DatabaseStatus:
    try:
        container.check_database()
    except Exception:
        logger.exception("Database health check failed")
        return DatabaseStatus(ok=False, detail="The database could not be read.")
    return DatabaseStatus(ok=True)


def build_health(container: Container) -> HealthData:
    """Probe every dependency. Never raises."""
    database = _database_status(container)
    vector_store = container.vectors.health()
    embeddings = container.embeddings.health()
    llm = container.llm.health()

    return HealthData(
        status="ok" if (database.ok and vector_store.ok and llm.ok) else "degraded",
        version=container.settings.app_version,
        api=ServiceStatus(ok=True),
        database=database,
        vector_store=VectorStoreStatus(
            ok=vector_store.ok,
            chunks_indexed=vector_store.chunks_indexed,
            detail=vector_store.detail,
        ),
        embeddings=EmbeddingStatus(
            model=embeddings.model,
            loaded=embeddings.loaded,
            dimension=embeddings.dimension,
        ),
        llm=LLMStatus(
            provider=llm.provider,
            model=llm.model,
            ok=llm.ok,
            reachable=llm.reachable,
            model_available=llm.model_available,
            detail=llm.detail,
            available_models=list(llm.available_models),
        ),
    )


@router.get(
    "/health",
    response_model=Envelope[HealthData],
    summary="Status of the API and its local dependencies",
)
def health(container: ContainerDep) -> Envelope[HealthData]:
    """Report API, database, vector store, embedding model and LLM status.

    Always answers ``200``: the app remains useful for uploading and indexing
    while Ollama is down, so an unavailable dependency is reported as
    ``"degraded"`` rather than as an HTTP failure.
    """
    return Envelope.ok(build_health(container))
