"""Application factory.

Run it with::

    .venv/Scripts/python -m uvicorn --factory app.main:create_app --reload

A factory (rather than a module-level ``app`` object) keeps importing this
module free of side effects -- tests import ``create_app`` and build an app
against a temporary data directory without touching the real one.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError as PydanticValidationError

from app.api.errors import register_exception_handlers
from app.api.routes import chat as chat_routes
from app.api.routes import documents as documents_routes
from app.api.routes import health as health_routes
from app.api.schemas import AppInfo, Envelope
from app.container import Container
from app.core.config import Settings
from app.core.logging import configure_logging
from app.db.base import init_schema
from app.llm.base import LLMProvider
from app.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)

#: Ships with the repository; no build step and no external assets.
STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(
    settings: Settings | None = None,
    *,
    embeddings: EmbeddingService | None = None,
    llm: LLMProvider | None = None,
) -> FastAPI:
    """Build the application. ``embeddings``/``llm`` exist for tests."""
    try:
        settings = settings or Settings()
    except PydanticValidationError as exc:
        # Missing or malformed environment configuration: report it plainly
        # instead of dying inside a stack trace.
        raise SystemExit(
            "Invalid configuration. Check your .env file against .env.example.\n"
            f"{exc}"
        ) from exc

    configure_logging(settings.log_level)
    # huggingface_hub sends telemetry unless told not to.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    settings.ensure_directories()
    settings.reset_temp_dir()

    container = Container.build(settings, embeddings=embeddings, llm=llm)
    init_schema(container.engine)

    def warm_up_embeddings() -> None:
        """Pay the embedding model's one-time initialisation off the hot path."""
        try:
            started = time.perf_counter()
            container.embeddings.embed_query("warm up")
            logger.info("Embedding model warm (%.1fs)", time.perf_counter() - started)
        except Exception:
            # A missing or unloadable model must not stop the API from serving
            # document management requests; the failure surfaces on /health.
            logger.warning("Embedding warm-up failed; the model will load on first use", exc_info=True)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "%s %s ready | data=%s | llm=%s",
            settings.app_name,
            settings.app_version,
            settings.data_dir,
            settings.ollama_model or "<not configured>",
        )
        if settings.embedding_warmup:
            threading.Thread(target=warm_up_embeddings, name="embedding-warmup", daemon=True).start()
        try:
            yield
        finally:
            container.close()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Local, privacy-preserving question answering over your own PDFs. "
            "Extraction, embeddings, vector search and generation all run on this machine."
        ),
        lifespan=lifespan,
    )
    app.state.container = container

    # CORS is closed by default: with ALLOWED_ORIGINS empty, no other origin can
    # read these responses from a browser. The UI is same-origin (served from
    # this process), so it needs no CORS at all -- the setting exists so a
    # developer running a separate front-end dev server can open exactly the
    # origin they name, rather than widening to `*`.
    origins = settings.cors_origins
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type"],
            # No cookies or Authorization headers are used, so credentials are
            # never allowed: an allowed origin could not ride an existing session.
            allow_credentials=False,
        )
        logger.info("CORS enabled for %d configured origin(s)", len(origins))

    @app.middleware("http")
    async def revalidate_ui_assets(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Make browsers revalidate the UI instead of reusing a stale copy.

        ``StaticFiles`` sends ``ETag`` and ``Last-Modified`` but no
        ``Cache-Control``, which lets a browser reuse a cached ``app.js``
        without asking -- so an updated UI can keep running the old JavaScript
        until the user happens to hard-refresh. ``no-cache`` means "revalidate
        before reuse": unchanged files still answer ``304`` and cost nothing,
        changed ones are fetched immediately.
        """
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    register_exception_handlers(app)
    app.include_router(health_routes.router)
    app.include_router(documents_routes.router)
    app.include_router(chat_routes.router)

    # Static assets (the whole UI is these three files).
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False, summary="Web UI")
    def index() -> FileResponse:
        """Serve the single-page UI. The JSON descriptor lives at /info."""
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/info", response_model=Envelope[AppInfo], tags=["system"], summary="Service descriptor")
    def info() -> Envelope[AppInfo]:
        return Envelope.ok(
            AppInfo(name=settings.app_name, version=settings.app_version, docs_url=app.docs_url or "/docs")
        )

    return app
