"""Composition root.

Everything the app needs is built once, in one place, and handed to the routes
through ``app.state``. Tests build a container with substitute embeddings and a
fake LLM by passing them to :func:`app.main.create_app` -- no globals, no
monkeypatching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.base import create_database, ping_database
from app.llm.base import LLMProvider
from app.llm.ollama_provider import OllamaProvider
from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_repository import VectorRepository
from app.services.chat import ChatService
from app.services.documents import DocumentService
from app.services.embedding import EmbeddingService, SentenceTransformerEmbedding
from app.services.pdf_extraction import PdfExtractor
from app.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    documents: DocumentRepository
    vectors: VectorRepository
    embeddings: EmbeddingService
    extractor: PdfExtractor
    document_service: DocumentService
    retrieval: RetrievalService
    llm: LLMProvider
    chat: ChatService

    @classmethod
    def build(
        cls,
        settings: Settings,
        *,
        embeddings: EmbeddingService | None = None,
        llm: LLMProvider | None = None,
    ) -> Container:
        engine, session_factory = create_database(settings)
        documents = DocumentRepository(session_factory)
        vectors = VectorRepository(
            persist_dir=settings.chroma_dir,
            collection_name=settings.collection_name,
        )
        embedding_service = embeddings or SentenceTransformerEmbedding(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )
        extractor = PdfExtractor(
            max_pages=settings.max_pdf_pages,
            min_chars=settings.min_extractable_chars,
        )
        llm_provider = llm or OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
            health_timeout_seconds=settings.ollama_health_timeout_seconds,
            context_window=settings.llm_context_window,
            keep_alive=settings.llm_keep_alive,
        )
        retrieval = RetrievalService(
            vectors=vectors,
            documents=documents,
            embeddings=embedding_service,
            top_k=settings.top_k,
            min_similarity=settings.retrieval_min_similarity,
        )
        return cls(
            settings=settings,
            engine=engine,
            session_factory=session_factory,
            documents=documents,
            vectors=vectors,
            embeddings=embedding_service,
            extractor=extractor,
            document_service=DocumentService(
                settings=settings,
                documents=documents,
                vectors=vectors,
                embeddings=embedding_service,
                extractor=extractor,
            ),
            retrieval=retrieval,
            llm=llm_provider,
            chat=ChatService(
                retrieval=retrieval,
                llm=llm_provider,
                max_context_chars=settings.max_context_chars,
                snippet_chars=settings.source_snippet_chars,
                temperature=settings.chat_temperature,
                max_tokens=settings.chat_max_tokens,
                context_window=settings.llm_context_window,
            ),
        )

    def check_database(self) -> None:
        """Raise if SQLite is not answering."""
        ping_database(self.engine)

    def close(self) -> None:
        self.llm.close()
        self.engine.dispose()
