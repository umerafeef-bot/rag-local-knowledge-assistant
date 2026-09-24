"""Semantic retrieval: question -> ranked, citable chunks.

The vector store returns the nearest neighbours whether or not they are
relevant, because that is what a nearest-neighbour search does. Two pieces of
work turn that into something trustworthy:

* **A similarity floor.** Chunks scoring below ``RETRIEVAL_MIN_SIMILARITY`` are
  dropped. Without it, asking about a topic that is not in the library still
  returns five chunks, and the model is then asked to answer from context that
  does not contain the answer.
* **Filename resolution.** Chunk metadata deliberately stores only
  ``document_id``, so the display name lives in one place (SQLite). Hits are
  resolved to names in a single query, not one per hit.

The retrieval is intentionally a single dense pass: no reranking model, no
hybrid BM25. Those are listed as future work in the README, not half-built
here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Sequence

from app.core.errors import NotFoundError, ValidationError
from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_repository import VectorRepository
from app.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One retrieved chunk of a document, with everything a citation needs."""

    document_id: str
    filename: str
    page: int
    chunk_index: int
    similarity: float
    text: str


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    """What a search found, plus the best score it saw *before* the floor.

    ``best_similarity`` exists so that a refusal can be explained instead of
    arriving as an unexplained "I found nothing". On a small or oddly worded
    corpus, the honest answer to "why did it refuse?" is often "the closest
    passage scored 0.28 against a floor of 0.30", and the caller cannot work
    that out from an empty list. ``None`` means the vector store returned no
    neighbours at all -- a different problem from "they all scored too low".
    """

    chunks: list[RetrievedChunk]
    best_similarity: float | None


class RetrievalService:
    def __init__(
        self,
        *,
        vectors: VectorRepository,
        documents: DocumentRepository,
        embeddings: EmbeddingService,
        top_k: int,
        min_similarity: float,
    ) -> None:
        self._vectors = vectors
        self._documents = documents
        self._embeddings = embeddings
        self._top_k = top_k
        self._min_similarity = min_similarity

    @property
    def min_similarity(self) -> float:
        return self._min_similarity

    def document_count(self) -> int:
        return self._documents.count()

    def search(
        self,
        question: str,
        *,
        document_ids: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return the relevant chunks for ``question``, best first.

        An empty list is a real answer: it means nothing in the library scored
        above the floor, so the caller must not guess. Callers that need to
        explain *why* it was empty should use :meth:`search_detailed`.
        """
        return self.search_detailed(question, document_ids=document_ids, top_k=top_k).chunks

    def search_detailed(
        self,
        question: str,
        *,
        document_ids: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> RetrievalOutcome:
        """Like :meth:`search`, but also report the best score seen.

        Raises :class:`ValidationError` for a blank question and
        :class:`NotFoundError` when a filter names a document that does not
        exist (silently returning nothing would look like an empty library).
        """
        question = question.strip()
        if not question:
            raise ValidationError("The question must not be empty.")

        if document_ids:
            self._assert_documents_exist(document_ids)

        limit = top_k if top_k is not None else self._top_k
        started = time.perf_counter()
        # The question is embedded with the same model that embedded the
        # chunks -- mixing models would put the vectors in different spaces.
        hits = self._vectors.search(
            self._embeddings.embed_query(question),
            top_k=limit,
            document_ids=document_ids,
        )
        relevant = [hit for hit in hits if hit.similarity >= self._min_similarity]

        filenames = self._documents.get_many({hit.document_id for hit in relevant})
        chunks: list[RetrievedChunk] = []
        for hit in relevant:
            document = filenames.get(hit.document_id)
            if document is None:
                # The document was deleted between the vector lookup and the
                # metadata lookup. Dropping the hit keeps its text out of an
                # answer that could no longer cite it.
                continue
            chunks.append(
                RetrievedChunk(
                    document_id=hit.document_id,
                    filename=document.filename,
                    page=hit.page,
                    chunk_index=hit.chunk_index,
                    similarity=hit.similarity,
                    text=hit.text,
                )
            )

        # Counts and scores only: never the question, never the text.
        logger.info(
            "Retrieved %d/%d chunks above %.2f in %.3fs",
            len(chunks),
            len(hits),
            self._min_similarity,
            time.perf_counter() - started,
        )
        return RetrievalOutcome(
            chunks=chunks,
            best_similarity=max((hit.similarity for hit in hits), default=None),
        )

    def _assert_documents_exist(self, document_ids: Sequence[str]) -> None:
        found = self._documents.get_many(document_ids)
        missing = [document_id for document_id in document_ids if document_id not in found]
        if missing:
            raise NotFoundError(
                f"No document exists with id '{missing[0]}'.",
            )
