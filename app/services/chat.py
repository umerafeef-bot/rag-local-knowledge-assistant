"""Grounded question answering: retrieve, build context, generate, cite.

The flow is deliberately linear and short enough to hold in your head:

    question -> RetrievalService.search -> build_context -> build_prompt
             -> LLMProvider.generate -> answer + sources

Three decisions are worth calling out.

**Retrieval decides whether to call the model at all.** If nothing scores above
the similarity floor, the service returns the refusal sentence itself and never
reaches the LLM. Asking a 3B model to answer from context that does not contain
the answer is the single most reliable way to get a confident fabrication, so
the call is skipped and the response is marked ``grounded: false``.

**An empty library is an error, not a refusal.** ``no_documents`` (409) tells
the client to offer an upload; a refusal reads as "your PDF does not cover
this", which would be wrong and unhelpful before anything is uploaded.

**No conversation history.** The MVP sends one self-contained prompt per
question. History costs tokens on a small model, and every extra turn is one
more opportunity for the model to answer from an earlier turn instead of from
the documents. It is a documented future improvement, not a hidden omission.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Sequence

from app.core.errors import NoDocumentsError, ValidationError
from app.llm.base import LLMProvider
from app.services.prompting import (
    INSUFFICIENT_CONTEXT_ANSWER,
    SYSTEM_PROMPT,
    build_context,
    build_prompt,
    estimate_prompt_tokens,
    prompt_fits_window,
)
from app.services.retrieval import RetrievalService
from app.utils.text_cleaning import make_snippet

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChatSource:
    """A cited passage: where it came from, how well it matched, what it said."""

    document_id: str
    filename: str
    page: int
    chunk_index: int
    similarity: float
    snippet: str


@dataclass(frozen=True, slots=True)
class ChatTimings:
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float

    def as_dict(self) -> dict[str, float]:
        return {
            "retrieval_seconds": self.retrieval_seconds,
            "generation_seconds": self.generation_seconds,
            "total_seconds": self.total_seconds,
        }


@dataclass(frozen=True, slots=True)
class ChatAnswer:
    answer: str
    #: True when ``answer`` was generated from retrieved document context.
    #: False means the app refused because retrieval found nothing -- the
    #: answer text is then this application's own sentence, not a model output.
    grounded: bool
    sources: tuple[ChatSource, ...]
    #: Chunks retrieved above the similarity floor.
    retrieved: int
    #: Chunks actually pasted into the prompt (the budget may have cut the tail).
    context_chunks: int
    context_chars: int
    context_truncated: bool
    prompt_tokens: int
    timings: ChatTimings


class ChatService:
    def __init__(
        self,
        *,
        retrieval: RetrievalService,
        llm: LLMProvider,
        max_context_chars: int,
        snippet_chars: int,
        temperature: float,
        max_tokens: int,
        context_window: int,
    ) -> None:
        self._retrieval = retrieval
        self._llm = llm
        self._max_context_chars = max_context_chars
        self._snippet_chars = snippet_chars
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._context_window = context_window

    def answer(self, question: str, *, document_ids: Sequence[str] | None = None) -> ChatAnswer:
        """Answer ``question`` from the indexed documents.

        Raises :class:`ValidationError` for a blank question,
        :class:`NoDocumentsError` when the library is empty,
        :class:`NotFoundError` when a filter names an unknown document, and
        :class:`ServiceUnavailableError` when the local model cannot answer.
        """
        started = time.perf_counter()
        question = question.strip()
        if not question:
            raise ValidationError("The question must not be empty.")

        if self._retrieval.document_count() == 0:
            raise NoDocumentsError(
                "No documents have been indexed yet, so there is nothing to answer from. "
                "Upload a PDF first."
            )

        retrieval_started = time.perf_counter()
        hits = self._retrieval.search(question, document_ids=document_ids)
        retrieval_seconds = time.perf_counter() - retrieval_started

        if not hits:
            # Nothing relevant: refuse here instead of asking the model to
            # invent an answer from irrelevant context.
            logger.info("No chunk passed the similarity floor; refusing without calling the model")
            return ChatAnswer(
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                grounded=False,
                sources=(),
                retrieved=0,
                context_chunks=0,
                context_chars=0,
                context_truncated=False,
                prompt_tokens=0,
                timings=ChatTimings(
                    retrieval_seconds=round(retrieval_seconds, 4),
                    generation_seconds=0.0,
                    total_seconds=round(time.perf_counter() - started, 4),
                ),
            )

        context = build_context(hits, max_chars=self._max_context_chars)
        prompt = build_prompt(question, context.text)
        prompt_tokens = estimate_prompt_tokens(prompt)

        if not prompt_fits_window(prompt, context_window=self._context_window):
            # Worth a warning, not a refusal: Ollama would truncate the prompt,
            # so a short or partial answer here is expected rather than a bug.
            logger.warning(
                "Prompt is ~%d tokens against a %d-token window; the model may truncate it. "
                "Lower MAX_CONTEXT_CHARS.",
                prompt_tokens,
                self._context_window,
            )

        generation_started = time.perf_counter()
        answer = self._llm.generate(
            prompt,
            system=SYSTEM_PROMPT,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        generation_seconds = time.perf_counter() - generation_started

        sources = tuple(
            ChatSource(
                document_id=chunk.document_id,
                filename=chunk.filename,
                page=chunk.page,
                chunk_index=chunk.chunk_index,
                similarity=round(chunk.similarity, 4),
                snippet=make_snippet(chunk.text, self._snippet_chars),
            )
            for chunk in context.used
        )

        logger.info(
            "Answered from %d chunks (%d context chars) in %.2fs",
            len(sources),
            context.chars,
            time.perf_counter() - started,
        )
        return ChatAnswer(
            answer=answer,
            grounded=True,
            sources=sources,
            retrieved=len(hits),
            context_chunks=len(context.used),
            context_chars=context.chars,
            context_truncated=context.truncated,
            prompt_tokens=prompt_tokens,
            timings=ChatTimings(
                retrieval_seconds=round(retrieval_seconds, 4),
                generation_seconds=round(generation_seconds, 4),
                total_seconds=round(time.perf_counter() - started, 4),
            ),
        )
