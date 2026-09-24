"""Test doubles.

The default test run must be offline, fast and independent of torch, so the
real embedding model and a real Ollama server are replaced by deterministic
stand-ins. Both satisfy the Protocols the application depends on.
"""

from __future__ import annotations

import hashlib
import io
import math
import re

from starlette.datastructures import Headers
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.core.errors import ServiceUnavailableError, ValidationError
from app.llm.base import LLMHealth
from app.services.embedding import EmbeddingHealth

_TOKEN = re.compile(r"[a-z0-9]+")


def make_upload(
    content: bytes,
    *,
    filename: str = "document.pdf",
    content_type: str = "application/pdf",
) -> StarletteUploadFile:
    """Build an UploadFile without going through HTTP."""
    return StarletteUploadFile(
        io.BytesIO(content),
        filename=filename,
        size=len(content),
        headers=Headers({"content-type": content_type}),
    )


class FakeEmbedding:
    """Bag-of-words hashing embedding.

    Not a semantic model, but token overlap makes cosine similarity behave
    sensibly enough to verify that the right chunk is stored and retrieved, and
    it is fully deterministic. ``dimension`` is a class attribute to match the
    ``EmbeddingService`` Protocol's ``model_name`` property.
    """

    model_name = "fake-bow-256"

    def __init__(self, dimension: int = 256) -> None:
        self._dimension = dimension
        self.embedded_document_count = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            vector[int.from_bytes(digest, "big") % self._dimension] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_documents(self, texts):  # type: ignore[no-untyped-def]
        self.embedded_document_count += len(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValidationError("The question must not be empty.")
        return self._vector(text)

    def health(self) -> EmbeddingHealth:
        return EmbeddingHealth(model=self.model_name, loaded=True, dimension=self._dimension)


class FakeLLMProvider:
    """Canned LLM backend with a scriptable health report."""

    provider_name = "fake"

    def __init__(
        self,
        *,
        model: str = "fake-model",
        reachable: bool = True,
        model_available: bool = True,
        answer: str = "Grounded answer.",
        error: Exception | None = None,
    ) -> None:
        self._model = model
        self._reachable = reachable
        self._model_available = model_available
        self._answer = answer
        self._error = error
        self.closed = False
        self.last_prompt: str | None = None
        self.last_system: str | None = None

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        self.last_prompt = prompt
        self.last_system = system
        if self._error is not None:
            raise self._error
        return self._answer

    def health(self) -> LLMHealth:
        return LLMHealth(
            provider=self.provider_name,
            model=self._model,
            ok=self._reachable and self._model_available,
            reachable=self._reachable,
            model_available=self._model_available,
            detail=None if self._reachable else "Ollama is not reachable.",
            available_models=(self._model,) if self._model_available else (),
        )

    def close(self) -> None:
        self.closed = True


class UnavailableLLM(FakeLLMProvider):
    """A backend that is up but refuses to answer (used for error paths)."""

    def __init__(self, message: str = "The local model is unavailable.") -> None:
        super().__init__(error=ServiceUnavailableError(message))
