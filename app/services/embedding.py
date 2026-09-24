"""Local text embeddings.

The application depends on the :class:`EmbeddingService` Protocol, not on
sentence-transformers, so tests (and any future provider) can substitute their
own implementation. The real implementation loads its model lazily: importing
the app must not pull torch into memory, and unit tests must not need it.

Vectors are L2-normalised, which makes the cosine similarity that ChromaDB
computes numerically cheap and keeps scores in a predictable [-1, 1] range.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Protocol, Sequence

from app.core.errors import ValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EmbeddingHealth:
    model: str
    loaded: bool
    dimension: int | None


class EmbeddingService(Protocol):
    """What the rest of the app needs from an embedding backend."""

    @property
    def model_name(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def health(self) -> EmbeddingHealth: ...


class SentenceTransformerEmbedding:
    """sentence-transformers running on CPU (no GPU or API required)."""

    def __init__(self, *, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None
        self._load_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def _loaded_model(self):  # type: ignore[no-untyped-def]
        """Load the model once, thread-safely, on first use."""
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s (device=%s)", self._model_name, self._device)
                    self._model = SentenceTransformer(self._model_name, device=self._device)
                    logger.info("Embedding model ready (dimension=%s)", self.dimension)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._loaded_model.get_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._loaded_model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValidationError("The question must not be empty.")
        return self.embed_documents([text])[0]

    def health(self) -> EmbeddingHealth:
        # Intentionally does not load the model: /health must stay fast.
        model = self._model
        return EmbeddingHealth(
            model=self._model_name,
            loaded=model is not None,
            dimension=int(model.get_embedding_dimension()) if model is not None else None,
        )
