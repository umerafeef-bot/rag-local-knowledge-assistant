"""Vector storage on top of a persistent local ChromaDB.

Privacy note: ChromaDB collects anonymised telemetry by default. It is switched
off explicitly here -- an app whose selling point is "nothing leaves your
machine" must not phone home.

Chunk metadata stays minimal (``document_id``, ``page``, ``chunk_index``);
the document's display name is looked up in SQLite, so a name is stored in
exactly one place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.utils.chunking import Chunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One retrieved chunk, ready to be turned into a citation."""

    text: str
    document_id: str
    page: int
    chunk_index: int
    similarity: float


@dataclass(frozen=True, slots=True)
class VectorStoreHealth:
    ok: bool
    chunks_indexed: int | None = None
    detail: str | None = None


class VectorRepository:
    def __init__(self, *, persist_dir: Path, collection_name: str) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            # Embeddings are unit vectors, so cosine is the right metric.
            metadata={"hnsw:space": "cosine"},
        )

    # --- writes ------------------------------------------------------------
    @staticmethod
    def _chunk_id(document_id: str, chunk_index: int) -> str:
        """Deterministic id, which keeps re-ingestion idempotent."""
        return f"{document_id}:{chunk_index}"

    def add_chunks(
        self,
        *,
        document_id: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        self._collection.add(
            ids=[self._chunk_id(document_id, chunk.index) for chunk in chunks],
            embeddings=[list(vector) for vector in embeddings],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "document_id": document_id,
                    "page": chunk.page,
                    "chunk_index": chunk.index,
                }
                for chunk in chunks
            ],
        )

    def delete_document(self, document_id: str) -> int:
        """Remove every chunk of a document. Returns how many were removed."""
        removed = self.count_by_document(document_id)
        if removed:
            self._collection.delete(where={"document_id": document_id})
        return removed

    # --- reads -------------------------------------------------------------
    def count_by_document(self, document_id: str) -> int:
        stored = self._collection.get(where={"document_id": document_id}, include=[])
        return len(stored.get("ids") or [])

    def total_chunks(self) -> int:
        return int(self._collection.count())

    def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[SearchHit]:
        """Nearest chunks by cosine similarity, optionally within a subset."""
        where: dict[str, Any] | None = None
        if document_ids:
            where = (
                {"document_id": document_ids[0]}
                if len(document_ids) == 1
                else {"document_id": {"$in": list(document_ids)}}
            )

        result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        texts = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[SearchHit] = []
        for text, metadata, distance in zip(texts, metadatas, distances):
            if not text or not metadata:
                continue
            hits.append(
                SearchHit(
                    text=text,
                    document_id=str(metadata["document_id"]),
                    page=int(metadata["page"]),
                    chunk_index=int(metadata["chunk_index"]),
                    # Cosine distance -> similarity, so the API reports a score
                    # where 1.0 means "identical".
                    similarity=1.0 - float(distance),
                )
            )
        return hits

    # --- health ------------------------------------------------------------
    def health(self) -> VectorStoreHealth:
        try:
            return VectorStoreHealth(ok=True, chunks_indexed=self.total_chunks())
        except Exception:
            logger.exception("Vector store health check failed")
            return VectorStoreHealth(ok=False, detail="The vector store could not be read.")
