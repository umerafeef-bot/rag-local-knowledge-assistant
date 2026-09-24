"""Request/response models and the single response envelope.

Every endpoint answers with the same shape::

    {"success": true,  "data": {...},  "error": null}
    {"success": false, "data": null,   "error": {"code": "...", "message": "..."}}

so a client needs exactly one branch to handle both cases.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

T = TypeVar("T")


class ErrorBody(BaseModel):
    code: str = Field(description="Stable, machine-readable error code.")
    message: str = Field(description="Human-readable explanation.")
    details: dict[str, Any] | None = Field(
        default=None, description="Optional structured context; never contains document text."
    )


class Envelope(BaseModel, Generic[T]):
    """The response shape shared by every endpoint."""

    success: bool
    data: T | None = None
    error: ErrorBody | None = None

    @classmethod
    def ok(cls, data: T) -> Envelope[T]:
        return cls(success=True, data=data)

    @classmethod
    def fail(
        cls, code: str, message: str, details: dict[str, Any] | None = None
    ) -> Envelope[T]:
        return cls(success=False, error=ErrorBody(code=code, message=message, details=details))


# --- documents -------------------------------------------------------------
class DocumentOut(BaseModel):
    """Document metadata. Never includes the document's text."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    page_count: int
    chunk_count: int
    char_count: int
    embedding_model: str
    created_at: datetime


class IngestionTimingsOut(BaseModel):
    upload_seconds: float
    extract_seconds: float
    chunk_seconds: float
    embed_seconds: float
    store_seconds: float
    total_seconds: float


class UploadData(BaseModel):
    document: DocumentOut
    timings: IngestionTimingsOut


class DocumentListData(BaseModel):
    documents: list[DocumentOut]
    total: int
    limit: int
    offset: int


class DeletedDocumentData(BaseModel):
    id: str
    deleted_vectors: int
    file_removed: bool


# --- chat ------------------------------------------------------------------
class ChatRequest(BaseModel):
    """One question, optionally restricted to a subset of the library."""

    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)] = Field(
        description="The question to answer from the indexed documents."
    )
    document_ids: list[str] | None = Field(
        default=None,
        max_length=50,
        description=(
            "Restrict retrieval to these documents. Omit or send an empty list "
            "to search the whole library."
        ),
    )


class ChatSourceOut(BaseModel):
    """A cited passage. A short excerpt, never the whole chunk."""

    document_id: str
    filename: str
    page: int
    chunk_index: int
    similarity: float = Field(description="Cosine similarity in [-1, 1]; higher is closer.")
    snippet: str


class ChatTimingsOut(BaseModel):
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float


class ChatData(BaseModel):
    answer: str
    grounded: bool = Field(
        description=(
            "True when the answer was generated from retrieved context. False when "
            "retrieval found nothing relevant: the answer is then this service's "
            "own refusal, not a model output."
        )
    )
    sources: list[ChatSourceOut]
    retrieved: int = Field(description="Chunks scoring above the similarity floor.")
    closest_similarity: float | None = Field(
        default=None,
        description=(
            "Best similarity any chunk scored before the floor was applied, so a "
            "refusal can be explained. Null when the search returned no neighbours "
            "at all, which is a different situation from 'all of them scored low'."
        ),
    )
    min_similarity: float = Field(
        default=0.0, description="The similarity floor in force for this request."
    )
    context_chunks: int = Field(description="Chunks actually sent to the model.")
    context_chars: int
    context_truncated: bool = Field(
        description="True when the context budget cut the retrieved context short."
    )
    prompt_tokens: int = Field(description="Estimated prompt size in tokens.")
    timings: ChatTimingsOut


# --- health ----------------------------------------------------------------
class ServiceStatus(BaseModel):
    ok: bool


class DatabaseStatus(BaseModel):
    ok: bool
    detail: str | None = None


class VectorStoreStatus(BaseModel):
    ok: bool
    chunks_indexed: int | None = None
    detail: str | None = None


class EmbeddingStatus(BaseModel):
    model: str
    loaded: bool
    dimension: int | None = None


class LLMStatus(BaseModel):
    provider: str
    model: str
    ok: bool
    reachable: bool
    model_available: bool
    detail: str | None = None
    available_models: list[str] = Field(default_factory=list)


class HealthData(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    api: ServiceStatus
    database: DatabaseStatus
    vector_store: VectorStoreStatus
    embeddings: EmbeddingStatus
    llm: LLMStatus


class AppInfo(BaseModel):
    name: str
    version: str
    docs_url: str
