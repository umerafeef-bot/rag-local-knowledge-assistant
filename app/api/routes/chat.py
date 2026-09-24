"""POST /chat -- grounded question answering over the indexed documents.

A plain ``def`` handler on purpose. Everything underneath is blocking: the
question is embedded on CPU, ChromaDB is queried in-process and Ollama is
called over a synchronous HTTP client. Running it in FastAPI's worker thread
pool keeps the event loop free to serve other requests while a model thinks.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import ContainerDep
from app.api.schemas import ChatData, ChatRequest, ChatSourceOut, ChatTimingsOut, Envelope

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=Envelope[ChatData],
    summary="Ask a question about your documents",
    responses={
        409: {
            "description": (
                "No documents are indexed yet (`no_documents`). Distinct from a "
                "grounded refusal, which is a 200 with `grounded: false`."
            )
        },
        404: {"description": "A document filter named an unknown document (`document_not_found`)"},
        422: {"description": "Empty or oversized question (`validation_error`)"},
        503: {
            "description": (
                "The local model cannot answer (`llm_unavailable`): Ollama is not "
                "running, the model is not pulled, or it timed out."
            )
        },
    },
)
def chat(container: ContainerDep, payload: ChatRequest) -> Envelope[ChatData]:
    """Embed the question, retrieve the best chunks, answer from them, cite them.

    Returns ``200`` with the shared envelope. When nothing in the library is
    relevant the answer is a refusal with ``grounded: false`` and no sources --
    the model is not called at all in that case, so there is nothing to
    hallucinate from.
    """
    # An empty filter list means "search everything", which is also the default.
    result = container.chat.answer(payload.question, document_ids=payload.document_ids or None)

    return Envelope.ok(
        ChatData(
            answer=result.answer,
            grounded=result.grounded,
            sources=[
                ChatSourceOut(
                    document_id=source.document_id,
                    filename=source.filename,
                    page=source.page,
                    chunk_index=source.chunk_index,
                    similarity=source.similarity,
                    snippet=source.snippet,
                )
                for source in result.sources
            ],
            retrieved=result.retrieved,
            closest_similarity=result.closest_similarity,
            min_similarity=result.min_similarity,
            context_chunks=result.context_chunks,
            context_chars=result.context_chars,
            context_truncated=result.context_truncated,
            prompt_tokens=result.prompt_tokens,
            timings=ChatTimingsOut(**result.timings.as_dict()),
        )
    )
