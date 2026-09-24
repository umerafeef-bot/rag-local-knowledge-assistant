"""Document endpoints: upload, list, detail, delete.

The upload handler is a plain ``def`` on purpose: ingestion is CPU-bound
(PDF parsing plus embedding), so FastAPI should run it in its worker thread pool
instead of blocking the event loop.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Query, UploadFile, status

from app.api.deps import ContainerDep
from app.api.schemas import (
    DeletedDocumentData,
    DocumentListData,
    DocumentOut,
    Envelope,
    IngestionTimingsOut,
    UploadData,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=Envelope[UploadData],
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a PDF",
    responses={
        409: {"description": "The same file is already in the library"},
        413: {"description": "File larger than MAX_UPLOAD_MB"},
        415: {"description": "Not a .pdf file"},
        422: {"description": "Corrupted, encrypted, empty or text-free PDF"},
    },
)
def upload_document(
    container: ContainerDep,
    file: Annotated[UploadFile, File(description="The PDF document to index.")],
) -> Envelope[UploadData]:
    """Validate, extract, clean, chunk, embed and store a PDF."""
    result = container.document_service.ingest(file)
    return Envelope.ok(
        UploadData(
            document=DocumentOut.model_validate(result.document),
            timings=IngestionTimingsOut(**result.timings.as_dict()),
        )
    )


@router.get(
    "",
    response_model=Envelope[DocumentListData],
    summary="List indexed documents",
)
def list_documents(
    container: ContainerDep,
    limit: Annotated[int, Query(ge=1, le=100, description="Page size.")] = 20,
    offset: Annotated[int, Query(ge=0, description="Documents to skip.")] = 0,
) -> Envelope[DocumentListData]:
    documents, total = container.document_service.list(limit=limit, offset=offset)
    return Envelope.ok(
        DocumentListData(
            documents=[DocumentOut.model_validate(document) for document in documents],
            total=total,
            limit=limit,
            offset=offset,
        )
    )


@router.get(
    "/{document_id}",
    response_model=Envelope[DocumentOut],
    summary="Get one document's metadata",
    responses={404: {"description": "No such document"}},
)
def get_document(container: ContainerDep, document_id: str) -> Envelope[DocumentOut]:
    return Envelope.ok(DocumentOut.model_validate(container.document_service.get(document_id)))


@router.delete(
    "/{document_id}",
    response_model=Envelope[DeletedDocumentData],
    summary="Delete a document and its vectors",
    responses={404: {"description": "No such document"}},
)
def delete_document(container: ContainerDep, document_id: str) -> Envelope[DeletedDocumentData]:
    """Remove the chunks from the vector store, the file and the metadata row."""
    result = container.document_service.delete(document_id)
    return Envelope.ok(
        DeletedDocumentData(
            id=result.document_id,
            deleted_vectors=result.deleted_vectors,
            file_removed=result.file_removed,
        )
    )
