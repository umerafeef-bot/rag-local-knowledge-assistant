"""End-to-end document API tests: upload -> list -> get -> delete, plus errors.

These run the real HTTP stack, the real SQLite database, the real ChromaDB
store and the real PDF parser; only the embedding model and Ollama are
substituted (see tests/fakes.py).
"""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from tests.pdf_factory import build_encrypted_pdf, build_pdf

PAGES = [
    "Chapter one introduces momentum and gravity in a short paragraph.",
    "Chapter two explains entropy and the gradient descent method.",
    "Chapter three measures the latency of vector retrieval over a corpus.",
]

ONE_MEGABYTE = 1024 * 1024


@pytest.fixture
def pdf_bytes() -> bytes:
    return build_pdf(PAGES)


def upload(
    client: TestClient,
    content: bytes,
    *,
    filename: str = "notes.pdf",
    content_type: str = "application/pdf",
):  # type: ignore[no-untyped-def]
    return client.post(
        "/documents/upload",
        files={"file": (filename, content, content_type)},
    )


def container(client: TestClient):  # type: ignore[no-untyped-def]
    return client.app.state.container


# --- happy path ------------------------------------------------------------
def test_upload_indexes_a_pdf(client: TestClient, pdf_bytes: bytes, settings: Settings) -> None:
    response = upload(client, pdf_bytes)

    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True

    document = body["data"]["document"]
    assert document["filename"] == "notes.pdf"
    assert document["content_type"] == "application/pdf"
    assert document["page_count"] == 3
    assert document["chunk_count"] >= 3
    assert document["char_count"] > 0
    assert document["size_bytes"] == len(pdf_bytes)
    assert document["sha256"] == hashlib.sha256(pdf_bytes).hexdigest()
    assert document["embedding_model"] == "fake-bow-256"
    assert body["data"]["timings"]["total_seconds"] > 0

    # The stored file is named after the document id, never the user's name.
    stored = settings.upload_dir / f"{document['id']}.pdf"
    assert stored.exists()
    assert stored.read_bytes() == pdf_bytes
    assert [path.name for path in settings.upload_dir.iterdir()] == [f"{document['id']}.pdf"]

    # The staged file is gone, and the chunks really are in the vector store.
    assert list(settings.temp_dir.iterdir()) == []
    assert container(client).vectors.count_by_document(document["id"]) == document["chunk_count"]


def test_upload_response_never_contains_document_text(
    client: TestClient, pdf_bytes: bytes
) -> None:
    """Privacy: metadata only -- no extracted text leaves the index."""
    response = upload(client, pdf_bytes)

    assert "gravity" not in response.text
    assert "entropy" not in response.text


def test_upload_sanitises_a_traversing_filename(client: TestClient, pdf_bytes: bytes) -> None:
    response = upload(client, pdf_bytes, filename="../../etc/passwd.pdf")

    assert response.status_code == 201
    assert response.json()["data"]["document"]["filename"] == "passwd.pdf"


def test_list_and_get(client: TestClient, pdf_bytes: bytes) -> None:
    document_id = upload(client, pdf_bytes).json()["data"]["document"]["id"]

    listing_response = client.get("/documents")
    listing = listing_response.json()
    assert listing["success"] is True
    assert listing["data"]["total"] == 1
    assert listing["data"]["limit"] == 20
    assert listing["data"]["offset"] == 0
    assert listing["data"]["documents"][0]["id"] == document_id
    # Metadata only: the extracted text is never serialised back out.
    assert "gravity" not in listing_response.text

    detail = client.get(f"/documents/{document_id}")
    assert detail.status_code == 200
    assert detail.json()["data"]["id"] == document_id


def test_empty_library_is_an_empty_list_not_an_error(client: TestClient) -> None:
    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json()["data"] == {
        "documents": [],
        "total": 0,
        "limit": 20,
        "offset": 0,
    }


def test_duplicate_upload_is_rejected(client: TestClient, pdf_bytes: bytes) -> None:
    first = upload(client, pdf_bytes).json()["data"]["document"]

    response = upload(client, pdf_bytes, filename="same-file-again.pdf")

    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "duplicate_document"
    assert error["details"]["existing_document_id"] == first["id"]

    # No second row and no duplicate vectors.
    assert client.get("/documents").json()["data"]["total"] == 1
    assert container(client).vectors.total_chunks() == first["chunk_count"]


def test_delete_removes_file_vectors_and_metadata(
    client: TestClient, pdf_bytes: bytes, settings: Settings
) -> None:
    document = upload(client, pdf_bytes).json()["data"]["document"]
    stored = settings.upload_dir / f"{document['id']}.pdf"

    response = client.delete(f"/documents/{document['id']}")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["id"] == document["id"]
    assert data["deleted_vectors"] == document["chunk_count"]
    assert data["file_removed"] is True

    assert not stored.exists()
    assert container(client).vectors.total_chunks() == 0
    assert container(client).vectors.count_by_document(document["id"]) == 0
    assert client.get("/documents").json()["data"]["total"] == 0
    assert client.get(f"/documents/{document['id']}").status_code == 404


def test_deleting_one_document_leaves_the_other_searchable(
    client: TestClient, pdf_bytes: bytes
) -> None:
    first = upload(client, pdf_bytes).json()["data"]["document"]
    other_pdf = build_pdf(["A completely different page about stochastic gradient descent."])
    second = upload(client, other_pdf, filename="other.pdf").json()["data"]["document"]

    client.delete(f"/documents/{first['id']}")

    vectors = container(client).vectors
    assert vectors.count_by_document(first["id"]) == 0
    assert vectors.count_by_document(second["id"]) == second["chunk_count"]
    assert client.get("/documents").json()["data"]["total"] == 1


def test_search_returns_the_right_page(client: TestClient, pdf_bytes: bytes) -> None:
    """Storage and metadata round-trip: a query must find its own page."""
    upload(client, pdf_bytes)
    vectors = container(client).vectors
    query_vector = container(client).embeddings.embed_query("latency of vector retrieval")

    hits = vectors.search(query_vector, top_k=3)

    assert hits, "expected at least one hit"
    assert hits[0].page == 3
    # Chunk indices are global across pages (one chunk per page here), which is
    # what makes the stored ids unique within a document.
    assert hits[0].chunk_index == 2
    assert -1.0 <= hits[0].similarity <= 1.0
    assert hits[0].text.startswith("Chapter three")


def test_search_can_be_filtered_to_one_document(client: TestClient, pdf_bytes: bytes) -> None:
    first = upload(client, pdf_bytes).json()["data"]["document"]
    other_pdf = build_pdf(["Nothing relevant here at all."])
    second = upload(client, other_pdf, filename="other.pdf").json()["data"]["document"]

    vectors = container(client).vectors
    query_vector = container(client).embeddings.embed_query("vector retrieval latency")

    hits = vectors.search(query_vector, top_k=5, document_ids=[second["id"]])

    assert hits
    assert {hit.document_id for hit in hits} == {second["id"]}
    assert first["id"] not in {hit.document_id for hit in hits}


# --- error paths -----------------------------------------------------------
def test_unknown_document_returns_404(client: TestClient) -> None:
    response = client.get("/documents/deadbeef")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "document_not_found"
    assert response.json()["data"] is None


def test_deleting_an_unknown_document_returns_404(client: TestClient) -> None:
    response = client.delete("/documents/deadbeef")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "document_not_found"


def test_corrupted_pdf_is_rejected(client: TestClient) -> None:
    response = upload(client, b"%PDF-1.4\ntruncated body\n%%EOF\n")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "pdf_corrupted"
    assert client.get("/documents").json()["data"]["total"] == 0


def test_encrypted_pdf_is_rejected(client: TestClient) -> None:
    response = upload(client, build_encrypted_pdf(PAGES, password="secret"))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "pdf_encrypted"


def test_image_only_pdf_is_rejected(client: TestClient) -> None:
    response = upload(client, build_pdf(["", ""]))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "no_extractable_text"


def test_oversized_upload_is_rejected(client: TestClient, settings: Settings) -> None:
    payload = b"%PDF-1.4\n" + b"x" * (settings.max_upload_bytes + 16)

    response = upload(client, payload)

    assert response.status_code == 413
    error = response.json()["error"]
    assert error["code"] == "file_too_large"
    assert error["details"]["limit_bytes"] == settings.max_upload_bytes
    assert list(settings.temp_dir.iterdir()) == []


def test_non_pdf_extension_is_rejected(client: TestClient, pdf_bytes: bytes) -> None:
    response = upload(client, pdf_bytes, filename="notes.txt")

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_file_type"


def test_mislabelled_pdf_extension_is_rejected(client: TestClient) -> None:
    response = upload(client, b"I am plain text pretending to be a PDF")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "not_a_pdf"


def test_missing_file_field_uses_the_error_envelope(client: TestClient) -> None:
    response = client.post("/documents/upload", data={"file": "not-a-file"})

    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["details"]["fields"]


@pytest.mark.parametrize("query", ["limit=0", "limit=500", "offset=-1"])
def test_pagination_bounds_are_validated(client: TestClient, query: str) -> None:
    response = client.get(f"/documents?{query}")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
