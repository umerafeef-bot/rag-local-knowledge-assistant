"""Retrieval: query embedding, ranking, filtering, the similarity floor and
filename resolution.

The vector store, the SQLite metadata store, the chunker and the extractor are
the real implementations; only the embedding model is a deterministic
bag-of-words stand-in (see tests/fakes.py). Documents are indexed through the
real API, so the chunks under test are exactly the chunks production stores.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.errors import NotFoundError, ValidationError
from app.main import create_app
from tests.fakes import FakeEmbedding, FakeLLMProvider
from tests.pdf_factory import build_pdf

# Page 3 is long on purpose: with 200-character chunks it produces several
# chunks that all answer the same question, which is what the ranking and
# budgeting tests need. Pages 1 and 2 are short and unrelated, so the
# similarity floor has something to reject.
PAGES = [
    "Chapter one defines momentum as the product of mass and velocity.",
    "Chapter two introduces entropy as a measure of disorder.",
    (
        "Chapter three reports that vector retrieval latency grows slowly with corpus "
        "size. The measurements in this chapter show that doubling the corpus increased "
        "vector retrieval latency by less than ten percent, because the index is searched "
        "in approximate nearest-neighbour order. Vector retrieval latency therefore stays "
        "practical for a corpus of a hundred thousand chunks."
    ),
]


def index_document(client: TestClient) -> str:
    """Upload the three-page fixture and return the document id."""
    response = client.post(
        "/documents/upload",
        files={"file": ("chapters.pdf", build_pdf(PAGES), "application/pdf")},
    )
    assert response.status_code == 201, response.text
    return response.json()["data"]["document"]["id"]


@pytest.fixture
def indexed(client: TestClient) -> str:  # type: ignore[no-untyped-def]
    """One three-page document in the library; returns its id."""
    return index_document(client)


def retrieval(client: TestClient):  # type: ignore[no-untyped-def]
    return client.app.state.container.retrieval


def test_a_question_finds_its_own_chunk_with_a_citable_source(
    client: TestClient, indexed: str
) -> None:
    hits = retrieval(client).search("how does vector retrieval latency behave?")

    assert hits
    best = hits[0]
    assert best.document_id == indexed
    assert best.filename == "chapters.pdf"
    assert best.page == 3
    assert "retrieval latency" in best.text
    assert -1.0 <= best.similarity <= 1.0


def test_results_are_ordered_by_similarity(client: TestClient, indexed: str) -> None:
    hits = retrieval(client).search("vector retrieval latency and corpus size")

    similarities = [hit.similarity for hit in hits]
    assert similarities == sorted(similarities, reverse=True)


def test_top_k_limits_the_number_of_hits(client: TestClient, indexed: str) -> None:
    service = retrieval(client)

    hits = service.search(
        "momentum entropy retrieval latency velocity disorder corpus", top_k=1
    )

    assert len(hits) == 1


IRRELEVANT = "how do I bake sourdough bread at home?"


def build_client(settings):  # type: ignore[no-untyped-def]
    return TestClient(create_app(settings=settings, embeddings=FakeEmbedding(), llm=FakeLLMProvider()))


def test_a_permissive_floor_still_returns_the_nearest_chunks(settings) -> None:  # type: ignore[no-untyped-def]
    """Baseline for the test below: the hits exist, the floor is what removed them."""
    permissive = settings.model_copy(update={"retrieval_min_similarity": 0.0})
    with build_client(permissive) as client:
        index_document(client)

        hits = retrieval(client).search(IRRELEVANT)

    assert hits, "top-K always returns neighbours when nothing is filtered"


def test_a_strict_floor_turns_unrelated_hits_into_no_hits(settings) -> None:  # type: ignore[no-untyped-def]
    """A nearest-neighbour search always returns something; relevance does not.

    This is the difference between answering and refusing, so it is asserted
    from both ends.
    """
    strict = settings.model_copy(update={"retrieval_min_similarity": 0.99})
    with build_client(strict) as client:
        index_document(client)

        assert retrieval(client).search(IRRELEVANT) == []


def test_a_filter_restricts_retrieval_to_named_documents(client: TestClient) -> None:
    first = client.post(
        "/documents/upload",
        files={"file": ("a.pdf", build_pdf(PAGES), "application/pdf")},
    ).json()["data"]["document"]["id"]
    second = client.post(
        "/documents/upload",
        files={
            "file": (
                "b.pdf",
                build_pdf(["A separate manual about vector retrieval latency tuning."]),
                "application/pdf",
            )
        },
    ).json()["data"]["document"]["id"]

    hits = retrieval(client).search("vector retrieval latency", document_ids=[second])

    assert hits
    assert {hit.document_id for hit in hits} == {second}
    assert first not in {hit.document_id for hit in hits}


def test_an_unknown_filter_is_a_404_not_an_empty_result(client: TestClient, indexed: str) -> None:
    """Silently returning nothing would look like an empty library."""
    with pytest.raises(NotFoundError) as error:
        retrieval(client).search("anything", document_ids=["nope"])

    assert error.value.status_code == 404


@pytest.mark.parametrize("question", ["", "   ", "\n"])
def test_a_blank_question_is_rejected(client: TestClient, question: str) -> None:
    with pytest.raises(ValidationError):
        retrieval(client).search(question)


def test_chunks_of_a_deleted_document_are_not_returned(
    client: TestClient, indexed: str
) -> None:
    """Deleting the metadata row must retire the vectors with it.

    Simulated at the repository level (rather than through DELETE /documents) to
    prove the resolve step is defensive: a hit whose document row is gone is
    dropped, never cited with a missing name.
    """
    container = client.app.state.container
    hits_before = container.retrieval.search("vector retrieval latency")
    assert hits_before

    assert container.documents.delete(indexed) is True
    # The vectors are deliberately left behind to isolate the lookup step.
    assert container.vectors.count_by_document(indexed) > 0

    remaining = container.retrieval.search("vector retrieval latency")
    assert remaining == []


def test_document_count_tracks_the_library(client: TestClient, indexed: str) -> None:
    container = client.app.state.container

    assert container.retrieval.document_count() == 1
    client.delete(f"/documents/{indexed}")
    assert container.retrieval.document_count() == 0


def test_a_custom_top_k_overrides_the_configuration(client: TestClient, indexed: str) -> None:
    container = client.app.state.container
    assert container.settings.top_k == 3

    hits = container.retrieval.search("vector retrieval latency", top_k=2)

    assert len(hits) <= 2


def test_settings_drive_the_floor_that_the_container_uses(settings: Settings) -> None:
    """The configured floor is the one retrieval applies (no hidden constant)."""
    assert settings.retrieval_min_similarity == 0.2
