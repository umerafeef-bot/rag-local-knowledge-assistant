"""POST /chat end to end: retrieval, prompt assembly, generation, citations.

The LLM is a recording fake, so no model is needed and the prompt the service
actually sent can be inspected. Everything else -- extraction, chunking, the
embedding stand-in, ChromaDB, SQLite, the HTTP layer -- is real.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.errors import ServiceUnavailableError
from app.main import create_app
from app.services.prompting import INSUFFICIENT_CONTEXT_ANSWER, SYSTEM_PROMPT
from tests.fakes import FakeEmbedding, FakeLLMProvider, UnavailableLLM
from tests.pdf_factory import build_pdf

# Page 3 is deliberately long enough (with 200-character test chunks) to yield
# several chunks, so retrieval returns more than the context budget can carry.
# Pages 1 and 2 share no vocabulary with the question, so the similarity floor
# has real noise to reject.
PAGES = [
    "Chapter one defines momentum as the product of mass and velocity.",
    "Chapter two introduces entropy as a measure of disorder in a closed system.",
    (
        "Chapter three reports that vector retrieval latency grows slowly with corpus "
        "size. The measurements in this chapter show that doubling the corpus increased "
        "vector retrieval latency by less than ten percent, because the index is searched "
        "in approximate nearest-neighbour order. Vector retrieval latency therefore stays "
        "practical for a corpus of a hundred thousand chunks."
    ),
]

QUESTION = "what does the text say about vector retrieval latency and corpus size?"


@pytest.fixture
def llm() -> FakeLLMProvider:
    return FakeLLMProvider(answer="Retrieval latency grows slowly with corpus size. [3]")


def upload(client: TestClient, pages, name: str = "chapters.pdf") -> str:  # type: ignore[no-untyped-def]
    response = client.post(
        "/documents/upload", files={"file": (name, build_pdf(pages), "application/pdf")}
    )
    assert response.status_code == 201, response.text
    return response.json()["data"]["document"]["id"]


def ask(client: TestClient, question: str = QUESTION, **extra):  # type: ignore[no-untyped-def]
    return client.post("/chat", json={"question": question, **extra})


# --- happy path ------------------------------------------------------------
def test_a_grounded_answer_comes_back_with_page_sources(
    client: TestClient, llm: FakeLLMProvider
) -> None:
    document_id = upload(client, PAGES)

    response = ask(client)

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["error"] is None

    data = body["data"]
    assert data["answer"] == "Retrieval latency grows slowly with corpus size. [3]"
    assert data["grounded"] is True
    assert data["sources"], "a grounded answer must cite something"
    assert data["retrieved"] >= 1
    assert data["context_chunks"] == len(data["sources"])
    assert data["context_chars"] > 0
    assert data["context_truncated"] is False
    assert data["prompt_tokens"] > 0

    best = data["sources"][0]
    assert best["document_id"] == document_id
    assert best["filename"] == "chapters.pdf"
    assert best["page"] == 3
    assert best["chunk_index"] >= 0
    assert -1.0 <= best["similarity"] <= 1.0
    assert best["snippet"]

    assert data["timings"]["retrieval_seconds"] >= 0
    assert data["timings"]["generation_seconds"] >= 0
    assert data["timings"]["total_seconds"] >= data["timings"]["retrieval_seconds"]


def test_sources_are_short_snippets_not_whole_chunks(
    client: TestClient, llm: FakeLLMProvider
) -> None:
    upload(client, PAGES)

    snippet = ask(client).json()["data"]["sources"][0]["snippet"]

    # Snippets are capped by SOURCE_SNIPPET_CHARS; an ellipsis is added on a cut.
    assert len(snippet) <= 83


def test_the_model_receives_the_grounding_rules_and_a_numbered_context(
    client: TestClient, llm: FakeLLMProvider
) -> None:
    upload(client, PAGES)

    ask(client)

    assert llm.last_system == SYSTEM_PROMPT
    prompt = llm.last_prompt
    assert prompt is not None
    assert "[1]" in prompt
    assert "chapters.pdf" in prompt
    assert QUESTION in prompt
    assert INSUFFICIENT_CONTEXT_ANSWER in prompt
    # The rule must be in the prompt as well as the system message: small models
    # weight the final user turn heavily.
    assert "only the context above" in prompt.lower()


def test_no_document_text_leaks_into_the_response_beyond_the_snippet(
    client: TestClient, llm: FakeLLMProvider
) -> None:
    upload(client, PAGES)

    body = ask(client).text

    assert "velocity" not in body  # page 1 is not a source for this question
    assert "disorder" not in body


# --- refusals and errors ---------------------------------------------------
def test_asking_before_uploading_is_a_conflict_with_an_actionable_code(
    client: TestClient, llm: FakeLLMProvider
) -> None:
    response = ask(client)

    assert response.status_code == 409
    body = response.json()
    assert body["success"] is False
    assert body["data"] is None
    assert body["error"]["code"] == "no_documents"
    assert "upload" in body["error"]["message"].lower()
    # Nothing was generated: the library is empty, not the documents unhelpful.
    assert llm.last_prompt is None


def test_an_irrelevant_question_refuses_instead_of_guessing(
    client: TestClient, llm: FakeLLMProvider
) -> None:
    upload(client, PAGES)

    response = ask(client, "how do I bake sourdough bread at home?")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["grounded"] is False
    assert data["answer"] == INSUFFICIENT_CONTEXT_ANSWER
    assert data["sources"] == []
    assert data["retrieved"] == 0
    assert data["context_chars"] == 0
    assert data["timings"]["generation_seconds"] == 0.0
    # The key property: the model was never asked, so it cannot hallucinate.
    assert llm.last_prompt is None


def test_a_model_failure_becomes_a_503_with_a_helpful_message(settings) -> None:  # type: ignore[no-untyped-def]
    app = create_app(
        settings=settings,
        embeddings=FakeEmbedding(),
        llm=UnavailableLLM("Ollama is not reachable at http://127.0.0.1:11434."),
    )
    with TestClient(app) as client:
        upload(client, PAGES)

        response = ask(client)

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "llm_unavailable"
    assert "127.0.0.1:11434" in error["message"]
    assert response.json()["data"] is None


def test_a_timeout_is_reported_as_unavailable_not_as_an_internal_error(settings) -> None:  # type: ignore[no-untyped-def]
    app = create_app(
        settings=settings,
        embeddings=FakeEmbedding(),
        llm=UnavailableLLM("The local model did not answer within 120s."),
    )
    with TestClient(app) as client:
        upload(client, PAGES)

        response = ask(client)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_unavailable"


@pytest.mark.parametrize("question", ["", "   ", "\n\t "])
def test_a_blank_question_is_rejected(client: TestClient, question: str) -> None:
    upload(client, PAGES)

    response = ask(client, question)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_an_oversized_question_is_rejected(client: TestClient) -> None:
    response = ask(client, "x" * 1001)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_a_missing_question_field_uses_the_error_envelope(client: TestClient) -> None:
    response = client.post("/chat", json={})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


# --- filtering and budgets -------------------------------------------------
def test_a_document_filter_limits_which_documents_can_be_cited(client: TestClient) -> None:
    latency_doc = upload(client, PAGES, name="retrieval.pdf")
    upload(client, ["A guide to momentum and velocity in classical mechanics."], name="physics.pdf")

    response = ask(client, document_ids=[latency_doc])

    assert response.status_code == 200
    sources = response.json()["data"]["sources"]
    assert sources
    assert {source["document_id"] for source in sources} == {latency_doc}
    assert {source["filename"] for source in sources} == {"retrieval.pdf"}


def test_an_empty_filter_list_searches_the_whole_library(client: TestClient) -> None:
    upload(client, PAGES)

    response = ask(client, document_ids=[])

    assert response.status_code == 200
    assert response.json()["data"]["grounded"] is True


def test_filtering_on_an_unknown_document_is_a_404(client: TestClient) -> None:
    upload(client, PAGES)

    response = ask(client, document_ids=["deadbeef"])

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "document_not_found"


def test_a_tight_context_budget_truncates_and_says_so(settings) -> None:  # type: ignore[no-untyped-def]
    tight = settings.model_copy(update={"max_context_chars": 260})
    app = create_app(settings=tight, embeddings=FakeEmbedding(), llm=FakeLLMProvider())
    with TestClient(app) as client:
        upload(client, PAGES)

        data = ask(client).json()["data"]

    assert data["context_truncated"] is True
    assert data["context_chars"] <= 260
    # Retrieval found more than the prompt could carry, and the response reports
    # both numbers rather than hiding the cut.
    assert data["retrieved"] > 1
    assert data["context_chunks"] < data["retrieved"]
    assert 0 < len(data["sources"]) == data["context_chunks"]


def test_the_pipeline_still_answers_when_the_budget_only_fits_one_chunk(settings) -> None:  # type: ignore[no-untyped-def]
    minimal = settings.model_copy(update={"max_context_chars": 250})
    app = create_app(settings=minimal, embeddings=FakeEmbedding(), llm=FakeLLMProvider())
    with TestClient(app) as client:
        upload(client, PAGES)

        data = ask(client).json()["data"]

    assert data["grounded"] is True
    assert len(data["sources"]) == 1
    assert data["context_truncated"] is True


def test_openapi_documents_the_chat_endpoint(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert "/chat" in schema["paths"]
    request = schema["paths"]["/chat"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    assert request["$ref"].endswith("ChatRequest")
