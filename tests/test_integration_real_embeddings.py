"""Integration tests that load the real sentence-transformers model.

Deselected by default (``addopts = "-m 'not integration and ...'"``) because the
first run downloads the model (~90 MB) and it makes the suite slower. Run them
with::

    .venv/Scripts/python -m pytest -m integration -s

Only the LLM is faked here; extraction, chunking, embedding, retrieval, context
assembly and the prompt are all the real implementations. Retrieval *quality*
needs a real model, so it is measured here rather than in the offline suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.fakes import FakeLLMProvider
from tests.pdf_factory import build_pdf

pytestmark = pytest.mark.integration

PAGES = [
    "Chapter one defines momentum as the product of mass and velocity.",
    "Chapter two introduces entropy as a measure of disorder in a system.",
    "Chapter three reports that vector retrieval latency grows slowly with corpus size.",
]


def test_real_pipeline_indexes_and_retrieves(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        chunk_size=300,
        chunk_overlap=60,
        top_k=3,
        ollama_model="",
        _env_file=None,
    )
    app = create_app(settings=settings, llm=FakeLLMProvider())
    with TestClient(app) as client:
        response = client.post(
            "/documents/upload",
            files={"file": ("chapters.pdf", build_pdf(PAGES), "application/pdf")},
        )
        assert response.status_code == 201, response.text
        document = response.json()["data"]["document"]
        assert document["chunk_count"] >= 3

        container = client.app.state.container
        embedder = container.embeddings
        # all-MiniLM-L6-v2 documented output width.
        assert embedder.dimension == 384

        query_vector = embedder.embed_query("what does the text say about retrieval latency?")
        hits = container.vectors.search(query_vector, top_k=3)

        assert hits
        assert hits[0].page == 3
        assert "latency" in hits[0].text
        print(
            f"\n[integration] model={embedder.model_name} "
            f"chunks={document['chunk_count']} top_similarity={hits[0].similarity:.4f}"
        )


def test_real_model_drives_a_grounded_answer_with_page_sources(tmp_path: Path) -> None:
    """The whole Day 2 path with real vectors and a fake (recording) LLM.

    This is what proves retrieval *quality*: a paraphrased question -- the kind a
    user actually asks -- has to retrieve the right page once the embedding model
    is a real one. The bag-of-words stand-in used by the offline suite cannot
    show that, because it matches tokens rather than meaning.
    """
    llm = FakeLLMProvider(answer="Retrieval latency grows slowly with the corpus. [3]")
    settings = Settings(
        data_dir=tmp_path / "data",
        chunk_size=300,
        chunk_overlap=60,
        top_k=3,
        retrieval_min_similarity=0.2,
        max_context_chars=2000,
        ollama_model="",
        _env_file=None,
    )
    app = create_app(settings=settings, llm=llm)
    with TestClient(app) as client:
        response = client.post(
            "/documents/upload",
            files={"file": ("chapters.pdf", build_pdf(PAGES), "application/pdf")},
        )
        assert response.status_code == 201, response.text

        # Deliberately paraphrased: shares almost no vocabulary with page 3.
        answer = client.post(
            "/chat", json={"question": "does searching a bigger collection take much longer?"}
        )

        assert answer.status_code == 200, answer.text
        data = answer.json()["data"]
        assert data["grounded"] is True, data
        assert data["sources"], "a real model must retrieve something for this question"
        assert data["sources"][0]["page"] == 3
        assert data["sources"][0]["similarity"] > 0.2

        # The prompt really contains the numbered, source-labelled context.
        assert llm.last_prompt is not None
        assert f"(file: chapters.pdf, page 3)" in llm.last_prompt
        print(
            f"\n[integration] question=paraphrased top_page={data['sources'][0]['page']} "
            f"similarity={data['sources'][0]['similarity']:.4f} "
            f"context_chars={data['context_chars']}"
        )


def test_an_unrelated_question_is_refused_by_the_real_model(tmp_path: Path) -> None:
    """The similarity floor has to reject noise with a real embedding model."""
    llm = FakeLLMProvider()
    settings = Settings(
        data_dir=tmp_path / "data",
        chunk_size=300,
        chunk_overlap=60,
        top_k=3,
        retrieval_min_similarity=0.30,
        ollama_model="",
        _env_file=None,
    )
    app = create_app(settings=settings, llm=llm)
    with TestClient(app) as client:
        client.post(
            "/documents/upload",
            files={"file": ("chapters.pdf", build_pdf(PAGES), "application/pdf")},
        )

        data = client.post(
            "/chat", json={"question": "how do I replace a bicycle inner tube?"}
        ).json()["data"]

    assert data["grounded"] is False
    assert data["sources"] == []
    assert llm.last_prompt is None, "the model must not be asked an unanswerable question"
