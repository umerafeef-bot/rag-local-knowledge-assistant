"""Integration test that needs a real Ollama server and a real model.

This is the only test in the suite that requires something the application
cannot install for itself, so it is marked ``integration_llm``, deselected by
default, and it *skips* (never fails) when Ollama is not reachable or the model
configured in ``OLLAMA_MODEL`` has not been pulled. A missing local model is a
setup state, not a defect in the code.

Prerequisites:

    ollama serve                      # or launch the Ollama desktop app
    ollama pull llama3.2:3b
    OLLAMA_MODEL=llama3.2:3b

Run it with::

    .venv/Scripts/python -m pytest -m integration_llm -s

It costs real time: the embedding model loads, and a 3B model on CPU answers in
tens of seconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.container import Container
from app.core.config import Settings
from app.core.errors import NoDocumentsError
from app.db.base import init_schema
from app.services.prompting import INSUFFICIENT_CONTEXT_ANSWER
from tests.fakes import make_upload
from tests.pdf_factory import build_pdf

pytestmark = pytest.mark.integration_llm

PAGES = [
    "Chapter one defines momentum as the product of mass and velocity.",
    "Chapter two introduces entropy as a measure of disorder in a closed system.",
    (
        "Chapter three reports that vector retrieval latency grows slowly with corpus size. "
        "Doubling the corpus increased vector retrieval latency by less than ten percent, "
        "because the index is searched in approximate nearest-neighbour order."
    ),
]

QUESTION = "how much did doubling the corpus increase vector retrieval latency?"


@pytest.fixture(scope="module")
def container(tmp_path_factory: pytest.TempPathFactory) -> Container:  # type: ignore[no-untyped-def]
    """A real container: real embeddings, real ChromaDB, real Ollama client."""
    data_dir = tmp_path_factory.mktemp("ollama-integration") / "data"
    settings = Settings(
        data_dir=data_dir,
        chunk_size=400,
        chunk_overlap=80,
        top_k=3,
        retrieval_min_similarity=0.2,
        max_context_chars=2000,
        log_level="WARNING",
        _env_file=None,
    )
    settings.ensure_directories()
    settings.reset_temp_dir()
    built = Container.build(settings)
    init_schema(built.engine)

    # Verify the local setup before loading anything heavy. Skipping here rather
    # than failing keeps the suite honest: nothing is proven either way.
    health = built.llm.health()
    if not health.ok:
        built.close()
        pytest.skip(
            f"Ollama is not ready: {health.detail} "
            f"(configured model: {health.model or '<unset>'})"
        )

    yield built
    built.close()


@pytest.fixture(scope="module")
def indexed(container: Container) -> str:
    """The library is populated once for this module, so test order cannot matter."""
    document_id = container.document_service.ingest(
        make_upload(build_pdf(PAGES), filename="chapters.pdf")
    ).document.id
    assert container.vectors.count_by_document(document_id) > 0
    return document_id


def test_a_real_model_answers_from_the_documents(container: Container, indexed: str) -> None:
    result = container.chat.answer(QUESTION)

    # Structural guarantees, not wording: what a 3B model says is its business,
    # but it must have been given real context and it must cite it.
    assert result.grounded is True
    assert result.sources, "a grounded answer must carry sources"
    assert any(source.page == 3 for source in result.sources)
    assert result.answer.strip()
    assert result.answer.strip() != INSUFFICIENT_CONTEXT_ANSWER
    assert result.timings.generation_seconds > 0

    print(
        f"\n[integration_llm] model={container.llm.model_name} "
        f"prompt_tokens={result.prompt_tokens} "
        f"retrieval={result.timings.retrieval_seconds:.2f}s "
        f"generation={result.timings.generation_seconds:.2f}s\n"
        f"  sources: {[(source.filename, source.page, round(source.similarity, 3)) for source in result.sources]}\n"
        f"  answer : {result.answer.strip()[:400]}"
    )


def test_a_real_model_refuses_a_question_the_documents_cannot_answer(
    container: Container, indexed: str
) -> None:
    """The refusal is decided by retrieval, so no model call should happen."""
    result = container.chat.answer("what is the boiling point of mercury?")

    assert result.grounded is False
    assert result.sources == ()
    assert result.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert result.timings.generation_seconds == 0.0


def test_asking_before_uploading_is_refused(tmp_path: Path) -> None:
    """An empty library is a 409-shaped error, not a grounded refusal."""
    settings = Settings(
        data_dir=tmp_path / "empty",
        ollama_model="",
        log_level="WARNING",
        _env_file=None,
    )
    settings.ensure_directories()
    empty = Container.build(settings)
    init_schema(empty.engine)
    try:
        with pytest.raises(NoDocumentsError) as error:
            empty.chat.answer("anything at all?")
        assert error.value.status_code == 409
    finally:
        empty.close()
