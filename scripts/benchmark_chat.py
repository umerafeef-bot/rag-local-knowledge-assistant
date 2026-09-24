"""Measure the answer path: retrieval, prompt assembly and (if available) generation.

Everything is real -- real PDF parsing, real chunking, real sentence-transformers
embeddings, real ChromaDB, real SQLite. The one component that needs a running
Ollama is generation, and this machine does not have one, so the script reports
that component separately instead of folding a guess into the total:

    # always works; measures the non-generation half
    .venv/Scripts/python -m scripts.benchmark_chat

    # once Ollama is installed and the model is pulled, the same command
    # additionally reports true end-to-end answer latency

Questions come from the evaluation corpus, so the numbers are comparable with
`scripts.evaluate_retrieval`.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

from app.container import Container
from app.core.config import Settings
from app.db.base import init_schema
from app.services.prompting import SYSTEM_PROMPT, build_context, build_prompt
from scripts.eval_corpus import QUESTIONS, build_corpus_pdf
from tests.fakes import make_upload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repeats", type=int, default=5, help="Passes over the question set.")
    parser.add_argument("--data-dir", type=Path, help="Throwaway store location.")
    parser.add_argument("--keep", action="store_true", help="Keep the data directory.")
    return parser.parse_args(argv)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarise(label: str, samples: list[float]) -> None:
    print(
        f"  {label:<34} median {statistics.median(samples) * 1000:7.1f} ms   "
        f"min {min(samples) * 1000:7.1f}   max {max(samples) * 1000:7.1f}   "
        f"p95 {percentile(samples, 0.95) * 1000:7.1f}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir or Path(tempfile.mkdtemp(prefix="docuask-chatbench-"))
    settings = Settings(
        data_dir=data_dir,
        log_level="WARNING",
        _env_file=None,
    )
    settings.ensure_directories()
    settings.reset_temp_dir()

    print("ENVIRONMENT")
    print(f"  platform            : {platform.platform()}")
    print(f"  python              : {sys.version.split()[0]} ({platform.machine()})")
    print(f"  embedding model     : {settings.embedding_model} on {settings.embedding_device}")
    print(f"  top_k / floor       : {settings.top_k} / {settings.retrieval_min_similarity}")
    print(f"  context budget      : {settings.max_context_chars} chars")
    print(f"  data directory      : {data_dir}")
    print()

    container = Container.build(settings)
    init_schema(container.engine)
    temporary = args.data_dir is None
    questions = [item.question for item in QUESTIONS if item.expected_page is not None]

    try:
        print("WARM-UP (one-time model initialisation)")
        started = time.perf_counter()
        container.embeddings.embed_query("warm up")
        print(f"  first encode() call : {time.perf_counter() - started:.2f} s\n")

        print("INGESTION")
        started = time.perf_counter()
        result = container.document_service.ingest(
            make_upload(build_corpus_pdf(), filename="bench-corpus.pdf")
        )
        print(
            f"  {result.document.page_count} pages, {result.document.char_count} chars -> "
            f"{result.document.chunk_count} chunks"
        )
        print(f"  extract             : {result.timings.extract_seconds:.3f} s")
        print(f"  clean + chunk       : {result.timings.chunk_seconds:.3f} s")
        print(f"  embed               : {result.timings.embed_seconds:.3f} s")
        print(f"  index + save        : {result.timings.store_seconds:.3f} s")
        print(f"  total (warm)        : {result.timings.total_seconds:.3f} s")
        print()

        # --- retrieval half -------------------------------------------------
        retrieval_samples: list[float] = []
        prompt_samples: list[float] = []
        for _ in range(max(1, args.repeats)):
            for question in questions:
                started = time.perf_counter()
                hits = container.retrieval.search(question)
                retrieval_samples.append(time.perf_counter() - started)

                started = time.perf_counter()
                context = build_context(hits, max_chars=settings.max_context_chars)
                build_prompt(question, context.text)
                prompt_samples.append(time.perf_counter() - started)

        print("ANSWER PATH (non-generation half)")
        print(f"  samples             : {len(retrieval_samples)} "
              f"({len(questions)} questions x {max(1, args.repeats)} passes)")
        summarise("retrieval (embed + search + resolve)", retrieval_samples)
        summarise("context + prompt assembly", prompt_samples)
        overhead = statistics.median(retrieval_samples) + statistics.median(prompt_samples)
        print(f"  {'fixed overhead per answer':<34} {overhead * 1000:7.1f} ms")
        print(f"  {'system prompt size':<34} {len(SYSTEM_PROMPT):7d} chars")
        print()

        # --- generation half ------------------------------------------------
        health = container.llm.health()
        print("GENERATION")
        if health.ok:
            generation_samples: list[float] = []
            total_samples: list[float] = []
            for _ in range(max(1, args.repeats)):
                for question in questions:
                    answer = container.chat.answer(question)
                    generation_samples.append(answer.timings.generation_seconds)
                    total_samples.append(answer.timings.total_seconds)
            summarise("LLM generation", generation_samples)
            summarise("end-to-end answer (retrieval + LLM)", total_samples)
            print(f"  model               : {health.model}")
        else:
            print("  NOT MEASURED.")
            print(f"  reason              : {health.detail}")
            print(
                f"  configured model    : {health.model or '<unset>'}\n"
                "  Install Ollama and pull the model, then re-run this script to get\n"
                "  a real end-to-end number. The overhead above is the part this\n"
                "  machine can measure today."
            )
        print()
    finally:
        container.close()
        if temporary and not args.keep:
            shutil.rmtree(data_dir, ignore_errors=True)
        else:
            print(f"Kept data directory: {data_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
