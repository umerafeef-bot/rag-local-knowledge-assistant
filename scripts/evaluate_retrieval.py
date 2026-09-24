"""Measure retrieval quality on this machine, with real components.

Runs the production pipeline: real PDF parsing, real cleaning and chunking, the
real sentence-transformers model, the real ChromaDB store, the real
``RetrievalService``. Only the questions and their expected pages are supplied by
a human -- that is the ground truth being tested against.

    # the built-in self-written corpus (10 pages, 13 questions)
    .venv/Scripts/python -m scripts.evaluate_retrieval

    # your own PDF and your own question/answer pairs
    .venv/Scripts/python -m scripts.evaluate_retrieval --pdf book.pdf --questions questions.json

The questions file is a JSON array; ``expected_page`` is 1-based, and ``null``
means "the documents do not answer this":

    [{"question": "what does the text say about caching?", "expected_page": 12},
     {"question": "how do I bake bread?", "expected_page": null}]

Two numbers come out:

* **hit-rate@K** -- for an answerable question, is the expected page among the
  pages of the top-K retrieved chunks? Compared against the chance rate of
  ``K / pages``, which is also printed.
* **abstention accuracy** -- for an out-of-corpus question, did retrieval return
  nothing above ``RETRIEVAL_MIN_SIMILARITY``? This is what stops the app from
  answering questions its documents cannot support.

Nothing here is cached or estimated: every number is measured in this process.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from app.container import Container
from app.core.config import Settings
from app.db.base import init_schema
from scripts.eval_corpus import QUESTIONS, build_corpus_pdf
from tests.fakes import make_upload

Question = tuple[str, int | None]


@dataclass(frozen=True, slots=True)
class QuestionResult:
    question: str
    expected_page: int | None
    returned_pages: tuple[int, ...]
    top_similarity: float | None
    rank: int | None
    latency_ms: float

    @property
    def hit(self) -> bool:
        return self.rank is not None

    @property
    def abstained(self) -> bool:
        return not self.returned_pages

    @property
    def correct(self) -> bool:
        """A hit for an answerable question, an abstention for an unanswerable one."""
        return self.abstained if self.expected_page is None else self.hit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pdf", type=Path, help="A real PDF to evaluate against.")
    parser.add_argument(
        "--questions",
        type=Path,
        help="JSON file of {question, expected_page} pairs (required with --pdf).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval depth (default: 5).")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help="Override RETRIEVAL_MIN_SIMILARITY for this run.",
    )
    parser.add_argument("--data-dir", type=Path, help="Throwaway store location.")
    parser.add_argument("--keep", action="store_true", help="Keep the data directory.")
    parser.add_argument(
        "--json",
        type=Path,
        help="Also write the full result set to this path as JSON.",
    )
    return parser.parse_args(argv)


def load_questions(path: Path) -> list[Question]:
    """Read and validate a question file. Bad input fails loudly, not silently."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise SystemExit(f"{path}: expected a non-empty JSON array")

    questions: list[Question] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict) or not str(entry.get("question", "")).strip():
            raise SystemExit(f"{path}: entry {index} needs a non-empty 'question'")
        expected = entry.get("expected_page")
        if expected is not None and not isinstance(expected, int):
            raise SystemExit(
                f"{path}: entry {index} 'expected_page' must be an integer page number or null"
            )
        questions.append((str(entry["question"]).strip(), expected))
    return questions


def evaluate(
    container: Container,
    questions: list[Question],
    *,
    top_k: int,
) -> list[QuestionResult]:
    results: list[QuestionResult] = []
    for question, expected_page in questions:
        started = time.perf_counter()
        hits = container.retrieval.search(question, top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000

        pages = tuple(hit.page for hit in hits)
        rank = pages.index(expected_page) + 1 if expected_page in pages else None
        results.append(
            QuestionResult(
                question=question,
                expected_page=expected_page,
                returned_pages=pages,
                top_similarity=hits[0].similarity if hits else None,
                rank=rank,
                latency_ms=latency_ms,
            )
        )
    return results


@dataclass(frozen=True, slots=True)
class SweepRow:
    """Retrieval quality at one candidate similarity floor."""

    threshold: float
    hits: int
    answerable: int
    abstained: int
    unanswerable: int

    @property
    def hit_rate(self) -> str:
        return f"{self.hits}/{self.answerable}" if self.answerable else "n/a"

    @property
    def abstention_rate(self) -> str:
        return f"{self.abstained}/{self.unanswerable}" if self.unanswerable else "n/a"


def threshold_sweep(
    container: Container,
    questions: list[Question],
    *,
    top_k: int,
    thresholds: tuple[float, ...],
) -> list[SweepRow]:
    """Re-score the same retrieval at several similarity floors.

    Each question is searched once with no floor, and then every candidate
    threshold is applied to those scores offline. Same query, same vectors, only
    the cut-off moves -- so the comparison is apples to apples, and the shipped
    default was chosen from evidence rather than guessed.
    """
    unfiltered: list[tuple[int | None, list[tuple[float, int]]]] = []
    for question, expected_page in questions:
        vector = container.embeddings.embed_query(question)
        hits = container.vectors.search(vector, top_k=top_k)
        unfiltered.append((expected_page, [(hit.similarity, hit.page) for hit in hits]))

    rows: list[SweepRow] = []
    for threshold in thresholds:
        hits = answerable = abstained = unanswerable = 0
        for expected_page, scored_pages in unfiltered:
            kept_pages = [page for score, page in scored_pages if score >= threshold]
            if expected_page is None:
                unanswerable += 1
                abstained += 1 if not kept_pages else 0
            else:
                answerable += 1
                hits += 1 if expected_page in kept_pages else 0
        rows.append(
            SweepRow(
                threshold=threshold,
                hits=hits,
                answerable=answerable,
                abstained=abstained,
                unanswerable=unanswerable,
            )
        )
    return rows


def print_header(container: Container, *, top_k: int, page_count: int, chunk_count: int) -> None:
    settings = container.settings
    print("ENVIRONMENT")
    print(f"  platform            : {platform.platform()}")
    print(f"  python              : {sys.version.split()[0]} ({platform.machine()})")
    print(f"  embedding model     : {settings.embedding_model} on {settings.embedding_device}")
    print(f"  chunking            : size={settings.chunk_size} overlap={settings.chunk_overlap}")
    print(f"  top_k               : {top_k}")
    print(f"  min similarity floor: {container.retrieval.min_similarity}")
    print(f"  corpus              : {page_count} pages -> {chunk_count} chunks")
    print()


def print_results(results: list[QuestionResult], *, top_k: int) -> None:
    print(f"PER QUESTION  (K={top_k}; 'rank' is where the expected page landed)")
    width = 62
    for number, result in enumerate(results, start=1):
        question = result.question
        if len(question) > width:
            question = f"{question[: width - 3]}..."
        top_score = f"{result.top_similarity:.3f}" if result.top_similarity is not None else "n/a"
        if result.expected_page is None:
            verdict = "ABSTAINED" if result.abstained else "RETRIEVED NOISE"
            detail = f"top score {top_score}"
        elif result.hit:
            verdict = "HIT"
            detail = f"page {result.expected_page} at rank {result.rank}, top score {top_score}"
        else:
            verdict = "MISS"
            shown = list(result.returned_pages)
            detail = f"expected page {result.expected_page}, got pages {shown} or none"
        print(f"  {number:2d}. {question:<{width}}  {verdict:<16} {detail}")
    print()


def print_summary(results: list[QuestionResult], *, top_k: int, page_count: int) -> None:
    answerable = [result for result in results if result.expected_page is not None]
    unanswerable = [result for result in results if result.expected_page is None]
    hits = [result for result in answerable if result.hit]
    latencies = [result.latency_ms for result in results]

    chance = min(1.0, top_k / page_count) if page_count else 0.0
    hit_rate = len(hits) / len(answerable) if answerable else 0.0
    reciprocal_ranks = [1.0 / result.rank for result in answerable if result.rank]
    mrr = statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    print("SUMMARY")
    print(f"  questions           : {len(results)} ({len(answerable)} answerable, "
          f"{len(unanswerable)} out-of-corpus)")
    print(f"  hit-rate@{top_k}          : {len(hits)}/{len(answerable)} = {hit_rate:.1%}")
    print(f"  chance rate@{top_k}       : {chance:.1%}  (K / pages, for scale)")
    print(f"  MRR                 : {mrr:.3f}")
    if unanswerable:
        abstained = [result for result in unanswerable if result.abstained]
        print(
            f"  abstention accuracy : {len(abstained)}/{len(unanswerable)} = "
            f"{len(abstained) / len(unanswerable):.1%}"
        )
    print(
        f"  retrieval latency   : median {statistics.median(latencies):.1f} ms "
        f"(min {min(latencies):.1f}, max {max(latencies):.1f})"
    )
    print()


def write_json(
    path: Path,
    results: list[QuestionResult],
    *,
    container: Container,
    top_k: int,
    page_count: int,
    chunk_count: int,
) -> None:
    answerable = [result for result in results if result.expected_page is not None]
    unanswerable = [result for result in results if result.expected_page is None]
    payload = {
        "embedding_model": container.settings.embedding_model,
        "chunk_size": container.settings.chunk_size,
        "chunk_overlap": container.settings.chunk_overlap,
        "top_k": top_k,
        "min_similarity": container.retrieval.min_similarity,
        "pages": page_count,
        "chunks": chunk_count,
        "hit_rate": (len([r for r in answerable if r.hit]) / len(answerable)) if answerable else None,
        "abstention_accuracy": (
            len([r for r in unanswerable if r.abstained]) / len(unanswerable)
            if unanswerable
            else None
        ),
        "questions": [
            {
                "question": result.question,
                "expected_page": result.expected_page,
                "rank": result.rank,
                "returned_pages": list(result.returned_pages),
                "top_similarity": result.top_similarity,
                "latency_ms": round(result.latency_ms, 3),
            }
            for result in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.pdf and not args.questions:
        raise SystemExit("--pdf needs --questions (the ground truth is yours to supply)")

    data_dir = args.data_dir or Path(tempfile.mkdtemp(prefix="docuask-eval-"))
    overrides: dict[str, object] = {"data_dir": data_dir, "ollama_model": "", "log_level": "WARNING"}
    if args.min_similarity is not None:
        overrides["retrieval_min_similarity"] = args.min_similarity
    settings = Settings(_env_file=None, **overrides)
    settings.ensure_directories()
    settings.reset_temp_dir()

    container = Container.build(settings)
    init_schema(container.engine)
    temporary = args.data_dir is None

    try:
        if args.pdf:
            payload = args.pdf.read_bytes()
            filename = args.pdf.name
            questions = load_questions(args.questions)
        else:
            payload = build_corpus_pdf()
            filename = "eval-corpus.pdf"
            questions = [(item.question, item.expected_page) for item in QUESTIONS]

        print("WARM-UP (one-time model initialisation, excluded from the numbers below)")
        warm_started = time.perf_counter()
        container.embeddings.embed_query("warm up")
        print(f"  first encode() call : {time.perf_counter() - warm_started:.2f} s\n")

        print("INGESTION")
        ingest_started = time.perf_counter()
        result = container.document_service.ingest(make_upload(payload, filename=filename))
        ingest_seconds = time.perf_counter() - ingest_started
        document = result.document
        print(
            f"  {filename}: {document.page_count} pages, {document.char_count} chars, "
            f"{document.chunk_count} chunks in {ingest_seconds:.2f} s "
            f"(embed {result.timings.embed_seconds:.2f} s)\n"
        )

        print_header(
            container,
            top_k=args.top_k,
            page_count=document.page_count,
            chunk_count=document.chunk_count,
        )

        results = evaluate(container, questions, top_k=args.top_k)
        print_results(results, top_k=args.top_k)
        print_summary(results, top_k=args.top_k, page_count=document.page_count)

        print("THRESHOLD SWEEP  (same retrieval, score floor applied offline)")
        print(f"  {'floor':>6}  {'hit-rate':>10}  {'abstained':>12}")
        for row in threshold_sweep(
            container,
            questions,
            top_k=args.top_k,
            thresholds=(0.0, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6),
        ):
            marker = "  <- shipped default" if abs(row.threshold - container.retrieval.min_similarity) < 1e-9 else ""
            print(
                f"  {row.threshold:6.2f}  {row.hit_rate:>10}  {row.abstention_rate:>12}{marker}"
            )

        if args.json:
            write_json(
                args.json,
                results,
                container=container,
                top_k=args.top_k,
                page_count=document.page_count,
                chunk_count=document.chunk_count,
            )
    finally:
        container.close()
        if temporary and not args.keep:
            shutil.rmtree(data_dir, ignore_errors=True)
        else:
            print(f"\nKept data directory: {data_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
