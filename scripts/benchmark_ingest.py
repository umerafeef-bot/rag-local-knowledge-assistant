"""Measure the real ingestion and retrieval latency of this machine.

No estimates, no hard-coded numbers: this script runs the production pipeline
(real PDF parsing, real chunking, real sentence-transformers embeddings, real
ChromaDB) and prints what it actually measured.

    # synthetic document, 12 pages
    .venv/Scripts/python -m scripts.benchmark_ingest --pages 12

    # your own PDF
    .venv/Scripts/python -m scripts.benchmark_ingest --pdf C:/docs/book.pdf

By default it works in a throwaway data directory, so your library is untouched.
"""

from __future__ import annotations

import argparse
import io
import platform
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

from starlette.datastructures import Headers
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.container import Container
from app.core.config import Settings
from app.db.base import init_schema
from tests.pdf_factory import build_pdf, synthetic_pages

DEFAULT_QUERY = "how does vector retrieval latency scale with corpus size?"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pdf", type=Path, help="A real PDF to ingest.")
    source.add_argument("--pages", type=int, default=12, help="Pages of synthetic text (default: 12).")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Query used for the search benchmark.")
    parser.add_argument("--repeats", type=int, default=10, help="Search repetitions (default: 10).")
    parser.add_argument("--data-dir", type=Path, help="Where to put the throwaway database and vectors.")
    parser.add_argument("--keep", action="store_true", help="Keep the data directory afterwards.")
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the warm-up pass so its cost lands inside the ingestion numbers.",
    )
    return parser.parse_args(argv)


def build_upload(payload: bytes, filename: str) -> StarletteUploadFile:
    return StarletteUploadFile(
        io.BytesIO(payload),
        filename=filename,
        size=len(payload),
        headers=Headers({"content-type": "application/pdf"}),
    )


def print_environment(settings: Settings) -> None:
    print("ENVIRONMENT")
    print(f"  platform          : {platform.platform()}")
    print(f"  python            : {sys.version.split()[0]} ({platform.machine()})")
    print(f"  embedding model   : {settings.embedding_model} on {settings.embedding_device}")
    print(f"  chunking          : size={settings.chunk_size} overlap={settings.chunk_overlap}")
    print(f"  top_k             : {settings.top_k}")
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    data_dir = args.data_dir or Path(tempfile.mkdtemp(prefix="docuask-bench-"))
    settings = Settings(
        data_dir=data_dir,
        ollama_model="",
        log_level="WARNING",
        _env_file=None,
    )
    settings.ensure_directories()
    settings.reset_temp_dir()

    print_environment(settings)
    print(f"data directory: {data_dir}\n")

    container = Container.build(settings)
    init_schema(container.engine)
    temporary = args.data_dir is None

    try:
        if args.pdf is not None:
            payload = args.pdf.read_bytes()
            filename = args.pdf.name
            page_note = "(page count reported by the parser)"
        else:
            pages = synthetic_pages(args.pages)
            payload = build_pdf(pages)
            filename = f"synthetic-{args.pages}-pages.pdf"
            page_note = f"({args.pages} generated pages)"

        if not args.no_warmup:
            print("MODEL WARM-UP (one-time per process)")
            warm_started = time.perf_counter()
            container.embeddings.embed_query("warm up")
            print(f"  first encode() call: {time.perf_counter() - warm_started:.2f} s")
            print()

        upload = build_upload(payload, filename)

        print("INGESTION")
        started = time.perf_counter()
        result = container.document_service.ingest(upload)
        wall_seconds = time.perf_counter() - started
        document = result.document
        timings = result.timings

        print(f"  file              : {filename} ({len(payload) / 1024:.1f} KiB)")
        print(f"  pages             : {document.page_count} {page_note}")
        print(f"  characters        : {document.char_count}")
        print(f"  chunks            : {document.chunk_count}")
        print(f"  stage timings     :")
        print(f"      validate/store: {timings.upload_seconds:.3f} s")
        print(f"      extract       : {timings.extract_seconds:.3f} s")
        print(f"      clean + chunk : {timings.chunk_seconds:.3f} s")
        print(f"      embed         : {timings.embed_seconds:.3f} s")
        print(f"      index + save  : {timings.store_seconds:.3f} s")
        print(f"  TOTAL (pipeline)  : {timings.total_seconds:.3f} s")
        print(f"  TOTAL (wall clock): {wall_seconds:.3f} s")
        if timings.embed_seconds > 0:
            print(f"  embedding rate    : {document.chunk_count / timings.embed_seconds:.1f} chunks/s")
        print()

        print("SEARCH")
        query_vector = container.embeddings.embed_query(args.query)
        samples: list[float] = []
        hits = []
        for _ in range(max(1, args.repeats)):
            search_started = time.perf_counter()
            hits = container.vectors.search(query_vector, top_k=settings.top_k)
            samples.append(time.perf_counter() - search_started)

        print(f"  query             : {args.query!r}")
        print(f"  repeats           : {len(samples)}")
        print(f"  latency (median)  : {statistics.median(samples) * 1000:.2f} ms")
        print(f"  latency (min/max) : {min(samples) * 1000:.2f} / {max(samples) * 1000:.2f} ms")
        print(f"  chunks in store   : {container.vectors.total_chunks()}")
        print("  top hit           :")
        for hit in hits[:1]:
            snippet = hit.text[:70].replace("\n", " ")
            print(f"      page {hit.page}, similarity {hit.similarity:.4f}: {snippet}...")
        print()

        print("Cleanup: deleting the benchmark document")
        container.document_service.delete(document.id)
        print(f"  chunks after delete: {container.vectors.total_chunks()}")
    finally:
        container.close()
        if temporary and not args.keep:
            shutil.rmtree(data_dir, ignore_errors=True)
        else:
            print(f"\nKept data directory: {data_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
