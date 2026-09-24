"""A tiny, dependency-free PDF writer used to build test fixtures.

Writing the fixtures ourselves (rather than committing binary PDFs or adding
reportlab) keeps the repository text-only and the suite fully offline. The
output is a plain uncompressed PDF 1.4 document using the standard Helvetica
font, which is exactly what ``pdfplumber`` and ``pypdf`` are expected to read.
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Sequence

_WORDS = (
    "gravity",
    "momentum",
    "entropy",
    "gradient",
    "retrieval",
    "index",
    "vector",
    "latency",
    "stochastic",
    "corpus",
    "embedding",
    "context",
    "chapter",
    "theorem",
    "because",
    "therefore",
    "measured",
    "document",
    "section",
    "explains",
)


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(text: str) -> bytes:
    """A single text object that draws each line on its own baseline."""
    parts = ["BT", "/F1 12 Tf", "72 748 Td", "14 TL"]
    for line in text.split("\n"):
        parts.append(f"({_escape(line)}) Tj")
        parts.append("T*")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1", errors="replace")


def build_pdf(pages: Sequence[str]) -> bytes:
    """Return a valid PDF whose pages contain exactly the given text."""
    page_ids = [4 + 2 * index for index in range(len(pages))]
    content_ids = [5 + 2 * index for index in range(len(pages))]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii"),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    }
    for position, text in enumerate(pages):
        stream = _content_stream(text)
        objects[page_ids[position]] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_ids[position]} 0 R >>"
        ).encode("ascii")
        objects[content_ids[position]] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += f"{number} 0 obj\n".encode("ascii") + objects[number] + b"\nendobj\n"

    xref_offset = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for number in range(1, size):
        out += f"{offsets[number]:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


def build_encrypted_pdf(pages: Sequence[str], *, password: str = "secret") -> bytes:
    """Wrap :func:`build_pdf` output in real password encryption."""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(io.BytesIO(build_pdf(pages)))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(user_password=password, owner_password=password)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def write_pdf(path: Path, pages: Sequence[str]) -> bytes:
    payload = build_pdf(pages)
    path.write_bytes(payload)
    return payload


def synthetic_pages(
    page_count: int,
    *,
    seed: int = 20260922,
    words_per_page: int = 180,
    header: str = "Advanced Retrieval Systems",
) -> list[str]:
    """Deterministic filler text, one shared header line per page.

    The repeated header is intentional: it exercises the boilerplate-removal
    step of the cleaning pipeline.
    """
    generator = random.Random(seed)
    pages: list[str] = []
    for page_number in range(1, page_count + 1):
        words = [generator.choice(_WORDS) for _ in range(words_per_page)]
        lines = [
            header,
            f"Chapter {page_number}",
            " ".join(words[: words_per_page // 2]),
            "".join(f" {word}" for word in words[words_per_page // 2 :]).strip(),
        ]
        pages.append("\n".join(lines))
    return pages
