"""Point the retrieval evaluation at a real book instead of synthetic text.

The self-written corpus in ``scripts/eval_corpus.py`` measures the pipeline on
pages whose ground truth was designed into them. This module does the opposite:
it takes a real book nobody wrote for this project -- Project Gutenberg #3300,
Adam Smith's *An Inquiry into the Nature and Causes of the Wealth of Nations*
(1776), public domain -- paginates it the way a PDF reader would, and derives
question/expected-page pairs from the text itself.

Every question carries an ``anchor``: a quote that must occur **exactly once**
in the paginated text. The expected page is computed from where the anchor
actually sits, so the ground truth cannot drift from the content. If an anchor
is missing, duplicated, or split by a page boundary, the script fails loudly
rather than writing a wrong page number.

Matching is done on a normalized view (casefolded, punctuation stripped) so
capitalization and 18th-century hyphenation cannot hide a real occurrence, and
per-page so an anchor spanning a page break can never match. Anchors were
chosen from famous passages and verified against the downloaded text.

    # download + paginate + write questions.json
    python -m scripts.book_corpus

    # then run the evaluation on the result
    python -m scripts.evaluate_retrieval --pdf <book.pdf> --questions <questions.json>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from tests.pdf_factory import build_pdf

#: Verified live on 2026-09-24: HTTP 200, 2.47 MB, ``Accept-Ranges: bytes``.
BOOK_URL = "https://www.gutenberg.org/cache/epub/3300/pg3300.txt"

#: Geometry of the generated PDF: 40 lines x 70 characters at 12pt Helvetica on
#: a US-Letter page (build_pdf starts at y=748 with 14pt leading, so 40 lines
#: fits comfortably above the 72pt margin).
PAGE_LINES = 40
PAGE_WIDTH = 70

START_MARKER = re.compile(r"\*\*\*\s*START OF THE PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE)
END_MARKER = re.compile(r"\*\*\*\s*END OF THE PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Anchor:
    """One evaluation item: the question and the unique quote that locates its answer."""

    question: str
    quote: str


#: Ten answerable questions (each anchored to one unique passage) plus two the
#: book cannot answer, which test abstention. Quotes are Smith's own wording
#: (checked case-insensitively, without punctuation).
ANCHORS: tuple[Anchor, ...] = (
    Anchor(
        "Do we get our dinner because those who provide it love us, or because they "
        "regard their own advantage?",
        "benevolence of the butcher",
    ),
    Anchor(
        "How does the text describe people ending up promoting an end they never intended?",
        "invisible hand",
    ),
    Anchor(
        "What are the two senses in which the word value can be used, according to the "
        "chapter on commodity value?",
        "has two different meanings",
    ),
    Anchor(
        "Where does the text describe the bargaining between buyers and sellers that "
        "settles the market price?",
        "the higgling and bargaining of the market",
    ),
    Anchor(
        "According to the text, is defending the nation more or less important than "
        "becoming rich?",
        "importance than opulence",
    ),
    Anchor(
        "What does the text call the attempt of kings and ministers to watch over the "
        "private economy of individuals?",
        "the highest impertinence and presumption",
    ),
    Anchor(
        "What does the text say should be established for the education of young people?",
        "institutions for the education of the youth",
    ),
    Anchor(
        "Where does the text object to forbidding a great people from making everything "
        "it can of its own produce?",
        "To prohibit a great people",
    ),
    Anchor(
        "How does the text say every man comes to be supplied with what he needs?",
        "Every man thus lives by exchanging",
    ),
    Anchor(
        "What three sources does the text name as the origin of the annual produce?",
        "The produce of land, mines, and fisheries",
    ),
    # Out-of-corpus: nothing above the similarity floor should be retrieved.
    Anchor("What is the boiling point of mercury in degrees Celsius?", ""),
    Anchor("How do I configure a firewall on my computer?", ""),
)


def normalize_key(text: str) -> str:
    """Casefolded, punctuation-free text: 'Invisible hand' == 'invisible hand'."""
    return re.sub(r"[^a-z0-9 ]+", " ", text.casefold()).strip()


def download(path: Path) -> bytes:
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"using cached download: {path}")
        return path.read_bytes()
    print(f"downloading {BOOK_URL} ...")
    request = urllib.request.Request(BOOK_URL, headers={"User-Agent": "Mozilla/5.0 (eval builder)"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    path.write_bytes(payload)
    print(f"  {len(payload)} bytes written")
    return payload


def normalize(raw: bytes) -> str:
    """Decode, strip the Gutenberg header/footer, and make the text PDF-safe."""
    text = raw.decode("utf-8")
    start = START_MARKER.search(text)
    end = END_MARKER.search(text)
    if not start or not end:
        raise SystemExit("could not locate the Project Gutenberg START/END markers")
    text = text[start.end() : end.start()]
    replacements = {
        "\r\n": "\n",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2014": "--", "\u2013": "-", "\u2026": "...",
        "\u00a0": " ", "\ufeff": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text


def paginate(text: str) -> list[str]:
    """Wrap the text into fixed-size pages, the way the generated PDF lays it out."""
    wrapper = textwrap.TextWrapper(width=PAGE_WIDTH, break_long_words=True, break_on_hyphens=False)
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if paragraph.strip():
            lines.extend(wrapper.wrap(paragraph))
        else:
            lines.append("")
    pages: list[str] = []
    for offset in range(0, len(lines), PAGE_LINES):
        page_text = "\n".join(lines[offset : offset + PAGE_LINES]).strip("\n")
        if page_text.strip():
            pages.append(page_text)
    return pages


def locate(pages: list[str], quote: str) -> int:
    """1-based page containing the quote, or a loud failure with the reason."""
    keys = [normalize_key(page) for page in pages]
    joined = "\f".join(keys)
    occurrences = joined.count(normalize_key(quote))
    if occurrences != 1:
        raise SystemExit(
            f"anchor {quote!r} occurs {occurrences} times, expected exactly once -- "
            "fix the anchor before trusting the ground truth"
        )
    return joined[: joined.index(normalize_key(quote))].count("\f") + 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(".venv/eval_book"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    text = normalize(download(args.out / "pg3300.txt"))
    pages = paginate(text)
    print(f"paginated into {len(pages)} pages ({len(text):,} chars of body text)")

    questions = []
    for anchor in ANCHORS:
        if not anchor.quote:
            questions.append({"question": anchor.question, "expected_page": None})
            print(f"  p.  --  (abstention case) {anchor.question[:58]}...")
            continue
        page = locate(pages, anchor.quote)
        questions.append({"question": anchor.question, "expected_page": page})
        print(f"  p.{page:>4}  {anchor.quote!r} -> {anchor.question[:58]}...")

    book_path = args.out / "wealth_of_nations.pdf"
    book_path.write_bytes(build_pdf(pages))
    questions_path = args.out / "questions.json"
    questions_path.write_text(json.dumps(questions, indent=2), encoding="utf-8")
    print(f"wrote {book_path} ({book_path.stat().st_size:,} bytes) and {questions_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
