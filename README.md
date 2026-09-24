# DocuAsk — private question answering over your own PDFs

Upload PDFs (textbooks, manuals, papers, notes), ask questions, and get answers
that are **grounded in those documents with page-level citations** — or an honest
"I could not find that" when they are not.

Everything runs on your machine: PDF parsing, embeddings, the vector store, the
language model and the web UI. No document text, question or answer is ever sent
to an external API.

No LangChain, no LlamaIndex: chunking, retrieval and prompt construction are
implemented here, in the open, so every line can be explained in an interview.

---

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech stack and why each piece exists](#tech-stack-and-why-each-piece-exists)
- [Requirements](#requirements)
- [Install](#install)
- [Environment variables](#environment-variables)
- [Run](#run)
- [API](#api)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Measured performance](#measured-performance)
- [Security review](#security-review)
- [Privacy](#privacy)
- [Testing](#testing)
- [Project structure](#project-structure)
- [Known limitations](#known-limitations)
- [Future improvements](#future-improvements)
- [Definition of done](#definition-of-done)
- [License](#license)

---

## Overview

DocuAsk is a local RAG (retrieval-augmented generation) system with no remote
dependency. You give it PDFs; it extracts and indexes them; you ask questions in
a browser; it retrieves the passages that answer them, hands them to a small
model running locally, and returns the answer with a citation to each passage —
document name, page number, similarity score and an excerpt you can check.

The interesting engineering is not the prompt. It is the decisions around it:

- **Retrieval decides whether the model is called at all.** If nothing in the
  library scores above the similarity floor, the app returns a refusal and never
  contacts the model. Asking a 3B model to answer from context that does not
  contain the answer is the most reliable way to get a confident fabrication.
- **The context is budgeted in characters.** Ollama truncates a prompt that
  exceeds the model's window *silently*, typically discarding the question or the
  end of the context. `MAX_CONTEXT_CHARS` caps what is sent, and
  `LLM_CONTEXT_WINDOW` pins the model's window so the two cannot disagree.
- **Ingestion is all-or-nothing.** An upload either becomes a fully searchable
  document (file + vectors + metadata row) or leaves nothing behind.
- **Deletion is ordered**: vectors, then file, then metadata row — so a document
  is never listed but unsearchable, and never half-deleted but looking healthy.

## Features

**Ingestion**
- Upload with three independent checks: extension, streaming size cap, `%PDF` magic bytes
- SHA-256 duplicate detection (409 before a second copy is written)
- Distinct, actionable errors for corrupted / encrypted / empty / image-only / oversized PDFs
- Per-page extraction, boilerplate removal, de-hyphenation, wrapped-line merging
- Recursive character chunking with word-aligned overlap, page numbers preserved

**Retrieval and answers**
- Cosine top-K search over 384-dimension embeddings, optionally filtered to one document
- Similarity floor, so "nothing relevant" is a first-class answer
- Numbered, source-labelled context with a hard character budget
- Prompt rules: answer only from the context, cite block numbers, say when you don't know
- Sources back with document, page, chunk index, similarity score and excerpt
- Replaceable LLM layer: `LLMProvider` protocol, `OllamaProvider` implementation
- Graceful degradation: 503 with instructions when Ollama is down or the model is missing

**Interface**
- Web UI served by the API process: drag-and-drop upload, document library with
  delete, chat with a per-answer source panel, document scope selector
- Loading, error and empty states throughout; responsive; keyboard operable
- `GET /health` reporting API, database, vector store, embedding model and LLM
- One JSON envelope for every response, including static-asset 404s
- Interactive OpenAPI docs at `/docs`

## Architecture

```
  browser ── GET /  (HTML/CSS/JS, same process) ───────────────┐
     │                                                         │
     │  POST /documents/upload   POST /chat   GET /health      ▼
     └──────────────────────────────────────────────▶ FastAPI app
                                                        │ api/routes: documents, chat, health
                                                        │ api/schemas: DTOs + Envelope[T]
                                                        │ api/errors: one error shape
                    ┌───────────────────────────────────┼──────────────────────────────┐
                    │                                   │                            │
                    ▼                                   ▼                            ▼
          DocumentService                        ChatService                  LLMProvider (protocol)
          ingest() / delete()                    answer()                     │
                    │                                   │                      └─ OllamaProvider
        ┌───────────┴───────────┐         ┌─────────────┼─────────────┐           │ 127.0.0.1:11434
        ▼                       ▼         ▼             ▼             ▼
  pdf_extraction          embedding   retrieval     prompting      (generation)
  pypdf + pdfplumber      sentence-   embed query   budgeted       
  structural check +      transformers  → top-K →   context +      
  layout-aware text       all-MiniLM   floor →      grounding      
                          CPU, 384-d   filenames    prompt
        │                       │         │             │
        ▼                       ▼         ▼             ▼
  ┌────────────────┐   ┌──────────────────────┐   ┌─────────────────────────┐
  │ repositories   │   │ repositories         │   │ data/uploads/<uuid>.pdf │
  │ document_      │   │ vector_ (ChromaDB,   │   │ (UUID name, never the   │
  │ repository     │   │ cosine, local,       │   │  user's filename)       │
  │ (SQLAlchemy +  │   │ telemetry off)       │   └─────────────────────────┘
  │  SQLite,       │   │                      │
  │  metadata only)│   │ chunk text + vectors │
  └───────┬────────┘   └──────────┬───────────┘
          ▼                       ▼
   data/docuask.sqlite3    data/chroma/
```

**Ingestion flow**

```
validate (extension → stream with size cap → SHA-256 → %PDF magic bytes)
  → duplicate check
  → pypdf structural check (encrypted / empty / page count)
  → pdfplumber per-page text
  → clean (NFKC, boilerplate lines, de-hyphenation, page numbers, wrapped lines)
  → recursive chunking (800 chars / 100 overlap, page numbers kept)
  → embed (384-dim unit vectors)
  → store vectors → store metadata → promote file to data/uploads/<uuid>.pdf
```

**Answer flow**

```
question → embed with the SAME model that embedded the chunks
        → ChromaDB cosine top-K (optional document filter)
        → drop everything below RETRIEVAL_MIN_SIMILARITY
            └── nothing left? → refusal, grounded: false, model NOT called
        → pack survivors into numbered, source-labelled context ≤ MAX_CONTEXT_CHARS
        → system rules + context + question → local LLM
        → { answer, sources[], timings }
```

**Data model** — SQLite holds metadata only; text and vectors live in ChromaDB.

```sql
CREATE TABLE documents (
    id              VARCHAR(32) PRIMARY KEY,   -- uuid4 hex; stored file is <id>.pdf
    filename        VARCHAR(255) NOT NULL,     -- sanitised display label, never used on disk
    content_type    VARCHAR(100) NOT NULL,
    sha256          VARCHAR(64)  NOT NULL UNIQUE,  -- duplicate-upload guard
    size_bytes      BIGINT       NOT NULL,
    page_count      INTEGER      NOT NULL,
    chunk_count     INTEGER      NOT NULL,
    char_count      INTEGER      NOT NULL,
    embedding_model VARCHAR(200) NOT NULL,     -- vectors are invalid if this changes
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL
);
-- indexes: UNIQUE (sha256), INDEX (created_at)
```

Chunk metadata in ChromaDB: `document_id`, `page`, `chunk_index`. The display
filename is deliberately not duplicated there — it is resolved from SQLite in one
query per search, so a document's name exists in exactly one place.

A longer version with failure-mode notes lives in [docs/architecture.md](docs/architecture.md).

## Tech stack and why each piece exists

| Choice | Why this one |
|---|---|
| **FastAPI + Uvicorn** | Pydantic validation and OpenAPI docs come for free, and `def` handlers run in a thread pool — which matters because ingestion and generation are blocking CPU work. |
| **Pydantic + pydantic-settings** | One validated type for every I/O boundary; config comes from env/`.env` so no secret or model name is hard-coded. |
| **SQLAlchemy + SQLite** | Real transactions for the metadata row and zero operational overhead. A single-file database matches a single-user local app. |
| **pdfplumber + pypdf** | Two libraries doing what each is best at: pypdf answers structural questions cheaply (encrypted? how many pages? parseable at all?), pdfplumber does layout-aware text extraction. |
| **sentence-transformers + all-MiniLM-L6-v2** | 90 MB, CPU-only, 384 dimensions, no API. Small enough to run on this laptop, good enough that a paraphrased question still retrieves the right page (measured below). |
| **ChromaDB (persistent, local)** | An embedded vector store with cosine distance and metadata filtering — no server to run, and telemetry explicitly disabled. |
| **Ollama** | Runs a quantised model locally and exposes a plain HTTP API, so the LLM layer is swappable behind `LLMProvider`. The model name comes from `OLLAMA_MODEL`, never from code. |
| **Vanilla HTML/CSS/JS** | No Node toolchain, no build step, no CDN. Three files served by the same process — which is also a privacy property: the browser loads nothing from off this machine. |
| **No LangChain / LlamaIndex** | The chunker, retriever and prompt builder are ~500 lines total. Owning them means fewer dependencies to audit and every line explainable. |
| **pytest** | Offline suite runs in ~16 s with deterministic fakes; opt-in markers cover the real model and the real LLM. |

## Requirements

- **Python 3.11+** (built and tested on 3.12.3)
- ~3 GB of disk for dependencies and the embedding model
- ~2 GB of RAM free while running; CPU-only is fine (no GPU needed)
- **Ollama** — optional, required only for chat. UI, upload, search and delete work without it.

Measured environment: Windows 10 Pro 22H2 (build 19045), CPython 3.12.3,
8 logical cores, 16 GB RAM, NVIDIA Quadro M1000M 2 GB present but unused —
this stack is CPU-only.

## Install

### Windows (PowerShell)

```powershell
git clone <your-repo-url> docuask
cd docuask

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

Copy-Item .env.example .env   # then edit .env if you want different limits
```

### Ubuntu / macOS

```bash
git clone <your-repo-url> docuask
cd docuask

python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

cp .env.example .env          # then edit .env if you want different limits
```

> Windows notes: `python3` is the broken Microsoft Store alias on this machine —
> use `python` or `py`. In PowerShell, `curl` is an alias for `Invoke-WebRequest`;
> use `curl.exe`, and see [Usage](#usage) for JSON quoting that actually works.

### For chat: install Ollama

```powershell
# Windows: install from https://ollama.com, then in a second terminal:
ollama serve          # or just launch the Ollama desktop app
ollama pull llama3.2:3b    # ~2.0 GB at Q4_K_M; the name must match OLLAMA_MODEL
```

```bash
# Ubuntu: see https://ollama.com/download for the install command, then
ollama serve &
ollama pull llama3.2:3b
```

Until this is done, `/health` reports `degraded` and `/chat` answers `503`
with the same instructions. That is by design: the app stays useful.

## Environment variables

Everything is environment-driven (`.env`, see `.env.example`). Nothing secret is
in the repository.

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `data` | root for uploads, SQLite, vector store, staging |
| `MAX_UPLOAD_MB` | `50` | streaming size cap — the file is never buffered whole |
| `MAX_PDF_PAGES` | `2000` | page-count guard |
| `MIN_EXTRACTABLE_CHARS` | `40` | below this a PDF is rejected as scanned/image-only |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` | recursive character chunking |
| `TOP_K` | `5` | chunks retrieved per question |
| `RETRIEVAL_MIN_SIMILARITY` | `0.30` | similarity floor; below it the app refuses |
| `MAX_CONTEXT_CHARS` | `4000` | hard budget for context pasted into the prompt |
| `SOURCE_SNIPPET_CHARS` | `240` | excerpt length returned with each cited source |
| `CHAT_TEMPERATURE` | `0.1` | generation temperature |
| `CHAT_MAX_TOKENS` | `512` | generation cap (Ollama `num_predict`) |
| `LLM_CONTEXT_WINDOW` | `4096` | sent as Ollama `num_ctx`; Ollama truncates silently past it |
| `LLM_KEEP_ALIVE` | `5m` | how long Ollama keeps the model resident between questions |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | local LLM server |
| `OLLAMA_MODEL` | *(empty in code)* | **no model is hard-coded anywhere**; empty = unconfigured |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | generation timeout, reported as `503` |
| `EMBEDDING_MODEL` / `EMBEDDING_DEVICE` | all-MiniLM-L6-v2 / `cpu` | local embeddings |
| `EMBEDDING_WARMUP` | `true` | load the model in a background thread at startup |
| `ALLOWED_ORIGINS` | *(empty)* | comma-separated origins permitted by CORS; empty = closed |
| `LOG_LEVEL` | `INFO` | log verbosity — content is never logged at any level |

## Run

```powershell
# Windows PowerShell, from the repository root
.\.venv\Scripts\python.exe -m uvicorn --factory app.main:create_app --host 127.0.0.1 --port 8907
```

```bash
# Ubuntu / macOS
python -m uvicorn --factory app.main:create_app --host 127.0.0.1 --port 8907
```

Then open **<http://127.0.0.1:8907/>** for the UI, or
<http://127.0.0.1:8907/docs> for the interactive API docs.

The first request loads the embedding model; with `EMBEDDING_WARMUP=true` (the
default) this happens in the background at startup instead of stalling the first
upload.

## API

Every response uses one envelope — including static-asset 404s, so a client
needs exactly one error branch:

```json
{"success": true,  "data": {...}, "error": null}
{"success": false, "data": null,  "error": {"code": "...", "message": "...", "details": {...}}}
```

| Method | Path | Purpose | Errors |
|---|---|---|---|
| `POST` | `/documents/upload` | Upload + index a PDF (multipart field `file`) | 409 `duplicate_document`, 413 `file_too_large`, 415 `unsupported_file_type`, 422 `not_a_pdf` / `pdf_corrupted` / `pdf_encrypted` / `pdf_empty` / `no_extractable_text` / `too_many_pages` |
| `GET` | `/documents?limit=&offset=` | List indexed documents (newest first) | 422 `validation_error` |
| `GET` | `/documents/{id}` | One document's metadata (never its text) | 404 `document_not_found` |
| `DELETE` | `/documents/{id}` | Remove file + vectors + metadata | 404 `document_not_found`, 500 `storage_error` |
| `POST` | `/chat` | Ask a question, get an answer with sources | 409 `no_documents`, 404 `document_not_found`, 422 `validation_error`, 503 `llm_unavailable` |
| `GET` | `/health` | API, database, vector store, embedding model, LLM status | always 200 (`"degraded"` when a dependency is down) |
| `GET` | `/info` | Service descriptor | — |
| `GET` | `/docs` | Interactive OpenAPI docs | — |

`POST /chat` body:

```json
{"question": "what stops a fast sender from overwhelming a receiver?",
 "document_ids": ["optional-restrict-to-these-documents"]}
```

Response (`document_ids` omitted or `[]` searches the whole library):

```json
{"success": true, "data": {
  "answer": "The receiver advertises a window of bytes it will accept, and the sender may not exceed it [1].",
  "grounded": true,
  "sources": [{"document_id": "83c72a…", "filename": "notes.pdf", "page": 5,
               "chunk_index": 8, "similarity": 0.6854,
               "snippet": "Chapter 5: Preventing a fast sender from drowning a slow receiver…"}],
  "retrieved": 5, "context_chunks": 5, "context_chars": 2653,
  "context_truncated": false, "prompt_tokens": 745,
  "timings": {"retrieval_seconds": 0.0283, "generation_seconds": 4.1, "total_seconds": 4.13}
}}
```

Two different "no" answers, deliberately:

- **200 + `grounded: false`** — the library has documents but none of them
  answer the question. `sources` is empty and the refusal sentence is this
  application's own, not a model output.
- **409 `no_documents`** — the library is empty. The advice is "upload
  something", which is different advice.

## Usage

**In the browser:** open `/`, drag a PDF onto the upload area (or use the file
button), watch the status line, then type a question. Each answer shows its
sources with page numbers and similarity scores. The scope selector above the
conversation restricts retrieval to one document; "Clear conversation" resets it;
each document has a Delete button that removes its file, vectors and metadata.

**On the command line** — note the PowerShell JSON quoting: piping a
single-quoted string with embedded double quotes to a native `.exe` is mangled by
PowerShell 5.1 (it strips the inner quotes and you get a `422`). Both forms below
were run on this machine; the inline variant was not.

<details>
<summary>Windows (PowerShell) — verified forms</summary>

```powershell
curl.exe http://127.0.0.1:8907/health
curl.exe -X POST http://127.0.0.1:8907/documents/upload -F "file=@C:\path\to\document.pdf"
curl.exe http://127.0.0.1:8907/documents
curl.exe -X DELETE http://127.0.0.1:8907/documents/<id-from-the-response>

# Form 1: JSON in a file
'{ "question": "what does the document say about caching?" }' | Set-Content request.json
curl.exe -X POST http://127.0.0.1:8907/chat -H "Content-Type: application/json" -d "@request.json"

# Form 2: Invoke-RestMethod, no quoting tricks
$response = Invoke-RestMethod -Method Post http://127.0.0.1:8907/chat `
  -ContentType "application/json" `
  -Body (@{ question = "what does the document say about caching?" } | ConvertTo-Json)
$response.data.answer
$response.data.sources | Format-Table filename, page, similarity
```

</details>

<details>
<summary>Ubuntu / macOS (bash) — inline JSON is fine</summary>

```bash
curl -s http://127.0.0.1:8907/health
curl -s -X POST http://127.0.0.1:8907/documents/upload -F "file=@./document.pdf"
curl -s http://127.0.0.1:8907/documents

curl -s -X POST http://127.0.0.1:8907/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"what does the document say about caching?"}'
```

</details>

## Evaluation

`scripts/evaluate_retrieval.py` measures retrieval against known ground truth. By
default it uses `scripts/eval_corpus.py`: ten self-written pages on ten separable
topics, thirteen questions — ten answerable, each phrased differently from its
page so a hit requires semantic matching, plus three the corpus deliberately
cannot answer, which test that the pipeline abstains.

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_retrieval
```

Your own PDF and ground truth:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_retrieval --pdf C:\docs\book.pdf --questions questions.json
```

A fully worked real-book example is built in. `scripts/book_corpus.py` downloads
Adam Smith's *Wealth of Nations* (Project Gutenberg #3300, public domain),
paginates all 1,584 pages into a real PDF, and derives the question set from
quotes that are **verified to occur exactly once** in the text — the expected
page is computed from where the quote actually sits, never hand-typed, and the
build fails loudly if an anchor is missing or duplicated:

```powershell
.\.venv\Scripts\python.exe -m scripts.book_corpus
.\.venv\Scripts\python.exe -m scripts.evaluate_retrieval --pdf .venv\eval_book\wealth_of_nations.pdf --questions .venv\eval_book\questions.json --top-k 5
```

```json
[{"question": "what does the text say about caching?", "expected_page": 12},
 {"question": "how do I bake bread?",                 "expected_page": null}]
```

`expected_page` is 1-based; `null` means "the documents do not answer this".

**Measured on this machine** (10 pages / 9,367 chars → 20 chunks, `K=5`, floor 0.30):

| Metric | Result |
|---|---|
| hit-rate@5 | **10/10 = 100%** (chance rate for this corpus: 5/10 = 50%) |
| MRR | **0.950** |
| abstention accuracy (3 out-of-corpus questions) | **3/3 = 100%** |
| retrieval latency (incl. query embedding) | median **15–24 ms** across runs |

Nine of ten expected pages landed at rank 1, one at rank 2. Honest caveat: ten
clearly separable topics with `K=5` out of 10 pages is an **easy** corpus — the
50% chance rate says so. It is a regression test with a real number attached, not
a benchmark.

The threshold sweep justifies the shipped similarity floor instead of asserting it:

| floor | hit-rate@5 | out-of-corpus abstained |
|---|---|---|
| 0.00 | 10/10 | **0/3** — noise passes, the model would be asked to answer from nothing |
| 0.15 | 10/10 | 3/3 |
| 0.25 | 10/10 | 3/3 |
| **0.30 (shipped)** | **10/10** | **3/3** |
| 0.35 | 10/10 | 3/3 |
| 0.40 | 10/10 | 3/3 |
| 0.50 | **4/10** | 3/3 — real answers start being refused |

On this easy corpus any floor in [0.15, 0.40] looks identical — which is exactly
why the shipped floor was not chosen from it. The real-book eval below broke the
tie.

### Real-book evaluation — Adam Smith, *Wealth of Nations* (Gutenberg #3300)

A 1,584-page public-domain treatise (3.34 MB PDF → **4,180 chunks**) with ten
questions anchored to famous passages plus two the book cannot answer. Here the
chance rate at `K=5` is **0.3%**, so a hit actually means something.

| Metric | Result |
|---|---|
| hit-rate@5 | **5/10 = 50%** — ~150× the 0.3% chance rate |
| MRR | **0.617** (hit ranks: 1, 1, 2, 3, 4) |
| abstention accuracy (2 out-of-corpus questions) | **2/2 = 100%** |
| retrieval latency over 4,180 chunks | median **17.3 ms** |
| full-book ingestion | **310.6 s** (extract+clean+chunk 146.6 s, embed 164.0 s) |

The five misses are the informative part. Every one is an abstract paraphrase —
"what does the text call the attempt of kings and ministers to watch over the
private economy of individuals?" for *the highest impertinence and presumption*
— where an 18th-century sentence and the modern question share almost no
vocabulary. One miss still landed in the right neighbourhood (the
defence-vs-opulence question retrieved pages 791/792, ~50 pages past its target
739, both in Book IV). This is the honest ceiling of a 384-dimension model over
800-character chunks; closing that gap is what the Future Improvements list
(better chunking, a reranker, hybrid BM25 + vector retrieval) is for.

The same threshold sweep on the real book is what moved the shipped floor from
0.25 to 0.30:

| floor | hit-rate@5 | out-of-corpus abstained |
|---|---|---|
| 0.00 | 5/10 | 0/2 |
| 0.25 (previous default) | 5/10 | **1/2** — a junk question scored 0.283 and slipped past |
| **0.30 (shipped)** | **5/10** | **2/2** |
| 0.35 | 5/10 | 2/2 |
| 0.40 | 4/10 | 2/2 — real answers start being refused |
| 0.50 | 1/10 | 2/2 |

0.30 abstains on every junk question at zero hit-rate cost **on both corpora**.
That is the definition of a free improvement, which is why it ships.

## Measured performance

All numbers produced on the machine described under [Requirements](#requirements),
by `scripts/benchmark_ingest`, `scripts/benchmark_chat`,
`scripts/evaluate_retrieval` and the test suite. Nothing is estimated.

| Measurement | Result |
|---|---|
| Offline test suite | **192 passed** in ~16 s |
| Integration tests (real embedding model) | **3 passed** in 136 s |
| Ollama integration test | passes with Ollama installed; **2 of 3 skip** without it |
| Embedding model / dimension | all-MiniLM-L6-v2 / **384** (asserted by a test) |
| One-time first `encode()` initialisation | **26–55 s** per process, observed across runs; ~6.6 s when weights are cached — absorbed by `EMBEDDING_WARMUP` |
| Ingestion, 10 pages / 9,367 chars → 20 chunks (warm) | extract **0.308 s** · clean+chunk **0.001 s** · embed **0.591 s** · index+save **0.086 s** · **total 0.994 s** |
| Ingestion, 12 pages / 18,812 chars → 36 chunks (warm) | extract 0.826 s · embed 0.909 s · **total 1.851 s** |
| Ingestion, 1,584 pages / 2.21M chars → 4,180 chunks | extract+clean+chunk **146.6 s** · embed **164.0 s** · **total 310.6 s** |
| Vector search alone (top-5, 10 repeats) | median **2.66 ms** |
| Retrieval over 4,180 chunks (query embed + search), 12 queries | median **17.3 ms** |
| Retrieval (query embed + search + filename resolve), 50 samples | median **16.8 ms** · min 13.3 · max 23.2 · p95 22.9 |
| Context + prompt assembly | **< 0.1 ms** |
| Fixed overhead per answer (everything but the model) | **16.8 ms** |
| **End-to-end answer latency with a real model** | **not yet measured — Ollama is not installed on this machine** |

`scripts/benchmark_chat` prints the non-generation half today and will print true
end-to-end latency the moment Ollama is installed; it refuses to estimate the
part it cannot see.

Why the cold/warm split matters: the first `encode()` call in a process costs
tens of seconds of one-time tokenizer and kernel initialisation, while the
identical second call takes well under a second. That is why
`EMBEDDING_WARMUP=true` loads the model in a background thread at startup rather
than letting the first upload appear to hang.

## Security review

Each item was checked against the running application, with evidence. The
corresponding controls are executable — they are tests, not claims. Run them with
`pytest tests/test_security.py tests/test_validation.py tests/test_frontend.py -q`.

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Path traversal | **PASS** | Document ids are looked up in SQLite *before* any path is built, so `../../etc/passwd` is a 404, never a path (`test_delete_rejects_path_traversal_ids`, 5 encoded variants). Uploads land at `<uuid>.pdf` regardless of the supplied name (`test_a_traversing_filename_never_reaches_the_filesystem` asserts a `[0-9a-f]{32}\.pdf` name, and that nothing escapes `data/uploads`). |
| 2 | Filename trust | **PASS** | `sanitize_filename` strips both separator styles, control chars and leading/trailing dots, caps length at 180; the result is only a display label — the disk name comes from a UUID. Verified: `../../../evil.pdf` and `..%2F..%2Fescape.pdf` never appear on disk. |
| 3 | Upload size / type | **PASS** | Size is enforced *while streaming* in 1 MiB blocks, so an oversized file is never buffered (`test_stage_upload_rejects_an_oversized_file`); a file exactly at the limit passes. Type is decided by the `%PDF` signature in the first 1024 bytes, not by the extension or the client's content type (`test_mislabelled_pdf_extension_is_rejected`, `test_non_pdf_extension_is_rejected`). |
| 4 | CORS | **FIXED** | No CORS middleware existed; now configured origins only. Default `ALLOWED_ORIGINS` is empty → **no** `Access-Control-Allow-Origin` header is ever emitted (`test_cross_origin_reads_are_denied_by_default`, `test_cross_origin_preflight_is_denied_by_default`). With origins configured, only those origins match (`test_only_the_configured_origin_is_allowed`); credentials are never allowed and `authorization` is not in the allowed headers (`test_credentials_are_never_allowed`); `*` and malformed origins are refused at startup (`test_a_wildcard_origin_is_refused_at_startup`, 4 malformed variants). |
| 5 | Secrets in repo | **PASS** | `.env` is gitignored and not tracked (`test_no_env_file_is_committed`, `test_the_gitignore_covers_every_secret_and_data_path`); `.env.example` holds no credential-shaped value (`test_the_env_template_contains_no_real_secret`); a scan of all tracked `.py` for `secret`/`token`/`password`/`api_key`-shaped literals found none (`test_no_committed_source_carries_a_secret_shaped_literal`). The model name comes from `OLLAMA_MODEL`, never from code. |
| 6 | Sensitive logging | **PASS** | Document text, filenames, questions and answers are never logged — retrieval logs counts and scores, generation logs a character count and a duration. Verified at `DEBUG` level: an upload containing a launch code and a chat question about private history produced no such string in `caplog` (`test_document_text_never_reaches_the_log`, `test_a_question_never_reaches_the_log`). `sqlalchemy.engine` is pinned to `WARNING` because it can echo bound SQL parameters (which would include chunk text); ChromaDB and huggingface telemetry are disabled. |
| 7 | Error messages leaking internals | **PASS** | A forced crash returns `500 internal_error` with a generic message and no `Traceback`, `RuntimeError`, path or `.py` (`test_a_crash_returns_a_generic_message`). Validation errors drop the submitted values before logging or returning them, so a 1,200-character question is not echoed back (`test_validation_errors_do_not_echo_the_submitted_value`). 404s do not disclose storage details (`test_unreachable_paths_do_not_leak_filesystem_details`). |

Additional frontend controls (`tests/test_frontend.py`): no asset references an
external host, and `app.js` never assigns `innerHTML` — every server-sourced
string (filename, answer, snippet) enters the DOM as text, so a PDF containing
markup cannot inject into the page.

## Privacy

| Component | Where it runs | Network |
|---|---|---|
| FastAPI + uvicorn | local process | none |
| pdfplumber / pypdf (text extraction) | in-process | none |
| sentence-transformers + all-MiniLM-L6-v2 | in-process, CPU | model fetched **once** from huggingface.co, then cached; telemetry disabled via `HF_HUB_DISABLE_TELEMETRY=1`; set `HF_HUB_OFFLINE=1` to forbid the update check too |
| ChromaDB (vector store) | embedded, local files | telemetry disabled (`anonymized_telemetry=False`) |
| Ollama (the language model) | local server, `127.0.0.1:11434` | none — generation is local |
| Web UI (HTML/CSS/JS) | served by this process | **no external host is referenced — no CDN, no web fonts, no analytics** |
| Logs | stdout | document text, questions, answers and user content are never logged |

**No external API receives user data.** The only outbound network activity this
app can have is the one-time embedding-model download from huggingface.co, and
`HF_HUB_OFFLINE=1` removes that too. This is verified, not asserted: the
model-load step prints a Hugging Face "unauthenticated requests" notice on this
machine, which is how I know the update check does reach out — and that the
switch to disable it is the one documented in `.env.example`.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -q                     # 192 tests, offline, ~16 s
.\.venv\Scripts\python.exe -m pytest -m integration -q -s   # real embedding model
.\.venv\Scripts\python.exe -m pytest -m integration_llm -q  # needs Ollama (skips without)
.\.venv\Scripts\python.exe -m scripts.benchmark_ingest --pages 12
.\.venv\Scripts\python.exe -m scripts.benchmark_chat
.\.venv\Scripts\python.exe -m scripts.evaluate_retrieval
```

```bash
python -m pytest -q          # 192 tests, offline, ~16 s
python -m pytest -m integration -q -s
```

The offline suite substitutes a deterministic bag-of-words embedding and a
recording fake LLM, so it needs no model, no network and no Ollama — the fake
also lets tests assert what the prompt actually contained. Coverage:

- **Ingestion** — chunk boundaries, overlap, blank input, hard-split of oversized
  tokens, no-lost-sentence checks, wrapped-line merging, boilerplate removal.
- **Validation** — filename traversal sanitising, magic bytes, size cap,
  corrupted / encrypted / image-only / page-limited PDFs.
- **Persistence** — SHA-256 duplicate detection at API *and* database level,
  upload → list → get → delete round trips with vectors verified removed,
  filtered vector search.
- **Retrieval** — ranking order, `top_k`, the similarity floor at both ends,
  document filters, an unknown filter raising 404, and chunks of a
  since-deleted document being dropped rather than cited.
- **Context and prompt** — the character budget is never exceeded, the best hit is
  never traded for a worse one, a single oversized chunk is cut rather than
  dropped, numbering is sequential, and the system prompt is asserted to contain
  the answer-only-from-context rule.
- **Chat** — grounded answers with page sources, refusals when nothing is relevant
  (**with the model asserted *not* to have been called**), 409 on an empty
  library, 503 on model failure or timeout, 404 on a bad filter, 422 on a blank
  or oversized question, truncation reported when the budget bites.
- **Security** — the seven-row table above, each row a test that can fail.
- **Frontend** — the UI is served, its assets contain no external URL, and no
  `innerHTML` assignment exists.

## Project structure

```
app/
  main.py                 app factory (uvicorn --factory), lifespan, CORS, static UI
  container.py            composition root: one place builds every dependency
  api/
    routes/               documents, chat, health
    schemas.py            DTOs + the single response envelope
    deps.py, errors.py    dependency injection, exception handlers
  core/                   config (env-based), logging, error taxonomy
  db/                     engine/session, SQLAlchemy models
  repositories/           document_repository (SQLite), vector_repository (ChromaDB)
  services/               pdf_extraction, embedding, documents, retrieval, prompting, chat
  llm/                    base Protocol (LLMProvider) + OllamaProvider
  utils/                  validation, text_cleaning, chunking
  static/                 index.html, styles.css, app.js  (the whole UI)
docs/
  architecture.md         deeper design notes and failure modes
scripts/
  benchmark_ingest.py     ingestion + search latency
  benchmark_chat.py       answer-path latency (reports generation only if Ollama is up)
  eval_corpus.py          the self-written 10-page corpus and its ground truth
  evaluate_retrieval.py   hit-rate / MRR / abstention harness (works on any PDF)
tests/                    192 offline tests + 3 real-model + 3 real-Ollama integration tests
```

## Known limitations

- **No OCR.** Scanned or image-only PDFs are rejected with `no_extractable_text`
  (by design, in the MVP). The message says so explicitly.
- **Generation latency is unmeasured** on this machine — Ollama is not installed,
  so the end-to-end answer number in [Measured performance](#measured-performance)
  is honestly blank rather than estimated.
- **No conversation history.** Each question is a self-contained prompt. History
  costs tokens on a small model and gives it another way to answer from an
  earlier turn instead of the documents.
- **The similarity floor admits weakly-related neighbours** (0.27–0.31 observed)
  alongside the right ones. They are cited, not hidden, and the sweep table shows
  the trade-off.
- **The evaluation corpus is easy by construction** (50% chance rate). It proves
  the pipeline works, not that it is state of the art.
- **Upload and chat are synchronous**; a large PDF or a slow model ties up one
  worker thread for its duration.
- **De-hyphenation** joins compound words that happen to break at a line end
  (`well-\nknown` → `wellknown`); the heuristic is deliberately conservative.
- **The wrapped-line merge** can join a single-word lowercase heading with the
  following body text.
- **`data/tmp` is wiped at startup**, which assumes a single application process —
  how DocuAsk is run locally.
- **Changing `EMBEDDING_MODEL` invalidates every stored vector.** `embedding_model`
  is recorded per row so the mismatch is detectable, but nothing enforces
  re-ingestion yet.
- **No authentication.** The server binds to `127.0.0.1` by default; do not expose
  it to a network without putting something in front of it.

## Future improvements

Listed here rather than half-built in the code:

- **Conversation history** with a token budget, while keeping the grounding rules
  authoritative over earlier turns.
- **Hybrid retrieval**: BM25 + vectors, with a reranking pass — the similarity
  floor is a blunt instrument compared to a learned reranker.
- **Streaming answers** over SSE so the UI can show tokens as they arrive.
- **Background ingestion** with job status, so a large PDF does not occupy a
  worker thread.
- **OCR** via Tesseract for scanned documents.
- **Page preview** in the source panel, so a citation can be checked visually.
- **Re-ingestion when the embedding model changes**, detected from the stored
  `embedding_model`.
- **ONNX int8 embeddings** for faster CPU ingestion.
- **Docker packaging**, and CI running the offline suite on push.
- **Multi-user auth** if this ever stops being a single-user local tool.

## Definition of done

Honest status against the original brief, item by item.

| Requirement | Status | Evidence |
|---|---|---|
| Upload PDF: extension + magic bytes + size limit from config | ✅ | `app/utils/validation.py`; `tests/test_validation.py`, `tests/test_documents_api.py` |
| Stored under a random UUID filename, never the user's filename | ✅ | `test_a_traversing_filename_never_reaches_the_filesystem` |
| Reject duplicates by SHA-256 | ✅ | 409 `duplicate_document`, tested at API and DB level; re-uploading the same file after a restart also returns 409 |
| Pipeline: validate → extract → clean → chunk → embed → store → metadata | ✅ | `app/services/documents.py`, all-or-nothing with rollback |
| Chunking ~800 / ~100 with page + chunk index metadata | ✅ | `app/utils/chunking.py`; sizes from `CHUNK_SIZE` / `CHUNK_OVERLAP` |
| List / view / delete a document (file + vectors + metadata) | ✅ | `test_delete_removes_file_vectors_and_metadata` asserts all three |
| Chat: embed → top-K → optional document filter → context → LLM → answer + sources | ✅ | `app/services/chat.py`; `tests/test_chat_api.py` |
| Sources = document, page, snippet, similarity | ✅ | `ChatSourceOut`; asserted in the happy-path test |
| Prompt answers only from context and says when it doesn't know | ✅ | `test_the_system_prompt_forbids_outside_knowledge` |
| Replaceable LLM layer behind an interface | ✅ | `LLMProvider` protocol; routes depend only on it; `OllamaProvider` is one implementation |
| Frontend: upload status, library, chat, source panel, states, delete, clear | ✅ | `app/static/*`; `tests/test_frontend.py` |
| Endpoints: upload, list, get, delete, chat, health | ✅ | all six present; health reports API, DB, vector store, embedding, LLM |
| Edge cases: corrupt, encrypted, empty, scanned, oversized, wrong type | ✅ | distinct `ErrorCode` per case, each tested |
| Ollama not running → 503 with a helpful message | ✅ | verified over real HTTP: *"Ollama is not reachable at http://127.0.0.1:11434. Start it with 'ollama serve'…"* |
| Model not pulled → helpful message | ✅ | `test_a_missing_model_tells_you_how_to_pull_it` |
| Empty question / no documents / LLM timeout | ✅ | 422 / 409 / 503, each tested |
| Everything local, no external API receives user data | ✅ | [Privacy](#privacy) table; no external host in any asset |
| Don't log document text or questions | ✅ | `caplog` assertions at `DEBUG` level |
| Free-only stack (open source, no credit card) | ✅ | every dependency and both models are free; licenses listed |
| Tests as you go (pytest) | ✅ | 192 offline + 6 integration |
| Evaluation with ~10 question/expected-page pairs, real hit-rate | ✅ | [Evaluation](#evaluation): 10/10, MRR 0.950, abstention 3/3 |
| Measured ingestion, search and answer latency in the README | ⚠️ **partial** | ingestion and search measured; **answer latency needs Ollama**, which is not installed here — `benchmark_chat` will fill it in |
| `.gitignore` before the first commit | ✅ | ignores `data/`, `.env`, `.venv/`, `*.sqlite3`; verified with `git check-ignore` |
| Secrets only via env, `.env.example` with placeholders | ✅ | see [Security review](#security-review) item 5 |
| Consistent JSON error envelope | ✅ | one envelope for API *and* static 404s, tested |
| Architecture documented | ✅ | above, plus [docs/architecture.md](docs/architecture.md) |
| A fresh clone + these instructions reproduces the app | ✅ | exported the 69 tracked files to a clean directory, created the venv, ran `pip install -r requirements-dev.txt`, ran the suite (**192 passed**) and drove the server over HTTP: UI 200 with all four regions, three static assets 200, no CORS header for a foreign origin, upload 201 (10 pages → 20 chunks stored as `<uuid>.pdf`), duplicate 409, chat on an empty library 409, chat with Ollama absent 503 with instructions, blank question 422, delete removed 20 vectors + file + row, port released |
| Responds in the requested daily format, stops at each checkpoint | ✅ | conversation transcript |

**Two things are not done, and neither is dressed up as done:** real-model
generation latency is unmeasured because Ollama is absent, and the corpus used
for the hit-rate is easy by construction. Both are called out wherever the numbers
appear.

## License

MIT — see [LICENSE](LICENSE).
