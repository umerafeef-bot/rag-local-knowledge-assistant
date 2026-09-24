# Architecture

The [README](../README.md) has the diagram and the API. This document is the
*why* behind the shape of the code, and what happens when things go wrong.

## Layering

```
routes  →  services  →  repositories  →  storage
   │           │             │
   │           │             └── SQLite (metadata), ChromaDB (text + vectors)
   │           └── pdf_extraction, embedding, retrieval, prompting, chat
   └── HTTP only: validation, envelope, status codes
```

- **Routes** own HTTP concerns: status codes, the envelope, DTO conversion. They
  contain no business logic — a handler is usually four lines.
- **Services** own use cases and the failure semantics that go with them.
- **Repositories** own storage access. `DocumentRepository` is SQLAlchemy; the
  `VectorRepository` wraps ChromaDB and hides its query shape.
- **`container.py`** is the composition root. One place builds every dependency
  and hands it to the routes through `app.state`, so tests substitute the whole
  graph by passing `embeddings=` / `llm=` to `create_app` instead of
  monkeypatching module globals.

The dependency direction is strictly downward, and `app/llm/base.py` plus
`app/services/embedding.py` are the two seams: application code imports the
`Protocol`, never the implementation. That is what lets the offline test suite
run without torch or Ollama.

## Key decisions

### Why the model is never named in code

`OLLAMA_MODEL` is the only source of the model name; empty means unconfigured and
`/health` reports it as such. A hard-coded model would silently break for anyone
whose machine has a different one pulled, and would put a choice that belongs to
the operator into the source.

### Why retrieval, not the model, decides to refuse

A nearest-neighbour search always returns *something*: given any question it will
happily hand back the five least-distant chunks, and a language model asked to
answer from irrelevant context will usually produce fluent text. So
`RETRIEVAL_MIN_SIMILARITY` cuts the tail, and when nothing survives,
`ChatService.answer` returns the refusal sentence itself with `grounded: false`
and **does not call the model**. This is asserted in tests by checking the fake
LLM's `last_prompt is None` — the property that matters is that no generation was
requested, not what the words were.

The floor was chosen from measurement rather than intuition; see the threshold
sweep in the README. Briefly: every value in [0.15, 0.40] gave 10/10 hit-rate and
3/3 abstention on the evaluation corpus, 0.0 abstained on nothing, and 0.5 started
refusing correct answers.

### Why the context is capped in characters

Ollama truncates a prompt larger than the model's context window silently. The
usual symptom is an answer that ignores the end of the context, or ignores the
question entirely — both look like model failure and are not. Two settings are
kept deliberately separate:

- `MAX_CONTEXT_CHARS` (4000) — what `build_context` is allowed to paste in.
- `LLM_CONTEXT_WINDOW` (4096) — sent as Ollama's `num_ctx`, so the window is
  pinned rather than whatever the version default happens to be.

The prompt is estimated in tokens (≈4 characters/token) and a warning is logged
if it looks too large for the window — a warning, not a failure, because
over-strictness would turn a working setup into an error. When the budget cuts
the tail, the response reports `context_truncated: true` and separate
`retrieved` / `context_chunks` counts, so a shortened context is visible rather
than inferred.

### Why ingestion is all-or-nothing

An upload that fails half-way is worse than one that fails: a document listed in
the UI but missing from the vector store looks like a bug in retrieval forever
after. So the file is staged under a temporary name and only promoted to
`data/uploads/<uuid>.pdf` after every earlier step succeeded; a failure during
indexing compensates by removing vectors and deleting the file. The duplicate
guard is the unique `sha256` index, not the earlier check — the check is a fast
path for the common case, the index is the arbiter under concurrency.

### Why deletion is ordered vectors → file → row

1. **Vectors first.** If this fails, nothing else is touched, so the document is
   still fully listed and fully searchable — a loud failure rather than a quiet
   half-state.
2. **File next.** A stray file with no row is invisible garbage; a row with no
   file is a visible lie. The former is preferable, and the response reports
   `file_removed: false` when Windows refuses (typically a PDF viewer holding the
   handle).
3. **Row last**, so a partially deleted document never looks healthy.

### Why the answer path is synchronous

`def` handlers run in FastAPI's thread pool, which keeps the event loop free for
`/health` and other requests while a blocking CPU job runs. The trade-off is that
one large upload occupies one worker thread for its duration — acceptable for a
single-user local app, and background ingestion is listed as a future improvement
rather than half-built now.

### Why static assets also use the envelope

`StaticFiles` raises an ordinary `HTTPException`, which the app's `Exception`
handler converts like any other failure. The alternative — a plain-text 404 for
assets and a JSON envelope everywhere else — would mean a client needs two error
paths. This was discovered by a test asserting the opposite; the test was wrong,
the behaviour is better, and both are now pinned down.

## Failure modes

| Failure | What happens | Where |
|---|---|---|
| Not a PDF / wrong extension | 415 or 422 with a specific code; staged file removed | `validation.py` |
| Oversized upload | 413 raised *during* the streaming read, so the file is never fully buffered | `stage_upload` |
| Corrupted / encrypted / empty / image-only PDF | 422 with a distinct `ErrorCode` per case | `pdf_extraction.py` |
| Duplicate upload | 409 naming the existing document; no second row, no second set of vectors | `documents.py` + unique index |
| Failure during indexing | Compensation: vectors deleted, file removed, no row created | `documents.py::_compensate` |
| Vector-store write fails during delete | 500, nothing else touched | `documents.py::delete` |
| Ollama unreachable | 503 `llm_unavailable` with the exact command to start it | `ollama_provider.py` |
| Model not pulled | 503 naming `ollama pull <model>` | `ollama_provider.py` (404 from `/api/generate`) |
| Generation timeout | 503 naming the timeout and the env var to raise | `ollama_provider.py` |
| Empty question | 422 from the schema; a second check in the service for non-HTTP callers | `schemas.py`, `chat.py` |
| No documents indexed | 409 `no_documents` — "upload something" is different advice from "not covered" | `chat.py` |
| Filter names an unknown document | 404 rather than a silent empty result, which would look like an empty library | `retrieval.py` |
| Document deleted between vector lookup and name lookup | Chunk dropped rather than cited without a name | `retrieval.py` |
| Embedding model unloadable at startup | Logged; `/health` reports it; upload/list/delete keep working | `main.py` |
| Any unhandled exception | 500 with a generic message; the traceback goes to the log only | `api/errors.py` |

## Concurrency and process model

One application process, which is what the defaults assume:

- `data/tmp` is wiped at startup — safe because it only ever holds in-flight
  uploads, and unsafe if two processes share a `DATA_DIR`.
- SQLite is used with short, single-statement sessions (one per repository
  call), so there is no long-lived transaction to contend over.
- ChromaDB is opened by one process and used in-process.
- The embedding model is loaded lazily behind a lock, and the warm-up thread is
  a daemon so it cannot block shutdown.

## Extension points

- **Different LLM**: implement `LLMProvider` (`generate`, `health`, `close`,
  `model_name`) and pass it to `create_app(llm=...)`. Nothing else changes.
- **Different embeddings**: implement `EmbeddingService` and pass
  `embeddings=...`. Change `EMBEDDING_MODEL` too — stored vectors are only
  comparable with the model that produced them, which is why each row records
  `embedding_model`.
- **Different chunking**: `app/utils/chunking.py` is pure functions over text with
  no storage or HTTP dependencies, so it is directly unit-testable and reusable.
- **Different UI**: the frontend is three static files against a documented JSON
  API; replacing it means changing what `GET /` serves.
