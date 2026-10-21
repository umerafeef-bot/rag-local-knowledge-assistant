# Security Review — RAG Local Knowledge Assistant (DocuAsk)

- **Audit date:** 2026-10-08
- **Auditor:** automated zero-gaps audit (Codebuff)
- **Project path:** `RAG Local Knowledge Assistant/`
- **Stack:** FastAPI (factory) · Uvicorn · SQLAlchemy + SQLite · ChromaDB (local, persistent) ·
  sentence-transformers (all-MiniLM-L6-v2, CPU) · pdfplumber/pypdf · Ollama (local LLM) ·
  vanilla HTML/CSS/JS UI
- **Entry:** `python -m uvicorn --factory app.main:create_app --host 127.0.0.1 --port 8907`
- **Branch:** `main` (2 commits)

---

## 1. Authentication — NOT APPLICABLE (by design)

- Single-user local tool bound to `127.0.0.1:8907`. The README states plainly: "No authentication.
  … do not expose it to a network without putting something in front of it."

## 2. Authorization — NOT APPLICABLE

- Single-user; no ownership model. Every document belongs to the one local operator.

## 3. Input validation — PASS (by inspection and by test design)

- Upload validated three ways: extension, **streaming** size cap (`MAX_UPLOAD_MB`), and `%PDF`
  magic bytes in the first 1024 bytes (not the extension or client content-type).
- Filenames are sanitised (`sanitize_filename`) and used only as a display label; the on-disk name is
  a UUID (`<id>.pdf`). Tests assert a traversing filename never reaches the filesystem.
- Document ids are resolved in SQLite **before** any path is built, so `../../etc/passwd` is a 404.
- Distinct errors for corrupted / encrypted / empty / image-only / oversized PDFs.

## 4. API security — PASS

- One JSON envelope for every response, including static 404s; a forced crash returns
  `500 internal_error` with a generic message (no traceback/path).
- Validation errors do not echo the submitted value.
- **CORS:** configured origins only; the default `ALLOWED_ORIGINS` is **empty → no
  `Access-Control-Allow-Origin` header is ever emitted**. Credentials are never allowed; a wildcard
  origin is refused at startup.

## 5. Secrets — PASS

- `.env` is gitignored and untracked (`git ls-files` shows only `.env.example`).
- No credential-shaped literals in tracked source; the LLM model name comes from `OLLAMA_MODEL`,
  never from code.

## 6. Dependencies

- `requirements.txt` pins exact versions: fastapi 0.141.1, chromadb 1.5.9, sentence-transformers 6.1.0,
  sqlalchemy 2.0.54, pdfplumber, pypdf, httpx, pydantic.
- **Blocker:** see F1 — `chromadb 1.5.9` crashes natively on this host, which prevents the test suite
  from running here.

## 7. Database security — PASS

- SQLite holds **metadata only**; chunk text and vectors live in ChromaDB. Document text is never
  duplicated into SQLite.
- `sha256` is `UNIQUE` (duplicate-upload guard); `created_at` indexed. Deletion is ordered
  (vectors → file → row) so a document is never listed-but-unsearchable.

## 8. Data / privacy — PASS

- **Nothing leaves the machine.** PDF parsing, embeddings, the vector store and the LLM all run
  locally; the UI references no external host (no CDN/web fonts/analytics).
- Telemetry disabled for ChromaDB (`anonymized_telemetry=False`); `sqlalchemy.engine` pinned to
  `WARNING` because it can echo bound parameters (i.e. chunk text).
- Document text, filenames, questions and answers are never logged (tested at `DEBUG`).
- The only outbound network activity is the one-time embedding-model download from huggingface.co,
  which `HF_HUB_OFFLINE=1` removes.

## 9. Deployment security

- Default bind `127.0.0.1`; no hardcoded personal paths in tracked source.
- `data/uploads/<uuid>.pdf`; `data/tmp` wiped at startup (single-process assumption, documented).

---

## Findings

### F1 — `chromadb 1.5.9` segfaults natively in this environment (**REMAINING — BLOCKER**, High)

- **Symptom:** every `Collection.add(...)` call crashes the interpreter with
  **Windows fatal exception: access violation** (exit code 139). The test suite dies mid-run.
- **Minimal reproduction (independent of this project):**

  ```python
  import chromadb, numpy as np
  c = chromadb.EphemeralClient()
  col = c.get_or_create_collection('test_col', metadata={'hnsw:space':'cosine'})
  v = np.zeros((3,384), dtype=np.float32); v[:,0]=1
  col.add(ids=['a','b','c'], embeddings=v.tolist(), metadatas=[{'p':1}]*3, documents=['x','y','z'])
  ```
  → access violation. `create_collection` succeeds; `add` crashes.
- **Isolation performed:**
  - Reproduced in a **clean venv outside the project** (`/tmp/chromatest`) with a freshly
    downloaded `chromadb==1.5.9` (`--no-cache-dir`): **still crashes** → not a corrupted venv or cache.
  - Reproduced with `numpy==2.2.6` as well as `2.5.3` → not a numpy-ABI mismatch.
  - `chromadb==0.5.23` **cannot be installed** on this host (`chroma-hnswlib` wheel build fails) →
    the older, non-Rust backend is not a drop-in workaround here.
  - `pip check` → "No broken requirements found".
- **Conclusion:** this is a **native incompatibility between chromadb 1.5.9's Rust bindings and this
  Windows host**, not a defect in the DocuAsk code. The application code uses only long-stable
  ChromaDB APIs (`PersistentClient`, `get_or_create_collection`, `add`, `query`, `get`, `delete`,
  `count`) and is correct as written.
- **Impact:** the 192-test offline suite (per the README) **could not be executed on this machine**;
  document ingestion cannot run here either. The project is therefore **unverified on this host**.
- **Recommended fix (human action):** verify on the documented Ubuntu target (the README's primary
  deployment path) or on a host with a working chromadb build; if it must run on this Windows
  machine, pin a chromadb version whose wheel works here, or run it under WSL2/Docker.
- **Automated:** no — requires an environment/build-toolchain decision.

### F2 — Timing/index artefacts

- `data/tmp` is wiped at startup, which assumes a single application process (documented as a
  local-run assumption). No change required.

---

## Verification evidence

| Check | Command | Result |
|---|---|---|
| Test suite | `pytest -q` | **BLOCKED** — native chromadb crash (exit 139) |
| ChromaDB minimal repro | clean venv + fresh wheel | crashes (see F1) |
| `pip check` | `python -m pip check` | No broken requirements |
| Static review | source + tests read | controls present as documented |

> Because the suite cannot run here, the README's security table (path traversal, CORS, size/type
> checks, layout-only logging, generic errors) was **reviewed in source and test code** but **not
> independently re-executed** on this host. Treat those rows as "documented and code-verified", not
> "re-run".

## Status legend

| Item | Status |
|---|---|
| Authentication | NOT APPLICABLE (local single-user) |
| Authorization | NOT APPLICABLE |
| Input validation | PASS (code-verified; not re-executed here) |
| API security | PASS (code-verified; CORS closed by default) |
| Secrets | PASS |
| Dependencies | **BLOCKED on this host** (chromadb native crash) |
| Database security | PASS |
| Data / privacy | PASS (fully local; no telemetry) |
| Deployment security | PASS (loopback; must sit behind auth if exposed) |

## Remaining risks / required human actions

1. **Resolve F1** and re-run the full suite on the target environment — this is the gate for
   declaring the project verified.
2. Do not expose the service to a network without an authenticating proxy in front.
3. Consider pinning `numpy`/`chromadb` to a combination validated on the deployment host and
   recording it in `requirements.txt`.

## Clearly distinguished

- **FIXED:** none required (no project code changed).
- **REMAINING:** F1 (environment/library blocker, High).
- **REQUIRES HUMAN ACTION:** run the suite on a compatible host; decide the deployment platform.
- **NOT APPLICABLE:** authentication, authorization.
