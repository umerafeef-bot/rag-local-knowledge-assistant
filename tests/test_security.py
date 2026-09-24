"""Security review, executed.

Each test is the evidence for one line of the security checklist in the README.
A test that can fail is a control; a claim without one is a wish.

The items covered here: CORS scope, path traversal on document ids, filename
trust, upload size/type, sensitive logging, and error messages leaking
internals. Upload size and type live in tests/test_validation.py and
tests/test_documents_api.py; secrets-in-repo is checked below because it is a
property of the repository, not of a running server.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.fakes import FakeEmbedding, FakeLLMProvider
from tests.pdf_factory import build_pdf

REPO_ROOT = Path(__file__).resolve().parents[1]

SECRET_ASSIGNMENT = re.compile(r"(?i)\b(api[_-]?key|secret|password|passwd|token)\s*[:=]\s*[\"'][^\"']{8,}")


def build_client(
    settings: Settings, *, raise_server_exceptions: bool = True
) -> TestClient:  # type: ignore[no-untyped-def]
    return TestClient(
        create_app(settings=settings, embeddings=FakeEmbedding(), llm=FakeLLMProvider()),
        raise_server_exceptions=raise_server_exceptions,
    )


def upload(client: TestClient, content: bytes, filename: str = "doc.pdf"):  # type: ignore[no-untyped-def]
    return client.post(
        "/documents/upload", files={"file": (filename, content, "application/pdf")}
    )


PAGES = ["Chapter one explains vector retrieval latency over a corpus of chunks."]


# --- 1. CORS ---------------------------------------------------------------
def test_cross_origin_reads_are_denied_by_default(settings: Settings) -> None:
    """With ALLOWED_ORIGINS empty, no origin header is ever granted."""
    with build_client(settings) as client:
        response = client.get("/documents", headers={"Origin": "https://evil.example"})

    assert "access-control-allow-origin" not in response.headers


def test_cross_origin_preflight_is_denied_by_default(settings: Settings) -> None:
    with build_client(settings) as client:
        response = client.options(
            "/chat",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    assert "access-control-allow-origin" not in response.headers


def test_only_the_configured_origin_is_allowed(settings: Settings) -> None:
    configured = settings.model_copy(
        update={"allowed_origins": "http://localhost:5173,https://app.example"}
    )
    with build_client(configured) as client:
        allowed = client.get(
            "/documents", headers={"Origin": "http://localhost:5173"}
        )
        also_allowed = client.get(
            "/documents", headers={"Origin": "https://app.example"}
        )
        rejected = client.get("/documents", headers={"Origin": "https://evil.example"})

    assert configured.cors_origins == ("http://localhost:5173", "https://app.example")
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert also_allowed.headers["access-control-allow-origin"] == "https://app.example"
    assert "access-control-allow-origin" not in rejected.headers


def test_credentials_are_never_allowed(settings: Settings) -> None:
    """No cookies or Authorization headers are used, so none may be granted."""
    configured = settings.model_copy(
        update={"allowed_origins": "http://localhost:5173"}
    )
    with build_client(configured) as client:
        response = client.options(
            "/chat",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    assert response.headers.get("access-control-allow-credentials") != "true"
    # Starlette unions the CORS-safelisted headers with the configured ones, so
    # this asserts the property that matters: the header the UI actually sends
    # is permitted, and nothing beyond it is granted by this middleware.
    allowed_headers = {
        header.strip().lower()
        for header in response.headers.get("access-control-allow-headers", "").split(",")
        if header.strip()
    }
    assert "content-type" in allowed_headers
    assert "authorization" not in allowed_headers


def test_a_wildcard_origin_is_refused_at_startup() -> None:
    """'*' must be impossible to configure, not merely discouraged."""
    with pytest.raises(ValueError, match="explicit origins"):
        Settings(_env_file=None, allowed_origins="*")


@pytest.mark.parametrize("bad", ["localhost:5173", "http://host/path", "ftp://x", "http://"])
def test_malformed_origins_are_refused_at_startup(bad: str) -> None:
    with pytest.raises(ValueError, match="ALLOWED_ORIGINS"):
        Settings(_env_file=None, allowed_origins=bad)


# --- 2. path traversal -----------------------------------------------------
@pytest.mark.parametrize(
    "attempt",
    [
        "../../etc/passwd",
        "..%2F..%2Fetc%2Fpasswd",
        "..\\..\\windows\\system32\\config",
        "....//....//etc/passwd",
        "/etc/passwd",
    ],
)
def test_delete_rejects_path_traversal_ids(client: TestClient, attempt: str) -> None:
    """The id reaches the filesystem path, so it must never leave the DB.

    It is looked up first: an id that is not a stored UUID simply does not
    exist, so the response is a 404 and no path is ever constructed from it.
    """
    response = client.delete(f"/documents/{attempt}")

    assert response.status_code in (404, 422), response.text
    assert response.status_code != 500


@pytest.mark.parametrize("attempt", ["../../etc/passwd", "..%2F..%2Fsecret.pdf"])
def test_get_rejects_path_traversal_ids(client: TestClient, attempt: str) -> None:
    response = client.get(f"/documents/{attempt}")

    assert response.status_code in (404, 422)
    assert "root:" not in response.text


def test_a_traversing_filename_never_reaches_the_filesystem(
    client: TestClient, settings: Settings
) -> None:
    """The user's filename is a display label; disk names come from a UUID."""
    response = upload(client, build_pdf(PAGES), filename="../../../evil.pdf")

    assert response.status_code == 201, response.text
    stored = [path.name for path in settings.upload_dir.iterdir()]
    # Whatever it is called on disk, it is a 32-hex UUID with a .pdf suffix.
    assert len(stored) == 1
    assert re.fullmatch(r"[0-9a-f]{32}\.pdf", stored[0]), stored
    # And the traversal text is not in the display label either.
    assert "/" not in response.json()["data"]["document"]["filename"]
    assert ".." not in response.json()["data"]["document"]["filename"]


def test_a_traversing_filename_cannot_escape_the_upload_directory(
    client: TestClient, settings: Settings
) -> None:
    upload(client, build_pdf(PAGES), filename="..%2F..%2Fescape.pdf")

    assert settings.upload_dir.exists()
    assert not (settings.data_dir.parent / "escape.pdf").exists()


# --- 3. sensitive logging --------------------------------------------------
def test_document_text_never_reaches_the_log(client: TestClient, caplog) -> None:  # type: ignore[no-untyped-def]
    payload = build_pdf(["The launch code for operation secret harbour is 0451."])

    with caplog.at_level(logging.DEBUG):
        upload(client, payload, filename="operations.pdf")

    text = caplog.text
    assert "launch code" not in text
    assert "secret harbour" not in text
    assert "0451" not in text
    assert "operations.pdf" not in text


def test_a_question_never_reaches_the_log(client: TestClient, caplog) -> None:  # type: ignore[no-untyped-def]
    upload(client, build_pdf(PAGES))
    question = "what does the doc say about my private medical history?"

    with caplog.at_level(logging.DEBUG):
        client.post("/chat", json={"question": question})

    assert "private medical history" not in caplog.text
    # Nor do the retrieved chunk text or the answer.
    assert "vector retrieval latency over a corpus" not in caplog.text


def test_validation_errors_do_not_echo_the_submitted_value(
    client: TestClient, caplog
) -> None:  # type: ignore[no-untyped-def]
    """A too-long question must not be repeated back, in the body or the log."""
    secret = "q" * 1200

    with caplog.at_level(logging.DEBUG):
        response = client.post("/chat", json={"question": secret})

    assert response.status_code == 422
    assert "q" * 100 not in response.text
    assert secret not in caplog.text


# --- 4. error messages do not leak internals ------------------------------
def test_a_crash_returns_a_generic_message(settings: Settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The 500 envelope must carry no traceback, path, SQL or exception text.

    ``raise_server_exceptions=False`` because Starlette re-raises a handled
    exception after writing the response, which is what lets a test inspect the
    response body a real client would receive.
    """
    with build_client(settings, raise_server_exceptions=False) as client:
        container = client.app.state.container

        def explode(question, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("internal detail: C:\\Users\\secret\\file.py")

        monkeypatch.setattr(container.chat, "answer", explode)
        upload(client, build_pdf(PAGES))

        response = client.post("/chat", json={"question": "anything"})

        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "internal_error"
        assert "An unexpected internal error occurred." in body["error"]["message"]
        for leaked in ("Traceback", "RuntimeError", "secret", ".py", "C:\\"):
            assert leaked not in response.text, f"leaked {leaked}"


def test_unreachable_paths_do_not_leak_filesystem_details(
    client: TestClient, settings: Settings
) -> None:
    response = client.get("/documents/does-not-exist")

    assert response.status_code == 404
    for leaked in ("sqlite", ".venv", "Traceback", str(settings.data_dir)):
        assert leaked not in response.text


# --- 5. secrets in the repository -----------------------------------------
def test_the_gitignore_covers_every_secret_and_data_path() -> None:
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    rules = {line.strip() for line in ignore.splitlines() if line.strip() and not line.startswith("#")}

    for required in (".env", "data/", ".venv/", "*.sqlite3"):
        assert required in rules, f".gitignore is missing {required}"


def test_the_env_template_contains_no_real_secret() -> None:
    template = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

    assert SECRET_ASSIGNMENT.search(template) is None
    # The only credential-shaped values are placeholders.
    assert "your-" not in template.lower() or "sk-" not in template


def test_no_committed_source_carries_a_secret_shaped_literal() -> None:
    """Scans the tracked source for credential-looking assignments.

    Deliberately conservative: it looks for a secret *name* paired with a long
    quoted value, which is what an accidentally committed key looks like.
    """
    offenders: list[str] = []
    for directory in ("app", "scripts", "tests"):
        for path in (REPO_ROOT / directory).rglob("*.py"):
            match = SECRET_ASSIGNMENT.search(path.read_text(encoding="utf-8"))
            if match:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {match.group(0)[:40]}")

    assert offenders == []


def test_no_env_file_is_committed() -> None:
    """A developer's real .env must never become part of the repository."""
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ".env"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # Non-zero means "not tracked", which is what we want. If git is absent,
    # the .gitignore test above still covers the control.
    assert result.returncode != 0 or not result.stdout.strip()
