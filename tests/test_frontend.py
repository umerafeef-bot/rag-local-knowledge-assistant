"""The bundled web UI: it must be served, and it must not phone home.

The UI is three static files (HTML, CSS, vanilla JS) served by the same process
that serves the API, so there is no build step and no CDN. These tests pin that
down: no external URL may appear anywhere in the assets, and no API string may
be assigned with ``innerHTML``.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

STATIC_DIR = Path(__file__).resolve().parents[1] / "app" / "static"
EXTERNAL_URL = re.compile(r"https?://")
INNER_HTML = re.compile(r"\.innerHTML\s*=")


# --- serving ---------------------------------------------------------------
def test_root_serves_the_ui(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>DocuAsk" in response.text
    assert 'id="upload-form"' in response.text
    assert 'id="messages"' in response.text
    assert 'id="doc-list"' in response.text


def test_the_three_assets_are_served(client: TestClient) -> None:
    assets = {
        "/static/app.js": "javascript",
        "/static/styles.css": "text/css",
        "/static/index.html": "text/html",
    }
    for path, content_type in assets.items():
        response = client.get(path)
        assert response.status_code == 200, path
        assert content_type in response.headers["content-type"], path


def test_the_ui_assets_must_be_revalidated_not_cached_blindly(client: TestClient) -> None:
    """An updated app.js must not be shadowed by a browser's stale copy.

    Without Cache-Control the browser is free to reuse a cached asset without
    asking, so a UI fix can appear to do nothing until a hard refresh.
    """
    for path in ("/", "/static/app.js", "/static/styles.css", "/static/index.html"):
        response = client.get(path)

        assert response.status_code == 200, path
        assert response.headers.get("cache-control") == "no-cache", path


def test_a_missing_asset_uses_the_shared_envelope(client: TestClient) -> None:
    """StaticFiles raises an HTTPException, which the app maps to the envelope.

    Asserted rather than assumed: an asset miss looks exactly like any other
    failed request, so a client needs one error branch, not two.
    """
    response = client.get("/static/nope.js")

    assert response.status_code == 404
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "http_error"


def test_the_service_descriptor_moved_to_info(client: TestClient) -> None:
    """/ is the UI now; the JSON descriptor it replaced still exists at /info."""
    response = client.get("/info")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["name"] == "DocuAsk"
    assert data["docs_url"] == "/docs"


def test_the_ui_route_is_not_in_the_openapi_schema(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert "/" not in paths
    assert "/info" in paths


# --- the UI must be self-contained -----------------------------------------
def test_no_asset_references_an_external_host(client: TestClient) -> None:
    """Privacy: nothing the browser loads may come from off this machine."""
    for path in ("/", "/static/app.js", "/static/styles.css"):
        body = client.get(path).text
        match = EXTERNAL_URL.search(body)
        assert match is None, f"{path} references an external host: {match and match.group(0)}"


def test_the_ui_builds_its_dom_without_inner_html() -> None:
    """XSS: API strings (filenames, answers, snippets) go in as text, not markup."""
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert INNER_HTML.search(script) is None, "app.js must never assign innerHTML"
    assert "document.write" not in script
    # Every server-sourced string is written through textContent / createElement.
    assert "textContent" in script
    assert "createElement" in script


def test_the_html_declares_its_language_and_viewport(client: TestClient) -> None:
    """Accessibility basics: a responsive viewport and a declared language."""
    body = client.get("/").text

    assert 'lang="en"' in body
    assert 'name="viewport"' in body
    assert 'name="description"' in body
