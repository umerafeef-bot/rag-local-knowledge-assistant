"""Ollama provider: request shape, health probing and error mapping.

A ``httpx.MockTransport`` runs the real provider code (real serialisation, real
status handling) without needing an Ollama server on the machine.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.core.errors import ServiceUnavailableError
from app.llm.ollama_provider import OllamaProvider

BASE_URL = "http://127.0.0.1:11434"


def build_provider(handler, *, model: str = "llama3.2:3b") -> tuple[OllamaProvider, httpx.Client]:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    provider = OllamaProvider(
        base_url=BASE_URL,
        model=model,
        timeout_seconds=5.0,
        health_timeout_seconds=5.0,
        client=client,
    )
    return provider, client


def tags_handler(model_names: tuple[str, ...]):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": name} for name in model_names]})

    return handler


# --- health ----------------------------------------------------------------
def test_health_reports_the_configured_model_as_available() -> None:
    provider, client = build_provider(tags_handler(("llama3.2:3b", "qwen2.5:3b")))
    try:
        health = provider.health()
    finally:
        client.close()

    assert health.ok is True
    assert health.reachable is True
    assert health.model_available is True
    assert health.provider == "ollama"
    assert set(health.available_models) == {"llama3.2:3b", "qwen2.5:3b"}
    assert health.detail is None


def test_a_tagless_model_matches_any_tag() -> None:
    provider, client = build_provider(tags_handler(("llama3.2:3b",)), model="llama3.2")
    try:
        assert provider.health().model_available is True
    finally:
        client.close()


def test_a_missing_model_tells_you_how_to_pull_it() -> None:
    provider, client = build_provider(tags_handler(("llama3.2:1b",)))
    try:
        health = provider.health()
    finally:
        client.close()

    assert health.ok is False
    assert health.reachable is True
    assert health.model_available is False
    assert health.detail is not None
    assert "ollama pull llama3.2:3b" in health.detail


def test_health_reports_an_unreachable_server() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider, client = build_provider(handler)
    try:
        health = provider.health()
    finally:
        client.close()

    assert health.ok is False
    assert health.reachable is False
    assert health.detail is not None
    assert BASE_URL in health.detail


def test_health_reports_a_server_error_status() -> None:
    provider, client = build_provider(lambda request: httpx.Response(500, text="boom"))
    try:
        health = provider.health()
    finally:
        client.close()

    assert health.reachable is False
    assert health.detail is not None
    assert "500" in health.detail


def test_health_reports_a_malformed_model_list() -> None:
    provider, client = build_provider(
        lambda request: httpx.Response(200, json={"models": "not-a-list"})
    )
    try:
        health = provider.health()
    finally:
        client.close()

    assert health.reachable is True
    assert health.model_available is False
    assert health.detail is not None


def test_health_reports_an_unset_model() -> None:
    provider, client = build_provider(tags_handler(("llama3.2:3b",)), model="")
    try:
        health = provider.health()
    finally:
        client.close()

    assert health.reachable is True
    assert health.model_available is False
    assert health.model == ""
    assert health.detail is not None
    assert "OLLAMA_MODEL" in health.detail


# --- generation ------------------------------------------------------------
def test_generate_sends_the_expected_payload() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"response": "Gravity pulls things down."})

    provider, client = build_provider(handler)
    try:
        answer = provider.generate(
            "Why do things fall?",
            system="Answer only from the context.",
            temperature=0.2,
            max_tokens=64,
        )
    finally:
        client.close()

    assert answer == "Gravity pulls things down."
    assert len(captured) == 1
    assert captured[0].url.path == "/api/generate"
    payload = json.loads(captured[0].content)
    assert payload["model"] == "llama3.2:3b"
    assert payload["prompt"] == "Why do things fall?"
    assert payload["system"] == "Answer only from the context."
    assert payload["stream"] is False
    assert payload["options"] == {"temperature": 0.2, "num_predict": 64}


def test_generate_omits_the_system_prompt_when_absent() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"response": "ok"})

    provider, client = build_provider(handler)
    try:
        provider.generate("Question?")
    finally:
        client.close()

    payload = json.loads(captured[0].content)
    assert "system" not in payload
    assert payload["options"] == {"temperature": 0.1}


def test_generate_reports_a_missing_model() -> None:
    provider, client = build_provider(
        lambda request: httpx.Response(404, json={"error": "model not found"})
    )
    try:
        with pytest.raises(ServiceUnavailableError) as error:
            provider.generate("Question?")
    finally:
        client.close()

    assert error.value.status_code == 503
    assert "ollama pull llama3.2:3b" in error.value.message


def test_generate_reports_a_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    provider, client = build_provider(handler)
    try:
        with pytest.raises(ServiceUnavailableError) as error:
            provider.generate("Question?")
    finally:
        client.close()

    assert "did not answer within" in error.value.message


def test_generate_reports_an_unreachable_server() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider, client = build_provider(handler)
    try:
        with pytest.raises(ServiceUnavailableError) as error:
            provider.generate("Question?")
    finally:
        client.close()

    assert "ollama serve" in error.value.message


def test_generate_reports_a_server_error() -> None:
    provider, client = build_provider(lambda request: httpx.Response(500, text="boom"))
    try:
        with pytest.raises(ServiceUnavailableError) as error:
            provider.generate("Question?")
    finally:
        client.close()

    assert error.value.status_code == 503


def test_generate_rejects_an_empty_answer() -> None:
    provider, client = build_provider(lambda request: httpx.Response(200, json={"response": "  "}))
    try:
        with pytest.raises(ServiceUnavailableError):
            provider.generate("Question?")
    finally:
        client.close()


def test_generate_without_a_configured_model_fails_fast() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request should be made")

    provider, client = build_provider(handler, model="")
    try:
        with pytest.raises(ServiceUnavailableError) as error:
            provider.generate("Question?")
    finally:
        client.close()

    assert "OLLAMA_MODEL" in error.value.message


def test_close_only_closes_clients_it_owns() -> None:
    provider, client = build_provider(tags_handler(("llama3.2:3b",)))
    provider.close()
    assert client.is_closed is False  # injected client belongs to the caller
    client.close()

    owned, _ = build_provider(tags_handler(("llama3.2:3b",)))
    owned.close()
