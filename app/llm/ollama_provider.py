"""Ollama implementation of :class:`~app.llm.base.LLMProvider`.

Talks to a local Ollama server over plain HTTP (default
``http://127.0.0.1:11434``). The model name comes exclusively from
``OLLAMA_MODEL`` -- nothing here assumes a particular model exists.

Error mapping matters more than the happy path: the user needs to be told which
one of "Ollama is not running", "the model is not pulled", "the model is slow"
happened, without leaking a stack trace.
"""

from __future__ import annotations

import logging
import time

import httpx

from app.core.errors import ServiceUnavailableError
from app.llm.base import LLMHealth

logger = logging.getLogger(__name__)

_MALFORMED_DETAIL = "Ollama returned a malformed response."


class OllamaProvider:
    """LLM provider backed by a local Ollama server.

    Talks plain HTTP with no client library: the API surface used here is two
    endpoints and one JSON body, so a dependency would buy nothing.
    """

    provider_name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        health_timeout_seconds: float,
        context_window: int | None = None,
        keep_alive: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model.strip()
        self._timeout = timeout_seconds
        self._health_timeout = health_timeout_seconds
        self._context_window = context_window
        self._keep_alive = keep_alive
        self._client = client or httpx.Client(base_url=self._base_url, timeout=timeout_seconds)
        self._owns_client = client is None

    @property
    def model_name(self) -> str:
        return self._model

    # --- generation --------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        if not self._model:
            raise ServiceUnavailableError(
                "OLLAMA_MODEL is not configured. Set it in .env (for example "
                "OLLAMA_MODEL=llama3.2:3b) and restart the app."
            )

        options: dict[str, object] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if self._context_window is not None:
            # Ollama truncates a prompt that exceeds the window without saying
            # so, which would silently drop retrieved context. Pinning it makes
            # the budget in MAX_CONTEXT_CHARS meaningful.
            options["num_ctx"] = self._context_window

        payload: dict[str, object] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system
        if self._keep_alive:
            # Keeps the model resident between questions instead of paying the
            # load cost again after a few idle minutes.
            payload["keep_alive"] = self._keep_alive

        try:
            started = time.perf_counter()
            response = self._client.post("/api/generate", json=payload, timeout=self._timeout)
        except httpx.TimeoutException as exc:
            # Both connect and read timeouts land here: the model may be slow to
            # load or genuinely too slow for this machine.
            raise ServiceUnavailableError(
                f"The local model did not answer within {self._timeout:.0f}s. "
                "Try a smaller model, or raise OLLAMA_TIMEOUT_SECONDS if the "
                "model is still loading."
            ) from exc
        except httpx.HTTPError as exc:
            raise ServiceUnavailableError(self._unreachable_message()) from exc

        if response.status_code == 404:
            raise ServiceUnavailableError(
                f"Ollama does not have the model '{self._model}'. "
                f"Pull it with: ollama pull {self._model}"
            )
        if response.status_code >= 400:
            logger.warning("Ollama returned HTTP %s for a generation request", response.status_code)
            raise ServiceUnavailableError(
                "Ollama rejected the generation request. Check the Ollama logs for details."
            )

        try:
            answer = str(response.json().get("response", ""))
        except (ValueError, AttributeError) as exc:
            raise ServiceUnavailableError(_MALFORMED_DETAIL) from exc

        if not answer.strip():
            raise ServiceUnavailableError("The local model returned an empty answer.")

        # Log the shape of the call, never the prompt or the answer.
        logger.info(
            "Ollama generated %d characters in %.2fs",
            len(answer),
            time.perf_counter() - started,
        )
        return answer.strip()

    # --- health ------------------------------------------------------------
    def health(self) -> LLMHealth:
        """Never raises: every failure becomes structured health data."""
        try:
            response = self._client.get("/api/tags", timeout=self._health_timeout)
        except httpx.HTTPError:
            return LLMHealth(
                provider=self.provider_name,
                model=self._model,
                ok=False,
                reachable=False,
                model_available=False,
                detail=f"Ollama is not reachable at {self._base_url}.",
            )

        if response.status_code >= 400:
            return LLMHealth(
                provider=self.provider_name,
                model=self._model,
                ok=False,
                reachable=False,
                model_available=False,
                detail=f"Ollama answered HTTP {response.status_code} at {self._base_url}.",
            )

        try:
            payload = response.json()
            available = tuple(
                str(entry.get("name", ""))
                for entry in payload.get("models", [])
                if entry.get("name")
            )
        except (ValueError, AttributeError):
            return LLMHealth(
                provider=self.provider_name,
                model=self._model,
                ok=False,
                reachable=True,
                model_available=False,
                detail="Ollama returned a model list that could not be read.",
            )

        if not self._model:
            return LLMHealth(
                provider=self.provider_name,
                model="",
                ok=False,
                reachable=True,
                model_available=False,
                detail="Ollama is running but OLLAMA_MODEL is not set.",
                available_models=available,
            )

        matched = self._matches(self._model, available)
        return LLMHealth(
            provider=self.provider_name,
            model=self._model,
            ok=matched,
            reachable=True,
            model_available=matched,
            detail=None if matched else f"Model '{self._model}' is not pulled yet. Run: ollama pull {self._model}",
            available_models=available,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def _matches(configured: str, available: tuple[str, ...]) -> bool:
        """A tag-less configuration matches any tag of that model."""
        if configured in available:
            return True
        if ":" not in configured:
            return any(name.split(":")[0] == configured for name in available)
        return False

    def _unreachable_message(self) -> str:
        return (
            f"Ollama is not reachable at {self._base_url}. Start it with "
            "'ollama serve' (or launch the Ollama desktop app) and try again."
        )


__all__ = ["OllamaProvider"]
