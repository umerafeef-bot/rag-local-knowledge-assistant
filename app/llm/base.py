"""The LLM boundary.

Application code imports these types only -- never ``OllamaProvider`` -- so the
backend can be swapped (llama.cpp server, a different OpenAI-compatible local
server, a recorded fixture in tests) without touching the routes or services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class LLMHealth:
    """Result of probing the LLM backend, reported verbatim by GET /health."""

    provider: str
    model: str
    ok: bool
    reachable: bool
    model_available: bool
    detail: str | None = None
    available_models: tuple[str, ...] = ()


@runtime_checkable
class LLMProvider(Protocol):
    """Answers questions from a prompt, locally."""

    @property
    def model_name(self) -> str:
        """The configured model, or an empty string when none is configured."""
        ...

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        """Return the model's answer.

        Raises :class:`~app.core.errors.ServiceUnavailableError` when the
        backend is unreachable, the model is missing, the call times out or the
        answer is empty.
        """
        ...

    def health(self) -> LLMHealth:
        """Probe the backend without generating anything. Must not raise."""
        ...

    def close(self) -> None:
        """Release network resources."""
        ...
