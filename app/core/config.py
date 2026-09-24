"""Environment-driven configuration.

Every knob is readable from an environment variable of the same name
(case-insensitive) or from a `.env` file in the working directory. No secret
and no model name is hard-coded in application logic: the LLM model in
particular only ever comes from ``OLLAMA_MODEL``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime settings. Instantiate once per process."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application -------------------------------------------------------
    app_name: str = "DocuAsk"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    # --- Security ---------------------------------------------------------
    allowed_origins: str = Field(
        default="",
        description=(
            "Comma-separated browser origins permitted by CORS, e.g. "
            "'http://localhost:5173,http://127.0.0.1:5173'. Empty (the default) "
            "means no cross-origin access at all, which is correct for the "
            "bundled UI because it is served from this same process."
        ),
    )

    # --- Storage -----------------------------------------------------------
    data_dir: Path = Path("data")
    max_upload_mb: int = Field(default=50, ge=1, le=2048)
    max_pdf_pages: int = Field(default=2000, ge=1)
    min_extractable_chars: int = Field(
        default=40,
        ge=1,
        description=(
            "Total extracted character count below which a PDF is treated as "
            "scanned / image-only. OCR is out of scope for the MVP."
        ),
    )

    # --- Chunking / retrieval ---------------------------------------------
    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=100, ge=0)
    top_k: int = Field(default=5, ge=1, le=50)
    retrieval_min_similarity: float = Field(
        default=0.30,
        ge=-1.0,
        le=1.0,
        description=(
            "Cosine-similarity floor. Retrieval is a nearest-neighbour search, "
            "so it always returns top-K chunks even when none of them are "
            "relevant; hits below this score are dropped, which is what lets "
            "the app say 'I found nothing' instead of feeding noise to the model."
        ),
    )
    max_context_chars: int = Field(
        default=4000,
        ge=200,
        le=20000,
        description="Hard budget for the retrieved context pasted into the prompt.",
    )
    source_snippet_chars: int = Field(
        default=240,
        ge=40,
        le=2000,
        description="Length of the text excerpt returned with each cited source.",
    )

    # --- Generation -------------------------------------------------------
    chat_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    chat_max_tokens: int = Field(
        default=512,
        ge=16,
        le=8192,
        description="Upper bound on generated tokens (Ollama option num_predict).",
    )
    llm_context_window: int = Field(
        default=4096,
        ge=512,
        description=(
            "Value sent as Ollama's num_ctx. Ollama's own default is 4096 "
            "tokens (2048 in older releases) and it truncates silently when a "
            "prompt exceeds it, so it is pinned explicitly here: everything in "
            "MAX_CONTEXT_CHARS has to survive into the model's window."
        ),
    )
    llm_keep_alive: str = Field(
        default="5m",
        description=(
            "How long Ollama keeps the model resident after a request "
            "(Ollama's keep_alive). Without it, a question asked after a few "
            "minutes of idling pays the model-load cost again."
        ),
    )

    # --- Embeddings (local, CPU) ------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    embedding_warmup: bool = Field(
        default=True,
        description=(
            "Load the embedding model in a background thread at startup. The "
            "first encode() call costs tens of seconds of one-time kernel/"
            "tokenizer initialisation (measured: ~35 s); warming up keeps the "
            "first upload from appearing to hang."
        ),
    )

    # --- Local LLM (Ollama) ------------------------------------------------
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = ""
    ollama_timeout_seconds: float = Field(default=120.0, gt=0)
    ollama_health_timeout_seconds: float = Field(default=3.0, gt=0)

    # --- Vector store ------------------------------------------------------
    collection_name: str = "docuask_chunks"

    @model_validator(mode="after")
    def _validate_chunking(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE "
                f"(got overlap={self.chunk_overlap}, size={self.chunk_size})"
            )
        return self

    @model_validator(mode="after")
    def _validate_cors(self) -> Settings:
        """Validate ALLOWED_ORIGINS eagerly, at startup.

        Failing fast means a typo stops the server with one clear line instead
        of silently allowing no origin (or, with '*', every origin) and
        surfacing later as a confusing browser error. The value itself is
        derived on read by ``cors_origins`` rather than cached, so a Settings
        rebuilt with ``model_copy`` cannot serve a stale list.
        """
        self._parse_origins()
        return self

    @property
    def cors_origins(self) -> tuple[str, ...]:
        """Validated origins to allow. Empty tuple means CORS stays closed."""
        return self._parse_origins()

    def _parse_origins(self) -> tuple[str, ...]:
        origins = tuple(
            part.strip() for part in self.allowed_origins.split(",") if part.strip()
        )
        for origin in origins:
            if origin == "*":
                raise ValueError(
                    "ALLOWED_ORIGINS must list explicit origins, not '*': a wildcard "
                    "would let any website read responses from this server."
                )
            parsed = urlparse(origin)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError(
                    f"ALLOWED_ORIGINS entry {origin!r} is not a valid origin. "
                    "Expected the form scheme://host[:port] with no path, for "
                    "example http://localhost:5173."
                )
            if parsed.path or parsed.query or parsed.fragment or parsed.username:
                raise ValueError(
                    f"ALLOWED_ORIGINS entry {origin!r} must be an origin only "
                    "(scheme://host[:port]) -- no path, query, fragment or credentials."
                )
        return origins

    # --- Derived paths -----------------------------------------------------
    @property
    def upload_dir(self) -> Path:
        """Final resting place of accepted PDFs, named by document UUID."""
        return self.data_dir / "uploads"

    @property
    def temp_dir(self) -> Path:
        """Staging area for in-flight uploads.

        Deliberately inside ``data_dir`` so that promoting a staged file to its
        final name is an atomic ``os.replace`` on the same filesystem.
        """
        return self.data_dir / "tmp"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "docuask.sqlite3"

    @property
    def database_url(self) -> str:
        return f"sqlite+pysqlite:///{self.database_path.resolve().as_posix()}"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def llm_is_configured(self) -> bool:
        return bool(self.ollama_model.strip())

    # --- Filesystem helpers ------------------------------------------------
    def ensure_directories(self) -> None:
        for directory in (self.data_dir, self.upload_dir, self.temp_dir, self.chroma_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def reset_temp_dir(self) -> None:
        """Drop staged uploads left behind by a previous crash.

        Safe because the staging directory only ever holds in-flight uploads.
        Assumes a single application process, which is how DocuAsk is run
        locally.
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
