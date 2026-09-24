"""Logging setup.

Privacy rule enforced here: DocuAsk never logs document text, questions or
answers. Third-party loggers that would do so are turned down to WARNING --
in particular ``sqlalchemy.engine``, which can log SQL statements together
with their bound parameters (which include chunk text).
"""

from __future__ import annotations

import logging
import logging.config

# Libraries that are chatty or that can echo user content at INFO/DEBUG.
_QUIET_LOGGERS = (
    "sqlalchemy.engine",
    "sqlalchemy.engine.Engine",
    "sqlalchemy.pool",
    "chromadb",
    "chromadb.telemetry",
    "httpx",
    "httpcore",
    "urllib3",
    "sentence_transformers",
    "transformers",
    "huggingface_hub",
    "filelock",
    "pdfminer",
    "PIL",
)

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """Install a single stdout handler and quieten noisy third parties."""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {"format": _LOG_FORMAT, "datefmt": _DATE_FORMAT},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {"handlers": ["console"], "level": level.upper()},
        }
    )
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
