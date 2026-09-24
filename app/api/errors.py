"""Exception handlers.

Three rules:

* every failure leaves the API as the same JSON envelope,
* no internal detail (traceback, SQL, path, library message) reaches the client,
* request-validation errors are logged and returned **without** echoing the
  submitted values -- an upload's bytes or a future chat question must not end
  up in a response body or a log line.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.schemas import Envelope, ErrorBody
from app.core.errors import AppError, ErrorCode

logger = logging.getLogger(__name__)


def _envelope_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = Envelope(success=False, error=ErrorBody(code=code, message=message, details=details))
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _handle_app_error)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _handle_validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, _handle_http_error)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _handle_unexpected_error)


async def _handle_app_error(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)  # registered for AppError only
    logger.warning("Request failed: %s - %s", exc.code.value, exc.message)
    return _envelope_response(
        status_code=exc.status_code,
        code=exc.code.value,
        message=exc.message,
        details=exc.details,
    )


async def _handle_validation_error(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # "input" and "ctx" are dropped on purpose: they can contain user data.
    details = [
        {
            "location": ".".join(str(part) for part in error.get("loc", ())),
            "message": error.get("msg", "invalid value"),
        }
        for error in exc.errors()
    ]
    logger.warning("Request validation failed: %s", details)
    return _envelope_response(
        status_code=422,
        code=ErrorCode.VALIDATION_ERROR.value,
        message="The request did not pass validation.",
        details={"fields": details},
    )


async def _handle_http_error(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    return _envelope_response(
        status_code=exc.status_code,
        code=ErrorCode.HTTP_ERROR.value,
        message=str(exc.detail),
        headers=getattr(exc, "headers", None),
    )


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # Log the path only: the query string can carry a question and the body can
    # carry a document.
    logger.exception("Unhandled error while processing %s %s", request.method, request.url.path)
    return _envelope_response(
        status_code=500,
        code=ErrorCode.INTERNAL_ERROR.value,
        message="An unexpected internal error occurred. Check the server logs.",
    )
