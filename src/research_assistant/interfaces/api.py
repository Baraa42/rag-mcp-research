"""Local HTTP interface over one serialized ResearchService instance."""

import argparse
import json
import logging
import math
import sys
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

import anyio
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse

from research_assistant.generation.answering import MAX_QUESTION_CHARACTERS
from research_assistant.service import ResearchService


class QuestionRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    question: str

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class SearchRequest(QuestionRequest):
    top_k: int = Field(default=5, ge=1, le=50)
    rerank: bool = True


class AnswerRequest(QuestionRequest):
    question: str = Field(max_length=MAX_QUESTION_CHARACTERS)


class StatusResponse(BaseModel):
    status: Literal["ok", "ready"]


class PassageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    chunk_id: str
    document_id: str
    source: str
    page_number: int | None
    section: str | None
    text: str


class SearchHit(PassageResponse):
    score: float = Field(allow_inf_nan=False)


class CitationResponse(PassageResponse):
    label: str


class SearchResponse(BaseModel):
    results: list[SearchHit]


class AnswerResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    answer: str
    sources: list[CitationResponse]


class RequestLoggingMiddleware:
    """One safe JSON record per HTTP request; no body/header/provider logging."""

    def __init__(self, app):
        self.app = app
        self.logger = logging.Logger("research_assistant.http", logging.INFO)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = uuid4().hex
        state = scope.setdefault("state", {})
        start = perf_counter()
        status = 500
        started = False

        async def send_response(message):
            nonlocal status, started
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
                started = True
            await send(message)

        try:
            await self.app(scope, receive, send_response)
        except anyio.get_cancelled_exc_class():
            state["error_category"] = "request_cancelled"
            if not started:
                status = 499
            raise
        except Exception:
            state["error_category"] = "internal_error"
            if started:
                raise
            await JSONResponse({"detail": "Internal server error"}, status_code=500)(
                scope, receive, send_response
            )
        finally:
            category = state.get("error_category")
            if category is None and status >= 400:
                category = {422: "invalid_request", 503: "service_unavailable"}.get(
                    status, "http_error"
                )
            self.logger.info(
                json.dumps(
                    {
                        "request_id": request_id,
                        "method": scope["method"],
                        # Never log arbitrary URLs or query strings containing user data.
                        "path": getattr(scope.get("route"), "path", "[unmatched]"),
                        "status": status,
                        "duration_ms": round((perf_counter() - start) * 1000, 3),
                        "error_category": category,
                    }
                )
            )


def backend_error(request: Request, error: Exception) -> HTTPException:
    """Classify provider failures without copying any provider response content."""
    if isinstance(error, APITimeoutError):
        status, category, message = 504, "upstream_timeout", "Answer provider timed out"
    elif isinstance(error, (RateLimitError, APIConnectionError)) or (
        isinstance(error, APIStatusError) and error.status_code >= 500
    ):
        status, category, message = (
            503,
            "upstream_unavailable",
            "Answer provider unavailable",
        )
    elif isinstance(error, APIError):
        status, category, message = (
            502,
            "upstream_error",
            "Answer provider request failed",
        )
    else:
        status, category, message = 500, "internal_error", "Research operation failed"
    request.state.error_category = category
    return HTTPException(status, message)


def create_app(index_path: Path, *, queue_wait_timeout: float = 30.0) -> FastAPI:
    """Create routes without loading the index; lifespan owns service resources."""
    if (
        type(queue_wait_timeout) not in (int, float)
        or not math.isfinite(queue_wait_timeout)
        or queue_wait_timeout <= 0
    ):
        raise ValueError("queue_wait_timeout must be a positive finite number")
    service: ResearchService | None = None
    lock: anyio.Lock | None = None
    available = False

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal service, lock, available
        lock = anyio.Lock()
        try:
            service = ResearchService(index_path)
        except Exception:
            raise RuntimeError("Research service initialization failed") from None
        available = True
        try:
            yield
        finally:
            available = False
            with anyio.CancelScope(shield=True):
                async with lock:
                    owned, service = service, None
                    if owned is not None:
                        try:
                            owned.close()
                        except Exception:
                            raise RuntimeError(
                                "Research service cleanup failed"
                            ) from None

    @asynccontextmanager
    async def service_access(request: Request):
        if not available or lock is None:
            raise HTTPException(503, "Research service unavailable")
        try:
            with anyio.fail_after(queue_wait_timeout):
                await lock.acquire()
        except TimeoutError:
            request.state.error_category = "queue_timeout"
            raise HTTPException(
                503, "Research service busy; queue wait timed out"
            ) from None
        # Release only after successful acquisition, never another request's lock.
        try:
            if not available or service is None:
                raise HTTPException(503, "Research service unavailable")
            yield service
        finally:
            lock.release()

    app = FastAPI(title="Research Assistant", lifespan=lifespan)
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/health", response_model=StatusResponse)
    async def health():
        return StatusResponse(status="ok")

    @app.get("/ready", response_model=StatusResponse)
    async def ready():
        if not available:
            raise HTTPException(503, "Research service unavailable")
        return StatusResponse(status="ready")

    @app.post("/search", response_model=SearchResponse)
    async def search(body: SearchRequest, request: Request):
        async with service_access(request) as active_service:
            try:
                operation = partial(
                    active_service.search, body.question, body.top_k, body.rerank
                )
                results = await anyio.to_thread.run_sync(
                    operation, abandon_on_cancel=False
                )
                return SearchResponse(
                    results=[SearchHit.model_validate(r) for r in results]
                )
            except HTTPException:
                raise
            except Exception as exc:
                raise backend_error(request, exc) from None

    @app.post("/answer", response_model=AnswerResponse)
    async def answer(body: AnswerRequest, request: Request):
        async with service_access(request) as active_service:
            try:
                operation = partial(active_service.answer, body.question)
                result = await anyio.to_thread.run_sync(
                    operation, abandon_on_cancel=False
                )
                return AnswerResponse.model_validate(result)
            except HTTPException:
                raise
            except Exception as exc:
                raise backend_error(request, exc) from None

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument(
        "--queue-wait-timeout", type=float, default=30.0, metavar="SECONDS"
    )
    args = parser.parse_args(argv)
    try:
        app = create_app(args.index, queue_wait_timeout=args.queue_wait_timeout)
    except ValueError:
        parser.error("--queue-wait-timeout must be a positive finite number")
    uvicorn.run(
        app, host="127.0.0.1", port=8000, workers=1, reload=False, access_log=False
    )


if __name__ == "__main__":
    main()
