"""Local HTTP interface over one serialized ResearchService instance."""

import argparse
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Literal

import anyio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

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


def create_app(index_path: Path) -> FastAPI:
    """Create routes without loading the index; lifespan owns service resources."""
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

    app = FastAPI(title="Research Assistant", lifespan=lifespan)

    @app.get("/health", response_model=StatusResponse)
    async def health():
        return StatusResponse(status="ok")

    @app.get("/ready", response_model=StatusResponse)
    async def ready():
        if not available:
            raise HTTPException(503, "Research service unavailable")
        return StatusResponse(status="ready")

    @app.post("/search", response_model=SearchResponse)
    async def search(body: SearchRequest):
        if not available or lock is None:
            raise HTTPException(503, "Research service unavailable")
        async with lock:
            if not available or service is None:
                raise HTTPException(503, "Research service unavailable")
            try:
                operation = partial(
                    service.search, body.question, body.top_k, body.rerank
                )
                results = await anyio.to_thread.run_sync(
                    operation, abandon_on_cancel=False
                )
                return SearchResponse(
                    results=[SearchHit.model_validate(r) for r in results]
                )
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(500, "Research search failed") from None

    @app.post("/answer", response_model=AnswerResponse)
    async def answer(body: AnswerRequest):
        if not available or lock is None:
            raise HTTPException(503, "Research service unavailable")
        async with lock:
            if not available or service is None:
                raise HTTPException(503, "Research service unavailable")
            try:
                operation = partial(service.answer, body.question)
                result = await anyio.to_thread.run_sync(
                    operation, abandon_on_cancel=False
                )
                return AnswerResponse.model_validate(result)
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(500, "Research answer generation failed") from None

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args(argv)
    uvicorn.run(
        create_app(args.index), host="127.0.0.1", port=8000, workers=1, reload=False
    )


if __name__ == "__main__":
    main()
