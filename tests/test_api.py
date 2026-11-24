"""Offline HTTP contracts, serialization and lifecycle tests."""

import threading
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import anyio
import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from research_assistant.generation.answering import CitationSource, GroundedAnswer
from research_assistant.interfaces import api
from research_assistant.retrieval.vector import SearchResult


@pytest.fixture
def setup(monkeypatch):
    service = Mock()
    service.search.return_value = [
        SearchResult("id", "doc", "paper.pdf", 3, None, "text", 0.5),
        SearchResult("id2", "doc2", "notes.md", None, "Intro", "second text", -0.2),
    ]
    service.answer.return_value = GroundedAnswer(
        "Answer [S1]",
        (CitationSource("S1", "id", "doc", "paper.pdf", 3, None, "text"),),
    )
    factory = Mock(return_value=service)
    monkeypatch.setattr(api, "ResearchService", factory)
    app = api.create_app(Path("index"))
    factory.assert_not_called()
    return app, service, factory


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_contracts_and_lifecycle(setup):
    app, service, factory = setup
    with TestClient(app) as client:
        factory.assert_called_once_with(Path("index"))
        assert set(client.get("/openapi.json").json()["paths"]) == {
            "/health",
            "/ready",
            "/search",
            "/answer",
        }
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"status": "ready"}
        service.search.assert_not_called()
        response = client.post("/search", json={"question": "q"})
        assert response.status_code == 200
        assert response.json() == {
            "results": [asdict(r) for r in service.search.return_value]
        }
        service.search.assert_called_once_with("q", 5, True)
        response = client.post("/answer", json={"question": "q"})
        assert response.status_code == 200
        assert response.json() == {
            "answer": "Answer [S1]",
            "sources": [asdict(s) for s in service.answer.return_value.sources],
        }
        service.answer.assert_called_once_with("q")
        service.search.return_value = []
        assert client.post(
            "/search", json={"question": "q", "top_k": 50, "rerank": False}
        ).json() == {"results": []}
        service.search.assert_called_with("q", 50, False)
        factory.assert_called_once()
        service.close.assert_not_called()
    service.close.assert_called_once()
    assert client.get("/ready").status_code == 503


@pytest.mark.parametrize(
    "endpoint,body",
    [
        ("/search", {"question": "q", "top_k": value})
        for value in (0, -1, 51, True, "5", 2.5)
    ]
    + [
        (endpoint, body)
        for endpoint in ("/search", "/answer")
        for body in (
            {},
            {"question": ""},
            {"question": "  "},
            {"question": 3},
            {"question": None},
            {"question": "q", "unknown": 1},
        )
    ]
    + [
        ("/search", {"question": "q", "rerank": "true"}),
        ("/answer", {"question": "x" * 2001}),
    ],
)
def test_validation(setup, endpoint, body):
    with TestClient(setup[0]) as client:
        assert client.post(endpoint, json=body).status_code == 422
    setup[1].search.assert_not_called()
    setup[1].answer.assert_not_called()


def test_answer_length_boundary(setup):
    with TestClient(setup[0]) as client:
        assert client.post("/answer", json={"question": "x" * 2000}).status_code == 200


def test_unavailable_before_startup(setup):
    client = TestClient(setup[0])
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    for path in ("/search", "/answer"):
        assert client.post(path, json={"question": "q"}).status_code == 503
    setup[2].assert_not_called()


@pytest.mark.parametrize(
    "endpoint,method", [("/search", "search"), ("/answer", "answer")]
)
def test_safe_backend_error(setup, endpoint, method):
    getattr(setup[1], method).side_effect = RuntimeError(
        "secret-key /private/path passage text"
    )
    with TestClient(setup[0]) as client:
        response = client.post(endpoint, json={"question": "q"})
        assert response.status_code == 500
        assert "secret" not in response.text and "private" not in response.text
    setup[1].close.assert_called_once()


def test_intentional_http_error(setup):
    setup[1].search.side_effect = HTTPException(503, "Unavailable")
    with TestClient(setup[0]) as client:
        response = client.post("/search", json={"question": "q"})
        assert response.status_code == 503


def test_failed_startup(setup):
    setup[2].side_effect = ValueError("secret path")
    with pytest.raises(RuntimeError, match="initialization failed") as exc:
        with TestClient(setup[0]):
            pass
    assert "secret" not in str(exc.value)
    setup[1].close.assert_not_called()


@pytest.mark.anyio
async def test_serialization_and_shutdown_waits(setup):
    app, service, _ = setup
    entered, release = threading.Event(), threading.Event()
    active = 0
    guard = threading.Lock()
    original = service.search.return_value

    def search(*args):
        nonlocal active
        with guard:
            active += 1
            assert active == 1
        entered.set()
        assert release.wait(5)
        with guard:
            active -= 1
        return original

    service.search.side_effect = search

    def answer(*args):
        assert active == 0
        return service.answer.return_value

    service.answer.side_effect = answer
    service.close.side_effect = lambda: (
        pytest.fail("closed while active") if active else None
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            async with anyio.create_task_group() as group:

                async def post(path):
                    response = await client.post(path, json={"question": "q"})
                    assert response.status_code == 200

                group.start_soon(post, "/search")
                assert await anyio.to_thread.run_sync(entered.wait, 5)
                group.start_soon(post, "/answer")
                await anyio.sleep(0)
                service.answer.assert_not_called()
                release.set()
    service.close.assert_called_once()


@pytest.mark.anyio
async def test_cancelled_request_finishes_before_cleanup(setup):
    app, service, _ = setup
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    result = service.search.return_value

    def search(*args):
        entered.set()
        assert release.wait(5)
        finished.set()
        return result

    service.search.side_effect = search
    service.close.side_effect = lambda: (
        pytest.fail("premature close") if not finished.is_set() else None
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            async with anyio.create_task_group() as group:

                async def request():
                    await client.post("/search", json={"question": "q"})

                group.start_soon(request)
                assert await anyio.to_thread.run_sync(entered.wait, 5)
                group.cancel_scope.cancel()
                service.close.assert_not_called()
                release.set()
    assert finished.is_set()
    service.close.assert_called_once()


def test_entry_point(monkeypatch):
    factory, run = Mock(), Mock()
    monkeypatch.setattr(api, "create_app", factory)
    monkeypatch.setattr(api.uvicorn, "run", run)
    api.main(["--index", "index"])
    run.assert_called_once_with(
        factory.return_value, host="127.0.0.1", port=8000, workers=1, reload=False
    )


@pytest.mark.anyio
async def test_shutdown_marks_unready_and_waits_for_worker(setup):
    app, service, _ = setup
    entered, release = threading.Event(), threading.Event()
    started, stop, stopped = anyio.Event(), anyio.Event(), anyio.Event()
    result = service.search.return_value

    def search(*args):
        entered.set()
        assert release.wait(5)
        return result

    service.search.side_effect = search

    async def lifecycle():
        async with app.router.lifespan_context(app):
            started.set()
            await stop.wait()
        stopped.set()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with anyio.create_task_group() as group:
            group.start_soon(lifecycle)
            await started.wait()

            async def request():
                response = await client.post("/search", json={"question": "q"})
                assert response.status_code == 200

            group.start_soon(request)
            assert await anyio.to_thread.run_sync(entered.wait, 5)
            stop.set()
            with anyio.fail_after(2):
                while (await client.get("/ready")).status_code != 503:
                    await anyio.sleep(0)
            assert not stopped.is_set()
            service.close.assert_not_called()
            assert (
                await client.post("/answer", json={"question": "q"})
            ).status_code == 503
            release.set()
            await stopped.wait()
    service.close.assert_called_once()
