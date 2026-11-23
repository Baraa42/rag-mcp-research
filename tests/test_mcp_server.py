"""In-memory MCP SDK tests with a mocked service; no models or API calls."""

import asyncio
import threading
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import pytest
from mcp import Client

from research_assistant.generation.answering import CitationSource, GroundedAnswer
from research_assistant.interfaces import mcp_server
from research_assistant.retrieval.vector import SearchResult


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def setup(monkeypatch):
    service = Mock()
    service.search.return_value = [
        SearchResult("b", "doc", "paper.pdf", 3, None, "Passage B", 2.1),
        SearchResult("a", "doc2", "notes.md", None, "Intro", "Passage A", 1.0),
    ]
    service.answer.return_value = GroundedAnswer(
        "Supported [S1].",
        (CitationSource("S1", "b", "doc", "paper.pdf", 3, None, "Passage B"),),
    )
    factory = Mock(return_value=service)
    monkeypatch.setattr(mcp_server, "ResearchService", factory)
    server = mcp_server.create_server(Path("test-index"))
    factory.assert_not_called()
    return server, service, factory


@pytest.mark.anyio
async def test_tools_schemas_and_lifecycle(setup):
    server, service, factory = setup
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
        assert {t.name for t in tools} == {"search_papers", "answer_question"}
        search = next(t for t in tools if t.name == "search_papers")
        assert set(search.input_schema["properties"]) == {"question", "top_k", "rerank"}
        assert search.input_schema["required"] == ["question"]
        assert search.input_schema["properties"]["question"]["type"] == "string"
        assert search.input_schema["properties"]["top_k"]["default"] == 5
        assert search.input_schema["properties"]["top_k"]["type"] == "integer"
        assert search.input_schema["properties"]["rerank"]["default"] is True
        assert search.input_schema["properties"]["rerank"]["type"] == "boolean"
        answer = next(t for t in tools if t.name == "answer_question")
        assert set(answer.input_schema["properties"]) == {"question"}
        assert answer.input_schema["required"] == ["question"]
        assert all(t.output_schema for t in tools)
        await client.call_tool("search_papers", {"question": "q"})
        await client.call_tool("answer_question", {"question": "q"})
        factory.assert_called_once_with(Path("test-index"))
        service.close.assert_not_called()
    service.close.assert_called_once()


@pytest.mark.anyio
async def test_search_output_and_defaults(setup):
    server, service, _ = setup
    async with Client(server) as client:
        result = await client.call_tool("search_papers", {"question": "q"})
        assert not result.is_error
        service.search.assert_called_once_with("q", 5, True)
        assert result.structured_content == {
            "results": [
                dict(rank=i, **asdict(r))
                for i, r in enumerate(service.search.return_value, 1)
            ]
        }
        service.search.return_value = []
        result = await client.call_tool(
            "search_papers", {"question": "q", "top_k": 25, "rerank": False}
        )
        service.search.assert_called_with("q", 25, False)
        assert result.structured_content == {"results": []}


@pytest.mark.anyio
async def test_answer_output(setup):
    server, service, _ = setup
    async with Client(server) as client:
        result = await client.call_tool("answer_question", {"question": "q"})
        assert not result.is_error
        assert result.structured_content == {
            "answer": "Supported [S1].",
            "sources": [asdict(s) for s in service.answer.return_value.sources],
        }
        service.answer.assert_called_once_with("q")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"question": 1},
        {"question": "q", "top_k": True},
        {"question": "q", "top_k": "5"},
        {"question": "q", "rerank": "false"},
    ],
)
async def test_input_types(setup, arguments):
    server, service, _ = setup
    async with Client(server) as client:
        result = await client.call_tool("search_papers", arguments)
        assert result.is_error
        service.search.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize("top_k", [0, -1])
async def test_service_validation_not_clamped(setup, top_k):
    server, service, _ = setup
    service.search.side_effect = ValueError("top_k must be a positive integer")
    async with Client(server) as client:
        result = await client.call_tool(
            "search_papers", {"question": "q", "top_k": top_k}
        )
        assert result.is_error
        assert "top_k must be a positive integer" in result.content[0].text
        service.search.assert_called_once_with("q", top_k, True)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error", [RuntimeError("secret-key traceback"), ValueError("secret-key")]
)
async def test_errors_redacted_and_cleanup(setup, error, capsys):
    server, service, _ = setup
    service.answer.side_effect = error
    async with Client(server) as client:
        result = await client.call_tool("answer_question", {"question": "q"})
        assert result.is_error
        assert "secret-key" not in str(result)
    service.close.assert_called_once()
    captured = capsys.readouterr()
    assert "secret-key" not in captured.out + captured.err


@pytest.mark.anyio
async def test_calls_are_serialized(setup):
    server, service, _ = setup
    active = 0
    guard = threading.Lock()

    def search(*args):
        nonlocal active
        with guard:
            active += 1
            assert active == 1
        time.sleep(0.02)
        with guard:
            active -= 1
        return []

    service.search.side_effect = search
    async with Client(server) as client:
        results = await asyncio.gather(
            *[client.call_tool("search_papers", {"question": "q"}) for _ in range(3)]
        )
        assert all(not r.is_error for r in results)
    assert service.search.call_count == 3


def test_entry_point_stdio_only(monkeypatch):
    server = Mock()
    factory = Mock(return_value=server)
    monkeypatch.setattr(mcp_server, "create_server", factory)
    mcp_server.main(["--index", "test-index"])
    factory.assert_called_once_with(Path("test-index"))
    server.run.assert_called_once_with(transport="stdio")


@pytest.mark.anyio
async def test_cleanup_on_session_exception(setup):
    server, service, _ = setup
    with pytest.raises(BaseExceptionGroup):
        async with Client(server):
            raise RuntimeError("session failed")
    service.close.assert_called_once()


@pytest.mark.anyio
async def test_application_diagnostics_use_logging(setup, capsys, caplog):
    server, service, _ = setup

    def search(*args):
        mcp_server.logger.warning("diagnostic")
        return []

    service.search.side_effect = search
    async with Client(server) as client:
        result = await client.call_tool("search_papers", {"question": "q"})
        assert not result.is_error
    captured = capsys.readouterr()
    assert "diagnostic" not in captured.out
    assert "diagnostic" in caplog.text


def test_startup_failure_is_sanitized(monkeypatch, capsys):
    server = Mock()
    server.run.side_effect = RuntimeError("secret-key")
    monkeypatch.setattr(mcp_server, "create_server", Mock(return_value=server))
    with pytest.raises(SystemExit) as exc:
        mcp_server.main(["--index", "test-index"])
    assert exc.value.code == 1
    output = capsys.readouterr()
    assert not output.out
    assert "secret-key" not in output.err
    assert "check index and service configuration" in output.err
