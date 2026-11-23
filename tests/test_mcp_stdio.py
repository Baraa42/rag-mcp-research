"""Actual uv/module stdio subprocess smoke tests; deterministic offline ML stub."""

import json
import shutil
from pathlib import Path

import anyio
import numpy as np
import pytest
from mcp import ClientSession
from mcp.client import stdio

from research_assistant.generation.answering import INSUFFICIENT_EVIDENCE
from research_assistant.ingest.chunking import Chunk
from research_assistant.retrieval.vector import VectorIndex


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "empty", [False, True], ids=["nonempty-retrieval", "empty-answer"]
)
async def test_stdio_subprocess(tmp_path, monkeypatch, empty):
    index_path = tmp_path / "index"
    chunks = (
        []
        if empty
        else [
            Chunk(
                "test-chunk",
                "test-doc",
                "test-paper.pdf",
                7,
                "Methods",
                0,
                "Positional encoding represents token order.",
            )
        ]
    )
    vectors = np.zeros((len(chunks), 384), dtype=np.float32)
    if chunks:
        vectors[:, 0] = 1
    VectorIndex(chunks, vectors).save(index_path)

    # Python's startup hook injects only test dependencies into the real command.
    # No production test flags, alternate service, or JSON-RPC implementation.
    hook = tmp_path / "hooks"
    hook.mkdir()
    marker = tmp_path / "lifecycle.txt"
    (hook / "sitecustomize.py").write_text(
        """
import logging
import os
from pathlib import Path
import numpy as np
from research_assistant import service

marker = Path(os.environ["MCP_TEST_MARKER"])
def record(event):
    with marker.open("a") as f:
        f.write(event + "\\n")

class DeterministicEmbedder:
    def __init__(self):
        record("embedder")
    def encode(self, texts):
        logging.getLogger("stdio-test").warning("Deterministic embedding invoked")
        result = np.zeros((len(texts), 384), dtype=np.float32)
        result[:, 0] = 1
        return result

def forbidden(*args, **kwargs):
    raise AssertionError("External models and API clients are forbidden")
service.Embedder = DeterministicEmbedder
service.CrossEncoderReranker = forbidden
service.OpenAI = forbidden
original_init = service.ResearchService.__init__
original_close = service.ResearchService.close

def tracked_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    record("loaded")

def tracked_close(self):
    original_close(self)
    record("closed")
service.ResearchService.__init__ = tracked_init
service.ResearchService.close = tracked_close
""",
        encoding="utf-8",
    )

    processes, lines = [], []
    # These private SDK hooks may change even when public MCP behavior still works.
    # On SDK upgrades, check hook compatibility before treating a failure as a
    # server regression. Retain this observation approach for now (see README).
    spawn = stdio._create_platform_compatible_process
    parse = stdio._parse_line

    async def observe_spawn(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        processes.append(process)
        return process

    def observe_line(line):
        lines.append(line)
        return parse(line)

    # Observe SDK transport internals, without replacing transport or protocol logic.
    monkeypatch.setattr(stdio, "_create_platform_compatible_process", observe_spawn)
    monkeypatch.setattr(stdio, "_parse_line", observe_line)
    uv = shutil.which("uv")
    assert uv, "Install uv to run the real stdio smoke test"
    params = stdio.StdioServerParameters(
        command=uv,
        args=[
            "run",
            "python",
            "-m",
            "research_assistant.interfaces.mcp_server",
            "--index",
            str(index_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={
            "PYTHONPATH": str(hook),
            "MCP_TEST_MARKER": str(marker),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "UV_OFFLINE": "1",
            "UV_NO_SYNC": "1",
            "UV_NO_ENV_FILE": "1",
        },
    )
    stderr_path = tmp_path / "stderr.log"
    with anyio.fail_after(45), stderr_path.open("w") as errlog:
        async with stdio.stdio_client(params, errlog=errlog) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                assert {t.name for t in tools} == {"search_papers", "answer_question"}
                for _ in range(2):
                    result = await session.call_tool(
                        "search_papers",
                        {
                            "question": "positional encoding",
                            "top_k": 1,
                            "rerank": False,
                        },
                    )
                    assert not result.is_error
                    passages = result.structured_content["results"]
                    if empty:
                        assert passages == []
                    else:
                        assert len(passages) == 1
                        assert passages[0] == {
                            "rank": 1,
                            "chunk_id": "test-chunk",
                            "document_id": "test-doc",
                            "source": "test-paper.pdf",
                            "page_number": 7,
                            "section": "Methods",
                            "text": chunks[0].text,
                            "score": pytest.approx(2 / 61),
                        }
                if empty:
                    result = await session.call_tool(
                        "answer_question", {"question": "q"}
                    )
                    assert not result.is_error
                    assert result.structured_content == {
                        "answer": INSUFFICIENT_EVIDENCE,
                        "sources": [],
                    }
    assert len(processes) == 1
    assert processes[0].returncode == 0  # EOF shutdown, not forced SIGTERM/SIGKILL.
    events = marker.read_text().splitlines()
    assert events.count("loaded") == events.count("closed") == 1
    assert events.count("embedder") == (0 if empty else 1)
    assert lines
    for line in lines:
        message = json.loads(line)  # Any unrelated stdout line fails the test.
        assert message["jsonrpc"] == "2.0"
        assert "id" in message or "method" in message
    diagnostics = stderr_path.read_text()
    if not empty:
        assert "Deterministic embedding invoked" in diagnostics
    assert "Traceback" not in diagnostics
