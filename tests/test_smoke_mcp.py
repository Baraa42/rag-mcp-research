"""Offline validation helpers only; never launch real-model smoke tests in CI."""

import importlib.util
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import anyio
import numpy as np
import pytest

from research_assistant.ingest.chunking import Chunk
from research_assistant.retrieval.vector import VectorIndex

SPEC = importlib.util.spec_from_file_location(
    "smoke_mcp", Path(__file__).resolve().parents[1] / "scripts/smoke_mcp.py"
)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


@pytest.fixture
def evidence():
    chunk = Chunk(
        chunk_id="id",
        document_id="doc",
        source="paper.pdf",
        page_number=3,
        section=None,
        text="Passage",
        chunk_index=0,
    )
    return {"id": chunk}, dict(asdict(chunk), rank=1, score=0.1)


def test_valid_search(evidence):
    chunks, result = evidence
    assert smoke.validate_search({"results": [result]}, chunks) == [result]


@pytest.mark.parametrize(
    "field,value",
    [
        ("chunk_id", "unknown"),
        ("source", "wrong.pdf"),
        ("rank", 2),
        ("score", float("nan")),
        ("score", True),
    ],
)
def test_bad_search(evidence, field, value):
    chunks, result = evidence
    result[field] = value
    with pytest.raises(smoke.SmokeFailure):
        smoke.validate_search({"results": [result]}, chunks)


def test_valid_answer(evidence, capsys):
    chunks, _ = evidence
    smoke.validate_answer(
        {
            "answer": "Supported [S1].",
            "sources": [dict(asdict(chunks["id"]), label="S1")],
        },
        chunks,
    )
    assert "Validated 1 citation" in capsys.readouterr().out


@pytest.mark.parametrize(
    "answer", ["Unknown [S2].", "Malformed [s1].", "Uncited answer."]
)
def test_bad_citations(evidence, answer):
    chunks, _ = evidence
    with pytest.raises(smoke.SmokeFailure):
        smoke.validate_answer(
            {"answer": answer, "sources": [dict(asdict(chunks["id"]), label="S1")]},
            chunks,
        )


def test_uncited_factual_answer_rejected(evidence):
    with pytest.raises(smoke.SmokeFailure):
        smoke.validate_answer({"answer": "Uncited fact.", "sources": []}, evidence[0])


@pytest.mark.parametrize(
    "field,value",
    [
        ("document_id", "other"),
        ("text", "Changed passage"),
        ("page_number", 4),
        ("section", "Other"),
        ("source", "other.pdf"),
    ],
)
def test_exact_passage_metadata(evidence, field, value):
    chunks, record = evidence
    record[field] = value
    with pytest.raises(smoke.SmokeFailure, match=field):
        smoke.validate_passage(record, chunks)


def test_markdown_metadata(evidence):
    chunk = replace(
        evidence[0]["id"], source="notes.md", page_number=None, section="Intro"
    )
    smoke.validate_passage(asdict(chunk), {"id": chunk})
    record = asdict(chunk)
    del record["page_number"]
    with pytest.raises(smoke.SmokeFailure, match="page_number"):
        smoke.validate_passage(record, {"id": chunk})


@pytest.mark.parametrize("filename", ["manifest.json", "chunks.jsonl", "vectors.faiss"])
def test_corrupt_index_rejected_before_subprocess(
    tmp_path, monkeypatch, evidence, filename
):
    from research_assistant.retrieval import embeddings, reranking

    forbidden = Mock(
        side_effect=AssertionError("Must not initialize models or subprocess")
    )
    monkeypatch.setattr(embeddings, "SentenceTransformer", forbidden)
    monkeypatch.setattr(reranking, "CrossEncoder", forbidden)
    monkeypatch.setattr(smoke, "stdio_client", forbidden)
    monkeypatch.setattr(smoke.shutil, "which", lambda _: "uv")
    vector = np.zeros((1, 384), dtype=np.float32)
    vector[0, 0] = 1
    VectorIndex([evidence[0]["id"]], vector).save(tmp_path)
    # Existing load validates a genuine index without constructing ML models.
    assert VectorIndex.load(tmp_path).chunks[0] == evidence[0]["id"]
    with (tmp_path / filename).open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError):
        anyio.run(smoke.run, tmp_path, False)
    forbidden.assert_not_called()
