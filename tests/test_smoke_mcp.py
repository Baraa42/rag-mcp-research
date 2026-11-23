"""Offline validation helpers only; never launch real-model smoke tests in CI."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "smoke_mcp", Path(__file__).resolve().parents[1] / "scripts/smoke_mcp.py"
)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


@pytest.fixture
def evidence():
    chunk = dict(
        chunk_id="id",
        document_id="doc",
        source="paper.pdf",
        page_number=3,
        section=None,
        text="Passage",
    )
    return {"id": chunk}, dict(chunk, rank=1, score=0.1)


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
        {"answer": "Supported [S1].", "sources": [dict(chunks["id"], label="S1")]},
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
            {"answer": answer, "sources": [dict(chunks["id"], label="S1")]}, chunks
        )


def test_uncited_factual_answer_rejected(evidence):
    with pytest.raises(smoke.SmokeFailure):
        smoke.validate_answer({"answer": "Uncited fact.", "sources": []}, evidence[0])
