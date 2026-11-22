"""Offline tests for manual evaluation exports; no model initialization."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from research_assistant.ingest.chunking import Chunk
from research_assistant.retrieval.vector import SearchResult

SPEC = importlib.util.spec_from_file_location(
    "export_eval_candidates",
    Path(__file__).resolve().parents[1] / "scripts" / "export_eval_candidates.py",
)
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


def result(key):
    return SearchResult(key, "doc", "paper.pdf", 3, "Intro", f"Text {key}", 0.5)


def test_pool_ranks_metadata_and_determinism():
    a, b, c = [result(key) for key in "abc"]
    rankings = {
        "faiss": [b, a, b],
        "bm25": [c, a],
        "hybrid": [a, b],
        "rerank": [replace(b, score=4.2), replace(a, score=-1.0)],
    }
    pooled = export.pool_candidates(rankings)
    assert [r["chunk_id"] for r in pooled] == ["a", "b", "c"]
    assert pooled[0] == {
        "chunk_id": "a",
        "source": "paper.pdf",
        "page_number": 3,
        "section": "Intro",
        "text": "Text a",
        "ranks": {"faiss": 2, "bm25": 2, "hybrid": 1, "rerank": 2},
        "cross_encoder_score": -1.0,
    }
    assert pooled[1]["ranks"] == {"faiss": 1, "bm25": None, "hybrid": 2, "rerank": 1}
    assert pooled[1]["cross_encoder_score"] == 4.2
    assert pooled[2]["cross_encoder_score"] is None
    assert pooled == export.pool_candidates(dict(reversed(list(rankings.items()))))
    assert b.score == 0.5


def test_conflicting_metadata():
    with pytest.raises(ValueError, match="Conflicting metadata"):
        export.pool_candidates(
            {"faiss": [result("a")], "bm25": [replace(result("a"), text="other")]}
        )


def test_empty_pool():
    assert export.pool_candidates({}) == []
    assert export.pool_candidates(dict.fromkeys(export.METHODS, [])) == []


def test_retrieval_calls():
    index, bm25, hybrid, embedder, reranker = [Mock() for _ in range(5)]
    index.search.return_value = [result("a")]
    bm25.search.return_value = [result("b")]
    hybrid.search.return_value = [result("a"), result("b")]
    reranker.rerank.return_value = [result("b"), result("a")]
    output = export.retrieve_candidates("q", index, bm25, hybrid, embedder, reranker)
    index.search.assert_called_once_with("q", embedder, 20)
    bm25.search.assert_called_once_with("q", 20)
    hybrid.search.assert_called_once_with("q", 20)
    reranker.rerank.assert_called_once_with("q", hybrid.search.return_value, 20)
    assert len(output) == 2


def chunk(key, source="paper.pdf"):
    return Chunk(key, "doc", source, 3, None, 0, f"Text {key}")


def test_source_export():
    index = SimpleNamespace(chunks=(chunk("b"), chunk("x", "other.pdf"), chunk("a")))
    assert export.export_source(index, "paper.pdf") == [
        {
            "chunk_id": key,
            "source": "paper.pdf",
            "page_number": 3,
            "section": None,
            "text": f"Text {key}",
        }
        for key in ("b", "a")
    ]
    assert export.export_source(index, "unknown.pdf") == []
    assert export.export_source(SimpleNamespace(chunks=()), "paper.pdf") == []


def test_source_cli_no_models(tmp_path, monkeypatch):
    monkeypatch.setattr(
        export.VectorIndex,
        "load",
        Mock(return_value=SimpleNamespace(chunks=(chunk("a"),))),
    )
    for name in ("Embedder", "CrossEncoderReranker", "BM25Index", "HybridRetriever"):
        monkeypatch.setattr(export, name, Mock(side_effect=AssertionError("No models")))
    path = tmp_path / "source.json"
    export.main(["--source", "paper.pdf", "--out", str(path)])
    data = json.loads(path.read_text())
    assert data["chunks"][0]["chunk_id"] == "a"
    original = path.read_bytes()
    with pytest.raises(SystemExit):
        export.main(["--source", "paper.pdf", "--out", str(path)])
    assert path.read_bytes() == original


def test_question_cli_empty_index(tmp_path, monkeypatch):
    monkeypatch.setattr(
        export.VectorIndex, "load", Mock(return_value=SimpleNamespace(chunks=()))
    )
    monkeypatch.setattr(
        export, "Embedder", Mock(side_effect=AssertionError("No models"))
    )
    questions = tmp_path / "questions.json"
    questions.write_text('{"Q01": "Question?"}')
    path = tmp_path / "nested" / "empty.json"
    export.main(
        ["--question-id", "Q01", "--questions", str(questions), "--out", str(path)]
    )
    data = json.loads(path.read_text())
    assert data["question_id"] == "Q01"
    assert data["question"] == "Question?"
    assert data["candidates"] == []
