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


V2 = Path(__file__).resolve().parents[1] / "eval" / "v2"


def test_validate_frozen_questions():
    questions = json.loads((V2 / "questions.json").read_bytes())
    metadata = json.loads((V2 / "metadata.json").read_bytes())
    baseline = json.loads((V2.parent / "questions.json").read_bytes())
    export.validate_batch_dataset(questions, metadata, baseline)

    override = dict(questions, Q01="Changed baseline question?")
    with pytest.raises(ValueError, match="Baseline question"):
        export.validate_batch_dataset(override, metadata, baseline)

    override = dict(questions, Q50=questions["Q49"])
    with pytest.raises(ValueError, match="Duplicate question"):
        export.validate_batch_dataset(override, metadata, baseline)

    override_metadata = dict(metadata, Q50=dict(metadata["Q50"], split="development"))
    with pytest.raises(ValueError, match="15 development"):
        export.validate_batch_dataset(questions, override_metadata, baseline)


def test_batch_reuses_models_and_blinds_outside_references(tmp_path, monkeypatch):
    picked, extra = result("picked"), result("extra")
    index = SimpleNamespace(chunks=[picked, extra])
    chunks = export.pool_candidates({"faiss": [picked], "rerank": [picked]})
    load = Mock(return_value=index)
    embedder, reranker, bm25, hybrid = Mock(), Mock(), Mock(), Mock()
    retrieve = Mock(return_value=chunks)
    monkeypatch.setattr(export.VectorIndex, "load", load)
    monkeypatch.setattr(export, "Embedder", embedder)
    monkeypatch.setattr(export, "CrossEncoderReranker", reranker)
    monkeypatch.setattr(export, "BM25Index", bm25)
    monkeypatch.setattr(export, "HybridRetriever", hybrid)
    monkeypatch.setattr(export, "retrieve_candidates", retrieve)
    monkeypatch.setattr(export, "validate_batch_dataset", Mock())
    questions = tmp_path / "questions.json"
    metadata = tmp_path / "metadata.json"
    questions.write_text(json.dumps({"Q01": "First?", "Q02": "Second?"}))
    metadata.write_text(
        json.dumps(
            {
                "Q01": {
                    "references": [
                        {
                            "chunk_id": extra.chunk_id,
                            "source": extra.source,
                            "page_number": 3,
                        }
                    ]
                },
                "Q02": {"references": []},
            }
        )
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "sha256": {"vectors.faiss": "fixture"},
                "model_name": "model",
                "model_revision": "revision",
            }
        )
    )
    output = tmp_path / "batch"
    export.export_batch(questions, metadata, tmp_path, output)
    load.assert_called_once_with(tmp_path)
    embedder.assert_called_once()
    reranker.assert_called_once()
    assert retrieve.call_count == 2
    for call in retrieve.call_args_list:
        assert call.args[1:] == (
            index,
            bm25.return_value,
            hybrid.return_value,
            embedder.return_value,
            reranker.return_value,
        )
    blind = json.loads((output / "blind" / "Q01.json").read_bytes())["candidates"]
    assert {r["chunk_id"] for r in blind} == {"picked", "extra"}
    assert all(
        set(r) == {"chunk_id", "source", "page_number", "section", "text"}
        for r in blind
    )
    assert blind == export.blind_candidates("Q01", list(reversed(blind)), blind)
    compact = json.loads((output / "rankings.json").read_bytes())
    assert compact["rankings"]["Q01"] == {
        "faiss": ["picked"],
        "bm25": [],
        "hybrid": [],
        "rerank": ["picked"],
    }
    summary = json.loads((output / "run.json").read_bytes())
    assert summary["failures"] == {}
    assert summary["questions"]["Q01"]["pooled"] == 1
    assert summary["questions"]["Q01"]["blind"] == 2

    retrieve.side_effect = RuntimeError("failed test query")
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="failed queries"):
        export.export_batch(questions, metadata, tmp_path, failed)
    summary = json.loads((failed / "run.json").read_bytes())
    assert set(summary["failures"]) == {"Q01", "Q02"}
    assert not (failed / "rankings.json").exists()
