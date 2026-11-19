"""Offline fusion arithmetic, retriever reuse, and CLI regression checks."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from research_assistant.bm25 import BM25Index
from research_assistant.chunking import Chunk
from research_assistant.hybrid import HybridRetriever, reciprocal_rank_fusion
from research_assistant.retrieval import SearchResult, VectorIndex


def result(key: str, score: float = 1) -> SearchResult:
    return SearchResult(key, "doc", "paper.pdf", 3, "Section", f"Text {key}", score)


def test_known_ranks_shared_and_singletons() -> None:
    a, b, c = result("a", 1000), result("b", 0.2), result("c", -2)
    fused = reciprocal_rank_fusion([[a, b], [c, a]], top_k=9)
    assert [r.chunk_id for r in fused] == ["a", "c", "b"]
    assert [r.score for r in fused] == pytest.approx([1 / 61 + 1 / 62, 1 / 61, 1 / 62])
    assert replace(fused[0], score=a.score) == a
    assert a.score == 1000


def test_ties_raw_scores_and_duplicate_ids() -> None:
    a, b = result("a", -100), result("b", 999)
    assert [r.chunk_id for r in reciprocal_rank_fusion([[b], [a]])] == ["a", "b"]
    assert reciprocal_rank_fusion([[b], [a]]) == reciprocal_rank_fusion([[a], [b]])
    fused = reciprocal_rank_fusion([[a, a, b], [replace(a, score=42)]], top_k=2)
    assert fused[0].score == pytest.approx(2 / 61)
    assert fused[1].score == pytest.approx(1 / 63)
    assert reciprocal_rank_fusion([[a]], rrf_k=0)[0].score == 1


def test_empty_partial_and_limit() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    assert len(reciprocal_rank_fusion([[], [result("a"), result("b")]], top_k=1)) == 1


def test_conflicting_metadata() -> None:
    with pytest.raises(ValueError, match="Conflicting"):
        reciprocal_rank_fusion(
            [[result("a")], [replace(result("a"), source="other.pdf")]]
        )


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_invalid_top_k(value: object) -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], top_k=value)


@pytest.mark.parametrize("value", [-1, True, 1.5, None])
def test_invalid_rrf_k(value: object) -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], rrf_k=value)


@pytest.mark.parametrize("value", [None, "rankings", [None], [["result"]]])
def test_invalid_rankings(value: object) -> None:
    with pytest.raises(TypeError):
        reciprocal_rank_fusion(value)


def test_invalid_id() -> None:
    with pytest.raises(ValueError, match="chunk_id"):
        reciprocal_rank_fusion([[result("")]])


def test_retriever_reuses_dependencies_and_expands_candidates() -> None:
    vector, lexical, embedder = Mock(), Mock(), Mock()
    vector.chunks = lexical.chunks = ()
    vector.search.return_value = [result("a")]
    lexical.search.return_value = [result("a")]
    retriever = HybridRetriever(vector, lexical, embedder)
    assert retriever.search("question")[0].score == pytest.approx(2 / 61)
    vector.search.assert_called_once_with("question", embedder, 20)
    lexical.search.assert_called_once_with("question", 20)
    retriever.search("question", top_k=25)
    vector.search.assert_called_with("question", embedder, 25)
    lexical.search.assert_called_with("question", 25)
    embedder.encode.assert_not_called()  # Delegated exclusively to VectorIndex.


@pytest.mark.parametrize(
    "kwargs",
    [{"candidate_k": 0}, {"candidate_k": True}, {"candidate_k": 1.5}, {"rrf_k": -1}],
)
def test_invalid_retriever_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        HybridRetriever(Mock(), Mock(), Mock(), **kwargs)


def test_mismatched_corpora() -> None:
    vector, lexical = Mock(), Mock()
    vector.chunks, lexical.chunks = ("a",), ("b",)
    with pytest.raises(ValueError, match="same ordered chunks"):
        HybridRetriever(vector, lexical, Mock())


def test_empty_index_no_inference() -> None:
    model = Mock()
    retriever = HybridRetriever(
        VectorIndex([], np.empty((0, 384))), BM25Index([]), model
    )
    assert retriever.search("question") == []
    model.encode.assert_not_called()
    for question in (None, "", "   "):
        with pytest.raises(ValueError):
            retriever.search(question)
    with pytest.raises(ValueError):
        retriever.search("question", top_k=0)


@pytest.mark.parametrize("method", [None, "vector", "bm25", "hybrid"])
def test_cli_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    method: str | None,
) -> None:
    from research_assistant import __main__ as cli

    chunks = [
        Chunk(str(i), "doc", "paper.pdf", i + 1, None, 0, text)
        for i, text in enumerate(["positional encoding", "memory", "attention"])
    ]
    vectors = np.eye(3, 384, dtype=np.float32)
    index = VectorIndex(chunks, vectors)
    index.save(tmp_path)
    model = Mock()
    model.encode.return_value = vectors[:1]
    factory = Mock(return_value=model)
    monkeypatch.setattr(cli, "Embedder", factory)
    arguments = ["search", "positional", "--index", str(tmp_path), "--top-k", "1"]
    if method:
        arguments.extend(["--method", method])
    cli.main(arguments)
    output = capsys.readouterr().out
    assert "paper.pdf | page 1" in output and "positional encoding" in output
    if method == "bm25":
        factory.assert_not_called()
    else:
        factory.assert_called_once()
        model.encode.assert_called_once_with(["positional"])
    if method == "hybrid":
        assert f"score={2 / 61:.6f}" in output
    elif method in (None, "vector"):
        assert "score=1.000000" in output
