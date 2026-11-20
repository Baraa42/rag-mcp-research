"""Offline cross-encoder contracts and optional real-model verification."""

import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from torch import nn

from research_assistant import reranking
from research_assistant.reranking import (
    RERANKER_NAME,
    RERANKER_REVISION,
    CrossEncoderReranker,
)
from research_assistant.retrieval import SearchResult


def candidate(i: int) -> SearchResult:
    return SearchResult(
        str(i), f"doc{i}", f"paper{i}.pdf", i + 1, "Section", f"Passage {i}", 0.01
    )


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    model = Mock()
    factory = Mock(return_value=model)
    monkeypatch.setattr(reranking, "CrossEncoder", factory)
    return model, factory


def test_order_metadata_batch_and_single_load(fake: tuple[Mock, Mock]) -> None:
    model, factory = fake
    model.predict.return_value = np.array([-3.0, 4.0, 2.0])
    reranker = CrossEncoderReranker(batch_size=2)
    inputs = [candidate(i) for i in range(3)]
    output = reranker.rerank("question", inputs, top_k=9)
    assert [r.chunk_id for r in output] == ["1", "2", "0"]
    assert [r.score for r in output] == [4, 2, -3]
    assert [replace(r, score=0.01) for r in output] == [inputs[1], inputs[2], inputs[0]]
    assert all(r.score == 0.01 for r in inputs)
    model.predict.assert_called_once_with(
        [("question", r.text) for r in inputs],
        batch_size=2,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    reranker.rerank("question", inputs)
    factory.assert_called_once()
    assert factory.call_args.args == (RERANKER_NAME,)
    assert factory.call_args.kwargs["revision"] == RERANKER_REVISION
    assert factory.call_args.kwargs["device"] == "cpu"
    assert isinstance(factory.call_args.kwargs["activation_fn"], nn.Identity)
    assert factory.call_args.kwargs["max_length"] == 512


def test_ties_and_limit(fake: tuple[Mock, Mock]) -> None:
    fake[0].predict.return_value = [[1], [1], [-1]]
    inputs = [candidate(2), candidate(0), candidate(1)]
    reranker = CrossEncoderReranker()
    assert [r.chunk_id for r in reranker.rerank("q", inputs, 2)] == ["2", "0"]
    assert reranker.rerank("q", inputs) == reranker.rerank("q", inputs)


def test_empty(fake: tuple[Mock, Mock]) -> None:
    assert CrossEncoderReranker().rerank("question", []) == []
    fake[0].predict.assert_not_called()


@pytest.mark.parametrize("scores", [[], [1], [1, 2, 3], [[1, 2]], [[1, 2], [3, 4]], 1])
def test_bad_score_count(fake: tuple[Mock, Mock], scores: object) -> None:
    fake[0].predict.return_value = scores
    with pytest.raises(ValueError, match="one score"):
        CrossEncoderReranker().rerank("q", [candidate(0), candidate(1)])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_scores(fake: tuple[Mock, Mock], value: float) -> None:
    fake[0].predict.return_value = [value]
    with pytest.raises(ValueError, match="non-finite"):
        CrossEncoderReranker().rerank("q", [candidate(0)])


@pytest.mark.parametrize("batch", [0, -1, True, 1.5, None, "16"])
def test_bad_batch(fake: tuple[Mock, Mock], batch: object) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        CrossEncoderReranker(batch_size=batch)
    fake[1].assert_not_called()


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, None])
def test_bad_top_k(fake: tuple[Mock, Mock], top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k"):
        CrossEncoderReranker().rerank("q", [], top_k)
    fake[0].predict.assert_not_called()


@pytest.mark.parametrize("question", [None, 42, "", "  "])
def test_bad_question(fake: tuple[Mock, Mock], question: object) -> None:
    with pytest.raises(ValueError, match="question"):
        CrossEncoderReranker().rerank(question, [])


@pytest.mark.parametrize("inputs", [None, "text", (), [None]])
def test_bad_candidates(fake: tuple[Mock, Mock], inputs: object) -> None:
    with pytest.raises(TypeError):
        CrossEncoderReranker().rerank("q", inputs)


def test_empty_text_and_duplicates(fake: tuple[Mock, Mock]) -> None:
    reranker = CrossEncoderReranker()
    with pytest.raises(ValueError, match="text"):
        reranker.rerank("q", [replace(candidate(0), text=" ")])
    with pytest.raises(ValueError, match="Duplicate"):
        reranker.rerank("q", [candidate(0), candidate(0)])


@pytest.mark.parametrize("top_k", [5, 25])
def test_cli_rerank_expands_depth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    top_k: int,
) -> None:
    from research_assistant import __main__ as cli

    index = Mock()
    index.chunks = ("chunk",)
    monkeypatch.setattr(cli.VectorIndex, "load", Mock(return_value=index))
    monkeypatch.setattr(cli, "BM25Index", Mock())
    model = Mock()
    monkeypatch.setattr(cli, "Embedder", Mock(return_value=model))
    first_stage = Mock()
    first_stage.search.return_value = [candidate(0)]
    hybrid_factory = Mock(return_value=first_stage)
    monkeypatch.setattr(cli, "HybridRetriever", hybrid_factory)
    reranker = Mock()
    reranker.rerank.return_value = [replace(candidate(0), score=4.2)]
    monkeypatch.setattr(cli, "CrossEncoderReranker", Mock(return_value=reranker))
    cli.main(
        [
            "search",
            "question",
            "--index",
            str(tmp_path),
            "--method",
            "rerank",
            "--top-k",
            str(top_k),
        ]
    )
    first_stage.search.assert_called_once_with("question", max(20, top_k))
    reranker.rerank.assert_called_once_with("question", [candidate(0)], top_k)
    assert "score=4.200000" in capsys.readouterr().out
    model.encode.assert_not_called()


@pytest.mark.parametrize("method", [None, "vector", "bm25", "hybrid"])
def test_old_modes_do_not_load_reranker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    method: str | None,
) -> None:
    from research_assistant import __main__ as cli

    index = Mock()
    index.chunks = ("chunk",)
    index.search.return_value = [candidate(0)]
    monkeypatch.setattr(cli.VectorIndex, "load", Mock(return_value=index))
    monkeypatch.setattr(cli, "Embedder", Mock())
    for name in ("BM25Index", "HybridRetriever"):
        service = Mock()
        service.search.return_value = [candidate(0)]
        monkeypatch.setattr(cli, name, Mock(return_value=service))
    factory = Mock(side_effect=AssertionError("Should not load reranker"))
    monkeypatch.setattr(cli, "CrossEncoderReranker", factory)
    args = ["search", "q", "--index", str(tmp_path)]
    if method:
        args += ["--method", method]
    cli.main(args)
    assert "Passage 0" in capsys.readouterr().out
    factory.assert_not_called()


@pytest.mark.skipif(
    os.environ.get("RUN_RERANKING_INTEGRATION") != "1", reason="Opt-in model download"
)
def test_real_model() -> None:
    reranker = CrossEncoderReranker(batch_size=2)
    inputs = [
        replace(candidate(0), text="Bananas are yellow fruit."),
        replace(candidate(1), text="Paris is the capital of France."),
    ]
    results = reranker.rerank("What is the capital of France?", inputs)
    assert results[0].chunk_id == "1"
    assert np.isfinite([r.score for r in results]).all()
    assert [replace(r, score=0.01) for r in results] == [inputs[1], inputs[0]]
    # Explicit Identity keeps scores as logits rather than sigmoid probabilities.
    assert isinstance(reranker._model.activation_fn, nn.Identity)
