"""Offline lexical ranking and CLI checks, without model downloads."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from research_assistant.bm25 import BM25Index, tokenize
from research_assistant.chunking import Chunk
from research_assistant.retrieval import VectorIndex


def chunks(*texts: str) -> list[Chunk]:
    return [
        Chunk(str(i), f"doc{i}", f"paper{i}.pdf", i + 1, None, 0, text)
        for i, text in enumerate(texts)
    ]


def test_keyword_case_unicode_and_technical_tokens() -> None:
    assert tokenize("Straße CAFÉ self-attention Q_K BM25 C++") == [
        "strasse",
        "café",
        "self",
        "attention",
        "q_k",
        "bm25",
        "c",
    ]
    index = BM25Index(chunks("Straße self-attention Q_K", "other", "unrelated"))
    assert index.search("STRASSE")[0].chunk_id == "0"
    assert index.search("SELF ATTENTION") == index.search("self-attention")
    assert index.search("q_k")[0].chunk_id == "0"
    assert index.search("q") == []


def test_rare_vs_common() -> None:
    index = BM25Index(
        chunks("rare common", "common other", "common else", "four", "five")
    )
    assert index.search("rare")[0].score > index.search("common")[0].score
    assert index.search("rare common")[0].chunk_id == "0"


def test_repeated_term_saturation_and_query_repetition() -> None:
    index = BM25Index(
        chunks("needle", "needle needle needle", "a", "b", "c", "d", "e"), b=0
    )
    results = index.search("needle")
    assert [r.chunk_id for r in results] == ["1", "0"]
    assert results[1].score < results[0].score < 3 * results[1].score
    repeated = index.search("needle needle")
    assert repeated[0].score == pytest.approx(2 * results[0].score)


def test_length_normalization() -> None:
    corpus = chunks("needle", "needle filler filler filler filler", "a", "b", "c")
    assert [r.chunk_id for r in BM25Index(corpus).search("needle")] == ["0", "1"]
    scores = BM25Index(corpus, b=0).search("needle")
    assert scores[0].score == scores[1].score


def test_empty_no_match_and_empty_token_documents() -> None:
    assert BM25Index([]).search("anything") == []
    assert BM25Index(chunks("!!!", "---")).search("anything") == []
    index = BM25Index(chunks("needle", "!!!", "other"))
    for query in ("", " \n", "!?", "absent"):
        assert index.search(query) == []
    assert [r.chunk_id for r in index.search("needle", 99)] == ["0"]


def test_nonpositive_matches_are_not_discarded() -> None:
    # Okapi IDF can be zero for half the documents and negative for tiny corpora.
    assert BM25Index(chunks("needle", "other")).search("needle")[0].score == 0
    result = BM25Index(chunks("needle")).search("needle")
    assert len(result) == 1 and result[0].score < 0


def test_ties_and_metadata() -> None:
    corpus = chunks("same", "same", "other", "different", "else")
    corpus[0] = replace(
        corpus[0], page_number=None, section="Heading", source="notes.md"
    )
    index = BM25Index(corpus)
    corpus.clear()
    results = index.search("same", 50)
    assert [r.chunk_id for r in results] == ["0", "1"]
    assert results == index.search("same", 50)
    first = results[0]
    assert (
        first.document_id,
        first.source,
        first.page_number,
        first.section,
        first.text,
    ) == ("doc0", "notes.md", None, "Heading", "same")
    assert index.search("same", 1) == results[:1]


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, None, "5"])
def test_invalid_top_k(top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k"):
        BM25Index([]).search("query", top_k)


@pytest.mark.parametrize("query", [None, 42, [], b"query"])
def test_invalid_question(query: object) -> None:
    with pytest.raises(TypeError, match="string"):
        BM25Index([]).search(query)


@pytest.mark.parametrize("corpus", [None, "text", (), ["text"]])
def test_invalid_corpus(corpus: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        BM25Index(corpus)


def test_duplicates_and_bad_metadata() -> None:
    chunk = chunks("text")[0]
    with pytest.raises(ValueError, match="Duplicate"):
        BM25Index([chunk, chunk])
    with pytest.raises(ValueError, match="page_number"):
        BM25Index([replace(chunk, page_number=0)])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k1": 0},
        {"k1": -1},
        {"k1": float("nan")},
        {"k1": True},
        {"b": -0.1},
        {"b": 1.1},
        {"b": float("inf")},
        {"b": "0.75"},
    ],
)
def test_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        BM25Index([], **kwargs)


def test_cli_bm25_without_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from research_assistant import __main__ as cli

    corpus = chunks("positional encoding", "memory", "attention")
    vectors = np.zeros((3, 384), dtype=np.float32)
    vectors[:, 0] = 1
    VectorIndex(corpus, vectors).save(tmp_path)
    factory = Mock(side_effect=AssertionError("BM25 must not load the model"))
    monkeypatch.setattr(cli, "Embedder", factory)
    cli.main(
        [
            "search",
            "positional",
            "--index",
            str(tmp_path),
            "--method",
            "bm25",
            "--top-k",
            "5",
        ]
    )
    output = capsys.readouterr().out
    assert "paper0.pdf | page 1" in output and "positional encoding" in output
    factory.assert_not_called()
