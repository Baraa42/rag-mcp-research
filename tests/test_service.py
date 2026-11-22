"""Offline service orchestration and resource lifecycle contracts."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from research_assistant import service
from research_assistant.answering import INSUFFICIENT_EVIDENCE, GroundedAnswer
from research_assistant.chunking import Chunk
from research_assistant.retrieval import SearchResult


@pytest.fixture
def setup(monkeypatch):
    chunk = Chunk("id", "doc", "paper.pdf", 3, "Intro", 0, "Original text")
    index = Mock(chunks=(chunk,))
    load = Mock(return_value=index)
    monkeypatch.setattr(service.VectorIndex, "load", load)
    factories = {}
    for name in (
        "Embedder",
        "BM25Index",
        "HybridRetriever",
        "CrossEncoderReranker",
        "OpenAI",
        "GroundedAnswerer",
    ):
        factory = Mock()
        monkeypatch.setattr(service, name, factory)
        factories[name] = factory
    result = SearchResult("id", "doc", "paper.pdf", 3, "Intro", "Original text", 0.04)
    factories["HybridRetriever"].return_value.search.return_value = [result]
    factories["CrossEncoderReranker"].return_value.rerank.return_value = [
        replace(result, score=2.0)
    ]
    return index, load, factories


def test_initialization_lookup_and_close_without_models(setup):
    index, load, factories = setup
    instance = service.ResearchService(Path("index"))
    load.assert_called_once_with(Path("index"))
    assert instance.get_passage("id") is index.chunks[0]
    with pytest.raises(KeyError, match="Unknown chunk ID"):
        instance.get_passage("missing")
    instance.close()
    instance.close()
    for factory in factories.values():
        factory.assert_not_called()


def test_hybrid_reuse_then_rerank(setup):
    index, load, factories = setup
    instance = service.ResearchService(Path("index"))
    hybrid = factories["HybridRetriever"].return_value
    expected = hybrid.search.return_value
    assert instance.search("q", 3, False) is expected
    assert instance.search("q", 3, False) is expected
    hybrid.search.assert_called_with("q", 3)
    factories["CrossEncoderReranker"].assert_not_called()
    factories["OpenAI"].assert_not_called()
    for name in ("Embedder", "BM25Index", "HybridRetriever"):
        factories[name].assert_called_once()
    factories["HybridRetriever"].assert_called_once_with(
        index, factories["BM25Index"].return_value, factories["Embedder"].return_value
    )
    for depth in (5, 25):
        output = instance.search("q", depth)
        hybrid.search.assert_called_with("q", max(20, depth))
        factories["CrossEncoderReranker"].return_value.rerank.assert_called_with(
            "q", expected, depth
        )
        assert replace(output[0], score=expected[0].score) == expected[0]
    factories["CrossEncoderReranker"].assert_called_once()
    load.assert_called_once()


@pytest.mark.parametrize("question", [None, "", " ", 3])
def test_invalid_question(setup, question):
    instance = service.ResearchService(Path("index"))
    with pytest.raises(ValueError, match="question"):
        instance.search(question)
    setup[2]["Embedder"].assert_not_called()


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, None])
def test_invalid_top_k(setup, top_k):
    with pytest.raises(ValueError, match="top_k"):
        service.ResearchService(Path("index")).search("q", top_k)


@pytest.mark.parametrize("chunk_id", [None, "", " ", 123])
def test_invalid_lookup_id(setup, chunk_id):
    with pytest.raises(ValueError, match="chunk_id"):
        service.ResearchService(Path("index")).get_passage(chunk_id)


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_id(setup, conflicting):
    index = setup[0]
    chunk = index.chunks[0]
    index.chunks = (chunk, replace(chunk, text="Different") if conflicting else chunk)
    with pytest.raises(ValueError, match="Duplicate chunk ID"):
        service.ResearchService(Path("index"))


def test_invalid_persisted_id(setup):
    setup[0].chunks = (replace(setup[0].chunks[0], chunk_id=""),)
    with pytest.raises(ValueError, match="chunk_id"):
        service.ResearchService(Path("index"))


def test_empty_index(setup):
    setup[0].chunks = ()
    instance = service.ResearchService(Path("index"))
    assert instance.search("q") == []
    assert instance.search("q", rerank=False) == []
    for factory in setup[2].values():
        factory.assert_not_called()


def test_empty_index_answer_without_models_or_api(setup):
    setup[0].chunks = ()
    instance = service.ResearchService(Path("index"))
    assert instance.answer("q") == GroundedAnswer(INSUFFICIENT_EVIDENCE, ())
    for factory in setup[2].values():
        factory.assert_not_called()


def test_answer_handoff_reuse_and_close(setup, monkeypatch):
    factories = setup[2]
    instance = service.ResearchService(Path("index"))
    passages = factories["HybridRetriever"].return_value.search.return_value
    search = Mock(return_value=passages)
    monkeypatch.setattr(instance, "search", search)
    answerer = factories["GroundedAnswerer"].return_value
    answerer.answer.return_value = GroundedAnswer("Answer", ())
    for _ in range(2):
        assert instance.answer("q") is answerer.answer.return_value
    search.assert_called_with("q", top_k=5, rerank=True)
    answerer.answer.assert_called_with("q", passages)
    factories["OpenAI"].assert_called_once_with(timeout=60.0, max_retries=2)
    factories["GroundedAnswerer"].assert_called_once_with(
        factories["OpenAI"].return_value
    )
    instance.close()
    instance.close()
    factories["OpenAI"].return_value.close.assert_called_once()


def test_failed_answerer_construction_closes_client(setup):
    factories = setup[2]
    factories["GroundedAnswerer"].side_effect = ValueError("Missing model")
    instance = service.ResearchService(Path("index"))
    with pytest.raises(ValueError, match="Missing model"):
        instance.answer("q")
    instance.close()
    factories["OpenAI"].return_value.close.assert_called_once()


def test_closed_service(setup):
    instance = service.ResearchService(Path("index"))
    instance.close()
    for call in (
        lambda: instance.search("q"),
        lambda: instance.answer("q"),
        lambda: instance.get_passage("id"),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            call()
