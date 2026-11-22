"""Exact retrieval/persistence contracts with offline synthetic vectors."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from research_assistant.ingest.chunking import Chunk, chunk_pages, load_tokenizer
from research_assistant.ingest.ingestion import ingest_directory
from research_assistant.retrieval.embeddings import Embedder
from research_assistant.retrieval.vector import VectorIndex


def corpus() -> tuple[list[Chunk], np.ndarray]:
    chunks = [
        Chunk(f"c{i}", f"d{i}", f"paper{i}.pdf", i + 1, None, 0, f"passage {i}")
        for i in range(3)
    ]
    vectors = np.zeros((3, 384), dtype=np.float64)
    vectors[0, :2] = [0.6, 0.8]
    vectors[1, 0] = 1
    vectors[2, 1] = 1
    return chunks, np.asfortranarray(vectors)


def query() -> np.ndarray:
    vector = np.zeros((1, 384), dtype=np.float32)
    vector[0, 0] = 1
    return vector


def test_ranking_metadata_order_and_determinism() -> None:
    chunks, vectors = corpus()
    index = VectorIndex(chunks, vectors)
    results = index.search_vector(query(), top_k=10)
    assert [r.chunk_id for r in results] == ["c1", "c0", "c2"]
    assert [r.score for r in results] == sorted(
        [r.score for r in results], reverse=True
    )
    np.testing.assert_allclose([r.score for r in results], [1, 0.6, 0])
    assert (
        results[0].document_id,
        results[0].source,
        results[0].page_number,
        results[0].text,
    ) == ("d1", "paper1.pdf", 2, "passage 1")
    assert index.search_vector(query(), 10) == results
    assert len(index.search_vector(query(), 1)) == 1
    chunks.clear()
    assert len(index.chunks) == 3


def test_search_embeds_only_question() -> None:
    index = VectorIndex(*corpus())
    embedder = Mock()
    embedder.encode.return_value = query()
    assert index.search("question", embedder, 2) == index.search_vector(query(), 2)
    embedder.encode.assert_called_once_with(["question"])


def test_empty_roundtrip(tmp_path: Path) -> None:
    index = VectorIndex([], np.empty((0, 384)))
    model = Mock()
    assert index.search("question", model) == []
    assert index.search_vector(query()) == []
    model.encode.assert_not_called()
    index.save(tmp_path)
    assert VectorIndex.load(tmp_path).search_vector(query()) == []


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3", None])
def test_invalid_top_k(value: object) -> None:
    index = VectorIndex(*corpus())
    with pytest.raises(ValueError, match="top_k"):
        index.search_vector(query(), value)
    with pytest.raises(ValueError, match="top_k"):
        index.search("question", Mock(), value)


@pytest.mark.parametrize("value", [None, "", "  ", 42])
def test_invalid_question(value: object) -> None:
    with pytest.raises(ValueError, match="question"):
        VectorIndex(*corpus()).search(value, Mock())


@pytest.mark.parametrize("shape", [(3, 383), (2, 384), (384,)])
def test_invalid_dimensions(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shape"):
        VectorIndex(corpus()[0], np.zeros(shape))
    with pytest.raises(ValueError, match="shape"):
        VectorIndex(*corpus()).search_vector(np.zeros(shape))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e100])
def test_nonfinite(value: float) -> None:
    chunks, vectors = corpus()
    vectors[0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        VectorIndex(chunks, vectors)
    q = query().astype(np.float64)
    q[0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        VectorIndex(*corpus()).search_vector(q)


def test_reject_unnormalized_and_duplicate_ids() -> None:
    chunks, vectors = corpus()
    with pytest.raises(ValueError, match="normalized"):
        VectorIndex(chunks, vectors * 2)
    with pytest.raises(ValueError, match="normalized"):
        VectorIndex(chunks, vectors).search_vector(query() * 0)
    chunks[1] = replace(chunks[1], chunk_id=chunks[0].chunk_id)
    with pytest.raises(ValueError, match="Duplicate"):
        VectorIndex(chunks, vectors)


def test_roundtrip(tmp_path: Path) -> None:
    chunks, vectors = corpus()
    chunks[0] = replace(
        chunks[0], source="notes.md", page_number=None, section="Résumé"
    )
    index = VectorIndex(chunks, vectors, max_tokens=100, overlap=20)
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert loaded.chunks == index.chunks
    assert loaded.search_vector(query()) == index.search_vector(query())
    assert (loaded.max_tokens, loaded.overlap) == (100, 20)
    assert {p.name for p in tmp_path.iterdir()} == {
        "vectors.faiss",
        "chunks.jsonl",
        "manifest.json",
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_name", "wrong"),
        ("model_revision", "wrong"),
        ("embedding_dimension", 128),
        ("chunk_count", 99),
        ("index_type", "IndexFlatL2"),
        ("document_ids", ["wrong"]),
        ("chunking", {"max_tokens": 1, "overlap": 30}),
    ],
)
def test_incompatible_manifest(tmp_path: Path, key: str, value: object) -> None:
    VectorIndex(*corpus()).save(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[key] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Cannot load"):
        VectorIndex.load(tmp_path)


@pytest.mark.parametrize("name", ["vectors.faiss", "chunks.jsonl"])
def test_corrupt_or_mixed_files(tmp_path: Path, name: str) -> None:
    VectorIndex(*corpus()).save(tmp_path)
    path = tmp_path / name
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="Checksum"):
        VectorIndex.load(tmp_path)


def refresh_checksum(directory: Path, name: str) -> None:
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["sha256"][name] = hashlib.sha256(
        (directory / name).read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    "kind", ["metric", "dimension", "count", "nonfinite", "unnormalized"]
)
def test_index_structure_checks(tmp_path: Path, kind: str) -> None:
    import faiss

    VectorIndex(*corpus()).save(tmp_path)
    index = (
        faiss.IndexFlatL2(384)
        if kind == "metric"
        else faiss.IndexFlatIP(128 if kind == "dimension" else 384)
    )
    values = np.zeros((2 if kind == "count" else 3, index.d), dtype=np.float32)
    values[:, 0] = np.nan if kind == "nonfinite" else 2 if kind == "unnormalized" else 1
    index.add(values)
    faiss.write_index(index, str(tmp_path / "vectors.faiss"))
    refresh_checksum(tmp_path, "vectors.faiss")
    with pytest.raises(ValueError, match="Cannot load"):
        VectorIndex.load(tmp_path)


def test_invalid_metadata_and_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Cannot load"):
        VectorIndex.load(tmp_path)
    VectorIndex(*corpus()).save(tmp_path)
    path = tmp_path / "chunks.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["page_number"] = "third"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    refresh_checksum(tmp_path, "chunks.jsonl")
    with pytest.raises(ValueError, match="page_number"):
        VectorIndex.load(tmp_path)


def test_cli_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from research_assistant.interfaces import cli

    chunks, vectors = corpus()
    monkeypatch.setattr(cli, "ingest_directory", lambda _: [])
    monkeypatch.setattr(cli, "chunk_pages", lambda _: chunks)
    model = Mock()
    model.encode.return_value = vectors
    monkeypatch.setattr(cli, "Embedder", lambda: model)
    cli.main(["index", "unused", "--out", str(tmp_path)])
    assert "Indexed 3 chunks" in capsys.readouterr().out
    model.encode.return_value = query()
    cli.main(["search", "question", "--index", str(tmp_path), "--top-k", "1"])
    output = capsys.readouterr().out
    assert "score=1.000000" in output and "paper1.pdf | page 2" in output
    assert "passage 1" in output


@pytest.mark.skipif(
    os.environ.get("RUN_RETRIEVAL_INTEGRATION") != "1", reason="Opt-in model download"
)
def test_real_end_to_end(tmp_path: Path) -> None:
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "cats.md").write_text(
        "# Cats\nA cat is sleeping on a sofa.", encoding="utf-8"
    )
    (papers / "markets.md").write_text(
        "# Markets\nStock markets rose after the earnings announcement.",
        encoding="utf-8",
    )
    pages = ingest_directory(papers)
    chunks = chunk_pages(pages, tokenizer=load_tokenizer())
    model = Embedder()
    index = VectorIndex(chunks, model.encode([c.text for c in chunks]))
    results = index.search("Where is the kitten resting?", model, 2)
    assert results[0].source == "cats.md"
    index.save(tmp_path / "index")
    restored = VectorIndex.load(tmp_path / "index").search(
        "Where is the kitten resting?", model, 2
    )
    assert [r.chunk_id for r in restored] == [r.chunk_id for r in results]
    np.testing.assert_allclose(
        [r.score for r in restored], [r.score for r in results], atol=1e-6
    )
