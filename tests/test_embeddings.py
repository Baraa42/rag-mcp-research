"""Offline embedding contract tests and an opt-in real-model sanity check."""

import os
from unittest.mock import Mock

import numpy as np
import pytest

from research_assistant import embeddings
from research_assistant.embeddings import Embedder
from research_assistant.model_config import MODEL_NAME, MODEL_REVISION


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    model = Mock()
    factory = Mock(return_value=model)
    monkeypatch.setattr(embeddings, "SentenceTransformer", factory)
    return model, factory


def test_empty_without_inference(fake_model: tuple[Mock, Mock]) -> None:
    model, factory = fake_model
    embedder = Embedder()
    result = embedder.encode([])
    assert result.shape == (0, 384)
    assert result.dtype == np.float32
    model.encode.assert_not_called()
    factory.assert_called_once_with(MODEL_NAME, revision=MODEL_REVISION, device="cpu")


def test_shape_dtype_order_normalization_and_batch(
    fake_model: tuple[Mock, Mock],
) -> None:
    model, factory = fake_model
    raw = np.zeros((3, 384), dtype=np.float64)
    raw[0, :2] = [3, 4]
    raw[1, 2] = 8
    raw[2, 3] = -5
    model.encode.return_value = raw
    embedder = Embedder(batch_size=2)
    texts = ["first", "second", "third"]
    result = embedder.encode(texts)
    assert result.shape == (3, 384)
    assert result.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(result[0, :2], [0.6, 0.8])
    assert result[1, 2] == 1 and result[2, 3] == -1
    model.encode.assert_called_once_with(
        texts,
        batch_size=2,
        convert_to_numpy=True,
        precision="float32",
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    embedder.encode(texts)
    factory.assert_called_once()


@pytest.mark.parametrize(
    "texts",
    [None, "sentence", ("tuple",), {"set"}, [1], [None], [b"bytes"], ["valid", []]],
)
def test_invalid_inputs(fake_model: tuple[Mock, Mock], texts: object) -> None:
    with pytest.raises(TypeError, match="string"):
        Embedder().encode(texts)
    fake_model[0].encode.assert_not_called()


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True, "32", None])
def test_invalid_batch_size(fake_model: tuple[Mock, Mock], batch_size: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        Embedder(batch_size=batch_size)
    fake_model[1].assert_not_called()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e100])
def test_nonfinite_output(fake_model: tuple[Mock, Mock], value: float) -> None:
    raw = np.ones((1, 384), dtype=np.float64)
    raw[0, 0] = value
    fake_model[0].encode.return_value = raw
    with pytest.raises(ValueError, match="non-finite"):
        Embedder().encode(["test"])


@pytest.mark.parametrize("shape", [(384,), (1, 383), (2, 384)])
def test_wrong_shape(fake_model: tuple[Mock, Mock], shape: tuple[int, ...]) -> None:
    fake_model[0].encode.return_value = np.ones(shape)
    with pytest.raises(ValueError, match="shape"):
        Embedder().encode(["test"])


def test_zero_norm(fake_model: tuple[Mock, Mock]) -> None:
    fake_model[0].encode.return_value = np.zeros((1, 384))
    with pytest.raises(ValueError, match="zero-norm"):
        Embedder().encode(["test"])


def test_large_finite_values(fake_model: tuple[Mock, Mock]) -> None:
    fake_model[0].encode.return_value = np.full((1, 384), 1e30, dtype=np.float32)
    result = Embedder().encode([""])
    assert np.isfinite(result).all()
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6)


@pytest.mark.skipif(
    os.environ.get("RUN_EMBEDDING_INTEGRATION") != "1", reason="Opt-in model download"
)
def test_real_model() -> None:
    embedder = Embedder(batch_size=2)
    vectors = embedder.encode(
        [
            "A cat is sleeping on the sofa.",
            "A kitten is resting on a couch.",
            "The stock market closed higher today.",
            "A cat is sleeping on the sofa.",
        ]
    )
    assert vectors.shape == (4, 384)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(vectors[0], vectors[3], atol=1e-6)
    # A relative sanity check for these examples, not a universal threshold.
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]
