"""CPU sentence embeddings with a checked, normalized NumPy output contract."""

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from research_assistant.model_config import (
    EMBEDDING_DIMENSION,
    MODEL_NAME,
    MODEL_REVISION,
)


class Embedder:
    """Load the pinned MiniLM model once and encode ordered lists of strings."""

    def __init__(self, batch_size: int = 32) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self._model = SentenceTransformer(
            MODEL_NAME, revision=MODEL_REVISION, device="cpu"
        )

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        """Return unit-length float32 rows in input order; accept empty strings.

        Empty lists bypass inference. Non-finite, zero-norm, or incorrectly
        shaped model output raises ValueError instead of returning bad vectors.
        """
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"texts[{index}] must be a string")
        if not texts:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        raw = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            precision="float32",
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            vectors = np.asarray(raw, dtype=np.float32)
        expected = (len(texts), EMBEDDING_DIMENSION)
        if vectors.shape != expected:
            raise ValueError(
                f"Expected embedding shape {expected}, got {vectors.shape}"
            )
        if not np.isfinite(vectors).all():
            raise ValueError("Model returned non-finite embeddings")
        # Accumulate in float64 to avoid overflow for large finite float32 rows.
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Model returned zero-norm embeddings")
        return (vectors / norms).astype(np.float32)
