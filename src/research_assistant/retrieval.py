"""Exact cosine retrieval with a checked FAISS row-to-chunk mapping."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from research_assistant.chunking import Chunk
from research_assistant.embeddings import Embedder
from research_assistant.model_config import (
    EMBEDDING_DIMENSION,
    MODEL_NAME,
    MODEL_REVISION,
)


@dataclass(frozen=True)
class SearchResult:
    """A retrieved source passage and its inner-product similarity."""

    chunk_id: str
    document_id: str
    source: str
    page_number: int | None
    section: str | None
    text: str
    score: float


def _vectors(values: NDArray, rows: int) -> NDArray[np.float32]:
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.ascontiguousarray(values, dtype=np.float32)
    if array.shape != (rows, EMBEDDING_DIMENSION):
        raise ValueError(f"Expected embedding shape ({rows}, {EMBEDDING_DIMENSION})")
    if not np.isfinite(array).all():
        raise ValueError("Embeddings must contain only finite values")
    if not np.allclose(np.linalg.norm(array.astype(np.float64), axis=1), 1, atol=1e-5):
        raise ValueError("Embeddings must be L2-normalized")
    return array


def _top_k(value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError("top_k must be a positive integer")


def _check_chunks(chunks: tuple[Chunk, ...]) -> None:
    for chunk in chunks:
        if not isinstance(chunk, Chunk):
            raise ValueError("Metadata must contain Chunk records")
        for name in ("chunk_id", "document_id", "source", "text"):
            value = getattr(chunk, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Chunk {name} must be a nonempty string")
        if chunk.page_number is not None and (
            type(chunk.page_number) is not int or chunk.page_number < 1
        ):
            raise ValueError("Invalid chunk page_number")
        if chunk.section is not None and not isinstance(chunk.section, str):
            raise ValueError("Invalid chunk section")
        if type(chunk.chunk_index) is not int or chunk.chunk_index < 0:
            raise ValueError("Invalid chunk_index")
    if len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk identifiers")


class VectorIndex:
    """Store normalized vectors in exact IndexFlatIP order alongside chunks."""

    def __init__(
        self,
        chunks: list[Chunk],
        vectors: NDArray,
        *,
        max_tokens: int = 180,
        overlap: int = 30,
    ) -> None:
        # Initialize the embedding runtime before FAISS: importing FAISS first
        # caused native crashes with the installed macOS wheels.
        import faiss

        if (
            type(max_tokens) is not int
            or type(overlap) is not int
            or not 2 < max_tokens <= 256
            or not 0 <= overlap < max_tokens - 2
        ):
            raise ValueError("Invalid MiniLM chunking configuration")
        self.chunks = tuple(chunks)
        _check_chunks(self.chunks)
        self.max_tokens = max_tokens
        self.overlap = overlap
        array = _vectors(vectors, len(self.chunks))
        self._index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self._index.add(array)

    def search(
        self, question: str, embedder: Embedder, top_k: int = 5
    ) -> list[SearchResult]:
        """Embed only the question; reuse previously indexed document vectors."""
        _top_k(top_k)
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        if not self.chunks:
            return []
        return self.search_vector(embedder.encode([question]), top_k)

    def search_vector(self, vector: NDArray, top_k: int = 5) -> list[SearchResult]:
        """Search one normalized (1, 384) query, without model inference."""
        _top_k(top_k)
        query = _vectors(vector, 1)
        if not self.chunks:
            return []
        scores, positions = self._index.search(query, min(top_k, len(self.chunks)))
        matches = sorted(
            zip(scores[0], positions[0]), key=lambda item: (-item[0], item[1])
        )
        results = []
        for score, position in matches:
            chunk = self.chunks[int(position)]
            results.append(
                SearchResult(
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source,
                    chunk.page_number,
                    chunk.section,
                    chunk.text,
                    float(score),
                )
            )
        return results

    def _manifest(self) -> dict:
        return {
            "format_version": 1,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "chunk_count": len(self.chunks),
            "chunking": {"max_tokens": self.max_tokens, "overlap": self.overlap},
            "index_type": "IndexFlatIP",
            "document_ids": sorted({c.document_id for c in self.chunks}),
        }

    def save(self, path: Path) -> None:
        """Write index and ordered JSONL, then a manifest with file checksums."""
        import faiss

        path.mkdir(parents=True, exist_ok=True)
        payloads = {
            "vectors.faiss": faiss.serialize_index(self._index).tobytes(),
            "chunks.jsonl": "".join(
                json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in self.chunks
            ).encode("utf-8"),
        }
        manifest = self._manifest()
        manifest["sha256"] = {
            name: hashlib.sha256(content).hexdigest()
            for name, content in payloads.items()
        }
        payloads["manifest.json"] = json.dumps(manifest, indent=2).encode("utf-8")
        for name, content in payloads.items():
            temporary = path / f".{name}.tmp"
            temporary.write_bytes(content)
            temporary.replace(path / name)

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        """Load trusted local artifacts, rejecting incompatible or mixed files."""
        import faiss

        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            for key, expected in {
                "format_version": 1,
                "model_name": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "index_type": "IndexFlatIP",
            }.items():
                if manifest[key] != expected or type(manifest[key]) is not type(
                    expected
                ):
                    raise ValueError(f"Incompatible manifest {key}")
            if type(manifest["chunk_count"]) is not int or manifest["chunk_count"] < 0:
                raise ValueError("Invalid manifest chunk_count")
            payloads = {}
            for name in ("vectors.faiss", "chunks.jsonl"):
                content = (path / name).read_bytes()
                if hashlib.sha256(content).hexdigest() != manifest["sha256"][name]:
                    raise ValueError(f"Checksum mismatch: {name}")
                payloads[name] = content
            chunks = [
                Chunk(**json.loads(line))
                for line in payloads["chunks.jsonl"].decode("utf-8").splitlines()
            ]
            index = faiss.deserialize_index(
                np.frombuffer(payloads["vectors.faiss"], dtype=np.uint8)
            )
            if (
                not isinstance(index, faiss.IndexFlatIP)
                or index.d != EMBEDDING_DIMENSION
                or index.metric_type != faiss.METRIC_INNER_PRODUCT
            ):
                raise ValueError("Incompatible FAISS index type, dimension, or metric")
            if index.ntotal != len(chunks):
                raise ValueError("FAISS row count differs from chunk metadata")
            # Revalidate vectors, metadata and configuration before exposing results.
            result = cls(
                chunks, index.reconstruct_n(0, index.ntotal), **manifest["chunking"]
            )
            if manifest != dict(result._manifest(), sha256=manifest["sha256"]):
                raise ValueError(
                    "Manifest does not match chunk metadata or configuration"
                )
            result._index = index
            return result
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
        ) as exc:
            raise ValueError(f"Cannot load index '{path}': {exc}") from exc
