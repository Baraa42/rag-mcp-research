"""Reusable, synchronous research orchestration with lazy model ownership."""

from pathlib import Path

from openai import OpenAI

from research_assistant.answering import GroundedAnswer, GroundedAnswerer
from research_assistant.bm25 import BM25Index
from research_assistant.chunking import Chunk
from research_assistant.embeddings import Embedder
from research_assistant.hybrid import HybridRetriever
from research_assistant.reranking import CrossEncoderReranker
from research_assistant.retrieval import SearchResult, VectorIndex, _top_k


class ResearchService:
    """Load one index snapshot; lazily reuse models for sequential requests.

    Call close() when finished. All operations except close() reject subsequent
    use. Concurrent requests and index reloads are not supported by this wrapper.
    """

    def __init__(self, index_path: Path) -> None:
        self._index = VectorIndex.load(index_path)
        self._chunks: dict[str, Chunk] = {}
        for chunk in self._index.chunks:
            if not isinstance(chunk.chunk_id, str) or not chunk.chunk_id.strip():
                raise ValueError("chunk_id must be a nonempty string")
            if chunk.chunk_id in self._chunks:
                raise ValueError(f"Duplicate chunk ID: {chunk.chunk_id}")
            self._chunks[chunk.chunk_id] = chunk
        self._hybrid: HybridRetriever | None = None
        self._reranker: CrossEncoderReranker | None = None
        self._client: OpenAI | None = None
        self._answerer: GroundedAnswerer | None = None
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ResearchService is closed")

    def search(
        self, question: str, top_k: int = 5, rerank: bool = True
    ) -> list[SearchResult]:
        """Search hybrid rankings, optionally reranking at least 20 candidates."""
        self._ensure_open()
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        _top_k(top_k)
        if type(rerank) is not bool:
            raise ValueError("rerank must be a boolean")
        if not self._chunks:
            return []
        if self._hybrid is None:
            self._hybrid = HybridRetriever(
                self._index, BM25Index(list(self._index.chunks)), Embedder()
            )
        candidates = self._hybrid.search(question, max(20, top_k) if rerank else top_k)
        if not rerank:
            return candidates
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker.rerank(question, candidates, top_k)

    def get_passage(self, chunk_id: str) -> Chunk:
        """Return the exact persisted chunk without initializing any models."""
        self._ensure_open()
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("chunk_id must be a nonempty string")
        try:
            return self._chunks[chunk_id]
        except KeyError:
            raise KeyError(f"Unknown chunk ID: {chunk_id}") from None

    def answer(self, question: str) -> GroundedAnswer:
        """Generate using five reranked passages and the existing citation rules."""
        self._ensure_open()
        passages = self.search(question, top_k=5, rerank=True)
        if self._answerer is None:
            client = OpenAI(timeout=60.0, max_retries=2)
            try:
                answerer = GroundedAnswerer(client)
            except Exception:
                client.close()
                raise
            self._client = client
            self._answerer = answerer
        return self._answerer.answer(question, passages)

    def close(self) -> None:
        """Close the owned API client once and release references to ML models."""
        if self._closed:
            return
        self._closed = True
        client, self._client = self._client, None
        self._answerer = None
        self._reranker = None
        self._hybrid = None
        if client is not None:
            client.close()
