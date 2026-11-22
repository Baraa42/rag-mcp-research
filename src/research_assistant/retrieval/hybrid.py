"""Rank-only fusion of existing lexical and vector retrieval results."""

from dataclasses import replace

from research_assistant.retrieval.bm25 import BM25Index
from research_assistant.retrieval.embeddings import Embedder
from research_assistant.retrieval.vector import SearchResult, VectorIndex, _top_k


def reciprocal_rank_fusion(
    rankings: list[list[SearchResult]], *, top_k: int = 5, rrf_k: int = 60
) -> list[SearchResult]:
    """Fuse 1-based ranks with equal weights; break ties by chunk ID.

    Duplicate IDs within one list contribute only at their first position;
    later positions retain their original ranks. Conflicting metadata is rejected.
    Raw method scores are ignored. Inputs are not modified.
    """
    _top_k(top_k)
    if type(rrf_k) is not int or rrf_k < 0:
        raise ValueError("rrf_k must be a nonnegative integer")
    if not isinstance(rankings, list):
        raise TypeError("rankings must be a list of ranked lists")
    records: dict[str, SearchResult] = {}
    scores: dict[str, float] = {}
    for ranking in rankings:
        if not isinstance(ranking, list):
            raise TypeError("Each ranking must be a list of SearchResult objects")
        seen = set()
        for rank, result in enumerate(ranking, start=1):
            if not isinstance(result, SearchResult):
                raise TypeError("Ranked entries must be SearchResult objects")
            if not isinstance(result.chunk_id, str) or not result.chunk_id.strip():
                raise ValueError("chunk_id must be a nonempty string")
            metadata = replace(result, score=0.0)
            chunk_id = result.chunk_id
            if chunk_id in records and records[chunk_id] != metadata:
                raise ValueError(f"Conflicting metadata for chunk_id {chunk_id!r}")
            records[chunk_id] = metadata
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rrf_k + rank)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [replace(records[key], score=scores[key]) for key in ordered[:top_k]]


class HybridRetriever:
    """Reuse existing indexes/model; embed only the question and fuse ranks."""

    def __init__(
        self,
        vector_index: VectorIndex,
        bm25_index: BM25Index,
        embedder: Embedder,
        *,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        if type(candidate_k) is not int or candidate_k <= 0:
            raise ValueError("candidate_k must be a positive integer")
        if type(rrf_k) is not int or rrf_k < 0:
            raise ValueError("rrf_k must be a nonnegative integer")
        if vector_index.chunks != bm25_index.chunks:
            raise ValueError("Hybrid indexes must contain the same ordered chunks")
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(self, question: str, top_k: int = 5) -> list[SearchResult]:
        """Expand candidate depth to top_k when larger than the configured depth."""
        _top_k(top_k)
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        depth = max(self.candidate_k, top_k)
        vector_results = self.vector_index.search(question, self.embedder, depth)
        lexical_results = self.bm25_index.search(question, depth)
        return reciprocal_rank_fusion(
            [vector_results, lexical_results], top_k=top_k, rrf_k=self.rrf_k
        )
