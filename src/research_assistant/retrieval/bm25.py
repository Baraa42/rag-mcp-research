"""Deterministic lexical retrieval over the same chunks as vector search."""

import math
import re

from rank_bm25 import BM25Okapi

from research_assistant.ingest.chunking import Chunk
from research_assistant.retrieval.vector import SearchResult, _check_chunks, _top_k


def tokenize(text: str) -> list[str]:
    """Casefold Unicode word runs; split punctuation/hyphens, keep underscores."""
    return re.findall(r"\w+", text.casefold())


class BM25Index:
    """Rank term-matching chunks with BM25Okapi; ties retain corpus order."""

    def __init__(
        self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75
    ) -> None:
        if not isinstance(chunks, list):
            raise TypeError("chunks must be a list of Chunk records")
        for name, value in (("k1", k1), ("b", b)):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Require k1 > 0 and 0 <= b <= 1")
        self.chunks = tuple(chunks)
        _check_chunks(self.chunks)
        tokens = [tokenize(chunk.text) for chunk in self.chunks]
        # BM25Okapi cannot initialize an empty vocabulary. Empty-token documents
        # otherwise retain their original positions in the corpus statistics.
        self._model = BM25Okapi(tokens, k1=k1, b=b) if any(tokens) else None

    def search(self, question: str, top_k: int = 5) -> list[SearchResult]:
        """Return lexical matches, including zero/negative scores if applicable.

        BM25 scores are not cosine similarities or calibrated probabilities.
        Empty/tokenless/no-match queries return no results.
        """
        _top_k(top_k)
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        terms = tokenize(question)
        if not terms or self._model is None:
            return []
        matching = [
            i
            for i, frequencies in enumerate(self._model.doc_freqs)
            if any(term in frequencies for term in terms)
        ]
        if not matching:
            return []
        scores = self._model.get_scores(terms)
        ranked = sorted(matching, key=lambda i: (-float(scores[i]), i))[:top_k]
        results = []
        for i in ranked:
            chunk = self.chunks[i]
            results.append(
                SearchResult(
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source,
                    chunk.page_number,
                    chunk.section,
                    chunk.text,
                    float(scores[i]),
                )
            )
        return results
