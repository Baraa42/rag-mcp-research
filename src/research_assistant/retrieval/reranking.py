"""CPU cross-encoder scoring, independent of first-stage retrieval."""

from dataclasses import replace

import numpy as np
from sentence_transformers import CrossEncoder
from torch import nn

from research_assistant.retrieval.vector import SearchResult, _top_k

RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


class CrossEncoderReranker:
    """Load one pinned model and replace candidate scores with raw logits."""

    def __init__(self, batch_size: int = 16) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self._model = CrossEncoder(
            RERANKER_NAME,
            revision=RERANKER_REVISION,
            device="cpu",
            max_length=512,
            activation_fn=nn.Identity(),
        )

    def rerank(
        self, question: str, candidates: list[SearchResult], top_k: int = 5
    ) -> list[SearchResult]:
        """Score query/passage pairs; ties retain original candidate order.

        Scores are raw relevance logits, not probabilities. Empty candidates
        bypass inference. Inputs and all non-score metadata remain unchanged.
        """
        _top_k(top_k)
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        if not isinstance(candidates, list):
            raise TypeError("candidates must be a list of SearchResult objects")
        seen = set()
        for candidate in candidates:
            if not isinstance(candidate, SearchResult):
                raise TypeError("Each candidate must be a SearchResult")
            if not isinstance(candidate.text, str) or not candidate.text.strip():
                raise ValueError("Candidate text must be a nonempty string")
            if (
                not isinstance(candidate.chunk_id, str)
                or not candidate.chunk_id.strip()
            ):
                raise ValueError("Candidate chunk_id must be a nonempty string")
            if candidate.chunk_id in seen:
                raise ValueError("Duplicate candidate chunk IDs")
            seen.add(candidate.chunk_id)
        if not candidates:
            return []
        pairs = [(question, candidate.text) for candidate in candidates]
        scores = np.asarray(
            self._model.predict(
                pairs,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float64,
        )
        if scores.shape == (len(candidates), 1):
            scores = scores[:, 0]
        if scores.shape != (len(candidates),):
            raise ValueError("Cross-encoder must return one score per candidate")
        if not np.isfinite(scores).all():
            raise ValueError("Cross-encoder returned non-finite scores")
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        return [replace(candidates[i], score=float(scores[i])) for i in order[:top_k]]
