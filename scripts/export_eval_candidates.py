"""Export pooled retrieval candidates or one document for human labeling."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from research_assistant.retrieval.bm25 import BM25Index
from research_assistant.retrieval.embeddings import Embedder
from research_assistant.retrieval.hybrid import HybridRetriever
from research_assistant.retrieval.reranking import CrossEncoderReranker
from research_assistant.retrieval.vector import SearchResult, VectorIndex

METHODS = ("faiss", "bm25", "hybrid", "rerank")
DEPTH = 20


def pool_candidates(rankings: dict[str, list[SearchResult]]) -> list[dict]:
    """Deduplicate by ID, retain first per-method ranks, sort by chunk ID."""
    records = {}
    metadata = {}
    for method in METHODS:
        for rank, result in enumerate(rankings.get(method, []), 1):
            chunk_id = result.chunk_id
            comparable = replace(result, score=0.0)
            if chunk_id in metadata and metadata[chunk_id] != comparable:
                raise ValueError(f"Conflicting metadata for {chunk_id}")
            metadata[chunk_id] = comparable
            if chunk_id not in records:
                records[chunk_id] = {
                    "chunk_id": chunk_id,
                    "source": result.source,
                    "page_number": result.page_number,
                    "section": result.section,
                    "text": result.text,
                    "ranks": dict.fromkeys(METHODS),
                    "cross_encoder_score": None,
                }
            record = records[chunk_id]
            if record["ranks"][method] is None:
                record["ranks"][method] = rank
                if method == "rerank":
                    record["cross_encoder_score"] = float(result.score)
    return [records[key] for key in sorted(records)]


def retrieve_candidates(
    question: str,
    index: VectorIndex,
    bm25: BM25Index,
    hybrid: HybridRetriever,
    embedder: Embedder,
    reranker: CrossEncoderReranker,
) -> list[dict]:
    """Pool four top-20 lists; reranking operates on the hybrid top 20."""
    vector_results = index.search(question, embedder, DEPTH)
    lexical_results = bm25.search(question, DEPTH)
    # TODO: hybrid.search() independently repeats these FAISS and BM25 searches.
    # Compute each ranking once and reuse for RRF and evaluation, preserving
    # identical ranking semantics and deterministic tie-breaking.
    hybrid_results = hybrid.search(question, DEPTH)
    # TODO: Only hybrid top 20 reach the cross-encoder; rank 21 cannot be promoted.
    # Compare depths 10, 20, 50, and possibly 100 using Recall@K, MRR@K, and
    # inference latency; larger candidate pools need not improve quality.
    reranked = reranker.rerank(question, hybrid_results, DEPTH)
    # TODO: Pooling FAISS/BM25 top 20 does not ensure exhaustive relevance coverage.
    # Inspect relevant source documents beyond the pool; track unjudged passages
    # separately from grade 0, never treating missing labels as irrelevant.
    return pool_candidates(
        {
            "faiss": vector_results,
            "bm25": lexical_results,
            "hybrid": hybrid_results,
            "rerank": reranked,
        }
    )


def export_source(index: VectorIndex, source: str) -> list[dict]:
    """Export exact source-path matches in persisted document/chunk order."""
    return [
        {
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "page_number": chunk.page_number,
            "section": chunk.section,
            "text": chunk.text,
        }
        for chunk in index.chunks
        if chunk.source == source
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--question", help="Literal question text")
    mode.add_argument("--question-id", help="ID from the questions JSON file")
    mode.add_argument(
        "--source", help="Exact persisted source path; exports all its chunks"
    )
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.json"))
    parser.add_argument("--index", type=Path, default=Path("data/index"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        question = args.question
        if args.question_id is not None:
            questions = json.loads(args.questions.read_text(encoding="utf-8"))
            if args.question_id not in questions:
                raise ValueError(f"Unknown question ID: {args.question_id}")
            question = questions[args.question_id]
        if args.source is None and (
            not isinstance(question, str) or not question.strip()
        ):
            raise ValueError("Provide a nonempty question")
        index = VectorIndex.load(args.index)
        if args.source is not None:
            chunks = export_source(index, args.source)
            if not chunks:
                raise ValueError(f"No chunks for exact source: {args.source}")
            output = {"source": args.source, "chunks": chunks}
        else:
            candidates = []
            if index.chunks:
                embedder = Embedder()
                bm25 = BM25Index(list(index.chunks))
                hybrid = HybridRetriever(index, bm25, embedder)
                candidates = retrieve_candidates(
                    question, index, bm25, hybrid, embedder, CrossEncoderReranker()
                )
            output = {
                "question_id": args.question_id,
                "question": question,
                "candidate_depth": DEPTH,
                "rerank_candidate_depth": DEPTH,
                "candidates": candidates,
            }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Refuse accidental replacement of a previous inspection export.
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            )
        print(f"Exported to {args.out}")
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
