"""Export pooled retrieval candidates or one document for human labeling."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from research_assistant.retrieval.bm25 import BM25Index
from research_assistant.retrieval.embeddings import Embedder
from research_assistant.retrieval.hybrid import HybridRetriever
from research_assistant.retrieval.reranking import (
    RERANKER_NAME,
    RERANKER_REVISION,
    CrossEncoderReranker,
)
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


def validate_batch_dataset(questions: dict, metadata: dict, baseline: dict) -> None:
    """Require the frozen v2 shape and preserve the five development anchors."""
    ids = {f"Q{i:02}" for i in range(1, 51)}
    if not all(isinstance(data, dict) for data in (questions, metadata)):
        raise ValueError("Questions and metadata must be objects keyed by question ID")
    if set(questions) != ids or set(metadata) != ids:
        raise ValueError("Expected matching question and metadata IDs Q01–Q50")
    normalized = set()
    for key, question in questions.items():
        entry = metadata[key]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Blank or invalid question: {key}")
        canonical = " ".join(question.casefold().split()).rstrip("?.!")
        if canonical in normalized:
            raise ValueError(f"Duplicate question text: {key}")
        normalized.add(canonical)
        if (
            not isinstance(entry, dict)
            or entry.get("split") not in ("development", "held_out")
            or entry.get("difficulty") not in ("easy", "medium", "hard")
            or not isinstance(entry.get("category"), str)
            or not entry["category"].strip()
            or not isinstance(entry.get("reference_answer"), str)
            or not entry["reference_answer"].strip()
            or type(entry.get("answerable")) is not bool
            or not isinstance(entry.get("references"), list)
            or bool(entry["references"]) != entry["answerable"]
        ):
            raise ValueError(f"Invalid metadata: {key}")
        if entry["answerable"] != (key < "Q45"):
            raise ValueError(f"Unexpected answerability: {key}")
        for reference in entry["references"]:
            if (
                not isinstance(reference, dict)
                or set(reference) != {"chunk_id", "source", "page_number"}
                or not isinstance(reference["chunk_id"], str)
                or not isinstance(reference["source"], str)
                or type(reference["page_number"]) is not int
                or reference["page_number"] < 1
            ):
                raise ValueError(f"Invalid reference pointer: {key}")
    if sum(m["split"] == "development" for m in metadata.values()) != 15:
        raise ValueError("Expected 15 development and 35 held-out questions")
    for key, question in baseline.items():
        if questions.get(key) != question or metadata[key]["split"] != "development":
            raise ValueError(
                f"Baseline question must be unchanged in development: {key}"
            )


def blind_candidates(
    question_id: str, candidates: list[dict], references: list[dict]
) -> list[dict]:
    """Strip ranking signals and mix independently found references into the pool."""
    fields = ("chunk_id", "source", "page_number", "section", "text")
    merged = {
        candidate["chunk_id"]: {field: candidate[field] for field in fields}
        for candidate in candidates + references
    }
    return sorted(
        merged.values(),
        key=lambda row: hashlib.sha256(
            f"v2-blind-1:{question_id}:{row['chunk_id']}".encode()
        ).digest(),
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def export_batch(
    questions_path: Path, metadata_path: Path, index_path: Path, output: Path
) -> None:
    """Extract candidate pools and blind batches with one shared model pair."""
    start = perf_counter()
    question_bytes = questions_path.read_bytes()
    metadata_bytes = metadata_path.read_bytes()
    questions, metadata = json.loads(question_bytes), json.loads(metadata_bytes)
    baseline_path = Path(__file__).resolve().parents[1] / "eval" / "questions.json"
    validate_batch_dataset(questions, metadata, json.loads(baseline_path.read_bytes()))
    index = VectorIndex.load(index_path)
    chunks = {chunk.chunk_id: chunk for chunk in index.chunks}
    if not chunks:
        raise ValueError("Batch extraction requires a nonempty index")
    references = {}
    for key, entry in metadata.items():
        references[key] = []
        for pointer in entry["references"]:
            chunk = chunks.get(pointer["chunk_id"])
            if (
                chunk is None
                or chunk.source != pointer["source"]
                or chunk.page_number != pointer["page_number"]
            ):
                raise ValueError(f"Reference does not match the persisted index: {key}")
            references[key].append(
                dict(pointer, section=chunk.section, text=chunk.text)
            )
    output.mkdir(parents=True, exist_ok=False)
    blind_dir = output / "blind"
    blind_dir.mkdir()
    manifest_bytes = (index_path / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    summary = {"questions": {}, "failures": {}}
    rankings = {
        "questions_sha256": hashlib.sha256(question_bytes).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "index_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "index_checksums": manifest["sha256"],
        "embedding_model": {
            "name": manifest["model_name"],
            "revision": manifest["model_revision"],
        },
        "reranker": {"name": RERANKER_NAME, "revision": RERANKER_REVISION},
        "candidate_depth": DEPTH,
        "rankings": {},
    }
    try:
        embedder = Embedder()
        bm25 = BM25Index(list(index.chunks))
        hybrid = HybridRetriever(index, bm25, embedder)
        reranker = CrossEncoderReranker()
        summary["setup_seconds"] = perf_counter() - start
        for key, question in sorted(questions.items()):
            question_start = perf_counter()
            try:
                candidates = retrieve_candidates(
                    question, index, bm25, hybrid, embedder, reranker
                )
            except (ValueError, TypeError, RuntimeError, OSError) as exc:
                summary["failures"][key] = {
                    "error": str(exc),
                    "seconds": perf_counter() - question_start,
                }
                continue
            header = {"question_id": key, "question": question}
            write_json(
                output / f"{key}.json",
                dict(header, candidate_depth=DEPTH, candidates=candidates),
            )
            blind = blind_candidates(key, candidates, references[key])
            write_json(blind_dir / f"{key}.json", dict(header, candidates=blind))
            rankings["rankings"][key] = {
                method: [
                    row["chunk_id"]
                    for row in sorted(
                        (row for row in candidates if row["ranks"][method] is not None),
                        key=lambda row: row["ranks"][method],
                    )
                ]
                for method in METHODS
            }
            summary["questions"][key] = {
                "seconds": perf_counter() - question_start,
                "pooled": len(candidates),
                "blind": len(blind),
            }
        if summary["failures"]:
            raise ValueError("Batch has failed queries. See run.json.")
        write_json(output / "rankings.json", rankings)
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        summary["batch_error"] = str(exc)
        raise
    finally:
        summary["total_seconds"] = perf_counter() - start
        write_json(output / "run.json", summary)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--batch", action="store_true", help="Export the frozen v2 set")
    mode.add_argument("--question", help="Literal question text")
    mode.add_argument("--question-id", help="ID from the questions JSON file")
    mode.add_argument(
        "--source", help="Exact persisted source path; exports all its chunks"
    )
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("eval/v2/metadata.json"))
    args = parser.parse_args(argv)
    try:
        if args.batch:
            export_batch(
                args.questions or Path("eval/v2/questions.json"),
                args.metadata,
                args.index or Path("data/index_v2"),
                args.out,
            )
            print(f"Exported batch to {args.out}")
            return
        question = args.question
        if args.question_id is not None:
            questions = json.loads(
                (args.questions or Path("eval/questions.json")).read_bytes()
            )
            if args.question_id not in questions:
                raise ValueError(f"Unknown question ID: {args.question_id}")
            question = questions[args.question_id]
        if args.source is None and (
            not isinstance(question, str) or not question.strip()
        ):
            raise ValueError("Provide a nonempty question")
        index = VectorIndex.load(args.index or Path("data/index"))
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
