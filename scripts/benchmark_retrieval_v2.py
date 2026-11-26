"""Measure warm in-process top-five retrieval on the frozen v2 benchmark."""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import product
from pathlib import Path
from random import Random
from statistics import mean, median, quantiles
from time import perf_counter_ns

TOP_K = 5
METHODS = ("faiss", "bm25", "hybrid", "rerank")


def measure(questions, methods, rankings, *, repetitions=5, seed=42) -> list[dict]:
    """Warm once per method, then shuffle a complete block for each repetition."""
    rng = Random(seed)
    question_ids = sorted(questions)
    for search in methods.values():
        search(questions[question_ids[0]])

    records = []
    for repetition in range(1, repetitions + 1):
        block = list(product(question_ids, METHODS))
        rng.shuffle(block)
        for qid, method in block:
            start = perf_counter_ns()
            results = methods[method](questions[qid])
            elapsed = perf_counter_ns() - start
            ids = [result.chunk_id for result in results]
            records.append(
                {
                    "repetition": repetition,
                    "question_id": qid,
                    "method": method,
                    "elapsed_ns": elapsed,
                    "result_count": len(ids),
                    "matches_frozen_top5": ids == rankings[qid][method][:TOP_K],
                }
            )
    return records


def summarize(records: list[dict]) -> dict:
    """Aggregate elapsed times across all questions and repetitions per method."""
    summary = {}
    for method in METHODS:
        rows = [row for row in records if row["method"] == method]
        values = [row["elapsed_ns"] / 1_000_000 for row in rows]
        summary[method] = {
            "count": len(rows),
            "median_ms": median(values),
            "p95_ms": quantiles(values, n=20, method="inclusive")[-1],
            "mean_ms": mean(values),
            "result_counts": dict(
                sorted(Counter(row["result_count"] for row in rows).items())
            ),
            "frozen_top5_matches": sum(row["matches_frozen_top5"] for row in rows),
        }
    return summary


def hardware() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    if platform.system() == "Darwin":
        output = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True,
            text=True,
            check=True,
        )
        machine, *_ = json.loads(output.stdout)["SPHardwareDataType"]
        for key in ("chip_type", "number_processors", "physical_memory"):
            if key in machine:
                info[key] = machine[key]
    return info


def run(
    index_path: Path, dataset: Path, output: Path, repetitions: int, seed: int
) -> None:
    from evaluate_retrieval_v2 import DEPTH, QUESTION_IDS, validate_fingerprints

    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise ValueError("Set OMP_NUM_THREADS=1 before launching the benchmark")
    os.environ["HF_HUB_OFFLINE"] = "1"
    import faiss
    import torch

    from research_assistant.retrieval.bm25 import BM25Index
    from research_assistant.retrieval.embeddings import Embedder
    from research_assistant.retrieval.hybrid import HybridRetriever
    from research_assistant.retrieval.reranking import (
        RERANKER_NAME,
        RERANKER_REVISION,
        CrossEncoderReranker,
    )
    from research_assistant.retrieval.vector import VectorIndex

    question_bytes = (dataset / "questions.json").read_bytes()
    questions = json.loads(question_bytes)
    snapshot = json.loads((dataset / "rankings.json").read_bytes())
    if set(questions) != QUESTION_IDS or set(snapshot["rankings"]) != QUESTION_IDS:
        raise ValueError("The frozen benchmark must contain all 50 questions")
    hashes = {
        "questions": hashlib.sha256(question_bytes).hexdigest(),
        "metadata": hashlib.sha256(
            (dataset / "metadata.json").read_bytes()
        ).hexdigest(),
    }
    validate_fingerprints(snapshot, hashes, index_path)
    if snapshot["candidate_depth"] != DEPTH or snapshot["reranker"] != {
        "name": RERANKER_NAME,
        "revision": RERANKER_REVISION,
    }:
        raise ValueError("The reranker or candidate depth does not match the snapshot")

    started_at = datetime.now(timezone.utc).isoformat()
    machine = hardware()
    index = VectorIndex.load(index_path)
    embedder = Embedder()
    lexical = BM25Index(list(index.chunks))
    hybrid = HybridRetriever(index, lexical, embedder)
    reranker = CrossEncoderReranker()
    methods = {
        "faiss": lambda question: index.search(question, embedder, TOP_K),
        "bm25": lambda question: lexical.search(question, TOP_K),
        "hybrid": lambda question: hybrid.search(question, TOP_K),
        "rerank": lambda question: reranker.rerank(
            question, hybrid.search(question, DEPTH), TOP_K
        ),
    }
    records = measure(
        questions, methods, snapshot["rankings"], repetitions=repetitions, seed=seed
    )

    report = {
        "started_at_utc": started_at,
        "scope": "warm in-process retrieval, including query inference where applicable",
        "hardware": machine,
        "software": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in (
                    "faiss-cpu",
                    "numpy",
                    "torch",
                    "sentence-transformers",
                    "transformers",
                    "rank-bm25",
                )
            },
        },
        "threads": {
            **{
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "TOKENIZERS_PARALLELISM",
                )
            },
            "faiss_openmp": faiss.omp_get_max_threads(),
            "torch_intraop": torch.get_num_threads(),
            "torch_interop": torch.get_num_interop_threads(),
        },
        "index_manifest_sha256": snapshot["index_manifest_sha256"],
        "questions_sha256": hashes["questions"],
        "embedding_model": dict(
            snapshot["embedding_model"], batch_size=embedder.batch_size
        ),
        "reranker_model": dict(snapshot["reranker"], batch_size=reranker.batch_size),
        "index_chunk_count": len(index.chunks),
        "question_count": len(questions),
        "top_k": TOP_K,
        "candidate_depth": DEPTH,
        "repetitions": repetitions,
        "seed": seed,
        "methods": summarize(records),
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "retrieval-latency.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(records[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    (output / "retrieval-latency.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["methods"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/index_v2"))
    parser.add_argument("--dataset", type=Path, default=Path("eval/v2"))
    parser.add_argument("--out", type=Path, default=Path("eval/v2"))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repetitions < 5:
        parser.error("--repetitions must be at least five")
    run(args.index, args.dataset, args.out, args.repetitions, args.seed)


if __name__ == "__main__":
    main()
