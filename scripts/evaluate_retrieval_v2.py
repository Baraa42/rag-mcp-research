"""Score the frozen ten-paper retrieval benchmark without running retrieval."""

import argparse
import hashlib
import json
import re
from collections import Counter
from itertools import combinations
from math import log2
from pathlib import Path
from random import Random
from statistics import mean, quantiles

METHODS = ("faiss", "bm25", "hybrid", "rerank")
DEPTH = 20
RRF_K = 60
QUESTION_IDS = {f"Q{i:02}" for i in range(1, 51)}
INDEX_FILES = ("vectors.faiss", "chunks.jsonl")
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10_000


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def is_hex(value: object, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None
    )


def validate_fingerprints(snapshot: dict, hashes: dict, index: Path | None) -> None:
    """Bind inputs to the snapshot and verify local index bytes if supplied."""
    for name in ("questions", "metadata"):
        if snapshot.get(f"{name}_sha256") != hashes[name]:
            raise ValueError(f"Fingerprint mismatch: {name}.json")
    checksums = snapshot.get("index_checksums")
    if (
        not isinstance(checksums, dict)
        or set(checksums) != set(INDEX_FILES)
        or not all(is_hex(value) for value in checksums.values())
        or not is_hex(snapshot.get("index_manifest_sha256"))
    ):
        raise ValueError("Invalid index fingerprints")
    for name in ("embedding_model", "reranker"):
        model = snapshot.get(name)
        if (
            not isinstance(model, dict)
            or not isinstance(model.get("name"), str)
            or not model["name"].strip()
            or not is_hex(model.get("revision"), 40)
        ):
            raise ValueError(f"Invalid model fingerprint: {name}")

    if index is None:
        return
    raw = (index / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != snapshot["index_manifest_sha256"]:
        raise ValueError("Fingerprint mismatch: index manifest")
    manifest = json.loads(raw, object_pairs_hook=unique_object)
    embedding = snapshot["embedding_model"]
    if (
        not isinstance(manifest, dict)
        or manifest.get("sha256") != checksums
        or manifest.get("model_name") != embedding["name"]
        or manifest.get("model_revision") != embedding["revision"]
    ):
        raise ValueError("Index manifest disagrees with the rankings snapshot")
    for filename, expected in checksums.items():
        if hashlib.sha256((index / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Fingerprint mismatch: {filename}")


def validate_rankings(rankings: dict, qid: str) -> set[str]:
    """Validate the four top-20 lists and their candidate-set relationships."""
    if not isinstance(rankings, dict) or set(rankings) != set(METHODS):
        raise ValueError(f"Missing or unknown retrieval method: {qid}")
    for method, ids in rankings.items():
        if (
            not isinstance(ids, list)
            or len(ids) > DEPTH
            or (method != "bm25" and len(ids) != DEPTH)
            or not all(is_hex(cid) for cid in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError(f"Invalid ranking IDs or depth: {qid}/{method}")
    fusion_scores: dict[str, float] = {}
    for method in ("faiss", "bm25"):
        for rank, cid in enumerate(rankings[method], 1):
            fusion_scores[cid] = fusion_scores.get(cid, 0.0) + 1 / (RRF_K + rank)
    expected_hybrid = sorted(fusion_scores, key=lambda cid: (-fusion_scores[cid], cid))
    if rankings["hybrid"] != expected_hybrid[:DEPTH]:
        raise ValueError(f"Hybrid order disagrees with frozen RRF settings: {qid}")
    if set(rankings["rerank"]) != set(rankings["hybrid"]):
        raise ValueError(f"Reranking must permute the hybrid candidates: {qid}")
    return set(fusion_scores)


def validate_judgments(
    questions: dict, metadata: dict, snapshot: dict, labels: list
) -> dict[str, dict[str, int]]:
    """Require one judgment for each ranked or independently identified candidate."""
    rankings = snapshot.get("rankings")
    if (
        type(snapshot.get("candidate_depth")) is not int
        or snapshot["candidate_depth"] != DEPTH
        or not all(isinstance(data, dict) for data in (questions, metadata, rankings))
        or set(questions) != QUESTION_IDS
        or set(questions) != set(metadata)
        or set(questions) != set(rankings)
    ):
        raise ValueError(
            "Expected matching question IDs Q01–Q50 and candidate depth 20"
        )
    if not isinstance(labels, list):
        raise ValueError("Labels must be a list")
    judgments: dict[str, dict[str, int]] = {qid: {} for qid in questions}
    for label in labels:
        if not isinstance(label, dict):
            raise ValueError("Each label must be an object")
        qid, cid, grade = (
            label.get(key) for key in ("question_id", "chunk_id", "grade")
        )
        if not isinstance(qid, str) or qid not in judgments or not is_hex(cid):
            raise ValueError(f"Unknown question or invalid chunk ID: {qid}/{cid}")
        if cid in judgments[qid]:
            raise ValueError(f"Duplicate label: {qid}/{cid}")
        if type(grade) is not int or grade not in (0, 1, 2):
            raise ValueError(f"Invalid grade: {qid}/{cid}")
        judgments[qid][cid] = grade

    splits = Counter()
    for qid, question in questions.items():
        item = metadata[qid]
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(item, dict)
            or item.get("split") not in ("development", "held_out")
            or type(item.get("answerable")) is not bool
            or not isinstance(item.get("category"), str)
            or not item["category"].strip()
            or not isinstance(item.get("references"), list)
            or bool(item["references"]) != item["answerable"]
        ):
            raise ValueError(f"Invalid question or metadata: {qid}")
        splits[item["split"], item["answerable"]] += 1
        pool = validate_rankings(rankings[qid], qid)
        references = set()
        for reference in item["references"]:
            cid = reference.get("chunk_id") if isinstance(reference, dict) else None
            if not is_hex(cid) or cid in references:
                raise ValueError(f"Invalid or duplicate reference: {qid}/{cid}")
            references.add(cid)
        expected = pool | references
        missing = expected - judgments[qid].keys()
        extra = judgments[qid].keys() - expected
        if missing or extra:
            raise ValueError(
                f"Label coverage mismatch for {qid}: {len(missing)} missing, {len(extra)} unexpected"
            )
    if splits != {
        ("development", True): 15,
        ("held_out", True): 29,
        ("held_out", False): 6,
    }:
        raise ValueError("Unexpected split or answerability counts")
    return judgments


def score_ranking(ranking: list[str], judgments: dict[str, int]) -> dict:
    """Score one ordered list using every judged passage to form the ideal list."""
    grades = [judgments[cid] for cid in ranking]
    ideal = sorted(judgments.values(), reverse=True)
    relevant_count = ideal.count(2)
    hits = [i for i, grade in enumerate(grades, 1) if grade == 2]

    def dcg(values: list[int], k: int) -> float:
        return sum((2**grade - 1) / log2(i + 2) for i, grade in enumerate(values[:k]))

    scores = {
        "recall_at_5": sum(rank <= 5 for rank in hits) / relevant_count
        if relevant_count
        else None,
        "recall_at_20": sum(rank <= 20 for rank in hits) / relevant_count
        if relevant_count
        else None,
        "rr_at_5": 1 / hits[0] if hits and hits[0] <= 5 else 0.0,
    }
    for k in (5, 10):
        denominator = dcg(ideal, k)
        scores[f"ndcg_at_{k}"] = dcg(grades, k) / denominator if denominator else None
    return scores


def macro_average(rows: list[dict]) -> dict:
    """Give each question equal weight and keep undefined means explicit."""
    methods = {}
    for method in METHODS:
        scores = [row["methods"][method] for row in rows]
        means = {}
        for metric in scores[0]:
            values = [row[metric] for row in scores]
            means["mrr_at_5" if metric == "rr_at_5" else metric] = (
                None if None in values else mean(values)
            )
        methods[method] = means
    return {"question_count": len(rows), "methods": methods}


def aggregate(rows: list[dict]) -> dict:
    answerable = [row for row in rows if row["answerable"]]
    groups = {
        "development": [row for row in answerable if row["split"] == "development"],
        "held_out_answerable": [
            row for row in answerable if row["split"] == "held_out"
        ],
        "all_answerable": answerable,
        "expected_unanswerable": [row for row in rows if not row["answerable"]],
    }
    aggregates = {}
    for name, group in groups.items():
        if not group:
            continue
        categories = sorted({row["category"] for row in group})
        aggregates[name] = macro_average(group)
        aggregates[name]["categories"] = {
            category: macro_average(
                [row for row in group if row["category"] == category]
            )
            for category in categories
        }
    return aggregates


def paired_bootstrap(rows: list[dict], *, seed: int, resamples: int) -> list[dict]:
    """Use a common resample of paired held-out questions for every comparison."""
    rng = Random(seed)
    samples = [rng.choices(range(len(rows)), k=len(rows)) for _ in range(resamples)]
    results = []
    for baseline, method in combinations(METHODS, 2):
        scores = [
            (row["methods"][method]["ndcg_at_5"], row["methods"][baseline]["ndcg_at_5"])
            for row in rows
        ]
        interval = delta = None
        if scores and all(a is not None and b is not None for a, b in scores):
            differences = [a - b for a, b in scores]
            delta = mean(differences)
            draws = [
                sum(differences[i] for i in sample) / len(rows) for sample in samples
            ]
            cuts = quantiles(draws, n=40, method="inclusive")
            interval = [cuts[0], cuts[-1]]
        results.append(
            {
                "method": method,
                "baseline": baseline,
                "difference": delta,
                "ci_95": interval,
            }
        )
    return results


def evaluate(dataset: Path, index: Path | None = None) -> dict:
    """Read and validate the frozen inputs, then compute the offline evaluation."""
    inputs, hashes = {}, {}
    for name in ("questions", "metadata", "rankings", "labels"):
        raw = (dataset / f"{name}.json").read_bytes()
        inputs[name] = json.loads(raw, object_pairs_hook=unique_object)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    snapshot = inputs["rankings"]
    if not isinstance(snapshot, dict):
        raise ValueError("Rankings snapshot must be an object")
    validate_fingerprints(snapshot, hashes, index)
    judgments = validate_judgments(
        inputs["questions"], inputs["metadata"], snapshot, inputs["labels"]
    )

    rows = []
    for qid, question in sorted(inputs["questions"].items()):
        ranks, grades = snapshot["rankings"][qid], judgments[qid]
        meta = inputs["metadata"][qid]
        pool = set().union(*ranks.values())
        counts = Counter(grades.values())
        rows.append(
            {
                "question_id": qid,
                "question": question,
                "split": meta["split"],
                "category": meta["category"],
                "answerable": meta["answerable"],
                "grade_counts": {str(grade): counts[grade] for grade in range(3)},
                "methods": {
                    method: score_ranking(ranks[method], grades) for method in METHODS
                },
                "grade_2_missed_by_all": sorted(
                    cid
                    for cid, grade in grades.items()
                    if grade == 2 and cid not in pool
                ),
            }
        )
    held_out = [row for row in rows if row["answerable"] and row["split"] == "held_out"]
    return {
        "input_sha256": hashes,
        "per_question": rows,
        "aggregates": aggregate(rows),
        "held_out_ndcg_at_5_bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "question_count": len(held_out),
            "comparisons": paired_bootstrap(
                held_out, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("eval/v2"))
    parser.add_argument(
        "--index", type=Path, help="Verify the source index files against the snapshot"
    )
    parser.add_argument(
        "--out", type=Path, help="Write JSON to this path instead of stdout"
    )
    args = parser.parse_args()
    try:
        report = (
            json.dumps(
                evaluate(args.dataset, args.index),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        if args.out is None:
            print(report, end="")
        else:
            args.out.write_text(report, encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
