"""Evaluate saved retrieval rankings against complete manual pool judgments."""

import argparse
import json
from pathlib import Path

METHODS = ("faiss", "bm25", "hybrid", "rerank")
QUESTION_IDS = {f"Q{i:02}" for i in range(1, 6)}


def evaluate(questions: dict, labels: list, pools: dict) -> list[dict]:
    """Validate inputs before computing per-question pooled-relevance metrics."""
    if not isinstance(questions, dict) or set(questions) != QUESTION_IDS:
        raise ValueError("Questions must contain exactly Q01–Q05")
    if set(pools) != set(questions):
        raise ValueError("Candidate pools must cover exactly the known questions")
    if not isinstance(labels, list):
        raise ValueError("Labels must be a list")
    judgments = {qid: {} for qid in questions}
    for label in labels:
        if not isinstance(label, dict):
            raise ValueError("Each label must be an object")
        qid, cid, grade = (label.get(k) for k in ("question_id", "chunk_id", "grade"))
        if not isinstance(qid, str) or qid not in questions:
            raise ValueError(f"Unknown question ID: {qid}")
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError("Label chunk_id must be a nonempty string")
        if cid in judgments[qid]:
            raise ValueError(f"Duplicate label: {qid}/{cid}")
        if type(grade) is not int or grade not in (0, 1, 2):
            raise ValueError(f"Invalid grade for {qid}/{cid}")
        judgments[qid][cid] = grade
    rows = []
    for qid in sorted(questions):
        pool = pools[qid]
        if pool.get("question_id") != qid or pool.get("question") != questions[qid]:
            raise ValueError(f"Question metadata mismatch: {qid}")
        candidates = pool.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"Missing candidates: {qid}")
        ranks = {method: {} for method in METHODS}
        ids = set()
        for candidate in candidates:
            cid = candidate.get("chunk_id")
            if not isinstance(cid, str) or not cid.strip() or cid in ids:
                raise ValueError(f"Invalid or duplicate candidate ID: {qid}/{cid}")
            ids.add(cid)
            ranking = candidate.get("ranks")
            if not isinstance(ranking, dict) or set(ranking) != set(METHODS):
                raise ValueError(f"Missing or invalid rankings: {qid}/{cid}")
            if all(value is None for value in ranking.values()):
                raise ValueError(f"Candidate absent from every ranking: {qid}/{cid}")
            for method, rank in ranking.items():
                if rank is None:
                    continue
                if type(rank) is not int or not 1 <= rank <= 20:
                    raise ValueError(f"Invalid rank: {qid}/{method}/{cid}")
                if rank in ranks[method]:
                    raise ValueError(f"Duplicate rank: {qid}/{method}/{rank}")
                ranks[method][rank] = cid
        missing = ids - judgments[qid].keys()
        if missing or not judgments[qid]:
            raise ValueError(
                f"Incomplete labels for {qid}: {len(missing)} unjudged candidates"
            )
        for method, ranking in ranks.items():
            if set(ranking) != set(range(1, len(ranking) + 1)):
                raise ValueError(f"Noncontiguous ranks: {qid}/{method}")
        relevant = {cid for cid, grade in judgments[qid].items() if grade == 2}
        # Additional manually inspected chunks outside the pool count in recall.
        for method, ranking in ranks.items():
            hits = sorted(rank for rank, cid in ranking.items() if cid in relevant)
            rows.append(
                {
                    "question_id": qid,
                    "method": method,
                    "relevant_count": len(relevant),
                    "recall_at_5": sum(r <= 5 for r in hits) / len(relevant)
                    if relevant
                    else None,
                    "recall_at_20": len(hits) / len(relevant) if relevant else None,
                    "rr_at_5": 1 / hits[0] if hits and hits[0] <= 5 else 0.0,
                }
            )
    return rows


def render_markdown(rows: list[dict]) -> str:
    """Render per-question metrics and equal-weight macro averages."""
    lines = [
        "# Offline retrieval evaluation",
        "",
        "Recall uses pooled manual judgments, not exhaustive corpus relevance.",
        "",
        "| Question | Method | Grade-2 count | Recall@5 | Recall@20 | RR@5 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    def display(value):
        return "N/A" if value is None else f"{value:.4f}"

    for row in rows:
        values = [display(row[k]) for k in ("recall_at_5", "recall_at_20", "rr_at_5")]
        lines.append(
            f"| {row['question_id']} | {row['method']} | {row['relevant_count']} | "
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "| Method | Macro Recall@5 | Macro Recall@20 | MRR@5 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        means = []
        for key in ("recall_at_5", "recall_at_20", "rr_at_5"):
            values = [row[key] for row in group]
            means.append(
                None if any(v is None for v in values) else sum(values) / len(values)
            )
        lines.append(f"| {method} | " + " | ".join(display(v) for v in means) + " |")
    lines += [
        "",
        "Recall is N/A when a question has no grade-2 judgments; its macro recall",
        "is also N/A rather than silently excluding that question. RR@5 remains 0.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.json"))
    parser.add_argument("--labels", type=Path, default=Path("eval/labels.json"))
    parser.add_argument("--exports", type=Path, default=Path("data/eval"))
    args = parser.parse_args(argv)
    try:
        questions = json.loads(args.questions.read_text(encoding="utf-8"))
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
        pools = {
            qid: json.loads((args.exports / f"{qid}.json").read_text(encoding="utf-8"))
            for qid in QUESTION_IDS
        }
        print(render_markdown(evaluate(questions, labels, pools)), end="")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
