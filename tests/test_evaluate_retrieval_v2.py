"""Validate the frozen-snapshot evaluator with offline synthetic inputs."""

import importlib.util
import json
from hashlib import sha256
from math import log2
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "evaluate_retrieval_v2",
    Path(__file__).resolve().parents[1] / "scripts/evaluate_retrieval_v2.py",
)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_graded_metrics_include_independent_judgments():
    grades = {"partial": 1, "off-topic": 0, "direct": 2, "outside": 2}
    scores = evaluation.score_ranking(["partial", "off-topic", "direct"], grades)
    assert scores["recall_at_5"] == scores["recall_at_20"] == 0.5
    assert scores["rr_at_5"] == pytest.approx(1 / 3)
    expected = (1 + 3 / log2(4)) / (3 + 3 / log2(3) + 1 / log2(4))
    assert scores["ndcg_at_5"] == pytest.approx(expected)
    assert scores["ndcg_at_10"] == pytest.approx(expected)


def test_top_five_ten_and_twenty_cutoffs():
    ranking = [str(i) for i in range(21)]
    grades = dict.fromkeys(ranking, 0)
    grades.update({"5": 2, "9": 2, "20": 2})
    scores = evaluation.score_ranking(ranking, grades)
    assert scores["recall_at_5"] == scores["rr_at_5"] == scores["ndcg_at_5"] == 0
    assert scores["recall_at_20"] == pytest.approx(2 / 3)
    ideal = 3 + 3 / log2(3) + 3 / log2(4)
    assert scores["ndcg_at_10"] == pytest.approx((3 / log2(7) + 3 / log2(11)) / ideal)


@pytest.mark.parametrize("grade, ndcg", [(0, None), (1, 1.0)])
def test_undefined_denominators(grade, ndcg):
    scores = evaluation.score_ranking(["a"], {"a": grade})
    assert scores["recall_at_5"] is scores["recall_at_20"] is None
    assert scores["rr_at_5"] == 0
    assert scores["ndcg_at_5"] == scores["ndcg_at_10"] == ndcg


def test_split_and_category_aggregation():
    def row(split, category, score, *, answerable=True):
        return {
            "split": split,
            "category": category,
            "answerable": answerable,
            "methods": dict.fromkeys(
                evaluation.METHODS, {"ndcg_at_5": score, "rr_at_5": score or 0}
            ),
        }

    summary = evaluation.aggregate(
        [
            row("development", "attention", 0.25),
            row("held_out", "speech", 0.75),
            row("held_out", "attention", 1.0),
            row("held_out", "other", None, answerable=False),
        ]
    )
    assert summary["development"]["question_count"] == 1
    assert summary["held_out_answerable"]["question_count"] == 2
    all_answerable = summary["all_answerable"]
    assert all_answerable["question_count"] == 3
    assert all_answerable["methods"]["faiss"]["ndcg_at_5"] == pytest.approx(2 / 3)
    attention = all_answerable["categories"]["attention"]
    assert attention["question_count"] == 2
    assert attention["methods"]["faiss"]["mrr_at_5"] == 0.625
    unanswerable = summary["expected_unanswerable"]
    assert unanswerable["question_count"] == 1
    assert unanswerable["methods"]["faiss"]["ndcg_at_5"] is None


@pytest.fixture
def snapshot():
    questions = {qid: f"Question {qid}" for qid in sorted(evaluation.QUESTION_IDS)}
    ranked = [f"{i:064x}" for i in range(20)]
    reference = "f" * 64
    metadata = {
        qid: {
            "split": "development" if qid <= "Q15" else "held_out",
            "category": "synthetic",
            "answerable": qid < "Q45",
            "references": [{"chunk_id": reference}] if qid < "Q45" else [],
        }
        for qid in questions
    }
    rankings = {
        "candidate_depth": 20,
        "rankings": {
            qid: {method: ranked.copy() for method in evaluation.METHODS}
            for qid in questions
        },
    }
    labels = [
        {"question_id": qid, "chunk_id": cid, "grade": int(cid == reference) * 2}
        for qid in questions
        for cid in (ranked + [reference] if qid < "Q45" else ranked)
    ]
    return questions, metadata, rankings, labels


def test_require_complete_ranked_and_independent_labels(snapshot):
    questions, metadata, rankings, labels = snapshot
    assert len(evaluation.validate_judgments(*snapshot)["Q01"]) == 21
    for missing in [labels[0], labels[20]]:
        incomplete = [label for label in labels if label != missing]
        with pytest.raises(ValueError, match="1 missing"):
            evaluation.validate_judgments(questions, metadata, rankings, incomplete)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("duplicate_label", "Duplicate label"),
        ("extra_label", "unexpected"),
        ("bad_grade", "Invalid grade"),
        ("duplicate_rank", "Invalid ranking IDs"),
        ("bad_fusion", "Hybrid order"),
        ("bad_rerank", "must permute"),
        ("duplicate_reference", "duplicate reference"),
    ],
)
def test_reject_inconsistent_identifiers(snapshot, mutation, message):
    _, metadata, rankings, labels = snapshot
    ranks = rankings["rankings"]["Q01"]
    if mutation == "duplicate_label":
        labels.append(labels[0].copy())
    elif mutation == "extra_label":
        labels.append(dict(labels[0], chunk_id="e" * 64))
    elif mutation == "bad_grade":
        labels[0]["grade"] = True
    elif mutation == "duplicate_rank":
        ranks["faiss"][0] = ranks["faiss"][1]
    elif mutation == "bad_fusion":
        ranks["hybrid"].reverse()
    elif mutation == "bad_rerank":
        ranks["rerank"][0] = "e" * 64
    else:
        metadata["Q01"]["references"] *= 2
    with pytest.raises(ValueError, match=message):
        evaluation.validate_judgments(*snapshot)


def test_reject_duplicate_json_ids():
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        json.loads('{"Q01": {}, "Q01": {}}', object_pairs_hook=evaluation.unique_object)


def test_validate_snapshot_fingerprints(tmp_path):
    hashes = {"questions": "a" * 64, "metadata": "b" * 64}
    model = {"name": "model", "revision": "1" * 40}
    checksums = {}
    for filename in evaluation.INDEX_FILES:
        content = filename.encode()
        (tmp_path / filename).write_bytes(content)
        checksums[filename] = sha256(content).hexdigest()
    manifest = json.dumps(
        {
            "sha256": checksums,
            "model_name": model["name"],
            "model_revision": model["revision"],
        }
    ).encode()
    (tmp_path / "manifest.json").write_bytes(manifest)
    snapshot = {
        "questions_sha256": hashes["questions"],
        "metadata_sha256": hashes["metadata"],
        "index_manifest_sha256": sha256(manifest).hexdigest(),
        "index_checksums": checksums,
        "embedding_model": model,
        "reranker": model,
    }
    evaluation.validate_fingerprints(snapshot, hashes, tmp_path)
    with pytest.raises(ValueError, match="questions.json"):
        evaluation.validate_fingerprints(
            snapshot, dict(hashes, questions="other"), None
        )
    (tmp_path / "vectors.faiss").write_bytes(b"changed")
    with pytest.raises(ValueError, match="vectors.faiss"):
        evaluation.validate_fingerprints(snapshot, hashes, tmp_path)


def test_paired_bootstrap_preserves_a_constant_difference():
    rows = [
        {
            "methods": {
                "faiss": {"ndcg_at_5": base},
                "bm25": {"ndcg_at_5": base},
                "hybrid": {"ndcg_at_5": base},
                "rerank": {"ndcg_at_5": base + 0.25},
            }
        }
        for base in (0.0, 0.25, 0.75)
    ]
    comparisons = evaluation.paired_bootstrap(rows, seed=42, resamples=100)
    rerank_hybrid = next(
        row
        for row in comparisons
        if (row["method"], row["baseline"]) == ("rerank", "hybrid")
    )
    assert rerank_hybrid["difference"] == 0.25
    assert rerank_hybrid["ci_95"] == [0.25, 0.25]
