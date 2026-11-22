"""Synthetic, offline metric and validation checks."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "evaluate_retrieval",
    Path(__file__).resolve().parents[1] / "scripts/evaluate_retrieval.py",
)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


@pytest.fixture
def inputs():
    questions = {f"Q{i:02}": f"Question {i}" for i in range(1, 6)}
    labels, pools = [], {}
    for qid, question in questions.items():
        candidates = []
        for rank in range(1, 7):
            cid = str(rank)
            candidates.append(
                {"chunk_id": cid, "ranks": dict.fromkeys(evaluation.METHODS, rank)}
            )
            labels.append(
                {
                    "question_id": qid,
                    "chunk_id": cid,
                    "grade": 2 if rank in (2, 6) else 1,
                }
            )
        pools[qid] = {
            "question_id": qid,
            "question": question,
            "candidates": candidates,
        }
    return questions, labels, pools


def test_metrics_and_macro(inputs):
    rows = evaluation.evaluate(*inputs)
    assert len(rows) == 20
    for row in rows:
        assert row["recall_at_5"] == 0.5
        assert row["recall_at_20"] == 1
        assert row["rr_at_5"] == 0.5
    assert "| faiss | 0.5000 | 1.0000 | 0.5000 |" in evaluation.render_markdown(rows)


def test_no_top_five_hit_and_external_relevant(inputs):
    questions, labels, pools = inputs
    for label in labels:
        if label["chunk_id"] == "2":
            label["grade"] = 0
    labels.append({"question_id": "Q01", "chunk_id": "outside-pool", "grade": 2})
    row = evaluation.evaluate(questions, labels, pools)[0]
    assert row["recall_at_5"] == 0
    assert row["recall_at_20"] == 0.5
    assert row["rr_at_5"] == 0


def test_no_relevant_passages(inputs):
    for label in inputs[1]:
        label["grade"] = 0
    rows = evaluation.evaluate(*inputs)
    assert rows[0]["recall_at_5"] is None
    assert rows[0]["rr_at_5"] == 0
    assert "| faiss | N/A | N/A | 0.0000 |" in evaluation.render_markdown(rows)


@pytest.mark.parametrize("grade", [-1, 3, True, 2.0, "2", None])
def test_invalid_grade(inputs, grade):
    inputs[1][0]["grade"] = grade
    with pytest.raises(ValueError, match="Invalid grade"):
        evaluation.evaluate(*inputs)


@pytest.mark.parametrize(
    "change, message",
    [
        ("duplicate", "Duplicate label"),
        ("missing", "Incomplete labels"),
        ("unknown", "Unknown question"),
        ("missing_question", "exactly Q01"),
        ("missing_ranks", "Missing or invalid rankings"),
        ("duplicate_rank", "Duplicate rank"),
        ("invalid_rank", "Invalid rank"),
        ("gap", "Noncontiguous"),
        ("metadata", "metadata mismatch"),
    ],
)
def test_validation(inputs, change, message):
    questions, labels, pools = inputs
    candidate = pools["Q01"]["candidates"][0]
    if change == "duplicate":
        labels.append(labels[0].copy())
    elif change == "missing":
        labels.pop()
    elif change == "unknown":
        labels[0]["question_id"] = "Q99"
    elif change == "missing_question":
        questions.pop("Q05")
    elif change == "missing_ranks":
        candidate["ranks"].pop("bm25")
    elif change == "duplicate_rank":
        candidate["ranks"]["faiss"] = 2
    elif change == "invalid_rank":
        candidate["ranks"]["faiss"] = True
    elif change == "gap":
        candidate["ranks"]["faiss"] = None
    elif change == "metadata":
        pools["Q01"]["question_id"] = "Q99"
    with pytest.raises(ValueError, match=message):
        evaluation.evaluate(questions, labels, pools)


def test_absent_method_results(inputs):
    for pool in inputs[2].values():
        for candidate in pool["candidates"]:
            candidate["ranks"]["bm25"] = None
    rows = evaluation.evaluate(*inputs)
    assert all(row["recall_at_20"] == 0 for row in rows if row["method"] == "bm25")
