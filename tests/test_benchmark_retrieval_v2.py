"""Exercise benchmark scheduling and aggregation without loading models."""

import importlib.util
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "benchmark_retrieval_v2",
    Path(__file__).resolve().parents[1] / "scripts/benchmark_retrieval_v2.py",
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_warmup_and_reproducible_complete_schedule(monkeypatch):
    ticks = iter(range(0, 1000, 10))
    monkeypatch.setattr(benchmark, "perf_counter_ns", lambda: next(ticks))
    calls = []

    def search(question, method):
        calls.append((question, method))
        return [SimpleNamespace(chunk_id=method)]

    questions = {"Q02": "second", "Q01": "first"}
    methods = {
        method: lambda question, method=method: search(question, method)
        for method in benchmark.METHODS
    }
    ranks = {qid: {method: [method] for method in methods} for qid in questions}
    ranks["Q02"]["bm25"] = ["different"]
    first = benchmark.measure(questions, methods, ranks, repetitions=2, seed=11)
    second = benchmark.measure(questions, methods, ranks, repetitions=2, seed=11)
    assert first == second
    assert calls[:4] == [("first", method) for method in methods]
    expected_pairs = {(qid, method) for qid in questions for method in methods}
    for repetition in (1, 2):
        block = [
            (row["question_id"], row["method"])
            for row in first
            if row["repetition"] == repetition
        ]
        assert Counter(block) == dict.fromkeys(expected_pairs, 1)
    assert all(row["elapsed_ns"] == 10 for row in first)
    assert sum(row["matches_frozen_top5"] for row in first) == 14


def test_summary_uses_warm_samples_and_interpolated_p95():
    records = [
        {
            "method": method,
            "elapsed_ns": 1_000_000 * value,
            "result_count": 5,
            "matches_frozen_top5": value > 1,
        }
        for method in benchmark.METHODS
        for value in range(1, 6)
    ]
    for values in benchmark.summarize(records).values():
        assert values["count"] == 5
        assert values["median_ms"] == values["mean_ms"] == 3.0
        assert values["p95_ms"] == pytest.approx(4.8)
        assert values["result_counts"] == {5: 5}
        assert values["frozen_top5_matches"] == 4
