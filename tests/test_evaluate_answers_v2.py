"""Exercise the answer study locally with fake clients and synthetic evidence."""

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import AuthenticationError

SPEC = importlib.util.spec_from_file_location(
    "evaluate_answers_v2",
    Path(__file__).resolve().parents[1] / "scripts/evaluate_answers_v2.py",
)
answers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(answers)
ENDPOINT = "https://api.openai.com/v1/responses"


def new_row():
    return {
        "question": "Question",
        "api_attempts": 0,
        "http_status": None,
        "response_status": None,
        "retrieved_sources": [],
        "citations": [],
        "estimated_cost_usd": None,
    }


def test_frozen_selection_has_twelve_held_out_questions():
    dataset = Path(__file__).resolve().parents[1] / "eval/v2"
    questions, meta, selection = [
        json.loads((dataset / f"{name}.json").read_text())
        for name in ("questions", "metadata", "answer-questions")
    ]
    cases = answers.selected_questions(questions, meta, selection)
    assert len(cases) == 12
    assert sum(case["expected_answerable"] for case in cases) == 6
    assert len({meta[case["question_id"]]["category"] for case in cases[:6]}) == 4
    invalid = dict(selection, answerable=["Q01"] + selection["answerable"][1:])
    with pytest.raises(ValueError, match="held-out"):
        answers.selected_questions(questions, meta, invalid)


def test_paid_call_budget_caps_one_per_question_and_twelve_total():
    budget = answers.RequestBudget()
    request = httpx.Request("POST", ENDPOINT)
    for _ in range(12):
        budget.current = new_row()
        budget.on_request(request)
        assert budget.current["api_attempts"] == 1
        with pytest.raises(RuntimeError, match="budget"):
            budget.on_request(request)
    budget.current = new_row()
    with pytest.raises(RuntimeError, match="budget"):
        budget.on_request(request)
    assert budget.total == 12
    with pytest.raises(RuntimeError, match="endpoint"):
        budget.on_request(httpx.Request("POST", "https://example.test/v1/responses"))


def test_record_model_response_and_usage_without_changing_request(monkeypatch):
    budget = answers.RequestBudget()
    budget.current = new_row()
    usage = {
        "input_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 100},
        "output_tokens": 500,
    }
    fake_response = SimpleNamespace(
        status="completed",
        model="gpt-5-mini",
        service_tier="default",
        output_text="Result [S1]",
        usage=SimpleNamespace(model_dump=lambda: usage),
    )
    payload = {
        "model": "gpt-5-mini",
        "input": '{"evidence":[{"label":"S1","text":"Evidence"}]}',
        "store": False,
    }
    sent = []
    closed = []

    def fake_openai(**options):
        assert options["max_retries"] == 0
        assert options["base_url"] == "https://api.openai.com/v1"
        transport = options["http_client"]

        def fake_create(**kwargs):
            sent.append(kwargs)
            request = httpx.Request("POST", ENDPOINT)
            events = transport.event_hooks
            events["request"][0](request)
            events["response"][0](httpx.Response(200))
            return fake_response

        def close():
            transport.close()
            closed.append(True)

        return SimpleNamespace(
            responses=SimpleNamespace(create=fake_create), close=close
        )

    monkeypatch.setattr(answers, "OpenAI", fake_openai)
    client = answers.RecordingClient(budget, timeout=60, max_retries=2)
    try:
        assert client.create(**payload) is fake_response
    finally:
        client.close()
    assert closed and sent == [payload]
    assert budget.current["http_status"] == 200
    assert budget.current["generation_latency_ms"] >= 0
    assert budget.current["retrieved_sources"][0]["text"] == "Evidence"
    assert budget.current["usage"] == usage
    assert budget.current["estimated_cost_usd"] == pytest.approx(0.0012275)
    assert answers.estimate_cost("other-model", "default", usage) is None
    assert answers.estimate_cost("gpt-5-mini", "priority", usage) is None


def test_auth_failure_blocks_remaining_questions_and_redacts_error():
    budget = answers.RequestBudget()
    rows = [new_row(), new_row()]
    request = httpx.Request("POST", ENDPOINT)

    def reject(question):
        budget.on_request(request)
        response = httpx.Response(401, request=request)
        budget.on_response(response)
        raise AuthenticationError("SECRET EXCEPTION TEXT", response=response, body=None)

    run = SimpleNamespace(answer=reject)
    answers.record_answers(run, budget, rows, "fallback", lambda: None)
    assert [row["status"] for row in rows] == ["error", "blocked"]
    assert [row["api_attempts"] for row in rows] == [1, 0]
    assert "SECRET EXCEPTION TEXT" not in repr(rows)


@dataclass
class Source:
    label: str
    text: str


def test_assessment_includes_passages_while_publishable_summary_excludes_them(tmp_path):
    rows = [dict(new_row(), reference_answer="Reference"), new_row()]
    rows[0]["retrieved_sources"] = [{"label": "S1", "text": "Full passage"}]
    outputs = iter(
        [
            SimpleNamespace(
                answer="Claim [S1]", sources=(Source("S1", "Full passage"),)
            ),
            SimpleNamespace(answer="Insufficient", sources=()),
        ]
    )
    service = SimpleNamespace(answer=lambda question: next(outputs))
    answers.record_answers(
        service, answers.RequestBudget(), rows, "Insufficient", lambda: None
    )
    answers.write_run(tmp_path, {}, rows)
    assessment = json.loads((tmp_path / "assessment.json").read_text())
    assert assessment["responses"][0]["citations"][0]["text"] == "Full passage"
    assert assessment["responses"][0]["reference_answer"] == "Reference"
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["statuses"] == {"answered": 1, "abstained": 1}
    assert summary["responses"][0]["citations"] == [{"label": "S1"}]
    assert "Full passage" not in repr(summary)
