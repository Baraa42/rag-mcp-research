"""Collect a fixed 12-question answer run for independent grounding review."""

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import httpx
from openai import APIError, APIStatusError, OpenAI

CALL_LIMIT = 12
PRICING_URL = "https://developers.openai.com/api/docs/models/gpt-5-mini"
MINI_USD_PER_MILLION = {"input": 0.25, "cached_input": 0.025, "output": 2.0}


def selected_questions(questions: dict, metadata: dict, selection: dict) -> list[dict]:
    answerable = selection["answerable"]
    unanswerable = selection["expected_unanswerable"]
    ids = answerable + unanswerable
    if (
        len(answerable) != 6
        or sorted(unanswerable) != [f"Q{i}" for i in range(45, 51)]
        or len(set(ids)) != CALL_LIMIT
        or any(qid not in questions or qid not in metadata for qid in ids)
        or any(
            metadata[qid]["split"] != "held_out"
            or metadata[qid]["answerable"] != (qid in answerable)
            for qid in ids
        )
    ):
        raise ValueError("Expected six answerable held-out questions and Q45–Q50")
    return [
        {
            "question_id": qid,
            "question": questions[qid],
            "reference_answer": metadata[qid]["reference_answer"],
            "expected_answerable": metadata[qid]["answerable"],
        }
        for qid in ids
    ]


def estimate_cost(model: str, tier: str | None, usage: dict | None) -> float | None:
    """Estimate standard GPT-5 mini charges from reported usage, including cached input."""
    if (
        model not in ("gpt-5-mini", "gpt-5-mini-2025-08-07")
        or tier not in (None, "default")
        or not usage
    ):
        return None
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
    rates = MINI_USD_PER_MILLION
    return (
        (usage["input_tokens"] - cached) * rates["input"]
        + cached * rates["cached_input"]
        + usage["output_tokens"] * rates["output"]
    ) / 1_000_000


class RequestBudget:
    """Permit one official Responses POST per case, counted at the HTTP boundary."""

    def __init__(self) -> None:
        self.total = 0
        self.current: dict | None = None

    def on_request(self, request: httpx.Request) -> None:
        record = self.current
        if (
            request.method != "POST"
            or request.url != "https://api.openai.com/v1/responses"
        ):
            raise RuntimeError("Unexpected endpoint for the answer evaluation")
        if record is None or record["api_attempts"] or self.total >= CALL_LIMIT:
            raise RuntimeError("Paid request budget exhausted")
        self.total += 1
        record["api_attempts"] += 1

    def on_response(self, response: httpx.Response) -> None:
        if self.current is not None:
            self.current["http_status"] = response.status_code


class RecordingClient:
    """Observe the existing SDK call and return its response unchanged."""

    def __init__(
        self, budget: RequestBudget, *, timeout: float, max_retries: int
    ) -> None:
        self.budget = budget
        self.responses = self
        # One measured request per question. Prevent both SDK retries and redirects.
        self.client = OpenAI(
            base_url="https://api.openai.com/v1",
            timeout=timeout,
            max_retries=0,
            http_client=httpx.Client(
                follow_redirects=False,
                event_hooks={
                    "request": [budget.on_request],
                    "response": [budget.on_response],
                },
            ),
        )

    def create(self, **kwargs):
        row = self.budget.current
        if row is None:
            raise RuntimeError("No active evaluation question")
        row["retrieved_sources"] = json.loads(kwargs["input"])["evidence"]
        start = perf_counter()
        try:
            response = self.client.responses.create(**kwargs)
        finally:
            row["generation_latency_ms"] = (perf_counter() - start) * 1000
        row["response_status"] = response.status
        row["response_model"] = response.model
        row["service_tier"] = getattr(response, "service_tier", None)
        row["usage"] = response.usage.model_dump() if response.usage else None
        row["estimated_cost_usd"] = estimate_cost(
            response.model, row["service_tier"], row["usage"]
        )
        row["raw_output"] = response.output_text
        return response

    def close(self) -> None:
        self.client.close()


def write_run(output: Path, run: dict, rows: list[dict]) -> None:
    """Checkpoint metrics and an assessment copy containing the exact evidence."""
    assessment = {"run": run, "responses": rows}
    summary_rows = []
    for row in rows:
        compact = {
            key: value
            for key, value in row.items()
            if key
            not in ("raw_output", "reference_answer", "retrieved_sources", "citations")
        }
        for field in ("retrieved_sources", "citations"):
            compact[field] = [
                {key: value for key, value in source.items() if key != "text"}
                for source in row[field]
            ]
        summary_rows.append(compact)
    estimates = [
        row["estimated_cost_usd"]
        for row in rows
        if row["estimated_cost_usd"] is not None
    ]
    summary = {
        "run": run,
        "api_attempts": sum(row["api_attempts"] for row in rows),
        "api_http_responses": sum(row["http_status"] is not None for row in rows),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "estimated_cost_usd": round(sum(estimates), 10) if estimates else None,
        "responses_with_cost_estimate": len(estimates),
        "responses": summary_rows,
    }
    for filename, payload in (
        ("assessment.json", assessment),
        ("summary.json", summary),
    ):
        path = output / filename
        pending = path.with_suffix(".tmp")
        pending.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        pending.replace(path)


def record_answers(
    service, budget: RequestBudget, rows: list[dict], fallback: str, checkpoint
) -> None:
    """Run sequentially through ResearchService and checkpoint each outcome."""
    blocker = None
    for row in rows:
        if blocker:
            row.update(status="blocked", error=blocker)
            checkpoint()
            continue
        budget.current = row
        start = perf_counter()
        try:
            result = service.answer(row["question"])
            row["answer"] = result.answer
            row["citations"] = [asdict(source) for source in result.sources]
            row["status"] = "abstained" if result.answer == fallback else "answered"
        except APIStatusError as exc:
            row.update(
                status="error", error=f"{type(exc).__name__} (HTTP {exc.status_code})"
            )
            if (
                exc.status_code in (401, 403, 404)
                or getattr(exc, "code", None) == "insufficient_quota"
            ):
                blocker = row["error"]
        except (APIError, ValueError, RuntimeError) as exc:
            # Exception messages can contain credentials, request inputs, or model output.
            row.update(status="error", error=type(exc).__name__)
            if row["response_status"] and row["response_status"] != "completed":
                row["error"] = f"Returned generation status {row['response_status']}"
            elif row["response_status"] == "completed":
                row["error"] = "Completed output rejected by answer validation"
        finally:
            row["service_latency_ms"] = (perf_counter() - start) * 1000
            budget.current = None
            checkpoint()


def run(index: Path, dataset: Path, output: Path) -> None:
    from evaluate_retrieval_v2 import validate_fingerprints

    os.environ["HF_HUB_OFFLINE"] = "1"
    names = ("questions", "metadata", "rankings", "answer-questions")
    raw = {name: (dataset / f"{name}.json").read_bytes() for name in names}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()}
    snapshot = json.loads(raw["rankings"])
    validate_fingerprints(snapshot, hashes, index)
    cases = selected_questions(
        *(
            json.loads(raw[name])
            for name in ("questions", "metadata", "answer-questions")
        )
    )
    rows = [
        dict(
            case,
            status="pending",
            answer=None,
            raw_output=None,
            retrieved_sources=[],
            citations=[],
            generation_latency_ms=None,
            service_latency_ms=None,
            usage=None,
            estimated_cost_usd=None,
            error=None,
            api_attempts=0,
            http_status=None,
            response_status=None,
            response_model=None,
            service_tier=None,
        )
        for case in cases
    ]
    model = os.environ.get("OPENAI_MODEL", "").strip()
    config_ok = bool(model and os.environ.get("OPENAI_API_KEY", "").strip())
    run_info = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "configured_model": model if config_ok else None,
        "input_sha256": hashes,
        "index_manifest_sha256": snapshot["index_manifest_sha256"],
        "api_attempt_limit": CALL_LIMIT,
        "max_attempts_per_question": 1,
        "pricing_source": PRICING_URL,
        "gpt_5_mini_standard_usd_per_million": MINI_USD_PER_MILLION,
        "review_status": "pending independent assessment",
    }
    output.mkdir(parents=True, exist_ok=False)

    def checkpoint() -> None:
        write_run(output, run_info, rows)

    if not config_ok:
        for row in rows:
            row.update(status="blocked", error="Set OPENAI_API_KEY and OPENAI_MODEL")
        checkpoint()
        return

    from research_assistant import service as service_module
    from research_assistant.generation.answering import INSUFFICIENT_EVIDENCE

    budget = RequestBudget()
    service = service_module.ResearchService(index)
    checkpoint()
    try:
        with patch.object(
            service_module, "OpenAI", lambda **kwargs: RecordingClient(budget, **kwargs)
        ):
            record_answers(service, budget, rows, INSUFFICIENT_EVIDENCE, checkpoint)
    finally:
        service.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true", help="Explicitly allow up to 12 paid calls"
    )
    parser.add_argument("--index", type=Path, default=Path("data/index_v2"))
    parser.add_argument("--dataset", type=Path, default=Path("eval/v2"))
    parser.add_argument("--out", type=Path, default=Path("data/eval_v2/s5f-answer-run"))
    args = parser.parse_args()
    if not args.live:
        parser.error("Use --live to authorize the bounded answer run")
    if os.environ.get("OMP_NUM_THREADS") != "1":
        parser.error("Set OMP_NUM_THREADS=1 before launching the answer run")
    run(args.index, args.dataset, args.out)
    # The summary contains no credentials or raw passages.
    summary = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "api_attempts",
                    "api_http_responses",
                    "statuses",
                    "estimated_cost_usd",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
