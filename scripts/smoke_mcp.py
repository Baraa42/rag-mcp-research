"""Opt-in real-model MCP stdio smoke test; generation requires --with-answer."""

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path
from time import perf_counter

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

QUESTION = "Why do Transformers need positional encoding?"
ROOT = Path(__file__).resolve().parents[1]


class SmokeFailure(ValueError):
    """A safe, locally generated diagnostic without provider error details."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def validate_passage(record: dict, chunks: dict) -> None:
    chunk = chunks.get(record.get("chunk_id"))
    check(chunk is not None, "Returned passage ID is absent from the local index.")
    for field in ("document_id", "source", "text", "page_number", "section"):
        check(
            record.get(field) == chunk[field],
            f"Passage {field} differs from local metadata.",
        )
    check(
        all(
            isinstance(record.get(k), str) and record[k].strip()
            for k in ("chunk_id", "document_id", "source", "text")
        ),
        "Passage has empty required metadata.",
    )
    page = record["page_number"]
    check(
        page is None or (type(page) is int and page > 0),
        "Invalid physical page number.",
    )


def validate_search(data: dict, chunks: dict) -> list[dict]:
    results = data.get("results")
    check(
        isinstance(results, list) and 0 < len(results) <= 5,
        "Expected 1–5 search results.",
    )
    seen = set()
    previous = math.inf
    for rank, result in enumerate(results, 1):
        validate_passage(result, chunks)
        check(
            type(result.get("rank")) is int and result["rank"] == rank,
            "Ranks are not sequential.",
        )
        check(result["chunk_id"] not in seen, "Duplicate search passage ID.")
        seen.add(result["chunk_id"])
        score = result.get("score")
        check(
            type(score) in (int, float) and math.isfinite(score),
            "Non-finite or invalid score.",
        )
        check(score <= previous, "Scores are not in descending order.")
        previous = score
    return results


def validate_answer(data: dict, chunks: dict) -> None:
    answer, sources = data.get("answer"), data.get("sources")
    check(isinstance(answer, str) and bool(answer.strip()), "Empty generated answer.")
    check(isinstance(sources, list), "Missing citation sources.")
    labels = []
    for source in sources:
        validate_passage(source, chunks)
        label = source.get("label")
        check(
            isinstance(label, str) and re.fullmatch(r"S[1-9][0-9]*", label) is not None,
            "Invalid source label.",
        )
        labels.append(label)
    check(len(set(labels)) == len(labels), "Duplicate source labels.")
    references = re.findall(r"\[[Ss][^\[\]]*(?:\]|$)", answer)
    check(
        all(re.fullmatch(r"\[S[1-9][0-9]*\]", ref) for ref in references),
        "Malformed answer citation.",
    )
    check(
        {ref[1:-1] for ref in references} == set(labels),
        "Answer citations and returned source labels do not match.",
    )
    if not labels:
        from research_assistant.generation.answering import INSUFFICIENT_EVIDENCE

        check(
            answer == INSUFFICIENT_EVIDENCE,
            "Uncited answer is not an insufficient-evidence response.",
        )
    # Citation membership does not judge factual support.
    print(answer)
    print(
        f"Validated {len(labels)} citation source(s); factual support is not verified."
    )


async def run(index: Path, with_answer: bool) -> None:
    uv = shutil.which("uv")
    check(uv is not None, "uv must be installed and on PATH.")
    chunks_list = [
        json.loads(line) for line in (index / "chunks.jsonl").read_text().splitlines()
    ]
    chunks = {c["chunk_id"]: c for c in chunks_list}
    check(
        bool(chunks) and len(chunks) == len(chunks_list),
        "Index must have unique, nonempty chunks.",
    )
    env = {
        name: os.environ[name]
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME")
        if name in os.environ
    }
    env.update(UV_NO_ENV_FILE="1", UV_NO_SYNC="1")
    if with_answer:
        for name in ("OPENAI_API_KEY", "OPENAI_MODEL"):
            check(
                bool(os.environ.get(name, "").strip()), f"Set {name} for --with-answer."
            )
            env[name] = os.environ[name]
    params = StdioServerParameters(
        command=uv,
        args=[
            "run",
            "python",
            "-m",
            "research_assistant.interfaces.mcp_server",
            "--index",
            str(index.resolve()),
        ],
        cwd=ROOT,
        env=env,
    )
    start = perf_counter()
    with anyio.fail_after(180):
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                print(f"Startup + MCP initialization: {perf_counter() - start:.3f}s")
                tools = (await session.list_tools()).tools
                check(
                    len(tools) == 2
                    and {t.name for t in tools} == {"search_papers", "answer_question"},
                    "Server must expose exactly search_papers and answer_question.",
                )
                previous = {}
                for rerank, phase in (
                    (False, "cold hybrid: embedder loads"),
                    (True, "first reranked: cross-encoder loads"),
                    (False, "warm hybrid"),
                    (True, "warm reranked"),
                ):
                    start = perf_counter()
                    response = await session.call_tool(
                        "search_papers",
                        {
                            "question": QUESTION,
                            "top_k": 5,
                            "rerank": rerank,
                        },
                    )
                    elapsed = perf_counter() - start
                    check(
                        not response.is_error,
                        "search_papers failed; check server configuration.",
                    )
                    results = validate_search(response.structured_content, chunks)
                    if rerank in previous:
                        check(
                            results == previous[rerank],
                            "Repeated search results changed.",
                        )
                    previous[rerank] = results
                    print(f"\n{phase}: {elapsed:.3f}s")
                    for result in results:
                        excerpt = " ".join(result["text"].split())[:180]
                        location = (
                            f"page {result['page_number']}"
                            if result["page_number"]
                            else f"section {result['section'] or '(untitled)'}"
                        )
                        print(
                            f"{result['rank']}. {result['source']} | {location} | score={result['score']:.6f}\n   {excerpt}"
                        )
                if with_answer:
                    response = await session.call_tool(
                        "answer_question", {"question": QUESTION}
                    )
                    check(
                        not response.is_error,
                        "answer_question failed; check generation configuration.",
                    )
                    validate_answer(response.structured_content, chunks)
    print("Smoke test passed; SDK client/subprocess cleanup completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "data/index")
    parser.add_argument(
        "--with-answer",
        action="store_true",
        help="Enable a potentially paid OpenAI generation call",
    )
    args = parser.parse_args()
    try:
        anyio.run(run, args.index, args.with_answer)
    except SmokeFailure as exc:
        parser.exit(1, f"Smoke test failed: {exc}\n")
    except Exception:
        parser.exit(
            1,
            "Smoke test failed; check index files, cached models and MCP server diagnostics.\n",
        )


if __name__ == "__main__":
    main()
