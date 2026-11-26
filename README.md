# rag-mcp-research

A Python research assistant for ML papers. It builds a local index, compares
four retrieval methods, and generates answers with citations back to the
retrieved passages. The pipeline uses FAISS, BM25, and Sentence Transformers
without LangChain or LlamaIndex.

**S1–S5 complete within the project scope.** The expanded corpus contains
**10 papers and 1,238 chunks**. The frozen benchmark has **50 questions and
1,665 AI-generated relevance judgments**.

## Architecture

```text
PDF / Markdown → source-aware chunks
Shared chunks → MiniLM + FAISS and BM25 → rank fusion
Fused candidates → cross-encoder → five passages → cited answer
```

Chunks retain physical PDF page numbers or Markdown headings. The tokenizer
uses a 180-token budget and 30-token overlap. The versioned FAISS
index stores 384-dimensional vectors and chunk provenance and verifies file
hashes on load. Both model revisions are pinned. The default answer path
reranks 20 hybrid candidates and sends the top five passages to the OpenAI
Responses API. [Pipeline and index details](docs/pipeline.md).

The CLI uses `ResearchService` for hybrid retrieval, reranking, and answering.
The same service backs two stdio MCP tools and a local FastAPI interface.
The servers serialize access with AnyIO locks, run blocking work in worker
threads, and wait for active operations before closing the service.
The HTTP interface adds request IDs, structured JSON logs, strict
validation, and queue-wait limits. The CPU Docker image runs as a non-root
user with a read-only index mount. Build, retrieval, logging, and shutdown
were verified on Docker Desktop. [Interface runbook and verification](docs/integration.md).

## Retrieval results

The frozen protocol uses 15 development questions, 29 answerable held-out
questions, and six held-out expected-unanswerable controls. The controls are
analyzed separately. The quality columns use the 29 answerable held-out
questions. Warm latency covers all 50 questions with five repetitions per
method on one Apple M5 Pro CPU, measured in-process with one OpenMP thread.

| Method | Held-out nDCG@5 | Held-out pooled Recall@20 | Warm median |
| --- | ---: | ---: | ---: |
| Exact FAISS cosine | 0.5525 | 0.6560 | 2.57 ms |
| BM25 | 0.6664 | **0.8049** | **1.54 ms** |
| Hybrid RRF | 0.6719 | 0.7706 | 4.28 ms |
| Hybrid + cross-encoder | **0.7404** | 0.7706 | 164.49 ms |

Reranking improved held-out nDCG@5 by **0.0685** over hybrid, with a paired
95% bootstrap interval of **[0.0140, 0.1246]** from 10,000 resamples. It added
about 160 ms at the median. BM25 had the highest pooled Recall@20 and the
lowest warm latency. All four methods missed **18 directly relevant
question–passage pairs**. The interval describes this small benchmark,
where questions share papers and overlapping evidence. [Evaluation protocol,
metrics and failure analysis](docs/evaluation-v2.md) · [Latency methodology and raw data](docs/performance-v2.md).

## Live answers

The fixed 12-question study made **12 OpenAI requests** at an estimated
**$0.01585** total. Six responses passed citation validation, five exact
abstentions were appropriate, and one response was incomplete. In the
AI-assisted review, Q35 cited true facts about buffering and timestamps
and omitted Whisper's acoustic features. Q38 gave a supported but
incomplete description of the wav2vec 2.0 objective. Q50 reached the
1,024-output-token cap and was rejected as an incomplete-response error.
The live run called `ResearchService` directly. The assessment awaits independent
human verification.
[Question-by-question assessment](docs/answer-assessment-v2.md) · [Latency, usage and cost](docs/performance-v2.md#live-results).

## Quickstart

Use Python 3.11 and `uv`. Place the ten PDFs in `data/papers/` using the
filenames and source links in the [corpus inventory](docs/corpus-v2.md#source-inventory).
The first index build or reranked search may download pinned models.

```sh
uv python install 3.11
uv sync --locked
uv run python -m research_assistant index data/papers --out data/index_v2
uv run python -m research_assistant search \
  "How does PagedAttention manage KV-cache memory?" \
  --index data/index_v2 --method rerank --top-k 5
```

Search supports `vector`, `bm25`, `hybrid`, and `rerank`. For `ask`, set
`OPENAI_API_KEY` and `OPENAI_MODEL` in the environment. An ignored local
`.env` can be loaded explicitly with `uv run --env-file .env`. Answering
with evidence sends the question and selected passages to OpenAI and can
incur charges.

```sh
uv run python -m research_assistant ask \
  "How does PagedAttention manage KV-cache memory?" --index data/index_v2
```

| Interface | Start here |
| --- | --- |
| MCP | [Configure a local stdio host](docs/integration.md#local-mcp-server) for `search_papers` and `answer_question` |
| FastAPI | `uv run python -m research_assistant.interfaces.api --index data/index_v2` serves `/docs`, `/health`, `/ready`, `/search`, and `/answer` on loopback |
| Docker | [Build and run the CPU image locally](docs/integration.md#cpu-docker-deployment) |

## Reproduce and verify

```sh
uv run pytest
uv run ruff check .
uv run python scripts/evaluate_retrieval_v2.py --out /tmp/rag-evaluation-v2.json
cmp eval/v2/results.json /tmp/rag-evaluation-v2.json
```

The evaluator uses the tracked snapshots and runs without the index, models,
or API access. Pass `--index data/index_v2` to verify the saved index bytes
against the recorded fingerprints. Ordinary tests run offline, including MCP
stdio subprocess and FastAPI tests. GitHub Actions installs locked dependencies
and runs pytest and Ruff on pushes and pull requests.
[CI workflow](.github/workflows/ci.yml) · [Frozen dataset and benchmarks](eval/v2/README.md) · [Original five-question baseline](docs/evaluation.md).

## Limitations and future work

Pooled judgments are incomplete and require human auditing. Candidate depth
limits coverage, and the final five passages can omit necessary evidence.
Citation validation only confirms that labels resolve to supplied chunks.
PDF extraction and fixed token windows can fragment tables, formulas, and
arguments. The ten-paper corpus and 12-question answer study support local
comparisons. The deployment is local and unauthenticated.

Future research could vary candidate depth and evidence count, compare
section-aware or layout-aware chunking, and evaluate abstentions and incomplete
responses separately on a larger, human-audited set.
