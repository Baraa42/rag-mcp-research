# Local interfaces and CPU Docker

Use the [quickstart](../README.md#quickstart) to install locked dependencies and build the expanded index before starting an interface. The S4 record below uses the original three-paper index.

## S4 HTTP and Docker integration evidence

S4 is complete. These manual observations were supplied by the project author
from Docker Desktop testing on **November 26, 2026**, against Docker implementation
commit `9cd411e4b5fe3caddfd838fd27c680afe57f3d89`. They were not rerun during the final
documentation audit. They demonstrate local integration, not production readiness
or statistically robust performance. Raw logs and full hardware/resource settings
were not supplied with this record.

### Verified integration

| Check | Observed result |
| --- | --- |
| Image build | Successful; 52.2 seconds |
| Container startup | Successful on Docker Desktop |
| `/health`, `/ready` | HTTP 200 for each |
| Invalid `/search`, `top_k=0` | HTTP 422 |
| Request IDs | Unique `X-Request-ID` response headers |
| JSON request logging | Status, route, duration and error category present |
| CPU model execution | Embedding and cross-encoder inference succeeded |
| Graceful shutdown | Exit code 0 |

`/ready` confirms successful index initialization, not warmed ML models or OpenAI
availability. No live paid HTTP or MCP answer-generation verification is claimed.

### Individual latency observations

| Operation | First request | Warm request |
| --- | ---: | ---: |
| Hybrid retrieval (`rerank=false`) | 14.444 s | 0.0283 s |
| Hybrid + cross-encoder (`rerank=true`) | 12.601 s | 0.1426 s |
| Reranking after container restart | 2.118 s | Not measured |

These are single request observations, not inference-only benchmarks, averages,
percentiles or service-level guarantees. First calls include lazy initialization
and may include cache/network work. The first reranked request occurred **after
the embedding model had already initialized**, so it is not a fully cold pipeline.
Warm requests reuse in-memory models; they are not directly comparable to startup.
The build duration is separate from all request timings.

After restart, logs showed weights loading, consistent with a persisted disk
cache and fresh in-memory initialization. This does **not** establish that zero
network requests occurred. The restart test initially produced HTTP `000` / an
empty reply, then successful HTTP 200 retrieval. `000` is a client-side indication
that no HTTP response status was obtained, not a server status code. Startup timing
is a plausible explanation, but the root cause was not conclusively established.

### Native versus Docker ranking consistency

Question: **How does FlashAttention reduce GPU memory access?**

Both macOS and Docker returned these same three chunk IDs in the same order:

| Rank | Chunk ID | Source | Physical PDF page |
| --- | --- | --- | ---: |
| 1 | `12e5aad683abf32851dfe8e26310de99bdc33cdf9cf067b4f31bd31ea4066db0` | flashattention.pdf | 1 |
| 2 | `0dc4806386f2c342c16b7f198f9e8c12a59579149c0ca83dbdafd186a8f90f3d` | flashattention.pdf | 22 |
| 3 | `3691d02db494d3957fc357a6579dd20841985014d0fef86464235fcb5ed54755` | flashattention.pdf | 22 |

Score differences were approximately `1.5e-6` or less. This supports ranking
consistency for this question only, not general cross-platform numerical
equivalence or improved retrieval quality.

### Reproduction

Use the [CPU Docker instructions](#cpu-docker-deployment) below. Mount the
original three-paper index from `data/index` to reproduce these S4 observations. Search can download
weights but requires no OpenAI credentials or paid API call. Repeated requests
reuse models; a restart retains the disk cache but discards in-memory state.

### Final engineering audit

- The multi-stage Dockerfile copies only the installed runtime environment into
  the final stage. Python and uv images are digest-pinned; runtime dependencies
  use `uv sync --locked --no-dev --no-editable`. CPU-only PyTorch is checked during
  the build. OS packages and the build backend are not snapshot-pinned, so builds
  are not claimed to be bit-for-bit reproducible.
- The installed package is under `/app/.venv`; execution uses `/app` as its working
  directory. The validated index is mounted read-only at `/data/index`.
  `HF_HOME=/home/app/.cache/huggingface` matches the writable volume and UID 10001.
  Bind-mount permissions remain the operator's responsibility.
- `.dockerignore` excludes local secrets, documents, indexes, caches and outputs;
  explicit Docker COPY inputs further restrict image contents. Tracked-file and
  common credential-pattern checks found no credentials, model weights, raw papers
  or generated indexes. This is a scoped audit, not a comprehensive secret scan
  of all history or a forensic inspection of a built image. Curated evaluation
  questions and manual labels are intentionally versioned.
- Native startup defaults to `127.0.0.1`; Docker binds internally to `0.0.0.0`
  while documented host publication is loopback-only. One worker serializes service
  calls, preserves lazy model reuse and waits for active operations during shutdown.
- Offline tests cover HTTP contracts, strict validation, metadata, service reuse,
  request IDs, safe logs/provider errors, lock timeouts, cancellation and shutdown.
  Existing MCP subprocess and retrieval tests remain in the suite. This audit
  changes documentation only; HTTP, MCP, retrieval and generation behavior is unchanged.

### Scope of the S4 observations

The service is local and unauthenticated, with no rate limiting or bounded queue
capacity. Queue timeout does not bound inference or total latency. Long operations
can delay shutdown or exceed Docker's stop grace period. Citation-ID validation
is not proof of factual support. The later S5 retrieval benchmark measures warmed in-process calls separately from
these HTTP observations. [S5F performance and answer evaluation](performance-v2.md).

## Local MCP server

Uses the official MCP Python SDK v2 `MCPServer` API (`mcp>=2,<3`, locked to
2.3.0), with no optional SDK extras. See the [official SDK documentation](https://py.sdk.modelcontextprotocol.io/).

```sh
uv run python -m research_assistant.interfaces.mcp_server --index data/index_v2
```

The process waits for an MCP host on stdin/stdout; stdout is reserved for the
protocol and diagnostics go to stderr. There is no HTTP listener. Configure a
local host to launch this command from the repository root (or use absolute
paths). As with `ask`, answer generation needs `OPENAI_API_KEY` and `OPENAI_MODEL`
in the process environment. `.env` is not automatically loaded; optionally use
`uv run --env-file .env python -m research_assistant.interfaces.mcp_server --index data/index_v2`.
Search alone needs neither OpenAI configuration nor paid API calls. Models may
be downloaded on first search. Answer calls with evidence can incur API charges.

### Local host example: VS Code Copilot

This native user configuration follows the [official VS Code MCP reference](https://code.visualstudio.com/docs/agents/reference/mcp-configuration)
(checked 2026-11-26). Merge it into `~/.copilot/mcp-config.json`
when `COPILOT_HOME` is unset; otherwise use `$COPILOT_HOME/mcp-config.json`.
Replace the absolute paths for your machine (`command -v uv` locates uv).

```json
{
  "mcpServers": {
    "researchAssistant": {
      "type": "stdio",
      "command": "/absolute/path/to/uv",
      "args": [
        "run", "python", "-m", "research_assistant.interfaces.mcp_server",
        "--index", "/absolute/path/to/rag-mcp-research/data/index_v2"
      ],
      "cwd": "/absolute/path/to/rag-mcp-research",
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "OPENAI_MODEL": "${OPENAI_MODEL}"
      },
      "tools": ["search_papers", "answer_question"]
    }
  }
}
```

`cwd` selects the uv project. Install dependencies with `uv sync --locked` there
first; the local index must exist. For answering, export credentials before
launching a fresh VS Code process from that shell:

```sh
export OPENAI_API_KEY='your-api-key'
export OPENAI_MODEL='your-model-id'
code /absolute/path/to/rag-mcp-research
```

Do not store secrets in the configuration. For retrieval-only use, omit `env`.
Select the Copilot harness and manage the server through **Chat: Open Customizations →
MCP Servers**. This format does not load `.env` automatically. The example is
verified against documentation, not an executed VS Code host session. Calling
`answer_question` with evidence may incur OpenAI charges.

### Tools and lifecycle

Exactly two tools are registered:

- `search_papers(question, top_k=5, rerank=True)` returns a structured `results`
  list with 1-based `rank`, `chunk_id`, `document_id`, `source`, `page_number`,
  `section`, full `text`, and method-specific `score`. Order and metadata are
  preserved; empty searches return an empty list. Invalid top_k is not clamped.
- `answer_question(question)` returns `answer` and `sources`, each containing
  `label`, `chunk_id`, `document_id`, `source`, `page_number`, `section`, and
  evidence `text`. Citation-ID validity does not prove factual support.

The SDK lifespan loads one persisted index through one `ResearchService`, reuses
it across requests, and closes it in `finally`. Imports and tool registration do
not load models or indexes. An AnyIO lock serializes both tools; synchronous
service work runs in a worker thread without abandoning it on cancellation.
Shutdown waits for active service work before closing owned resources. Expected
input errors are reported safely; backend errors are sanitized rather than
returning provider messages, environment values, or stack traces. Our
application diagnostics use Python logging to stderr. Application code never
redirects process-wide stdout or file descriptors; third-party/native stdout
writes are not intercepted by our code. Such writes can still affect protocol
reliability (including before SDK transport startup); configure those libraries
to use stderr rather than relying on SDK-specific transport handling. Forced
process termination cannot promise cleanup, and an active inference/API request can delay shutdown.

Offline tests use the SDK's in-memory client and a fake service to verify schemas,
structured output, reuse, serialization, errors, and cleanup. No paid API calls
are made. The real stdio smoke tests launch the actual `uv run python -m` server
command through the official SDK subprocess transport. They exercise nonempty FAISS +
BM25 hybrid retrieval twice with a deterministic test embedder, plus empty-index
answering without API access. They check JSON-RPC stdout, stderr diagnostics,
service reuse, cleanup, and zero-exit EOF shutdown. Test-only Python startup
hooks inject dependencies; production code has no test mode. The tests observe
two private SDK transport helpers, so SDK upgrades may require test adjustments.
They do not validate real model quality or paid answer generation. Run them with:

```sh
uv run pytest tests/test_mcp_stdio.py -v
```

The tests require `uv` on PATH and installed project dependencies; no model
downloads or API credentials are needed. No HTTP transport, authentication, or
passage-lookup tool is exposed here.

### Opt-in real-model MCP smoke test

Run manually against the existing three-paper index (never run this script in CI):

```sh
uv run python scripts/smoke_mcp.py --index data/index
# Require cached models and prohibit Hugging Face downloads:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/smoke_mcp.py
```

The script starts the actual server through the official SDK stdio transport,
initializes MCP, checks the exact two tools, and searches for
"Why do Transformers need positional encoding?" with and without reranking.
Both queries are repeated on the same connection. The parent uses
`VectorIndex.load()` to validate the manifest, checksums, vector dimensions, and
chunk alignment, then builds a lookup of the original `Chunk` objects. This loads
the index without initializing embedding or reranking models in the parent.
It prints top-five passages and validates exact IDs, document IDs, source, text,
physical page, and section against those chunks, plus finite descending scores,
sequential ranks, unique results, and unchanged repeated results. The SDK context
managers close the session and subprocess, including on failure. A 180-second
overall timeout bounds the run. Dependencies must already be installed.

The first hybrid call includes lazy embedder initialization; the first reranked
call includes lazy cross-encoder initialization. Subsequent calls use warmed
models. Times are explicitly labeled single-run, client-observed round-trip latencies
including serialization, not rigorous inference benchmarks. Parent index validation
is excluded from MCP startup and request timings. On the local cached corpus (2026-11-23), startup + MCP
initialization took 3.178s, cold hybrid 0.204s, first reranked 0.198s, warm hybrid
0.005s, and warm reranked 0.093s. These are single-run observations, not a latency
distribution. The direct positional-encoding explanation from Attention page 6
was hybrid rank 2 (RRF 0.030478) and reranked rank 1 (raw score 2.534949).

Generation is disabled by default; the script does not forward OpenAI credentials
to the child unless explicitly enabled. To opt into a potentially paid call,
export `OPENAI_API_KEY` and `OPENAI_MODEL` and run:

```sh
uv run python scripts/smoke_mcp.py --with-answer
```

This checks returned citation labels and source metadata, not factual support.
The generation path was not executed during the retrieval-only verification.
`answer_question` has offline mocked and empty-index stdio coverage, but no live
paid API generation over MCP has been verified.
A legitimate insufficient-evidence response may have no sources. Errors fail
with diagnostics without printing API credentials or raw provider exceptions.

## Local HTTP API

```sh
uv run python -m research_assistant.interfaces.api --index data/index_v2
```

Uvicorn binds to `127.0.0.1:8000` with one worker and reload disabled. Interactive
OpenAPI documentation is at `http://127.0.0.1:8000/docs`. Module import and
`create_app(index_path)` do not initialize the index or models; lifespan loads
one index through `ResearchService`. Models remain lazy. Both operations share
one AnyIO lock and run in non-abandoned worker threads. Shutdown marks the
service unavailable, waits for active work, then closes it once.

```sh
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"Why use positional encoding?","top_k":5,"rerank":true}'
# Potentially paid: configure OPENAI_API_KEY and OPENAI_MODEL before server startup.
curl -X POST http://127.0.0.1:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"Why use positional encoding?"}'
```

| Endpoint | Contract |
| --- | --- |
| `GET /health` | `{"status":"ok"}`; no inference or API access |
| `GET /ready` | `{"status":"ready"}` when the persisted index loaded; 503 if unavailable |
| `POST /search` | Required nonblank string `question`; integer `top_k=5` in 1–50; boolean `rerank=true`. Returns ordered `results` with all SearchResult fields |
| `POST /answer` | Required nonblank string `question`, maximum 2,000 characters. Returns `answer` and complete citation `sources`, including labels and evidence text |

Request types are strict; unknown fields and invalid bodies return 422.
Unavailable service and queue-wait expiry return 503. OpenAI timeouts return 504;
connection failures, rate limits, and provider 5xx responses return 503; other
OpenAI API errors return 502. Unexpected backend/output failures return 500.
Messages are generic, without provider exception bodies or internal paths. A failed startup
aborts server startup rather than claiming readiness. Readiness confirms index
loading only, not warmed models, API credentials, or external API availability.
`.env` is not automatically loaded; export variables or explicitly use uv's
`--env-file .env` option. Citation validity is not proof of factual support.

Every HTTP response includes a server-generated `X-Request-ID` (client-supplied
IDs are ignored). Standard Python logging writes one JSON request record to
stderr; Uvicorn access logging is disabled by the entry point to avoid duplicate
request records. Logs contain no request bodies, headers, passage text, query
strings, or raw provider errors. Paths use matched route templates; unknown
paths are recorded as `[unmatched]`. Example record (illustrative, not a benchmark):

```json
{"request_id":"a891e9b4732d4e618f075a1b209cd638","method":"POST","path":"/search","status":200,"duration_ms":125.4,"error_category":null}
```

Error categories are `invalid_request`, `service_unavailable`, `queue_timeout`,
`upstream_timeout`, `upstream_unavailable`, `upstream_error`, and `internal_error`.
Other HTTP errors use `http_error`. Cancelled requests log `request_cancelled`;
499 denotes cancellation before a response starts, not a delivered response.

`/search` and `/answer` wait at most 30 seconds to acquire the shared service
lock. Configure this positive finite duration at startup:

```sh
uv run python -m research_assistant.interfaces.api --index data/index_v2 --queue-wait-timeout 10
```

Python callers can use `create_app(index_path, queue_wait_timeout=10)`.
This bounds **queue waiting only**, not model execution or end-to-end latency.
Workers remain non-abandoned; Python threads and native inference are not forcibly
interrupted. The existing OpenAI client retains its 60-second SDK timeout and two
retries, so total API time can exceed 60 seconds. Shutdown still waits for active
operations; waiting requests recheck availability after acquiring the lock.

This is a local, single-process API with no authentication, hard inference
timeout, bounded queue capacity, or rate limiting. Do not expose
it publicly or add workers as a workaround for serialized service access.
Active inference/API calls can delay shutdown. Offline tests mock ResearchService;
no paid HTTP answer-generation call has been verified.

## CPU Docker deployment

Prerequisites: a running Docker Engine/Desktop, network access for dependency
installation, and a trusted persisted index built with this project. Run these
commands from the repository root. The image uses digest-pinned Python 3.11.17
and uv 0.13.5, installs locked runtime dependencies without the development group,
and runs as UID/GID `10001:10001`. Linux PyTorch uses the official CPU wheel index;
its existing 2.15.0 version is now an explicit dependency to select that source
using [uv's PyTorch configuration](https://docs.astral.sh/uv/guides/integration/pytorch/).
The build asserts that PyTorch has no CUDA runtime. No model weights are downloaded
at build time. Debian's `libgomp1` supplies OpenMP support; OS package repositories
and the package build backend are not snapshot-pinned, so this is not a claim of
bit-for-bit image reproducibility. Image digests require deliberate security updates.

```sh
docker build -t rag-mcp-research:local .
docker volume create rag-mcp-hf-cache
docker run -d --name rag-mcp-api \
  -p 127.0.0.1:8000:8000 \
  --mount "type=bind,src=$(pwd)/data/index_v2,dst=/data/index,readonly" \
  --mount type=volume,src=rag-mcp-hf-cache,dst=/home/app/.cache/huggingface \
  rag-mcp-research:local
```

The index directory must contain `manifest.json`, `chunks.jsonl`, and
`vectors.faiss`. Startup validates their checksums and compatibility. The mount
is read-only and must be readable/traversable by UID 10001. No papers, local
indexes, credentials, caches, or benchmark outputs are copied into the image.
Only the installed application and runtime dependencies are copied from the build stage.

The fresh named volume inherits ownership from the image's cache directory.
[`HF_HOME`](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)
sets `/home/app/.cache/huggingface` as the download location; model
loading remains lazy. The first search may download embedding weights, and the
first reranked search may download cross-encoder weights. The cache persists across
container recreation. For a bind-mounted cache instead, provision a writable
directory owned by UID/GID 10001 on the host; Docker does not fix bind-mount
permissions. If overriding `HF_HOME`, mount a writable directory at that same path.

For optional paid `/answer` generation, export `OPENAI_API_KEY` and `OPENAI_MODEL`
in the calling shell, then add `-e OPENAI_API_KEY -e OPENAI_MODEL` before the image
name in `docker run`. Never pass secrets as image build arguments or copy `.env`
into the image. The application does not automatically load `.env`. Docker
administrators can inspect runtime environment variables; this is local development
credential handling, not a production secret store.

The container binds to `0.0.0.0:8000` internally; the published port above is
restricted to host loopback. Direct Python startup still defaults to `127.0.0.1`.
One Uvicorn worker owns one service. Exec-form startup forwards SIGTERM to Python;
shutdown waits for active non-abandoned operations before closing the service.
Queue waiting still defaults to 30 seconds. Override it by appending
`--index /data/index --host 0.0.0.0 --queue-wait-timeout 10` after the image name.

Smoke checks (no paid call or model download required):

```sh
# Poll for index initialization (60 attempts); fail visibly if it never succeeds.
ready=0
for attempt in $(seq 1 60); do
  if curl --fail --silent --max-time 1 http://127.0.0.1:8000/ready >/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
[ "$ready" -eq 1 ] || { docker logs rag-mcp-api; exit 1; }
curl --fail -i http://127.0.0.1:8000/health
curl --fail -i http://127.0.0.1:8000/ready
curl -i -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"test","top_k":0}'
docker logs rag-mcp-api
```

Expect 200 for health/readiness, 422 for the invalid request, `X-Request-ID` on
all three responses, and JSON request records in stderr logs. `/ready` confirms
index initialization only, not model warm-up or provider availability. After an
idle stop, expect completed application shutdown and exit code 0. Docker may
force-kill work that outlasts the stop grace period; adjust it to workload needs.
This unauthenticated single-process service must not be exposed publicly. No
Compose file is needed for this single-container setup.

Manual Docker Desktop verification on November 26, 2026 confirmed build/startup,
health/readiness 200, invalid search 422, request IDs, JSON logs, CPU embedding and
reranking, and shutdown exit code 0. The build took 52.2 seconds. See the
[compact results table, latency observations and native/Docker comparison](#verified-integration).
These supplied manual observations supersede the earlier daemon-unavailable check;
they were not rerun in the documentation audit.

Optional real retrieval after readiness (may download weights; no OpenAI call):

```sh
for rerank in false true; do
  curl --fail -i -w '\nClient total: %{time_total} s\n' \
    http://127.0.0.1:8000/search \
    -H 'Content-Type: application/json' \
    -d "{\"question\":\"How does FlashAttention reduce GPU memory access?\",\"top_k\":3,\"rerank\":$rerank}"
done
```

Repeat to observe warmed requests. First hybrid and first reranked calls initialize
different models; a persisted disk cache does not preserve in-memory models across
restart. Individual times are not rigorous benchmarks or performance guarantees.

When finished, stop and remove the container (the named model-cache volume remains):

```sh
docker stop --timeout 120 rag-mcp-api
docker logs rag-mcp-api
docker inspect --format '{{.State.ExitCode}}' rag-mcp-api
docker rm rag-mcp-api
```
