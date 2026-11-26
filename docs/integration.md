# S4 HTTP and Docker integration evidence

S4 is complete. These manual observations were supplied by the project author
from Docker Desktop testing on **November 26, 2026**, against Docker implementation
commit `9cd411e4b5fe3caddfd838fd27c680afe57f3d89`. They were not rerun during the final
documentation audit. They demonstrate local integration, not production readiness
or statistically robust performance. Raw logs and full hardware/resource settings
were not supplied with this record.

## Verified integration

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

## Individual latency observations

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

## Native versus Docker ranking consistency

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

## Reproduction

Use the [README Docker instructions](../README.md#cpu-docker-deployment-s4-step-4)
for image build, loopback-only publishing, read-only index mount, persistent
writable cache, readiness polling, HTTP requests and shutdown. Search can download
weights but requires no OpenAI credentials or paid API call. Repeated requests
reuse models; a restart retains the disk cache but discards in-memory state.

## Final engineering audit

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

## Remaining limits and S5 handoff

The service is local and unauthenticated, with no rate limiting or bounded queue
capacity. Queue timeout does not bound inference or total latency. Long operations
can delay shutdown or exceed Docker's stop grace period. Citation-ID validation
is not proof of factual support. The five-question pooled evaluation is small
and may miss relevant passages outside the pool. S5 should expand labeled retrieval
evaluation and controlled measurements; no new experiments were performed here.
