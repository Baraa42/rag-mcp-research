# S5F retrieval latency and answer grounding

## Warm retrieval benchmark

The run used the frozen 1,238-chunk ten-paper index and all 50 evaluation
questions. It loaded the index and the pinned CPU embedder and cross-encoder
once, built the BM25 index once, and warmed each method with one query.
Five measurement blocks each contained all 200 question–method pairs,
shuffled by a deterministic random generator with seed 42. The benchmark
used the existing retrieval classes, requested five results from each method,
and passed 20 hybrid candidates to the cross-encoder. All 1,000 calls
returned five results in the same order as the corresponding frozen top five.

Timing starts immediately before the in-process retrieval call and ends when
it returns. It includes query embedding, lexical ranking, fusion, and/or
cross-encoder inference as applicable. Index and model initialization, downloads,
warm-up, result validation, and file output are outside the timed interval.
These are synchronous, single-request measurements at the retrieval layer.
An HTTP round trip also includes server routing, serialization, and transport.

The November 26, 2026 run used an Apple M5 Pro CPU, 18 logical cores, 48 GB RAM,
and macOS 26.5.2. Python was 3.11.17, FAISS 1.16.0, NumPy 2.4.6,
PyTorch 2.15.0, Sentence Transformers 6.3.0, Transformers 5.23.0, and
rank-bm25 0.2.2. OMP, Accelerate, and OpenBLAS were configured for one
thread and tokenizer parallelism was disabled. FAISS reported one OpenMP
thread and PyTorch one intra-op thread with an 18-thread inter-op pool.
The models were cached and Hugging Face was in offline mode.
[Run metadata and statistics](../eval/v2/retrieval-latency.json).

| Retrieval method | Calls | Median (ms) | p95 (ms) | Mean (ms) | Held-out nDCG@5 | Held-out pooled Recall@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FAISS | 250 | 2.57 | 2.94 | 2.59 | 0.5525 | 0.6560 |
| BM25 | 250 | 1.54 | 2.09 | 1.54 | 0.6664 | 0.8049 |
| Hybrid RRF | 250 | 4.28 | 5.07 | 4.32 | 0.6719 | 0.7706 |
| Hybrid + cross-encoder | 250 | 164.49 | 174.95 | 164.93 | 0.7404 | 0.7706 |

Latency aggregates combine the 250 observations per method and use linear
interpolation for p95. See the [raw CSV in execution order](../eval/v2/retrieval-latency.csv)
and [S5E quality results](evaluation-v2.md). The quality columns use the 29
answerable held-out questions, while timings cover all 50.

BM25 was fastest and had the highest pooled Recall@20 in this snapshot.
Hybrid cost about 2.7 ms more than BM25 at the median and had similar held-out
nDCG@5. The paired interval for their quality difference spans zero.
Reranking cost about 160 ms more than hybrid at the median and improved
held-out nDCG@5 by 0.0685, with identical Recall@20 because the candidate
set stayed the same. The recall and ranking trade-offs apply to this corpus,
model configuration, and candidate depth.

## Bounded answer evaluation

The [fixed selection](../eval/v2/answer-questions.json) was written before
any API calls. It covers four research categories. Q06 compares multi-head
and single-head attention, Q13 compares split-K and split-Q, and Q33 covers
the query-placement intervention. Q21 checks dense retrieval architecture,
Q35 asks for Whisper's acoustic representation, and Q38 asks about the
wav2vec 2.0 objective. Q45–Q50 are the expected-unanswerable controls.

The runner used one `ResearchService` instance with its existing 20-candidate
hybrid retrieval, cross-encoder reranking, top-five evidence, prompt,
1,024-output-token limit, and citation validation. The model came from
`OPENAI_MODEL`. The evaluation transport allowed one Responses POST per
question, with SDK retries and redirects disabled and a hard cap of 12 POSTs.
There were no judging calls. Generation timers measure the outbound SDK
request and response, including network and provider time. The separately
recorded service times include retrieval and lazy setup on the first request.

### Live results

The configured `gpt-5-mini` returned snapshot `gpt-5-mini-2025-08-07`
at the default service tier. All 12 Responses POSTs returned HTTP 200.
Six produced answers accepted by the application, five returned the exact
insufficient-evidence fallback, and one returned an incomplete response.
Q50's incomplete response reported 1,024 output tokens and the application
rejected it. It is counted as an error. There were no retries.

| Question | Outcome | Cited labels | SDK round trip (s) | Input tokens | Output tokens |
| --- | --- | --- | ---: | ---: | ---: |
| Q06 | Accepted with valid citation IDs | S1, S2, S4, S5 | 11.99 | 1,659 | 656 |
| Q13 | Accepted with valid citation IDs | S1, S2 | 7.08 | 1,634 | 656 |
| Q21 | Accepted with valid citation IDs | S1, S4 | 6.05 | 1,607 | 517 |
| Q33 | Accepted with valid citation IDs | S2, S3 | 4.90 | 1,564 | 402 |
| Q35 | Accepted with valid citation IDs | S1, S3 | 5.37 | 1,676 | 412 |
| Q38 | Accepted with valid citation IDs | S1, S2 | 9.63 | 1,527 | 871 |
| Q45 | Exact abstention | — | 2.58 | 1,629 | 174 |
| Q46 | Exact abstention | — | 2.28 | 1,668 | 166 |
| Q47 | Exact abstention | — | 1.97 | 1,670 | 131 |
| Q48 | Exact abstention | — | 2.56 | 1,631 | 198 |
| Q49 | Exact abstention | — | 3.15 | 1,790 | 256 |
| Q50 | Incomplete response | — | 11.59 | 1,660 | 1,024 |

The SDK round trip had a median of **5.13 s** and ranged from **1.97 s**
to **11.99 s** across all 12 responses. The complete in-process answer call
had a median of **5.32 s**. Every question received the same five ordered
evidence chunks as the frozen reranked list. Usage totaled **19,715 input**
and **5,463 output tokens** with zero reported cached input tokens.
At standard GPT-5 mini rates of $0.25 per million input tokens,
$0.025 per million cached input tokens, and $2.00 per million output tokens,
the usage-based estimate is **$0.01585475 USD**, including the incomplete
response. The estimate uses provider-reported usage and can differ from
the billed amount. [OpenAI GPT-5 mini pricing](https://developers.openai.com/api/docs/models/gpt-5-mini).

The [answer records](../eval/v2/answer-results.json) include accepted text,
source IDs and provenance, citations, per-query usage, latency, and errors.
The local `data/eval_v2/s5f-answer-run/assessment.json` adds each reference
answer and the exact text of the retrieved and cited passages. It remains
ignored because it contains the full evidence. For independent review,
compare the answer with its reference, check claim support in each cited
passage, and note missing evidence or claims that exceed the source.
Assess expected-unanswerable cases separately, with service errors distinct
from abstentions.

Q35 illustrates a citation-valid answer with a substantive evidence gap.
The answer cites Whisper's 30-second buffering and 20-ms timestamp resolution.
The reference asks for its log-Mel spectrogram input. S5E found that the
grade-2 spectral-feature passage fell to reranked position 6, outside the
five chunks used for generation. The cited chunks were both judged grade 1.
The remaining answer records are available for claim-level review.
[S5E passage analysis](evaluation-v2.md#passage-level-examples).

## Limitations

These warm CPU results describe one sequential run on one machine.
They exclude cold starts, concurrent requests, HTTP, and deployment overhead.
The 12 fixed answer questions share a small ten-paper corpus. Answer latency
also includes variable network and provider time. Citation validation confirms
that source labels resolve within the supplied evidence. Independent review
is still required to assess factual support, correctness, and completeness.
The five exact abstentions describe this run and this control set.

## Reproduce

Run from the repository root with the validated index and cached model
revisions. Use a new output directory if keeping the published artifacts.

```sh
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 \
  uv run python scripts/benchmark_retrieval_v2.py --out data/eval_v2/latency-repeat
```

For a separate paid run, explicitly supply `OPENAI_API_KEY` and `OPENAI_MODEL`
through the environment or use `uv run --env-file .env` with an ignored local
file. The runner requires `--live` and a fresh output directory. It writes
full evidence and a summary with source IDs and metadata locally after each
outcome.

```sh
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 \
  uv run --env-file .env python scripts/evaluate_answers_v2.py --live \
  --out data/eval_v2/answer-repeat
```
