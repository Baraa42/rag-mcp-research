# Offline retrieval evaluation

Recall uses pooled manual judgments, not exhaustive corpus relevance.

| Question | Method | Grade-2 count | Recall@5 | Recall@20 | RR@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| Q01 | faiss | 1 | 0.0000 | 1.0000 | 0.0000 |
| Q01 | bm25 | 1 | 1.0000 | 1.0000 | 1.0000 |
| Q01 | hybrid | 1 | 1.0000 | 1.0000 | 1.0000 |
| Q01 | rerank | 1 | 1.0000 | 1.0000 | 1.0000 |
| Q02 | faiss | 1 | 1.0000 | 1.0000 | 0.3333 |
| Q02 | bm25 | 1 | 1.0000 | 1.0000 | 0.3333 |
| Q02 | hybrid | 1 | 1.0000 | 1.0000 | 1.0000 |
| Q02 | rerank | 1 | 1.0000 | 1.0000 | 0.2500 |
| Q03 | faiss | 6 | 0.1667 | 0.8333 | 0.2500 |
| Q03 | bm25 | 6 | 0.5000 | 0.6667 | 1.0000 |
| Q03 | hybrid | 6 | 0.5000 | 0.6667 | 1.0000 |
| Q03 | rerank | 6 | 0.6667 | 0.6667 | 1.0000 |
| Q04 | faiss | 7 | 0.2857 | 0.8571 | 0.3333 |
| Q04 | bm25 | 7 | 0.4286 | 1.0000 | 1.0000 |
| Q04 | hybrid | 7 | 0.2857 | 1.0000 | 1.0000 |
| Q04 | rerank | 7 | 0.2857 | 1.0000 | 1.0000 |
| Q05 | faiss | 3 | 0.0000 | 0.6667 | 0.0000 |
| Q05 | bm25 | 3 | 0.3333 | 1.0000 | 0.3333 |
| Q05 | hybrid | 3 | 0.0000 | 0.6667 | 0.0000 |
| Q05 | rerank | 3 | 0.3333 | 0.6667 | 0.3333 |

| Method | Macro Recall@5 | Macro Recall@20 | MRR@5 |
| --- | ---: | ---: | ---: |
| faiss | 0.2905 | 0.8714 | 0.1833 |
| bm25 | 0.6524 | 0.9333 | 0.7333 |
| hybrid | 0.5571 | 0.8667 | 0.8000 |
| rerank | 0.6571 | 0.8667 | 0.7167 |

Recall is N/A when a question has no grade-2 judgments; its macro recall
is also N/A rather than silently excluding that question. RR@5 remains 0.

## Methodology

Run `uv run python scripts/evaluate_retrieval.py` from the repository root.
The script uses only Python's standard library and saved JSON; it does not run
retrievers, embedding models, cross-encoders, or an LLM. Optional arguments are
`--questions`, `--labels`, and `--exports`. The input labels were supplied by the
user in `eval_labels_Q01_Q05.json`, imported without changing their judgments.

All 151 pooled candidates have manual labels: Q01 35, Q02 31, Q03 32,
Q04 27, Q05 26. Only grade 2 is relevant; grades 0 and 1 are nonrelevant
for these binary metrics. Recall@K divides grade-2 hits within rank K by
all grade-2 judgments for that question. RR@5 is the inverse rank of the
first grade-2 result within five, or zero. Macro averages weight all five
questions equally; MRR@5 is the macro average of RR@5. Calculations use full
precision, with four decimal places displayed.

Validation requires Q01–Q05, matching question metadata, complete pool label
coverage, unique label/candidate IDs, integer grades 0–2, and all four ranking
fields. Null means absent from that method, not an unjudged passage. Non-null
ranks must be unique, contiguous positive integers no greater than 20.
Additional manually judged passages outside the pool may contribute to the
recall denominator, allowing later document-wide inspection to expand judgments.

## Observations and limitations

BM25 has the highest pooled Recall@20 here; reranking has the highest
Recall@5 by a small margin, while hybrid has the highest MRR@5. These five
questions do not establish general superiority. For Q01, BM25, hybrid, and
reranking all place the single grade-2 passage first; FAISS misses it in the
top five. For Q02, reranking moves the first relevant result from hybrid rank
1 to rank 4. For Q05, reranking finds a grade-2 result in its top five where
hybrid does not.

Recall uses pooled judgments and may overestimate true corpus-wide recall:
relevant passages omitted by all methods are not in this denominator. No
exhaustive corpus inspection is claimed. The reranker only reorders hybrid
top-20 candidates, so their Recall@20 is identical by construction. The
pool is bounded by the FAISS/BM25 top-20 union. Overlapping chunks count as
separate passages, and manual grades can be subjective. Unjudged passages
are never automatically assigned grade 0. These results evaluate saved
rankings, not live performance, latency, or generated-answer correctness.

## Input fingerprints

Candidate exports remain local and Git-ignored. Preserve them to reproduce
these exact results. SHA-256 hashes identify the inputs used:

| File | SHA-256 |
| --- | --- |
| `eval/questions.json` | `b438fc755764a6646950fd2b5b90493705f9f1a89e0d95abd66849cf548e8df3` |
| `eval/labels.json` | `f6c8d7180bb5a9f910002516d9da20e036d30a86cb0ea8ddfa1f85dabb606b8f` |
| `data/eval/Q01.json` | `ab9329681407d55c420643b8a5f8be92336df25c5b6636c8497a775e7a26d45d` |
| `data/eval/Q02.json` | `4196be8afcef189fd6d6761170b284d4b674fc0141abf6d07a09ff69e3639984` |
| `data/eval/Q03.json` | `8214df3a370441bb08a9f76b0c4d593cb5c5360ca69be606520f4ab970da08b7` |
| `data/eval/Q04.json` | `ef9f41612eee97ba3e684dc2fdd47c466df28b2e6df40c84dafe9e7b0ae6ee70` |
| `data/eval/Q05.json` | `c04a56f37ecd84f2ab186b28a331ae5536b02c47ade1a3a74dfd35fc110bbf32` |
