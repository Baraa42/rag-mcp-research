# Ten-paper retrieval evaluation preparation

This is the frozen S5C question set for the [expanded corpus](../../docs/corpus-v2.md).
The five-question benchmark and its original evaluation artifacts remain in place.

- `questions.json` maps question IDs to text.
- `metadata.json` records difficulty, category, split, expected answerability,
  reference answers and pointers to independently selected passages.
- `rankings.json` records the method-specific ordered chunk IDs and fingerprints
  for the questions, metadata, index files, index manifest and pinned models.
  Array positions are 1-based ranks. BM25 may return fewer than 20 results.
- `labels.json` records AI-generated semantic relevance judgments for the
  blinded pool. The [labeling report](labeling.md) describes the rubric, grade
  distributions, and cases flagged for human audit.
- `results.json` records per-question metrics, split and category averages,
  and paired bootstrap intervals. See the [evaluation report](../../docs/evaluation-v2.md)
  for the results, failure analysis, and reproduction commands.

Question text, evidence and splits were frozen before batch retrieval. Reference
passages were selected by inspecting local paper sections and source-ordered
chunks with targeted keyword searches, without consulting candidate rankings.
They are starting points for human judgments and may provide partial or
complementary evidence. The dataset does not claim exhaustive evidence coverage.

| Category | Development | Held out |
| --- | ---: | ---: |
| Transformer | 2 | 4 |
| Attention systems | 6 | 6 |
| Retrieval and context | 5 | 10 |
| Speech | 2 | 9 |
| Out of corpus | 0 | 6 |
| **Total** | **15** | **35** |

Development contains 5 easy, 6 medium and 4 hard questions. Held out contains
9 easy, 15 medium and 11 hard questions. Development includes Q01–Q05 unchanged
and groups overlapping PagedAttention, ColBERT and Whisper smoke-test themes.
Held-out questions should remain excluded from tuning. This split is a protocol
for future work on the same corpus. The committed files are publicly inspectable.

## Changes to the draft

| ID | Revision and reason |
| --- | --- |
| Q13 | Use the paper's split-K and split-Q terminology and specify warp-level partitioning. |
| Q17 | Ask about backward-pass parallelism to distinguish it from Q12's occupancy question. |
| Q19 | Ask about position-wise feed-forward processing. The draft latency question largely repeated Q03's memory-traffic rationale. |
| Q20 | Ask about skipping and masking tiles to separate the question from Q08's basic decoder masking. |
| Q22 | Ask why RAG freezes the document encoder. The marginalization wording overlapped Q05. |
| Q26 | Ask about mask-token query augmentation. The original repeated the ColBERT smoke-test mechanism and overlapped Q24. |
| Q27 | Ask about punctuation filtering to make the storage question concrete and source-specific. |
| Q32 | Ask how answer passages and distractors are selected. The broad synthesis overlapped Q30. |
| Q33 | Ask about the measured query-placement intervention. The draft repeated Q28's position effect and invited speculation. |
| Q44 | Ask for the convolution module operations to distinguish it from Q41's local/global explanation. |

## Expected answerability

The first 44 questions have reference evidence in the ten papers. For Q45–Q50,
the audit checked paper topics, relevant sections and normalized text across
the persisted chunks for the requested entities and measurements. GPT-4 appears
in discussions, references and model evaluations, with no training-energy total.
Whisper's transcription results provide no matching Darija telephone experiment
or Scribe v2 benchmark. PagedAttention provides block-size trade-offs and its own
measurements, with no GPT-5/B200 result. The corpus supplied no evidence for
global workspace theory, exact Scribe v2 dataset composition or GPT-5.2 attention
architecture. These are expected-unanswerable **within this corpus**. PDF text
extraction and finite inspection limit certainty. Revisit any case if annotators
find contrary evidence.

## Reproduce the batch

From the repository root, run `uv sync --locked` and prepare `data/index_v2/`
using the corpus instructions. With both model revisions cached:

```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 \
  uv run python scripts/export_eval_candidates.py --batch --out data/eval_v2
```

Use a fresh output directory for each run. For November 26, 2026, the run
completed all 50 questions on an Apple M5 Pro CPU in **8.60 seconds**, including
**0.13 seconds** of setup inside the exporter. This is one run with cached
weights. The batch loads one validated index and reuses one embedder and one
cross-encoder. It preserves FAISS, BM25 and hybrid top-20 retrieval and reranks
the same 20 hybrid candidates. Hybrid retrieval still performs its own first
stage through the existing implementation.

- `data/eval_v2/Q01.json` through `Q50.json` contain 1,646 pooled question–chunk
  pairs, deduplicated per question, with ranks and cross-encoder scores.
- `data/eval_v2/blind/Q01.json` through `Q50.json` contain 1,665 pairs, including
  19 independently identified references outside the retrieval pools.
- `data/eval_v2/run.json` records total and per-question durations, counts and
  failures. `data/eval_v2/rankings.json` is the local copy of the tracked snapshot.

## Blind labeling

Give labelers only the files in `data/eval_v2/blind/`. Each contains the question
ID, question text and shuffled passages with chunk ID, source, physical PDF page
or section, and text. The deterministic order is a SHA-256 sort keyed by the
question ID and chunk ID. Independently selected references are mixed in without
flags. Keep metadata and rankings separate from the labeling view.

Use grade **2** for direct support, **1** for relevant context or partial evidence,
and **0** for irrelevant or misleading passages. Save judgments separately and
key them by question ID and chunk ID. Human auditing remains necessary before
using the AI-generated labels for evaluation. Leave unjudged passages distinct
from grade 0. For expected-unanswerable questions, grade the evidence that is
present and flag any passage that actually answers the question. Later metric
calculations need completed judgments and an explicit policy for questions with
zero relevant passages.
