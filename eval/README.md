# Manual retrieval evaluation preparation

Run from the repository root after `uv sync --locked`, using the existing
persisted index. This step exports evidence for human judgment, not metrics or
automatic labels. No OpenAI client or LLM is used. Local embedding and reranking
models may download their pinned weights on first use.

```sh
uv run python scripts/export_eval_candidates.py --question-id Q01 --out data/eval/Q01.json
uv run python scripts/export_eval_candidates.py --question-id Q02 --out data/eval/Q02.json
uv run python scripts/export_eval_candidates.py --question-id Q03 --out data/eval/Q03.json
uv run python scripts/export_eval_candidates.py --question-id Q04 --out data/eval/Q04.json
uv run python scripts/export_eval_candidates.py --question-id Q05 --out data/eval/Q05.json
```

Defaults: `--questions eval/questions.json`, `--index data/index`. Alternatively,
use `--question "Your question"`. Output paths must not already exist, to avoid
accidentally replacing inspection files. `data/eval/` exports are Git-ignored.

The pool combines FAISS top 20, BM25 top 20, hybrid RRF top 20, and cross-encoder
reranking of those same 20 hybrid candidates. It preserves the existing retrieval
algorithms. Reranking changes order, not membership of that hybrid candidate set.
Hybrid itself draws from the top 20 FAISS and BM25 results, so this pool contains
at most 40 unique chunks. BM25 may return fewer than 20 matches.

Records are deduplicated by chunk ID and sorted lexicographically by ID, rather
than by any method's score. Ranks are 1-based; absence is `null`. Raw cross-encoder
scores are logits, not relevance grades or probabilities. Other raw scores are
omitted. Repeated runs on fixed rankings produce identical output; inference
across hardware/library versions can still change rankings.

Illustrative candidate structure (placeholders, not a real export):

```json
{
  "question_id": "Q01",
  "question": "Why does the Transformer need positional encoding?",
  "candidate_depth": 20,
  "rerank_candidate_depth": 20,
  "candidates": [
    {
      "chunk_id": "<actual chunk ID from your export>",
      "source": "<source filename>",
      "page_number": null,
      "section": null,
      "text": "<full passage text>",
      "ranks": {"faiss": null, "bm25": null, "hybrid": null, "rerank": null},
      "cross_encoder_score": null
    }
  ]
}
```

## Labeling

Read the question and passage, consulting its original PDF page or Markdown
section when needed. Assign one grade per `(question_id, chunk_id)`:

- **2:** Directly supports answering the question.
- **1:** Relevant context but insufficient alone.
- **0:** Irrelevant or misleading for the question.

For initial binary metrics, only grade 2 will count as relevant. Metrics are not
implemented here. Unjudged passages are not automatically grade 0.

Keep manual judgments in a separate file such as `eval/labels.json`, never in
candidate ranks or scores. Example template, **not verified ground truth**:

```json
[
  {
    "question_id": "Q01",
    "chunk_id": "<copy an inspected chunk ID>",
    "grade": null,
    "notes": "<explain your manual judgment>"
  }
]
```

Replace `null` with 2 only after confirming the passage directly supports an
answer, 1 for supporting context alone, or 0 otherwise. Do not derive grades from
ranks or model scores. Hide the ranks and score fields while judging where
practical to reduce anchoring; retain them for later analysis. No labels have
been assigned by this tool. Keep the original index/chunking configuration for
reproducibility: rebuilding the corpus can change IDs and invalidate judgments.

## Inspect beyond the pool

All methods can miss relevant evidence. Inspect a likely source document when
a pool seems incomplete, rather than requiring all 419 chunks for every question:

```sh
uv run python scripts/export_eval_candidates.py \
  --source attention_is_all_you_need.pdf --out data/eval/attention-all.json
```

`--source` matches the exact persisted source path (including any subdirectories).
This exports every chunk of that document in persisted order, including IDs,
physical pages/sections, and full text. It loads the validated FAISS persistence
but does not initialize embedding or reranking models. An unknown source fails
clearly. Add manually discovered relevant IDs to the separate labels file.

Pooling reduces dependence on a single ranking, but it does not create unbiased
or exhaustive ground truth. Shared first-stage candidates, overlapping chunks,
PDF extraction errors, and limited document inspection remain coverage limits.
