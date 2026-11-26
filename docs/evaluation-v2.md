# S5E expanded retrieval evaluation

On the 29 answerable held-out questions, hybrid plus cross-encoder reranking
had the highest mean nDCG@5 at **0.7404**. BM25 had the highest pooled Recall@20
at **0.8049**. Hybrid's held-out mean nDCG@5 was **0.6719**, close to BM25's
**0.6664**. The paired interval for that difference spans zero.

## Data and protocol

The [ten-paper corpus](corpus-v2.md) has 1,238 chunks. Each frozen ranking lists
up to 20 candidates. The cross-encoder reorders the same 20 hybrid candidates.
There are 1,646 ranked question–chunk pairs in the union of the four methods and
19 additional independently identified pairs. The [labels](../eval/v2/labels.json)
cover all 1,665 pairs with 875 grade-0, 532 grade-1, and 258 grade-2 judgments.

The [label review](../eval/v2/labeling.md#s5e-review-of-the-six-flagged-judgments)
records all six decisions. Q02's all-pairs description and Q20's boundary-only
excerpt moved from grade 2 to 1. The original FlashAttention-2 PDF contains the
reversed inequality. The other Q20 excerpt retains grade 2 for its correct
explanation of skipping future tiles. These judgments remain AI-generated and
require human auditing.

- **Pooled Recall@5 and @20** divide the number of retrieved grade-2 chunks
  by all judged grade-2 chunks for the question, including independent evidence.
- **MRR@5** averages the reciprocal rank of the first grade-2 chunk in the top
  five. A query with no grade-2 hit contributes zero.
- **nDCG@5 and @10** use gain `2**grade - 1` and discount `1 / log2(rank + 1)`.
  The ideal list is sorted by grade from all judged candidates for the question,
  including candidates outside the rankings.
- A zero recall denominator or zero ideal DCG produces `null` in JSON and
  **N/A** in tables. Macro means give each question equal weight. A mean stays
  undefined if any of its constituent values is undefined.

The evaluator checks the question and metadata SHA-256 values against the
snapshot, validates method depths and unique IDs, reconstructs the frozen RRF
ordering from the two first-stage lists, and checks that reranking is a
permutation of the hybrid candidates. Labels must cover exactly the union of
the ranks and the independent references for each question. With the local
index supplied, it additionally verifies the manifest, model identity recorded
in the manifest, and both index file hashes. Model provenance comes from the
snapshot. The evaluation runs entirely on saved artifacts.

## Macro results

[Full per-question metrics and split/category aggregates](../eval/v2/results.json)
are saved at full precision. Category means below show nDCG@5. The JSON
includes all five metrics by category within each split.

| Split | N | Method | Pooled R@5 | Pooled R@20 | MRR@5 | nDCG@5 | nDCG@10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Development | 15 | FAISS | 0.1803 | 0.5860 | 0.3856 | 0.4349 | 0.4747 |
| Development | 15 | BM25 | 0.2766 | 0.7355 | 0.6778 | 0.6056 | 0.6197 |
| Development | 15 | Hybrid RRF | 0.2786 | 0.6882 | 0.5222 | 0.5378 | 0.5764 |
| Development | 15 | Hybrid + cross-encoder | 0.3061 | 0.6882 | 0.7300 | 0.6374 | 0.6507 |
| Held-out answerable | 29 | FAISS | 0.3495 | 0.6560 | 0.6149 | 0.5525 | 0.5560 |
| Held-out answerable | 29 | BM25 | 0.5066 | 0.8049 | 0.7414 | 0.6664 | 0.6413 |
| Held-out answerable | 29 | Hybrid RRF | 0.5027 | 0.7706 | 0.7328 | 0.6719 | 0.6645 |
| Held-out answerable | 29 | Hybrid + cross-encoder | 0.5167 | 0.7706 | 0.7759 | 0.7404 | 0.7264 |
| All answerable | 44 | FAISS | 0.2918 | 0.6321 | 0.5367 | 0.5125 | 0.5283 |
| All answerable | 44 | BM25 | 0.4282 | 0.7813 | 0.7197 | 0.6456 | 0.6340 |
| All answerable | 44 | Hybrid RRF | 0.4263 | 0.7425 | 0.6610 | 0.6262 | 0.6344 |
| All answerable | 44 | Hybrid + cross-encoder | 0.4449 | 0.7425 | 0.7602 | 0.7053 | 0.7006 |

| Split | Category | N | FAISS | BM25 | Hybrid RRF | Hybrid + cross-encoder |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Development | Attention systems | 6 | 0.5067 | 0.6395 | 0.5950 | 0.6794 |
| Development | Retrieval and context | 5 | 0.4534 | 0.6838 | 0.4875 | 0.7054 |
| Development | Speech | 2 | 0.1904 | 0.3125 | 0.3312 | 0.4075 |
| Development | Transformer | 2 | 0.4182 | 0.6011 | 0.6989 | 0.5713 |
| Held-out answerable | Attention systems | 6 | 0.4195 | 0.7172 | 0.5532 | 0.7583 |
| Held-out answerable | Retrieval and context | 10 | 0.5744 | 0.6685 | 0.7356 | 0.7772 |
| Held-out answerable | Speech | 9 | 0.5994 | 0.6149 | 0.6528 | 0.6565 |
| Held-out answerable | Transformer | 4 | 0.5921 | 0.7005 | 0.7338 | 0.8102 |
| All answerable | Attention systems | 12 | 0.4631 | 0.6784 | 0.5741 | 0.7188 |
| All answerable | Retrieval and context | 15 | 0.5340 | 0.6736 | 0.6529 | 0.7533 |
| All answerable | Speech | 11 | 0.5251 | 0.5599 | 0.5943 | 0.6112 |
| All answerable | Transformer | 6 | 0.5341 | 0.6674 | 0.7222 | 0.7306 |

## Paired uncertainty on held-out nDCG@5

The percentile intervals use 10,000 resamples of the 29 answerable held-out
questions with replacement. Each draw uses the same resampled questions for
both methods. The seed is 42 and the endpoints are the 2.5th and 97.5th
percentiles. Each difference takes the first method minus the second.

| Comparison | Mean difference | 95% interval |
| --- | ---: | ---: |
| BM25 − FAISS | +0.1138 | [+0.0157, +0.2134] |
| Hybrid RRF − FAISS | +0.1194 | [+0.0588, +0.1844] |
| Hybrid + cross-encoder − FAISS | +0.1878 | [+0.1063, +0.2720] |
| Hybrid RRF − BM25 | +0.0056 | [-0.0651, +0.0760] |
| Hybrid + cross-encoder − BM25 | +0.0740 | [+0.0234, +0.1278] |
| Hybrid + cross-encoder − Hybrid RRF | +0.0685 | [+0.0140, +0.1246] |

These intervals describe sensitivity to which questions from this set are sampled.
Questions share papers and overlapping evidence. Resampling individual questions
does not model that dependence or label uncertainty. The intervals are unadjusted
for multiple comparisons and do not support broad statistical generalization.

## Expected-unanswerable questions

Q45–Q50 have zero grade-2 judgments. Pooled recall is **N/A** for each question
and method, and reciprocal rank is zero. The four queries whose candidates are
all grade 0 also have **N/A** nDCG. Q48 and Q50 have grade-1 candidates, so their
nDCG is defined. The six-question macro nDCG is **N/A** under the stated
aggregation rule.

| Question | FAISS grade-1 hits @5 | BM25 | Hybrid RRF | Hybrid + cross-encoder |
| --- | ---: | ---: | ---: | ---: |
| Q45 | 0 | 0 | 0 | 0 |
| Q46 | 0 | 0 | 0 | 0 |
| Q47 | 0 | 0 | 0 | 0 |
| Q48 | 4 | 3 | 5 | 5 |
| Q49 | 0 | 0 | 0 | 0 |
| Q50 | 1 | 0 | 1 | 2 |

Q48 surfaces Whisper's transcript-normalization discussion and results from
other telephone and multilingual datasets. Those passages have no Moroccan
Darija result for the named benchmark. Q50 surfaces the historical vLLM
block-size experiments, which use other models and hardware. Each method
returned 20 passages on each of these six queries. Abstention requires a
separate evaluation of answer generation.

## Failure analysis

Using nDCG@5 across the 44 answerable questions, hybrid beat both standalone
methods on 12, fell below the better standalone method on 26, and tied the
better one on 6. Reranking improved over hybrid on 32, worsened on 9, and tied
on 3. Recall@20 is identical for hybrid and reranking because their candidate
sets match.

| Outcome | Questions |
| --- | --- |
| Hybrid above both standalone methods | Q01, Q02, Q06, Q09, Q16, Q21, Q22, Q23, Q27, Q33, Q34, Q35 |
| Hybrid below the better standalone method | Q03, Q04, Q05, Q07, Q08, Q10, Q11, Q12, Q14, Q17, Q18, Q19, Q20, Q24, Q25, Q26, Q28, Q29, Q31, Q36, Q37, Q38, Q39, Q40, Q41, Q43 |
| Reranking improves hybrid | Q01, Q03, Q04, Q05, Q07, Q08, Q09, Q10, Q11, Q12, Q13, Q14, Q15, Q16, Q17, Q19, Q20, Q22, Q23, Q24, Q25, Q27, Q28, Q29, Q31, Q33, Q34, Q36, Q37, Q39, Q41, Q43 |
| Reranking worsens hybrid | Q02, Q06, Q18, Q21, Q30, Q32, Q35, Q38, Q40 |

### Passage-level examples

Passage IDs below are 12-character prefixes. Page numbers refer to physical
PDF pages. The explanations follow from passage content and rank movements.

1. **Q21, RAG's dense retrieval architecture.** Hybrid raised nDCG@5 to 0.7995
   from 0.2183 for FAISS and 0.5269 for BM25. The DPR bi-encoder description
   (`257653b856be`, RAG p. 3) was FAISS rank 11 and BM25 rank 1, and stayed at
   hybrid rank 1. A passage naming the separate BERT encoders
   (`721eddcaffce`, p. 4) moved from FAISS 16 and BM25 5 to hybrid 2.
   FAISS rank 1 discusses index compression without describing the architecture.
   The missing next chunk on p. 3 (`fd7fe01f0c7e`) repeats the BERT encoder
   detail and adds the maximum inner-product search procedure. This is both
   overlapping evidence and a missed extension of the explanation.
2. **Q17, FlashAttention-2 backward parallelism.** BM25 ranked the actual
   one-thread-block-per-column explanation and atomic `dQ` adds first
   (`53b8b9e6ec5d`, FlashAttention-2 p. 8). FAISS omitted it. Hybrid moved it
   to rank 8, while a passage on forward softmax and incremental output
   accumulation from the first FlashAttention paper (`bd9238056b07`, p. 22)
   took rank 1 after appearing at FAISS rank 1 and BM25 rank 6. Agreement on
   nearby terminology displaced backward-specific evidence. Reranking raised
   the direct passage to rank 4.
3. **Q25, ColBERT's MaxSim formula.** Hybrid put a dataset description first
   (`5b6445a43353`, ColBERT p. 6). Its top five contained no grade-2 passage.
   The explanation that sums each query token's best match
   (`408f2ddff353`, p. 4) and the equation (`f99f1ca9d344`, p. 5) moved
   from hybrid ranks 7 and 6 to reranked ranks 1 and 2. These passages cover
   the desired computation in complementary prose and notation.
4. **Q35, Whisper's acoustic features.** The only grade-2 passage
   (`0eeaf6edb79b`, Whisper p. 3) specifies an 80-channel log-Mel spectrogram
   with 25 ms windows and 10 ms stride. It fell from hybrid rank 3 to reranked
   rank 6. A relevant but incomplete description of 30-second buffering
   (`57d1fa2c2db1`, p. 14) stayed first. Reranking also raised an English
   transcription benchmark (`f900fdc9ea13`, p. 15) from 17 to 2. The promoted
   passages concern audio handling and evaluation, leaving the requested
   spectral representation unstated.
5. **Q03, FlashAttention's GPU memory transfers.** The method passage
   (`0fd4561dd294`, FlashAttention p. 4) explains loading Q, K, and V tiles
   from HBM to SRAM and rescaling tile outputs. All methods missed it at 20.
   Hybrid still retrieved a figure caption with the key tiling explanation
   (`b3fd576c2ae5`, p. 2) at rank 4. Its first two results described
   FlashAttention-2 shared-memory changes. The miss costs chunk recall even
   though another passage provides part of the same answer.
6. **Q43, Whisper versus wav2vec 2.0.** Hybrid's first hit
   (`1ea0dc6f49d7`, Whisper p. 10) is a bibliography entry containing the
   other model's name and self-supervision terminology. Its top five had
   direct evidence from the Whisper side only. Reranking put the wav2vec 2.0
   masked-latent contrastive-learning conclusion (`dbeab9afc82a`, p. 8) at
   rank 2 and Whisper's weak-supervision discussion (`36e18a899e1d`, p. 9)
   at rank 4. The two passages together cover both sides. The same paper's
   comparative WER table was reranked first and supplies no training
   supervision comparison.

### Grade-2 evidence missed by all methods

All four rankings omitted **18 judged grade-2 question–chunk pairs** across
15 questions, representing 15 distinct chunks. These pairs come from independent
references outside the retrieval pool. Their full IDs are available under
`grade_2_missed_by_all` in the [results JSON](../eval/v2/results.json).

| Question | Missed passage prefix and source |
| --- | --- |
| Q03 | `0fd4561dd294` (flashattention.pdf, p. 4) |
| Q08 | `93af9ce7d1a9` (attention_is_all_you_need.pdf, p. 5) |
| Q09 | `0fd4561dd294` (flashattention.pdf, p. 4) |
| Q10 | `242a811e9b1d` (flashattention.pdf, p. 5) |
| Q11 | `2408e83902c0` (flashattention2.pdf, p. 5), `81a83de75dd1` (flashattention2.pdf, p. 5) |
| Q15 | `4394eab264ea` (pagedattention.pdf, p. 7) |
| Q18 | `0fd4561dd294` (flashattention.pdf, p. 4) |
| Q21 | `8a37f81ce1e1` (rag_2020.pdf, p. 2), `fd7fe01f0c7e` (rag_2020.pdf, p. 3) |
| Q28 | `e0050053a2f0` (lost_in_the_middle.pdf, p. 2) |
| Q34 | `450a044f81e6` (whisper.pdf, p. 1) |
| Q36 | `f33763a85091` (whisper.pdf, p. 3) |
| Q38 | `a84f64b500a7` (wav2vec2.pdf, p. 3) |
| Q39 | `9574b2e1f569` (wav2vec2.pdf, p. 1) |
| Q41 | `6724c2025666` (conformer.pdf, p. 2) |
| Q43 | `39830a82ceb0` (whisper.pdf, p. 2), `9574b2e1f569` (wav2vec2.pdf, p. 1) |

## Limitations

Pooled judgments can omit relevant passages outside the four rankings and
the independent references. Pooled recall can therefore overstate corpus-wide
recall. Overlapping chunks can earn multiple gains for restating one fact.
A grade-2 chunk can cover only one side of a comparison, so hit rates alone
cannot describe coverage of the complete answer. The same ten papers support
many questions, PDF extraction sometimes splits equations or statements, and
the judgments have yet to receive a human audit.

## Reproduce

From the repository root, evaluate the four tracked inputs using Python 3.11.
The second command additionally checks the original local index bytes.

```sh
uv run python scripts/evaluate_retrieval_v2.py --out eval/v2/results.json
uv run python scripts/evaluate_retrieval_v2.py --index data/index_v2 --out /tmp/s5e-index-checked.json
cmp eval/v2/results.json /tmp/s5e-index-checked.json
```

For passage analysis, consult the local blinded exports under
`data/eval_v2/blind/` or locate chunk IDs in the validated index's
`chunks.jsonl`. The [original five-question benchmark](evaluation.md) remains
independently reproducible with its original evaluator and judgments.
