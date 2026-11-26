# S5D semantic relevance judgments

These labels are AI-generated semantic judgments. Human auditing remains
necessary before relying on them for retrieval comparisons.

## Method

The model assessed each question and passage in the 50 blinded JSON files
under `data/eval_v2/blind/`. It used the question and extracted passage text,
with source and page identifying the evidence. It had no retrieval ranks,
scores, method identities, reference answers, or expected-answerability
metadata in its grading input. Each record in [labels.json](labels.json)
contains the question ID, chunk ID, grade, and a short justification.

- **2** means direct support for the requested answer or a central component.
  For a cross-document question, a passage can directly support one side of
  the comparison.
- **1** means useful context or incomplete support.
- **0** means irrelevant or misleading for that question.

Validation confirmed exactly 50 questions and 1,665 unique question–chunk
pairs. Every blinded candidate has exactly one label. Every grade is an
integer from 0 to 2, and every record has a nonempty justification.
The counts cover the exported pool. Passages outside it remain unjudged.

## Grade distribution

| Question | Grade 0 | Grade 1 | Grade 2 | Total |
| --- | ---: | ---: | ---: | ---: |
| Q01 | 21 | 12 | 2 | 35 |
| Q02 | 16 | 12 | 5 | 33 |
| Q03 | 13 | 14 | 7 | 34 |
| Q04 | 4 | 10 | 13 | 27 |
| Q05 | 7 | 12 | 6 | 25 |
| Q06 | 20 | 8 | 7 | 35 |
| Q07 | 24 | 5 | 2 | 31 |
| Q08 | 25 | 10 | 4 | 39 |
| Q09 | 7 | 18 | 7 | 32 |
| Q10 | 19 | 9 | 9 | 37 |
| Q11 | 19 | 14 | 3 | 36 |
| Q12 | 23 | 6 | 3 | 32 |
| Q13 | 28 | 4 | 2 | 34 |
| Q14 | 0 | 18 | 10 | 28 |
| Q15 | 7 | 16 | 6 | 29 |
| Q16 | 3 | 16 | 9 | 28 |
| Q17 | 17 | 14 | 2 | 33 |
| Q18 | 8 | 14 | 18 | 40 |
| Q19 | 18 | 11 | 3 | 32 |
| Q20 | 18 | 15 | 2 | 35 |
| Q21 | 13 | 14 | 10 | 37 |
| Q22 | 21 | 11 | 1 | 33 |
| Q23 | 20 | 13 | 6 | 39 |
| Q24 | 7 | 12 | 12 | 31 |
| Q25 | 15 | 13 | 6 | 34 |
| Q26 | 19 | 7 | 3 | 29 |
| Q27 | 18 | 13 | 1 | 32 |
| Q28 | 2 | 12 | 15 | 29 |
| Q29 | 17 | 12 | 8 | 37 |
| Q30 | 8 | 13 | 8 | 29 |
| Q31 | 16 | 12 | 7 | 35 |
| Q32 | 21 | 10 | 3 | 34 |
| Q33 | 11 | 19 | 3 | 33 |
| Q34 | 23 | 7 | 3 | 33 |
| Q35 | 31 | 4 | 1 | 36 |
| Q36 | 25 | 8 | 4 | 37 |
| Q37 | 20 | 9 | 3 | 32 |
| Q38 | 17 | 9 | 5 | 31 |
| Q39 | 19 | 7 | 8 | 34 |
| Q40 | 13 | 10 | 6 | 29 |
| Q41 | 12 | 12 | 10 | 34 |
| Q42 | 11 | 8 | 9 | 28 |
| Q43 | 21 | 12 | 6 | 39 |
| Q44 | 6 | 19 | 2 | 27 |
| Q45 | 40 | 0 | 0 | 40 |
| Q46 | 35 | 0 | 0 | 35 |
| Q47 | 40 | 0 | 0 | 40 |
| Q48 | 17 | 17 | 0 | 34 |
| Q49 | 36 | 0 | 0 | 36 |
| Q50 | 24 | 9 | 0 | 33 |
| **Total** | **875** | **530** | **260** | **1,665** |

## Human audit shortlist

Notes prefixed with `Review:` flag uncertain judgments or cases where the
source context could change how a passage is graded. Chunk IDs below use
their first 12 characters. Consult the original PDF page for the Q20
inequalities, where the extracted wording appears to reverse row and column.

| Question | Chunk ID prefix | Source | Grade | Reason to review |
| --- | --- | --- | ---: | --- |
| Q01 | `9ef24abb4a74` | attention_is_all_you_need.pdf, p. 6 | 1 | identifies the architectural premise, but cuts off before explaining the token-order consequence. |
| Q02 | `629b8e2e72e6` | attention_is_all_you_need.pdf, p. 5 | 2 | the all-positions interaction directly establishes the quadratic number of pairs, although the per-pair dimension is omitted. |
| Q05 | `5dea07809384` | rag_2020.pdf, p. 7 | 1 | qualitative examples and different document posteriors are suggestive, but the excerpt does not independently identify the distinction. |
| Q20 | `5461e5af2f56` | flashattention2.pdf, p. 6 | 2 | explicitly describes masking only the boundary tile for square tiles, despite a potentially reversed inequality in the extracted prose. |
| Q20 | `520f1e8871c6` | flashattention2.pdf, p. 6 | 2 | explicitly skips wholly future tiles. The following row/column inequality in the same excerpt appears reversed. |
| Q50 | `d0b63074348e` | pagedattention.pdf, p. 12 | 1 | Gives 16 as the original vLLM default and reports workload-dependent results. The experiments do not establish an optimum for GPT-5 on a B200. |

### Out-of-corpus questions

No candidate for Q45–Q50 directly supported its requested answer. Q48
retains grade-1 context about Whisper evaluations on other datasets, and
Q50 retains grade-1 context about PagedAttention block-size trade-offs and
historical defaults.
