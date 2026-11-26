# S5G answer-quality assessment

This is an AI-assisted review of the 12 saved S5F responses. It has yet to receive
independent human verification. The review compared each response with the frozen
reference answer and all five exact passages in the local S5F assessment. Source
labels below refer to their position in the evidence for that question. No new
generation or judging API requests were made.
[Tracked responses and source IDs](../eval/v2/answer-results.json) ·
[Fixed question selection](../eval/v2/answer-questions.json) ·
[Frozen references](../eval/v2/metadata.json).

Correctness and completeness are judged against the question and reference.
Factual support is checked separately against the cited text. Citation validation
only confirms that labels resolve to supplied passages. For unsupported queries,
an exact insufficient-evidence response counts as an appropriate abstention.
An incomplete API response counts as an error.

## Six answerable questions

| Question | Correctness and completeness | Evidential support and failure analysis |
| --- | --- | --- |
| **Q06 · Why use multiple attention heads?** | **Correct and complete.** Explains the different subspaces and positions, and the effect of averaging in one head. | S2 directly supports the core explanation. S4 supports the effective-resolution point, S1 supports comparable computation with smaller heads, and S5 reports the 0.9 BLEU gap in the paper's ablation. That empirical comparison is specific to the reported setting. An answer was appropriate. |
| **Q13 · Split-K versus split-Q** | **Correct and complete.** Identifies what is partitioned, who can access each input, and why split-Q avoids the forward reduction. | S2 describes K/V partitioning and shared-memory communication. S1 describes Q partitioning and reduced shared-memory traffic. The communication claim applies to the forward pass. S3 mentions synchronization in the backward pass. An answer was appropriate. |
| **Q21 · RAG dense retriever architecture** | **Correct with minor omissions.** Names the DPR bi-encoder, separates query and document encoders, and explains dot-product scoring. It leaves out BERT-base specificity and the explicit top-k maximum inner-product search over the index. | S1 supports the dual-encoder architecture and the score formula. S4 supports the dense Wikipedia index and DPR identity. The delivered response covers the core architecture. An answer was appropriate. |
| **Q33 · Query before and after context** | **Correct with a minor omission.** Explains the contrast between substantial gains for key-value retrieval and little improvement in multi-document QA robustness. It omits the reference's near-perfect key-value result. | S2 supports the position-dependent QA result and S3 supports the large key-value gain. S3 ends before the more specific result, which the five supplied passages do not establish. The qualitative wording stays within the evidence. An answer was appropriate. |
| **Q35 · Whisper audio input representation** | **Incorrect for the question.** Treats a 30-second input segment as the acoustic representation. Omits the 80-channel log-magnitude Mel spectrogram, 16 kHz input, 25 ms window and 10 ms stride. | S1 supports segmentation and S3 supports 20 ms timestamp quantization. Those are true adjacent facts, yet neither establishes the acoustic features. The direct passage was reranked sixth and fell outside the five supplied passages. The generator then answered a neighboring question. Acknowledging insufficient evidence would have been appropriate. |
| **Q38 · wav2vec 2.0 contrastive objective** | **Correct at a high level, incomplete.** Identifies masked latents and jointly learned quantization. It omits the central prediction task of identifying the true quantized latent among distractors and the temperature-scaled cosine scoring. | S1 and S2 support the high-level summary. The five passages do not describe the target-versus-distractor comparison or loss formula. The answer appropriately signals that the formula is unavailable, but its caveat could also have named the missing training objective detail. S5E identified a direct method passage missed by all four top-20 lists. |

## Six expected-unanswerable questions

| Question | Outcome and completeness | Evidence and failure analysis |
| --- | --- | --- |
| **Q45 · Global workspace theory** | **Appropriate abstention.** The exact fallback fully conveys that the supplied evidence is insufficient. | The retrieved material concerns global audio context, language-model position effects and speech tasks. It supplies no account of conscious access. The response made no external factual claim and included no citations. |
| **Q46 · GPT-4 training electricity** | **Appropriate abstention.** The exact fallback avoids an unsupported numerical estimate. | The five passages discuss training duration or compute throughput for GPT-2 or other GPT-style workloads. They supply neither GPT-4 training energy nor the data needed to calculate it. No citations or factual claims were generated. |
| **Q47 · Scribe v2 training composition** | **Appropriate abstention.** The exact fallback answers within the evidence boundary. | Whisper and wav2vec 2.0 training data and processing details refer to other systems. No passage gives a Scribe v2 dataset breakdown. The response used no citations and made no substituted claim. |
| **Q48 · Whisper WER on the named Darija benchmark** | **Appropriate abstention.** The exact fallback avoids substituting a result from another benchmark. | The evidence includes Fleurs, Kincaid46 and WER normalization. It supplies no measurement for Moroccan Darija telephone conversations on the named Scribe v2 benchmark. No external metric was claimed. |
| **Q49 · GPT-5.2 attention layers** | **Appropriate abstention.** The exact fallback avoids inferring private architectural details from related models. | The retrieved passages cover the original Transformer and FlashAttention variants. They do not establish GPT-5.2's attention design. No citations or factual claims were generated. |
| **Q50 · GPT-5 on B200 block size** | **Incomplete-response error.** The application delivered no answer or citations. | S2 and S5 discuss block-size trade-offs for other workloads. S4 describes an A100 configuration. With no GPT-5/B200 measurement, abstention would have been appropriate. The retained partial text begins to acknowledge the gap. The provider returned `incomplete` at the 1,024-output-token cap, with 960 reasoning tokens reported. The service rejected the partial response. |

The six completed, citation-valid answers included two complete answers, two
correct answers with minor omissions, one incomplete answer and one answer
addressing the wrong property. Five of the six expected-unanswerable controls
produced appropriate abstentions. Q50 was a generation error. These counts
describe a single small, fixed sample. The evidence audit does not establish
population-level answer accuracy or abstention reliability.

The local `data/eval_v2/s5f-answer-run/assessment.json` contains the original
answers, exact retrieved and cited text, and reference answers. It remains
outside Git. [S5F setup, usage and latency](performance-v2.md).
