# Cross-encoder reranking: four-query comparison

Measured 2026-11-20 on Apple M5 Pro CPU, macOS 26.5.2 arm64, six PyTorch threads. The same 419 persisted chunks are used for BM25, FAISS cosine, hybrid RRF, and hybrid plus cross-encoder. No document embeddings were recomputed.

Model: `cross-encoder/ms-marco-MiniLM-L6-v2` at verified Hugging Face revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Batch size **16**, **20 hybrid candidates** per query, pair-token limit **512**, explicit identity activation for raw relevance logits. These scores are not probabilities and cannot be directly compared across methods.

## Timing methodology

Persisted-index loading, BM25 construction, imports, downloads, and model initialization were completed before query timing. One full-pipeline warm-up per question preceded **10 timed repetitions per question**. Candidate retrieval includes query embedding, BM25, FAISS, and RRF. Cross-encoder timing instruments `predict()` itself, including pair tokenization, batching, model inference, and NumPy conversion. Warm end-to-end time measures hybrid retrieval plus the full rerank wrapper, including validation, sorting, and result construction; it excludes CLI startup, disk loading, and printing. Medians need not sum exactly.

- Initial model-file download/cache preparation: **25.096 s**.
- Embedder initialization from existing cache: **1.416 s**.
- Cross-encoder initialization after prefetch: **2.732 s**.

| Question | Retrieval median ms | Predict median ms | Full rerank median ms | Warm end-to-end median ms | End-to-end min–max ms |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 3.619 | 86.860 | 86.904 | 90.536 | 88.223–92.664 |
| 2 | 3.515 | 88.639 | 88.697 | 92.196 | 88.430–94.754 |
| 3 | 3.838 | 94.775 | 94.830 | 98.613 | 92.581–107.552 |
| 4 | 3.722 | 94.501 | 94.557 | 98.218 | 90.492–113.107 |

These are measurements on one machine, not service-level guarantees. Each timed call processed 20 candidates. Cross-encoder inference dominates query latency. The CLI loads both models per invocation, so its wall time is longer than these warm measurements.

## Top-five comparison

All 80 results follow. Excerpts are the first 160 characters with whitespace collapsed, HTML-special characters escaped, and nonprinting characters displayed as escapes; ellipses mark truncation. Pages are physical PDF pages. Manual observations below are not ground-truth labels or aggregate retrieval metrics. Four queries do not establish overall improvement.

### 1. What is the computational complexity of self-attention?

BM25 puts a repeated-text figure fragment first; FAISS puts the quadratic-complexity abstract first; hybrid puts the per-layer complexity table first. The cross-encoder instead ranks a self-attention/recurrent-layer comparison first and moves the table to rank 4. It also promotes a Linformer bibliography fragment to rank 3. This is mixed quality, not an unambiguous improvement.

**BM25**

| Rank | Source | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `attention_is_all_you_need.pdf` | 15 | 14.743746 | Input-Input Layer5 The Law will never be perfect , but its application should be just - this is what we are missing , in my opinion . &lt;EOS&gt; &lt;pad&gt; The Law will n… |
| 2 | `attention_is_all_you_need.pdf` | 6 | 14.543732 | -range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer t… |
| 3 | `attention_is_all_you_need.pdf` | 6 | 13.396904 | Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the re… |
| 4 | `attention_is_all_you_need.pdf` | 7 | 12.976347 | k) convolutional layers in the case of contiguous kernels, or O(logk(n)) in the case of dilated convolutions [18], increasing the length of the longest paths be… |
| 5 | `attention_is_all_you_need.pdf` | 6 | 12.158000 | a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata. One is the total comput… |

**FAISS cosine**

| Rank | Source | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `flashattention.pdf` | 1 | 0.592781 | FlashAttention: Fast and Memory-Eﬃcient Exact Attention with IO-Awareness Tri Daoy, Daniel Y. Fuy, Stefano Ermony, Atri Rudraz, and Christopher Réy yDepartment … |
| 2 | `attention_is_all_you_need.pdf` | 6 | 0.588325 | a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata. One is the total comput… |
| 3 | `attention_is_all_you_need.pdf` | 6 | 0.583313 | Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the re… |
| 4 | `flashattention.pdf` | 24 | 0.578041 | and they start out being in HBM, so if the algorithm computes exact attention it must incur at least Ω„𝑁𝑑” HBM accesses. This is a contradiction. \u0003 Proof of The… |
| 5 | `attention_is_all_you_need.pdf` | 6 | 0.576004 | -range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer t… |

**Hybrid RRF**

| Rank | Source | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `attention_is_all_you_need.pdf` | 6 | 0.031746 | Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the re… |
| 2 | `attention_is_all_you_need.pdf` | 6 | 0.031514 | a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata. One is the total comput… |
| 3 | `attention_is_all_you_need.pdf` | 6 | 0.031514 | -range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer t… |
| 4 | `flashattention.pdf` | 1 | 0.031319 | FlashAttention: Fast and Memory-Eﬃcient Exact Attention with IO-Awareness Tri Daoy, Daniel Y. Fuy, Stefano Ermony, Atri Rudraz, and Christopher Réy yDepartment … |
| 5 | `flashattention.pdf` | 24 | 0.029911 | and they start out being in HBM, so if the algorithm computes exact attention it must incur at least Ω„𝑁𝑑” HBM accesses. This is a contradiction. \u0003 Proof of The… |

**Hybrid + cross-encoder (raw logits)**

| Rank | Source | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `attention_is_all_you_need.pdf` | 6 | 4.774297 | -range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer t… |
| 2 | `attention_is_all_you_need.pdf` | 6 | 4.137313 | a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata. One is the total comput… |
| 3 | `flashattention.pdf` | 15 | 3.094463 | , Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020. [85] Samuel Williams, Andrew Wat… |
| 4 | `attention_is_all_you_need.pdf` | 6 | 2.622953 | Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the re… |
| 5 | `attention_is_all_you_need.pdf` | 7 | 2.380284 | length n is smaller than the representation dimensionality d, which is most often the case with sentence representations used by state-of-the-art models in mach… |

### 2. How does FlashAttention reduce memory transfers?

BM25 puts the IO-aware tiling explanation first but includes an unrelated RAG hit; FAISS leads with memory-access discussion. Hybrid leads with the tiling/recomputation comparison. Reranking places the memory-access discussion first, the HBM/SRAM tiling explanation second, and the comparison third. All reranked top-five sources are FlashAttention, but some excerpts remain less direct or contain PDF artifacts.

**BM25**

| Rank | Source | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `flashattention.pdf` | 1 | 8.199309 | problem by trading oﬀ model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making a… |
| 2 | `flashattention.pdf` | 22 | 6.983434 | B.5 Comparison with Rabe and Staats [66] We describe here some similarities and diﬀerences between our FlashAttention algorithm and the algorithm of Rabe and St… |
| 3 | `rag_2020.pdf` | 6 | 6.793812 | The Sun Also Rises" indicating the title "The Sun Also Rises" is stored in BART’s parameters. Similarly, BART will complete the partial decoding "The Sun Also R… |
| 4 | `flashattention.pdf` | 1 | 6.083321 | FlashAttention: Fast and Memory-Eﬃcient Exact Attention with IO-Awareness Tri Daoy, Daniel Y. Fuy, Stefano Ermony, Atri Rudraz, and Christopher Réy yDepartment … |
| 5 | `flashattention.pdf` | 31 | 6.062949 | For each baseline, we increase sequence length until it runs out of memory on the GPU, except for the following exceptions: The Megatron implementation does not… |

**FAISS cosine**

| Rank | Source | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `flashattention.pdf` | 22 | 0.712328 | ) while FlashAttention focuses on reducing memory accesses (the number of memory reads/writes). As mentioned in Section 2, the amount of memory access is the pr… |
| 2 | `flashattention.pdf` | 10 | 0.675855 | s begin to cross over with FlashAttention at sequences between 512 and 1024. On the other hand, block-sparse FlashAttention is faster than all implementations o… |
| 3 | `flashattention.pdf` | 22 | 0.613865 | B.5 Comparison with Rabe and Staats [66] We describe here some similarities and diﬀerences between our FlashAttention algorithm and the algorithm of Rabe and St… |
| 4 | `flashattention.pdf` | 2 | 0.601980 | that is 2-4\u0002 faster than even FlashAttention, scaling up to sequence length of 64k. We prove that block-sparse FlashAttention has better IO complexity than Flas… |
| 5 | `flashattention.pdf` | 2 | 0.601317 | FlashAttention Memory Hierarchy with Bandwidth & Memory Size Attention on GPT-2 FlashAttentionPyTorch Time (ms) Matmul Mask Softmax Dropout Matmul Fused Kernel … |

**Hybrid RRF**

| Rank | Source | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `flashattention.pdf` | 22 | 0.032002 | B.5 Comparison with Rabe and Staats [66] We describe here some similarities and diﬀerences between our FlashAttention algorithm and the algorithm of Rabe and St… |
| 2 | `flashattention.pdf` | 2 | 0.029631 | s (&gt;1 TB) 0 5 10 15 Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large 𝑁\u0002𝑁 attention matrix (dotted box) on (relatively) slow GP… |
| 3 | `flashattention.pdf` | 22 | 0.016393 | ) while FlashAttention focuses on reducing memory accesses (the number of memory reads/writes). As mentioned in Section 2, the amount of memory access is the pr… |
| 4 | `flashattention.pdf` | 1 | 0.016393 | problem by trading oﬀ model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making a… |
| 5 | `flashattention.pdf` | 10 | 0.016129 | s begin to cross over with FlashAttention at sequences between 512 and 1024. On the other hand, block-sparse FlashAttention is faster than all implementations o… |

**Hybrid + cross-encoder (raw logits)**

| Rank | Source | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `flashattention.pdf` | 22 | 6.720028 | ) while FlashAttention focuses on reducing memory accesses (the number of memory reads/writes). As mentioned in Section 2, the amount of memory access is the pr… |
| 2 | `flashattention.pdf` | 1 | 4.106818 | problem by trading oﬀ model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making a… |
| 3 | `flashattention.pdf` | 22 | 3.573207 | B.5 Comparison with Rabe and Staats [66] We describe here some similarities and diﬀerences between our FlashAttention algorithm and the algorithm of Rabe and St… |
| 4 | `flashattention.pdf` | 1 | 1.214695 | FlashAttention: Fast and Memory-Eﬃcient Exact Attention with IO-Awareness Tri Daoy, Daniel Y. Fuy, Stefano Ermony, Atri Rudraz, and Christopher Réy yDepartment … |
| 5 | `flashattention.pdf` | 2 | 1.176761 | s (&gt;1 TB) 0 5 10 15 Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large 𝑁\u0002𝑁 attention matrix (dotted box) on (relatively) slow GP… |

### 3. How does RAG combine parametric and non-parametric memory?

BM25, hybrid, and reranked results all put the direct parametric seq2seq/non-parametric Wikipedia-index explanation first; FAISS ranks it third. Reranking adds introductory abstract material near the top. There is considerable overlap between passages; the result is not evidence of general improvement.

**BM25**

| Rank | Source | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `rag_2020.pdf` | 2 | 24.240438 | non-parametric memory to the “workhorse of NLP,” i.e. sequence-to-sequence (seq2seq) models. We endow pre-trained, parametric-memory generation models with a no… |
| 2 | `rag_2020.pdf` | 1 | 24.073023 | ability to access and precisely manipulate knowl- edge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-speciﬁc arch… |
| 3 | `rag_2020.pdf` | 6 | 23.805355 | The Sun Also Rises" indicating the title "The Sun Also Rises" is stored in BART’s parameters. Similarly, BART will complete the partial decoding "The Sun Also R… |
| 4 | `rag_2020.pdf` | 9 | 23.721329 | ﬁnding RAG more factual and speciﬁc. We conducted an thorough investigation of the learned retrieval component, validating its effectiveness, and we illustrated… |
| 5 | `rag_2020.pdf` | 19 | 20.402689 | 5 model with the closest number of parameters to our models is T5-large (770M parameters), which achieves a score of 28.9 EM on Natural Questions [52], substant… |

**FAISS cosine**

| Rank | Source | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `rag_2020.pdf` | 3 | 0.516974 | ·\|x)) pη(z\|x) N∏ i pθ(yi\|x,z,y1:i−1) RAG-Token Model In the RAG-Token model we can draw a different latent document for each target token and marginalize accord… |
| 2 | `rag_2020.pdf` | 2 | 0.514233 | architectures to enrich systems with non-parametric memory which are trained from scratch for speciﬁc tasks, e.g. memory networks [64, 55], stack- augmented net… |
| 3 | `rag_2020.pdf` | 2 | 0.504367 | non-parametric memory to the “workhorse of NLP,” i.e. sequence-to-sequence (seq2seq) models. We endow pre-trained, parametric-memory generation models with a no… |
| 4 | `rag_2020.pdf` | 9 | 0.499228 | ﬁnding RAG more factual and speciﬁc. We conducted an thorough investigation of the learned retrieval component, validating its effectiveness, and we illustrated… |
| 5 | `rag_2020.pdf` | 6 | 0.489177 | The Sun Also Rises" indicating the title "The Sun Also Rises" is stored in BART’s parameters. Similarly, BART will complete the partial decoding "The Sun Also R… |

**Hybrid RRF**

| Rank | Source | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `rag_2020.pdf` | 2 | 0.032266 | non-parametric memory to the “workhorse of NLP,” i.e. sequence-to-sequence (seq2seq) models. We endow pre-trained, parametric-memory generation models with a no… |
| 2 | `rag_2020.pdf` | 2 | 0.031281 | architectures to enrich systems with non-parametric memory which are trained from scratch for speciﬁc tasks, e.g. memory networks [64, 55], stack- augmented net… |
| 3 | `rag_2020.pdf` | 6 | 0.031258 | The Sun Also Rises" indicating the title "The Sun Also Rises" is stored in BART’s parameters. Similarly, BART will complete the partial decoding "The Sun Also R… |
| 4 | `rag_2020.pdf` | 9 | 0.031250 | ﬁnding RAG more factual and speciﬁc. We conducted an thorough investigation of the learned retrieval component, validating its effectiveness, and we illustrated… |
| 5 | `rag_2020.pdf` | 19 | 0.030310 | 5 model with the closest number of parameters to our models is T5-large (770M parameters), which achieves a score of 28.9 EM on Natural Questions [52], substant… |

**Hybrid + cross-encoder (raw logits)**

| Rank | Source | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `rag_2020.pdf` | 2 | 7.064810 | non-parametric memory to the “workhorse of NLP,” i.e. sequence-to-sequence (seq2seq) models. We endow pre-trained, parametric-memory generation models with a no… |
| 2 | `rag_2020.pdf` | 2 | 6.502433 | architectures to enrich systems with non-parametric memory which are trained from scratch for speciﬁc tasks, e.g. memory networks [64, 55], stack- augmented net… |
| 3 | `rag_2020.pdf` | 1 | 6.496347 | ability to access and precisely manipulate knowl- edge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-speciﬁc arch… |
| 4 | `rag_2020.pdf` | 9 | 5.627413 | ﬁnding RAG more factual and speciﬁc. We conducted an thorough investigation of the learned retrieval component, validating its effectiveness, and we illustrated… |
| 5 | `rag_2020.pdf` | 1 | 5.478311 | non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever. We com- pare two RAG formulations, one which condition… |

### 4. Why does the Transformer use positional encoding?

Yes: the direct section 3.5 explanation on Attention page 6 moves from hybrid rank 2 to reranked rank 1, score 3.436294. It was BM25 rank 1 and absent from FAISS top five. The less-relevant page-8 dropout passage moves from hybrid rank 1 to reranked rank 3, score 1.625410. Other high-ranked passages are contextual rather than direct answers; reranking does not remove every weak result.

**BM25**

| Rank | Source | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `attention_is_all_you_need.pdf` | 6 | 15.931601 | n/r) 3.5 Positional Encoding Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must… |
| 2 | `attention_is_all_you_need.pdf` | 9 | 12.388983 | attention key size dk hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than do… |
| 3 | `attention_is_all_you_need.pdf` | 6 | 12.128041 | Epos. We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 … |
| 4 | `attention_is_all_you_need.pdf` | 6 | 11.382642 | e functions of different frequencies: P E(pos,2i) = sin(pos/100002i/dmodel ) P E(pos,2i+1) = cos(pos/100002i/dmodel ) where pos is the position and i is the dim… |
| 5 | `attention_is_all_you_need.pdf` | 8 | 10.634578 | of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional enco… |

**FAISS cosine**

| Rank | Source | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `flashattention.pdf` | 16 | 0.476756 | , Mohammad Shoeybi, Tom Goldstein, Anima Anandkumar, and Bryan Catanzaro. Long-short transformer: Eﬃcient transformers for language and vision. Advances in Neur… |
| 2 | `attention_is_all_you_need.pdf` | 3 | 0.474649 | Figure 1: The Transformer - model architecture. The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected l… |
| 3 | `flashattention.pdf` | 17 | 0.461168 | it performs almost as well as the (dense) FlashAttention on the Long-range Arena tasks. Eﬃcient Transformer. Transformer-based models have become the most widel… |
| 4 | `attention_is_all_you_need.pdf` | 5 | 0.459070 | ×dmodel . In this work we employ h = 8 parallel attention layers, or heads. For each of these we use dk = dv = dmodel/h = 64. Due to the reduced dimension of ea… |
| 5 | `attention_is_all_you_need.pdf` | 8 | 0.439497 | of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional enco… |

**Hybrid RRF**

| Rank | Source | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `attention_is_all_you_need.pdf` | 8 | 0.030769 | of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional enco… |
| 2 | `attention_is_all_you_need.pdf` | 6 | 0.030679 | n/r) 3.5 Positional Encoding Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must… |
| 3 | `flashattention.pdf` | 9 | 0.029418 | work, all transformer models have either run out of memory, or only achieved random performance [80]. There has been a search for alternative architectures that… |
| 4 | `attention_is_all_you_need.pdf` | 9 | 0.028992 | Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translat… |
| 5 | `attention_is_all_you_need.pdf` | 6 | 0.028531 | Epos. We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 … |

**Hybrid + cross-encoder (raw logits)**

| Rank | Source | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `attention_is_all_you_need.pdf` | 6 | 3.436294 | n/r) 3.5 Positional Encoding Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must… |
| 2 | `attention_is_all_you_need.pdf` | 2 | 1.910475 | the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and … |
| 3 | `attention_is_all_you_need.pdf` | 8 | 1.625410 | of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional enco… |
| 4 | `attention_is_all_you_need.pdf` | 5 | 1.481391 | ×dmodel . In this work we employ h = 8 parallel attention layers, or heads. For each of these we use dk = dv = dmodel/h = 64. Due to the reduced dimension of ea… |
| 5 | `attention_is_all_you_need.pdf` | 3 | 1.266150 | Figure 1: The Transformer - model architecture. The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected l… |

## Limitations

The reranker cannot introduce candidates absent from the first-stage pool. MS MARCO training does not guarantee scientific-domain relevance, and source layout artifacts and overlapping chunks persist. Pair inputs exceeding 512 tokens are truncated. The constructor loads a model even if a later rerank call has no candidates; only prediction is bypassed. Reuse long-lived instances for warm latency. This implementation provides neither calibrated confidence nor factual-answer verification.
