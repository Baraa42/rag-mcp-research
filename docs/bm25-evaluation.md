# BM25 versus FAISS: four-query comparison

Measured 2026-11-19 on Apple M5 Pro CPU. Both methods use the identical 419 saved Chunk records from the three papers; FAISS uses the existing persisted embeddings. BM25 parameters: k1=1.5, b=0.75, rank-bm25 0.2.2 default epsilon=0.25. No stemming, lemmatization, or stopword removal. The vector results are freshly queried, not copied from an earlier report.

**BM25 and cosine scores have different scales and are not directly comparable.** These are manually inspected examples, not a labeled benchmark or evidence of general superiority.

## Performance

Construction: 30 timed builds after one warm-up; median **7.823 ms**, min **7.614 ms**, max **8.198 ms**. Includes chunk validation, tokenization, and BM25 statistics; excludes persisted-index loading and Python imports.

Each query: 200 repetitions after one warm-up. Includes query tokenization, matching, scoring, sorting, and SearchResult construction. Timings were collected without allocation tracing. Model loading/inference is not involved in BM25.

| Query | Median ms | Min ms | Max ms |
| --- | ---: | ---: | ---: |
| 1 | 0.276 | 0.260 | 0.357 |
| 2 | 0.212 | 0.202 | 0.297 |
| 3 | 0.302 | 0.287 | 0.394 |
| 4 | 0.246 | 0.235 | 0.311 |

A separate tracemalloc run measured **2,558,731 bytes** retained and **3,736,243 bytes** peak incremental traced allocation during BM25 construction. Existing chunks were already allocated and are shared. These figures exclude interpreter/import overhead, source-chunk storage, persisted FAISS loading, native allocations not traced by Python, and embedding model memory; they are not process RSS.

## Results

The following tables show all 40 results: top five for each question under each method. Excerpts are the first 260 characters with whitespace collapsed and nonprinting characters escaped; ellipses mark truncation. Source text and IDs are not modified in the index. All pages are physical PDF page numbers.

### 1. What is the computational complexity of self-attention?

BM25 rank 1 is an attention-visualization fragment with repeated common words, not a complexity explanation. BM25 ranks 2–5 contain useful complexity discussion/table context. FAISS ranks 1 and 3 directly state quadratic complexity; its rank 4 discusses IO complexity. No stopwords were removed, so generic question terms can contribute to noisy BM25 matches.

**BM25**

| Rank | Paper | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `attention_is_all_you_need.pdf` | 15 | 14.743746 | Input-Input Layer5 The Law will never be perfect , but its application should be just - this is what we are missing , in my opinion . &lt;EOS&gt; &lt;pad&gt; The Law will never be perfect , but its application should be just - this is what we are missing , in my opinion .… |
| 2 | `attention_is_all_you_need.pdf` | 6 | 14.543732 | -range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types. As noted in Table 1, a self-attention layer connects all positions with a constant number of s… |
| 3 | `attention_is_all_you_need.pdf` | 6 | 13.396904 | Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the representation dimension, k is the kernel size of convolutions and r the size of the neighborhood in r… |
| 4 | `attention_is_all_you_need.pdf` | 7 | 12.976347 | k) convolutional layers in the case of contiguous kernels, or O(logk(n)) in the case of dilated convolutions [18], increasing the length of the longest paths between any two positions in the network. Convolutional layers are generally more expensive than recur… |
| 5 | `attention_is_all_you_need.pdf` | 6 | 12.158000 | a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata. One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as meas… |

**FAISS cosine**

| Rank | Paper | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `flashattention.pdf` | 1 | 0.592781 | FlashAttention: Fast and Memory-Eﬃcient Exact Attention with IO-Awareness Tri Daoy, Daniel Y. Fuy, Stefano Ermony, Atri Rudraz, and Christopher Réy yDepartment of Computer Science, Stanford University zDepartment of Computer Science and Engineering, University… |
| 2 | `attention_is_all_you_need.pdf` | 6 | 0.588325 | a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata. One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as meas… |
| 3 | `attention_is_all_you_need.pdf` | 6 | 0.583313 | Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the representation dimension, k is the kernel size of convolutions and r the size of the neighborhood in r… |
| 4 | `flashattention.pdf` | 24 | 0.578041 | and they start out being in HBM, so if the algorithm computes exact attention it must incur at least Ω„𝑁𝑑” HBM accesses. This is a contradiction. \u0003 Proof of Theorem 5. The IO complexity of the attention backward is very similar to the IO complexity of the atte… |
| 5 | `attention_is_all_you_need.pdf` | 6 | 0.576004 | -range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types. As noted in Table 1, a self-attention layer connects all positions with a constant number of s… |

### 2. How does FlashAttention reduce memory transfers?

BM25 rank 1 directly explains tiling to reduce reads/writes between HBM and SRAM; rank 2 is also relevant. BM25 rank 3 is an unrelated RAG passage. FAISS keeps all five hits in FlashAttention, but its top result discusses accesses rather than the tiling mechanism. These methods have different strengths on this example.

**BM25**

| Rank | Paper | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `flashattention.pdf` | 1 | 8.199309 | problem by trading oﬀ model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms IO- aware—accounting for reads and writes between levels of GPU memory. We propo… |
| 2 | `flashattention.pdf` | 22 | 6.983434 | B.5 Comparison with Rabe and Staats [66] We describe here some similarities and diﬀerences between our FlashAttention algorithm and the algorithm of Rabe and Staats [66]. Conceptually, both FlashAttention and Rabe and Staats [66] operate on blocks of the atten… |
| 3 | `rag_2020.pdf` | 6 | 6.793812 | The Sun Also Rises" indicating the title "The Sun Also Rises" is stored in BART’s parameters. Similarly, BART will complete the partial decoding "The Sun Also Rises" is a novel by this author of "A with "The Sun Also Rises" is a novel by this author of "A Fare… |
| 4 | `flashattention.pdf` | 1 | 6.083321 | FlashAttention: Fast and Memory-Eﬃcient Exact Attention with IO-Awareness Tri Daoy, Daniel Y. Fuy, Stefano Ermony, Atri Rudraz, and Christopher Réy yDepartment of Computer Science, Stanford University zDepartment of Computer Science and Engineering, University… |
| 5 | `flashattention.pdf` | 31 | 6.062949 | For each baseline, we increase sequence length until it runs out of memory on the GPU, except for the following exceptions: The Megatron implementation does not support sequence lengths longer than 2048. Block-Sparse (OpenAI) does not support sequence lengths … |

**FAISS cosine**

| Rank | Paper | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `flashattention.pdf` | 22 | 0.712328 | ) while FlashAttention focuses on reducing memory accesses (the number of memory reads/writes). As mentioned in Section 2, the amount of memory access is the primary determining factor of runtime. Reducing memory accesses also necessarily reduces the total amo… |
| 2 | `flashattention.pdf` | 10 | 0.675855 | s begin to cross over with FlashAttention at sequences between 512 and 1024. On the other hand, block-sparse FlashAttention is faster than all implementations of exact, sparse, and approximate attention that we know of, across all sequence lengths. Memory Foot… |
| 3 | `flashattention.pdf` | 22 | 0.613865 | B.5 Comparison with Rabe and Staats [66] We describe here some similarities and diﬀerences between our FlashAttention algorithm and the algorithm of Rabe and Staats [66]. Conceptually, both FlashAttention and Rabe and Staats [66] operate on blocks of the atten… |
| 4 | `flashattention.pdf` | 2 | 0.601980 | that is 2-4\u0002 faster than even FlashAttention, scaling up to sequence length of 64k. We prove that block-sparse FlashAttention has better IO complexity than FlashAttention by a factor proportional to the sparsity ratio. We discuss further extensions to other op… |
| 5 | `flashattention.pdf` | 2 | 0.601317 | FlashAttention Memory Hierarchy with Bandwidth & Memory Size Attention on GPT-2 FlashAttentionPyTorch Time (ms) Matmul Mask Softmax Dropout Matmul Fused Kernel Q: N x d V: N X d KT: d x N QKT: N x N sm(QKT)V: N x d Outer Loop Copy Block to SRAM Copy Outer Loop… |

### 3. How does RAG combine parametric and non-parametric memory?

BM25 ranks 1 and 2 directly describe the parametric seq2seq model plus non-parametric Wikipedia index; rank 3 gives a concrete example of their interaction. FAISS ranks the direct architecture explanation third, below token marginalization and introductory discussion. Both methods retrieve relevant RAG material.

**BM25**

| Rank | Paper | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `rag_2020.pdf` | 2 | 24.240438 | non-parametric memory to the “workhorse of NLP,” i.e. sequence-to-sequence (seq2seq) models. We endow pre-trained, parametric-memory generation models with a non-parametric memory through a general-purpose ﬁne-tuning approach which we refer to as retrieval-aug… |
| 2 | `rag_2020.pdf` | 1 | 24.073023 | ability to access and precisely manipulate knowl- edge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-speciﬁc architectures. Additionally, providing provenance for their decisions and updating their world knowledge… |
| 3 | `rag_2020.pdf` | 6 | 23.805355 | The Sun Also Rises" indicating the title "The Sun Also Rises" is stored in BART’s parameters. Similarly, BART will complete the partial decoding "The Sun Also Rises" is a novel by this author of "A with "The Sun Also Rises" is a novel by this author of "A Fare… |
| 4 | `rag_2020.pdf` | 9 | 23.721329 | ﬁnding RAG more factual and speciﬁc. We conducted an thorough investigation of the learned retrieval component, validating its effectiveness, and we illustrated how the retrieval index can be hot-swapped to update the model without requiring any retraining. In… |
| 5 | `rag_2020.pdf` | 19 | 20.402689 | 5 model with the closest number of parameters to our models is T5-large (770M parameters), which achieves a score of 28.9 EM on Natural Questions [52], substantially below the 44.5 that RAG-Sequence achieves, indicating that hybrid parametric/non- parametric m… |

**FAISS cosine**

| Rank | Paper | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `rag_2020.pdf` | 3 | 0.516974 | ·\|x)) pη(z\|x) N∏ i pθ(yi\|x,z,y1:i−1) RAG-Token Model In the RAG-Token model we can draw a different latent document for each target token and marginalize accordingly. This allows the generator to choose content from several documents when producing an answer. … |
| 2 | `rag_2020.pdf` | 2 | 0.514233 | architectures to enrich systems with non-parametric memory which are trained from scratch for speciﬁc tasks, e.g. memory networks [64, 55], stack- augmented networks [25] and memory layers [30]. In contrast, we explore a setting where both parametric and non-p… |
| 3 | `rag_2020.pdf` | 2 | 0.504367 | non-parametric memory to the “workhorse of NLP,” i.e. sequence-to-sequence (seq2seq) models. We endow pre-trained, parametric-memory generation models with a non-parametric memory through a general-purpose ﬁne-tuning approach which we refer to as retrieval-aug… |
| 4 | `rag_2020.pdf` | 9 | 0.499228 | ﬁnding RAG more factual and speciﬁc. We conducted an thorough investigation of the learned retrieval component, validating its effectiveness, and we illustrated how the retrieval index can be hot-swapped to update the model without requiring any retraining. In… |
| 5 | `rag_2020.pdf` | 6 | 0.489177 | The Sun Also Rises" indicating the title "The Sun Also Rises" is stored in BART’s parameters. Similarly, BART will complete the partial decoding "The Sun Also Rises" is a novel by this author of "A with "The Sun Also Rises" is a novel by this author of "A Fare… |

### 4. Why does the Transformer use positional encoding?

Positional-encoding retrieval improved for this question: BM25 rank 1 is section 3.5 on page 6, explaining the need to inject token order because the model has no recurrence or convolution. Ranks 2–4 discuss learned/sinusoidal encodings. FAISS ranks a bibliography fragment first and does not retrieve the direct explanation in its top five; rank 5 merely mentions dropout on positional encodings.

**BM25**

| Rank | Paper | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `attention_is_all_you_need.pdf` | 6 | 15.931601 | n/r) 3.5 Positional Encoding Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To t… |
| 2 | `attention_is_all_you_need.pdf` | 9 | 12.388983 | attention key size dk hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models… |
| 3 | `attention_is_all_you_need.pdf` | 6 | 12.128041 | Epos. We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence … |
| 4 | `attention_is_all_you_need.pdf` | 6 | 11.382642 | e functions of different frequencies: P E(pos,2i) = sin(pos/100002i/dmodel ) P E(pos,2i+1) = cos(pos/100002i/dmodel ) where pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelength… |
| 5 | `attention_is_all_you_need.pdf` | 8 | 10.634578 | of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of Pdrop = 0.1. Labe… |

**FAISS cosine**

| Rank | Paper | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `flashattention.pdf` | 16 | 0.476756 | , Mohammad Shoeybi, Tom Goldstein, Anima Anandkumar, and Bryan Catanzaro. Long-short transformer: Eﬃcient transformers for language and vision. Advances in Neural Information Processing Systems, 34, 2021. 16 |
| 2 | `attention_is_all_you_need.pdf` | 3 | 0.474649 | Figure 1: The Transformer - model architecture. The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively… |
| 3 | `flashattention.pdf` | 17 | 0.461168 | it performs almost as well as the (dense) FlashAttention on the Long-range Arena tasks. Eﬃcient Transformer. Transformer-based models have become the most widely-used architecture in natural language processing [22] and computer vision [24, 91]. However, one o… |
| 4 | `attention_is_all_you_need.pdf` | 5 | 0.459070 | ×dmodel . In this work we employ h = 8 parallel attention layers, or heads. For each of these we use dk = dv = dmodel/h = 64. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensio… |
| 5 | `attention_is_all_you_need.pdf` | 8 | 0.439497 | of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of Pdrop = 0.1. Labe… |
