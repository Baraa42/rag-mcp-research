# RRF hybrid retrieval: four-query review

2026-11-19. All methods use the same 419 persisted chunks. Vector and BM25 top five are shown alongside hybrid top five. Hybrid combines the top 20 from each method with equal weights and rrf_k=60; all ranks are 1-based. Model settings, embeddings, FAISS index, chunking, and BM25 tokenization are unchanged.

This is a manual review of four examples, not a labeled evaluation or a claim of general improvement. Cosine, BM25, and RRF scores are method-specific and not directly comparable.

Excerpts show the first 240 characters with whitespace collapsed; ellipses indicate truncation. Control characters are escaped for readability. Physical PDF page numbers are shown. The tables below contain all 60 retrieved results.

## 1. What is the computational complexity of self-attention?

Hybrid puts the per-layer complexity table first (rank 3 in both component lists). The noisy BM25 rank-1 figure fragment is absent from hybrid top five. Other returned passages discuss complexity or IO, though not all directly answer the question.

### BM25

| Rank | Filename | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `attention_is_all_you_need.pdf` | 15 | 14.74374550 | Input-Input Layer5 The Law will never be perfect , but its application should be just - this is what we are missing , in my opinion . &lt;EOS&gt; &lt;pad&gt; The Law will never be perfect , but its application should be just - this is what we are missi… |
| 2 | `attention_is_all_you_need.pdf` | 6 | 14.54373178 | -range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types. As noted in Table 1, a self-attention layer connects all positions with a … |
| 3 | `attention_is_all_you_need.pdf` | 6 | 13.39690406 | Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the representation dimension, k is the kernel size of convolutions and r the size of t… |
| 4 | `attention_is_all_you_need.pdf` | 7 | 12.97634743 | k) convolutional layers in the case of contiguous kernels, or O(logk(n)) in the case of dilated convolutions [18], increasing the length of the longest paths between any two positions in the network. Convolutional layers are generally more … |
| 5 | `attention_is_all_you_need.pdf` | 6 | 12.15800029 | a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata. One is the total computational complexity per layer. Another is the amount of computation that can be p… |

### FAISS cosine

| Rank | Filename | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `flashattention.pdf` | 1 | 0.59278095 | FlashAttention: Fast and Memory-Eﬃcient Exact Attention with IO-Awareness Tri Daoy, Daniel Y. Fuy, Stefano Ermony, Atri Rudraz, and Christopher Réy yDepartment of Computer Science, Stanford University zDepartment of Computer Science and Eng… |
| 2 | `attention_is_all_you_need.pdf` | 6 | 0.58832514 | a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata. One is the total computational complexity per layer. Another is the amount of computation that can be p… |
| 3 | `attention_is_all_you_need.pdf` | 6 | 0.58331305 | Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the representation dimension, k is the kernel size of convolutions and r the size of t… |
| 4 | `flashattention.pdf` | 24 | 0.57804120 | and they start out being in HBM, so if the algorithm computes exact attention it must incur at least Ω„𝑁𝑑” HBM accesses. This is a contradiction. \u0003 Proof of Theorem 5. The IO complexity of the attention backward is very similar to the IO co… |
| 5 | `attention_is_all_you_need.pdf` | 6 | 0.57600439 | -range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types. As noted in Table 1, a self-attention layer connects all positions with a … |

### Hybrid RRF

| Rank | Filename | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `attention_is_all_you_need.pdf` | 6 | 0.03174603 | Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the representation dimension, k is the kernel size of convolutions and r the size of t… |
| 2 | `attention_is_all_you_need.pdf` | 6 | 0.03151365 | a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata. One is the total computational complexity per layer. Another is the amount of computation that can be p… |
| 3 | `attention_is_all_you_need.pdf` | 6 | 0.03151365 | -range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types. As noted in Table 1, a self-attention layer connects all positions with a … |
| 4 | `flashattention.pdf` | 1 | 0.03131882 | FlashAttention: Fast and Memory-Eﬃcient Exact Attention with IO-Awareness Tri Daoy, Daniel Y. Fuy, Stefano Ermony, Atri Rudraz, and Christopher Réy yDepartment of Computer Science, Stanford University zDepartment of Computer Science and Eng… |
| 5 | `flashattention.pdf` | 24 | 0.02991071 | and they start out being in HBM, so if the algorithm computes exact attention it must incur at least Ω„𝑁𝑑” HBM accesses. This is a contradiction. \u0003 Proof of Theorem 5. The IO complexity of the attention backward is very similar to the IO co… |

Hybrid candidate provenance (`—` means outside that method’s top 20):

| Hybrid rank | Chunk ID | Vector rank | BM25 rank |
| ---: | --- | ---: | ---: |
| 1 | `9ef24abb4a7408c0a44b6e63725a7eb7415f75dc4e3737f57ee0bc6f369ac5c3` | 3 | 3 |
| 2 | `755e622da4c22f588eaabc63c156edcd9ef9028bbe444af6dcaad67152570f0d` | 2 | 5 |
| 3 | `bb7536c18827fb2f46a688e80adf85a3566c7ede19b9fee5b460c3d9522549c0` | 5 | 2 |
| 4 | `b6c013339afdb8d8d9736a16a53aa75626bf848e8fdc3728781f2e72df29cd14` | 1 | 7 |
| 5 | `9f0c978fb81b9be88a9a072f8abe728d0c451507a1d60f479f0f2f1059b1ab50` | 4 | 10 |

## 2. How does FlashAttention reduce memory transfers?

Hybrid puts the tiling/recomputation comparison first and a tiling figure caption second. It also keeps the vector rank-1 memory-access passage and BM25 rank-1 IO-aware explanation. The unrelated RAG hit from BM25 top five is absent. Fusion does not reconstruct the PDF formatting artifacts.

### BM25

| Rank | Filename | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `flashattention.pdf` | 1 | 8.19930910 | problem by trading oﬀ model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms IO- aware—accounting for reads and writes between levels of … |
| 2 | `flashattention.pdf` | 22 | 6.98343354 | B.5 Comparison with Rabe and Staats [66] We describe here some similarities and diﬀerences between our FlashAttention algorithm and the algorithm of Rabe and Staats [66]. Conceptually, both FlashAttention and Rabe and Staats [66] operate on… |
| 3 | `rag_2020.pdf` | 6 | 6.79381152 | The Sun Also Rises" indicating the title "The Sun Also Rises" is stored in BART’s parameters. Similarly, BART will complete the partial decoding "The Sun Also Rises" is a novel by this author of "A with "The Sun Also Rises" is a novel by th… |
| 4 | `flashattention.pdf` | 1 | 6.08332078 | FlashAttention: Fast and Memory-Eﬃcient Exact Attention with IO-Awareness Tri Daoy, Daniel Y. Fuy, Stefano Ermony, Atri Rudraz, and Christopher Réy yDepartment of Computer Science, Stanford University zDepartment of Computer Science and Eng… |
| 5 | `flashattention.pdf` | 31 | 6.06294881 | For each baseline, we increase sequence length until it runs out of memory on the GPU, except for the following exceptions: The Megatron implementation does not support sequence lengths longer than 2048. Block-Sparse (OpenAI) does not suppo… |

### FAISS cosine

| Rank | Filename | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `flashattention.pdf` | 22 | 0.71232843 | ) while FlashAttention focuses on reducing memory accesses (the number of memory reads/writes). As mentioned in Section 2, the amount of memory access is the primary determining factor of runtime. Reducing memory accesses also necessarily r… |
| 2 | `flashattention.pdf` | 10 | 0.67585462 | s begin to cross over with FlashAttention at sequences between 512 and 1024. On the other hand, block-sparse FlashAttention is faster than all implementations of exact, sparse, and approximate attention that we know of, across all sequence … |
| 3 | `flashattention.pdf` | 22 | 0.61386526 | B.5 Comparison with Rabe and Staats [66] We describe here some similarities and diﬀerences between our FlashAttention algorithm and the algorithm of Rabe and Staats [66]. Conceptually, both FlashAttention and Rabe and Staats [66] operate on… |
| 4 | `flashattention.pdf` | 2 | 0.60197973 | that is 2-4\u0002 faster than even FlashAttention, scaling up to sequence length of 64k. We prove that block-sparse FlashAttention has better IO complexity than FlashAttention by a factor proportional to the sparsity ratio. We discuss further ex… |
| 5 | `flashattention.pdf` | 2 | 0.60131675 | FlashAttention Memory Hierarchy with Bandwidth & Memory Size Attention on GPT-2 FlashAttentionPyTorch Time (ms) Matmul Mask Softmax Dropout Matmul Fused Kernel Q: N x d V: N X d KT: d x N QKT: N x N sm(QKT)V: N x d Outer Loop Copy Block to … |

### Hybrid RRF

| Rank | Filename | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `flashattention.pdf` | 22 | 0.03200205 | B.5 Comparison with Rabe and Staats [66] We describe here some similarities and diﬀerences between our FlashAttention algorithm and the algorithm of Rabe and Staats [66]. Conceptually, both FlashAttention and Rabe and Staats [66] operate on… |
| 2 | `flashattention.pdf` | 2 | 0.02963126 | s (&gt;1 TB) 0 5 10 15 Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large 𝑁\u0002𝑁 attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of th… |
| 3 | `flashattention.pdf` | 22 | 0.01639344 | ) while FlashAttention focuses on reducing memory accesses (the number of memory reads/writes). As mentioned in Section 2, the amount of memory access is the primary determining factor of runtime. Reducing memory accesses also necessarily r… |
| 4 | `flashattention.pdf` | 1 | 0.01639344 | problem by trading oﬀ model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms IO- aware—accounting for reads and writes between levels of … |
| 5 | `flashattention.pdf` | 10 | 0.01612903 | s begin to cross over with FlashAttention at sequences between 512 and 1024. On the other hand, block-sparse FlashAttention is faster than all implementations of exact, sparse, and approximate attention that we know of, across all sequence … |

Hybrid candidate provenance (`—` means outside that method’s top 20):

| Hybrid rank | Chunk ID | Vector rank | BM25 rank |
| ---: | --- | ---: | ---: |
| 1 | `3691d02db494d3957fc357a6579dd20841985014d0fef86464235fcb5ed54755` | 3 | 2 |
| 2 | `b3fd576c2ae53763570eaaddf60a4c1f7efbfa80e2ac65ee38197726ff731cba` | 8 | 7 |
| 3 | `0dc4806386f2c342c16b7f198f9e8c12a59579149c0ca83dbdafd186a8f90f3d` | 1 | — |
| 4 | `12e5aad683abf32851dfe8e26310de99bdc33cdf9cf067b4f31bd31ea4066db0` | — | 1 |
| 5 | `830e6f9a908b4e66bbe5b71534b4fcd833a7ce92d5bbd2f92cd313ea4a82fe11` | 2 | — |

## 3. How does RAG combine parametric and non-parametric memory?

Hybrid puts the direct parametric/non-parametric architecture explanation first, as BM25 does; vector retrieval ranked it third. Other hits provide contextual discussion, an example, conclusions, and model-size comparisons. This is not a measured general improvement.

### BM25

| Rank | Filename | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `rag_2020.pdf` | 2 | 24.24043822 | non-parametric memory to the “workhorse of NLP,” i.e. sequence-to-sequence (seq2seq) models. We endow pre-trained, parametric-memory generation models with a non-parametric memory through a general-purpose ﬁne-tuning approach which we refer… |
| 2 | `rag_2020.pdf` | 1 | 24.07302328 | ability to access and precisely manipulate knowl- edge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-speciﬁc architectures. Additionally, providing provenance for their decisions and updating t… |
| 3 | `rag_2020.pdf` | 6 | 23.80535462 | The Sun Also Rises" indicating the title "The Sun Also Rises" is stored in BART’s parameters. Similarly, BART will complete the partial decoding "The Sun Also Rises" is a novel by this author of "A with "The Sun Also Rises" is a novel by th… |
| 4 | `rag_2020.pdf` | 9 | 23.72132901 | ﬁnding RAG more factual and speciﬁc. We conducted an thorough investigation of the learned retrieval component, validating its effectiveness, and we illustrated how the retrieval index can be hot-swapped to update the model without requirin… |
| 5 | `rag_2020.pdf` | 19 | 20.40268912 | 5 model with the closest number of parameters to our models is T5-large (770M parameters), which achieves a score of 28.9 EM on Natural Questions [52], substantially below the 44.5 that RAG-Sequence achieves, indicating that hybrid parametr… |

### FAISS cosine

| Rank | Filename | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `rag_2020.pdf` | 3 | 0.51697409 | ·\|x)) pη(z\|x) N∏ i pθ(yi\|x,z,y1:i−1) RAG-Token Model In the RAG-Token model we can draw a different latent document for each target token and marginalize accordingly. This allows the generator to choose content from several documents when p… |
| 2 | `rag_2020.pdf` | 2 | 0.51423347 | architectures to enrich systems with non-parametric memory which are trained from scratch for speciﬁc tasks, e.g. memory networks [64, 55], stack- augmented networks [25] and memory layers [30]. In contrast, we explore a setting where both … |
| 3 | `rag_2020.pdf` | 2 | 0.50436705 | non-parametric memory to the “workhorse of NLP,” i.e. sequence-to-sequence (seq2seq) models. We endow pre-trained, parametric-memory generation models with a non-parametric memory through a general-purpose ﬁne-tuning approach which we refer… |
| 4 | `rag_2020.pdf` | 9 | 0.49922818 | ﬁnding RAG more factual and speciﬁc. We conducted an thorough investigation of the learned retrieval component, validating its effectiveness, and we illustrated how the retrieval index can be hot-swapped to update the model without requirin… |
| 5 | `rag_2020.pdf` | 6 | 0.48917738 | The Sun Also Rises" indicating the title "The Sun Also Rises" is stored in BART’s parameters. Similarly, BART will complete the partial decoding "The Sun Also Rises" is a novel by this author of "A with "The Sun Also Rises" is a novel by th… |

### Hybrid RRF

| Rank | Filename | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `rag_2020.pdf` | 2 | 0.03226646 | non-parametric memory to the “workhorse of NLP,” i.e. sequence-to-sequence (seq2seq) models. We endow pre-trained, parametric-memory generation models with a non-parametric memory through a general-purpose ﬁne-tuning approach which we refer… |
| 2 | `rag_2020.pdf` | 2 | 0.03128055 | architectures to enrich systems with non-parametric memory which are trained from scratch for speciﬁc tasks, e.g. memory networks [64, 55], stack- augmented networks [25] and memory layers [30]. In contrast, we explore a setting where both … |
| 3 | `rag_2020.pdf` | 6 | 0.03125763 | The Sun Also Rises" indicating the title "The Sun Also Rises" is stored in BART’s parameters. Similarly, BART will complete the partial decoding "The Sun Also Rises" is a novel by this author of "A with "The Sun Also Rises" is a novel by th… |
| 4 | `rag_2020.pdf` | 9 | 0.03125000 | ﬁnding RAG more factual and speciﬁc. We conducted an thorough investigation of the learned retrieval component, validating its effectiveness, and we illustrated how the retrieval index can be hot-swapped to update the model without requirin… |
| 5 | `rag_2020.pdf` | 19 | 0.03030999 | 5 model with the closest number of parameters to our models is T5-large (770M parameters), which achieves a score of 28.9 EM on Natural Questions [52], substantially below the 44.5 that RAG-Sequence achieves, indicating that hybrid parametr… |

Hybrid candidate provenance (`—` means outside that method’s top 20):

| Hybrid rank | Chunk ID | Vector rank | BM25 rank |
| ---: | --- | ---: | ---: |
| 1 | `a8e0bf30821689c5f17aa7be16179839a227b181d009b7cf463ccdf37955b724` | 3 | 1 |
| 2 | `d5aee2beeb08fb778006e3cfee27fcba0c8125900772e4e906e96192b5d3e6f1` | 2 | 6 |
| 3 | `4ce0da25f0e147e9f2f73b2ccdbaf41df42bcfc3e2ede96f6ea620e113b2dbea` | 5 | 3 |
| 4 | `41b5e640790284b6600080d9a252e21177a3b9d7b33a95c2b589efb4df0bec5c` | 4 | 4 |
| 5 | `ebb8f13bbc2ea67c08a2d880563b4b62b1548842029cbed7c086563f6daebc7e` | 7 | 5 |

## 4. Why does the Transformer use positional encoding?

The direct positional-encoding explanation is hybrid rank 2, from BM25 rank 1 and vector rank 10: 1/61 + 1/70 = 0.03067916. The top hybrid hit concerns dropout on positional encodings: rank 5 in both lists gives 2/65 = 0.03076923. Thus the explanation enters hybrid top five, but consensus still favors a less direct passage above it. Ranks 3 and 4 are tangential context, not direct explanations.

### BM25

| Rank | Filename | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `attention_is_all_you_need.pdf` | 6 | 15.93160050 | n/r) 3.5 Positional Encoding Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens i… |
| 2 | `attention_is_all_you_need.pdf` | 9 | 12.38898319 | attention key size dk hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as exp… |
| 3 | `attention_is_all_you_need.pdf` | 6 | 12.12804082 | Epos. We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extr… |
| 4 | `attention_is_all_you_need.pdf` | 6 | 11.38264233 | e functions of different frequencies: P E(pos,2i) = sin(pos/100002i/dmodel ) P E(pos,2i+1) = cos(pos/100002i/dmodel ) where pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds to a sinu… |
| 5 | `attention_is_all_you_need.pdf` | 8 | 10.63457813 | of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate … |

### FAISS cosine

| Rank | Filename | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `flashattention.pdf` | 16 | 0.47675613 | , Mohammad Shoeybi, Tom Goldstein, Anima Anandkumar, and Bryan Catanzaro. Long-short transformer: Eﬃcient transformers for language and vision. Advances in Neural Information Processing Systems, 34, 2021. 16 |
| 2 | `attention_is_all_you_need.pdf` | 3 | 0.47464931 | Figure 1: The Transformer - model architecture. The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Fi… |
| 3 | `flashattention.pdf` | 17 | 0.46116829 | it performs almost as well as the (dense) FlashAttention on the Long-range Arena tasks. Eﬃcient Transformer. Transformer-based models have become the most widely-used architecture in natural language processing [22] and computer vision [24,… |
| 4 | `attention_is_all_you_need.pdf` | 5 | 0.45907003 | ×dmodel . In this work we employ h = 8 parallel attention layers, or heads. For each of these we use dk = dv = dmodel/h = 64. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attentio… |
| 5 | `attention_is_all_you_need.pdf` | 8 | 0.43949670 | of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate … |

### Hybrid RRF

| Rank | Filename | Page | Score | Passage excerpt |
| ---: | --- | ---: | ---: | --- |
| 1 | `attention_is_all_you_need.pdf` | 8 | 0.03076923 | of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate … |
| 2 | `attention_is_all_you_need.pdf` | 6 | 0.03067916 | n/r) 3.5 Positional Encoding Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens i… |
| 3 | `flashattention.pdf` | 9 | 0.02941813 | work, all transformer models have either run out of memory, or only achieved random performance [80]. There has been a search for alternative architectures that can model such long context [37]. We present here the ﬁrst result of Transforme… |
| 4 | `attention_is_all_you_need.pdf` | 9 | 0.02899160 | Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, accord… |
| 5 | `attention_is_all_you_need.pdf` | 6 | 0.02853124 | Epos. We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extr… |

Hybrid candidate provenance (`—` means outside that method’s top 20):

| Hybrid rank | Chunk ID | Vector rank | BM25 rank |
| ---: | --- | ---: | ---: |
| 1 | `ef66a89a6fbe1db2d1f14444362a7ff39fd5df9c654f23ed229dff3171cb025c` | 5 | 5 |
| 2 | `3faa9ef937eedd5b6e0c94af2cfcba9ce055a427152f6ea4738fece24f0b0425` | 10 | 1 |
| 3 | `e0b4715969828a714d086ec0bf1fe5bcda6576d18dddd5aaa6de979377c8b7ee` | 7 | 9 |
| 4 | `3c930f1ae7df782407de5e074e0c8fe0f0e02be2828e269ab32957d991dff346` | 8 | 10 |
| 5 | `26d9e5f44a368f513f167499ded29bd637ff2509a4102089d2faae66148c74b0` | 19 | 3 |
