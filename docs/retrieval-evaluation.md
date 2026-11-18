# Step 5: real-corpus retrieval review

Measured on 2026-11-18 with Apple M5 Pro CPU, six PyTorch threads, batch size 32, pinned MiniLM, 180-token chunks and 30-token overlap. Three PDFs produced 68 pages and 419 chunks. These four manually inspected questions are illustrative examples, not a labeled benchmark; no precision/recall metrics are claimed.

Model initialization and downloads completed before inference timers. Document embedding includes tokenization and normalization. Raw FAISS construction measures `IndexFlatIP(384)` plus `add`; raw search measures `index.search` only. Measurements are single calls with no explicit warm-up or averaging, so the slower first search includes first-use overhead.

| Operation | Time |
| --- | ---: |
| Document embedding | 1.706088 s |
| Raw FAISS construction | 0.013958 ms |
| Wrapper construction including validation | 0.416583 ms |

Saved size: `vectors.faiss` = 643,629 bytes; `chunks.jsonl` = 383,485 bytes; `manifest.json` = 727 bytes; total = 1,027,841 bytes. All four queries returned exactly equal result records and scores before/after reload.

Full retrieved chunks follow. Trailing spaces are omitted; otherwise text is unchanged except that nonprinting control characters are displayed as literal escapes such as `\u0002`; these artifacts originate in PDF extraction. Physical PDF page numbers are used.

## 1. What is the computational complexity of self-attention?

Query embedding: 2.968583 ms. Raw FAISS search: 0.988084 ms.

Manual assessment: Ranks 1 and 3 directly support quadratic self-attention complexity; rank 3 contains the per-layer O(n² · d) table (flattened by PDF extraction). Rank 2 discusses computational-complexity criteria but does not itself give the formula. Useful results, with the most precise passage ranked third.

### Rank 1: flashattention.pdf, page 1, score 0.592781

Chunk ID: `b6c013339afdb8d8d9736a16a53aa75626bf848e8fdc3728781f2e72df29cd14`

```text
FlashAttention: Fast and Memory-Eﬃcient Exact Attention
with IO-Awareness
Tri Daoy, Daniel Y. Fuy, Stefano Ermony, Atri Rudraz, and Christopher Réy
yDepartment of Computer Science, Stanford University
zDepartment of Computer Science and Engineering, University at Buﬀalo, SUNY
{trid,danfu}@cs.stanford.edu, ermon@stanford.edu, atri@buffalo.edu,
chrismre@cs.stanford.edu
June 24, 2022
Abstract
Transformers are slow and memory-hungry on long sequences, since the time and memory complexity
of self-attention are quadratic in sequence length. Approximate attention methods have attempted
to address this problem by trading oﬀ model quality to reduce the compute complexity, but often do
not achieve wall-clock speedup. We argue that a missing principle is
```

### Rank 2: attention_is_all_you_need.pdf, page 6, score 0.588325

Chunk ID: `755e622da4c22f588eaabc63c156edcd9ef9028bbe444af6dcaad67152570f0d`

```text
a hidden
layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we
consider three desiderata.
One is the total computational complexity per layer. Another is the amount of computation that can
be parallelized, as measured by the minimum number of sequential operations required.
The third is the path length between long-range dependencies in the network. Learning long-range
dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the
ability to learn such dependencies is the length of the paths forward and backward signals have to
traverse in the network. The shorter these paths between any combination of positions in the input
and output sequences, the easier it is to learn long-range dependencies [12]. Hence we also compare
the maximum path length between any two input and output positions in networks composed of the
different layer
```

### Rank 3: attention_is_all_you_need.pdf, page 6, score 0.583313

Chunk ID: `9ef24abb4a7408c0a44b6e63725a7eb7415f75dc4e3737f57ee0bc6f369ac5c3`

```text
Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations
for different layer types. n is the sequence length, d is the representation dimension, k is the kernel
size of convolutions and r the size of the neighborhood in restricted self-attention.
Layer Type Complexity per Layer Sequential Maximum Path Length
Operations
Self-Attention O(n2 · d) O(1) O(1)
Recurrent O(n · d2) O(n) O(n)
Convolutional O(k · n · d2) O(1) O(logk(n))
Self-Attention (restricted) O(r · n · d) O(1) O(n/r)
3.5 Positional Encoding
Since our model contains no recurrence and no convolution, in order for the model
```

## 2. How does FlashAttention reduce memory transfers?

Query embedding: 2.757917 ms. Raw FAISS search: 0.021250 ms.

Manual assessment: Ranks 1 and 3 are relevant: they discuss memory accesses, tiling, avoiding storage of the attention matrix, and recomputation. Rank 2 is related but emphasizes memory footprint and speed rather than the mechanism of reducing transfers. The clearest mechanism appears at rank 3.

### Rank 1: flashattention.pdf, page 22, score 0.712328

Chunk ID: `0dc4806386f2c342c16b7f198f9e8c12a59579149c0ca83dbdafd186a8f90f3d`

```text
) while FlashAttention focuses on reducing memory accesses
(the number of memory reads/writes). As mentioned in Section 2, the amount of memory access is the
primary determining factor of runtime. Reducing memory accesses also necessarily reduces the total amount
of memory required (e.g., if an operation incurs 𝐴 memory accesses, then its total memory requirement is at
most 𝐴). As a result, FlashAttention is faster than standard attention (2-4\u0002) while Rabe and Staats [66]
is around the same speed or slightly slower than standard attention. In terms of total memory required, both
methods oﬀer substantial memory saving.
The second diﬀerence between the two methods is the way information is summarized from each block
to pass to the next block. Rabe and Staats [66] summarizes each block with its temporary output along

```

### Rank 2: flashattention.pdf, page 10, score 0.675855

Chunk ID: `830e6f9a908b4e66bbe5b71534b4fcd833a7ce92d5bbd2f92cd313ea4a82fe11`

```text
s begin to cross over with
FlashAttention at sequences between 512 and 1024. On the other hand, block-sparse FlashAttention
is faster than all implementations of exact, sparse, and approximate attention that we know of, across all
sequence lengths.
Memory Footprint. Figure 3 (right) shows the memory footprint of FlashAttention and block-sparse
FlashAttention compared to various exact, approximate, and sparse attention baselines. FlashAttention
and block-sparse FlashAttention have the same memory footprint, which grows linearly with sequence
length. FlashAttention is up to 20\u0002 more memory eﬃcient than exact attention baselines, and is more
memory-eﬃcient than the approximate attention baselines. All other algorithms except for Linformer run
out of memory on an A100 GPU before 64K, and FlashAttention is still 2\u0002 more
```

### Rank 3: flashattention.pdf, page 22, score 0.613865

Chunk ID: `3691d02db494d3957fc357a6579dd20841985014d0fef86464235fcb5ed54755`

```text
B.5 Comparison with Rabe and Staats [66]
We describe here some similarities and diﬀerences between our FlashAttention algorithm and the algorithm
of Rabe and Staats [66].
Conceptually, both FlashAttention and Rabe and Staats [66] operate on blocks of the attention matrix
using the well-established technique of tiling (or softmax scaling) [51, 60]. To reduce the memory footprint,
both methods avoid storing the large attention matrix in the forward pass and recompute it in the backward
pass.
The ﬁrst major diﬀerence is that Rabe and Staats [66] focuses on the reducing the total memory footprint
(maximum amount of GPU memory required) while FlashAttention focuses on reducing memory accesses
(the number of memory reads/writes). As mentioned in Section 2, the amount
```

## 3. How does RAG combine parametric and non-parametric memory?

Query embedding: 2.801875 ms. Raw FAISS search: 0.019916 ms.

Manual assessment: All three results concern RAG. Rank 1 explains token-level document marginalization; rank 2 discusses the motivation for combining memories. Rank 3 most directly answers the question by identifying the seq2seq transformer and the Wikipedia vector index. The most direct explanation is not ranked first.

### Rank 1: rag_2020.pdf, page 3, score 0.516974

Chunk ID: `d2493bc654fbfa6bbfd13bcd88ad014b133b97e12023177e2ef523ff82fdcf3d`

```text
·|x))
pη(z|x)
N∏
i
pθ(yi|x,z,y1:i−1)
RAG-Token Model In the RAG-Token model we can draw a different latent document for each
target token and marginalize accordingly. This allows the generator to choose content from several
documents when producing an answer. Concretely, the top K documents are retrieved using the
retriever, and then the generator produces a distribution for the next output token for each document,
before marginalizing, and repeating the process with the following output token, Formally, we deﬁne:
pRAG-Token(y|x) ≈
N∏
i
∑
z∈top-k(p(·|x))
pη(z|x)pθ(yi|x,z,y1:i−1)
Finally, we note that RAG can be
```

### Rank 2: rag_2020.pdf, page 2, score 0.514233

Chunk ID: `d5aee2beeb08fb778006e3cfee27fcba0c8125900772e4e906e96192b5d3e6f1`

```text
architectures to enrich systems with non-parametric
memory which are trained from scratch for speciﬁc tasks, e.g. memory networks [64, 55], stack-
augmented networks [25] and memory layers [30]. In contrast, we explore a setting where both
parametric and non-parametric memory components are pre-trained and pre-loaded with extensive
knowledge. Crucially, by using pre-trained access mechanisms, the ability to access knowledge is
present without additional training.
Our results highlight the beneﬁts of combining parametric and non-parametric memory with genera-
tion for knowledge-intensive tasks—tasks that humans could not reasonably be expected to perform
without access to an external knowledge source. Our RAG models achieve state-of-the-art results
on open Natural Questions [29], WebQuestions [3] and CuratedTre
```

### Rank 3: rag_2020.pdf, page 2, score 0.504367

Chunk ID: `a8e0bf30821689c5f17aa7be16179839a227b181d009b7cf463ccdf37955b724`

```text
non-parametric memory to the “workhorse of NLP,” i.e. sequence-to-sequence (seq2seq) models.
We endow pre-trained, parametric-memory generation models with a non-parametric memory through
a general-purpose ﬁne-tuning approach which we refer to as retrieval-augmented generation (RAG).
We build RAG models where the parametric memory is a pre-trained seq2seq transformer, and the
non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural
retriever. We combine these components in a probabilistic model trained end-to-end (Fig. 1). The
retriever (Dense Passage Retriever [26], henceforth DPR) provides latent documents conditioned on
the input, and the seq2se
```

## 4. Why does the Transformer use positional encoding?

Query embedding: 2.702958 ms. Raw FAISS search: 0.020750 ms.

Manual assessment: The top three do not answer why positional encoding is needed. Rank 1 is a bibliography fragment, rank 2 describes encoder/decoder architecture, and rank 3 discusses efficient Transformers. This is an observed retrieval failure, not a relevant answer.

### Rank 1: flashattention.pdf, page 16, score 0.476756

Chunk ID: `ed5fdb8012401fdee79757975169709c50897bacb5cc01f350c43d53ebac89a6`

```text
, Mohammad Shoeybi, Tom Goldstein, Anima Anandkumar, and
Bryan Catanzaro. Long-short transformer: Eﬃcient transformers for language and vision. Advances in
Neural Information Processing Systems, 34, 2021.
16
```

### Rank 2: attention_is_all_you_need.pdf, page 3, score 0.474649

Chunk ID: `e33cafc3f629595bb4844499adb6e74bb38fda01b6b96dc0c7662c733e8151a9`

```text
Figure 1: The Transformer - model architecture.
The Transformer follows this overall architecture using stacked self-attention and point-wise, fully
connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1,
respectively.
3.1 Encoder and Decoder Stacks
Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two
sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-
wise fully connected feed-forward network. We employ a residual connection [11] around each of
the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is
LayerNorm(x + Sublayer(x)), where Sublayer(x) is the
```

### Rank 3: flashattention.pdf, page 17, score 0.461168

Chunk ID: `2a8aa3d087dc8713a0777ab7917385ca5c7890d94943be7d527b246b3bd23cad`

```text
it performs almost as well as the (dense) FlashAttention on the Long-range Arena tasks.
Eﬃcient Transformer. Transformer-based models have become the most widely-used architecture in
natural language processing [22] and computer vision [24, 91]. However, one of their computational bottlenecks
is that their time and memory scales quadratic in the sequence length. There are numerous approaches to
overcome this bottleneck, including approximation with hashing (i.e., sparse) such as Reformer [51] and
Smyrf [19] and with low-rank approximation such as Performer [12, 54]. One can even combine sparse and
low-rank approximation for better accuracy (e.g., Longformer [3], BigBird [92], Scatterbrain [9], Long-short
transformer [94],
```
