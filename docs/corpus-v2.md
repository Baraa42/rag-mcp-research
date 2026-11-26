# S5B research corpus

The expanded corpus uses the ten local PDFs in `data/papers/`. Its index is
`data/index_v2/`. The original three-paper index remains at `data/index/`.
The five baseline questions and labels in `eval/`, candidate exports in
`data/eval/`, and [evaluation report](evaluation.md) matched their pre-build
SHA-256 checksums after the expansion. All 419 original chunk records also
matched the corresponding records in the expanded index.

## Source inventory

Fingerprints are SHA-256 hashes of the exact local PDF bytes and are also the
document IDs. Paper links identify the publications. Downloaded revisions or
publisher copies may have different fingerprints. Page counts are physical
PDF pages. All 176 pages yielded nonempty text.

| PDF | Corpus | Pages | Chunks | Extraction warnings |
| --- | --- | ---: | ---: | --- |
| [attention_is_all_you_need.pdf](https://arxiv.org/abs/1706.03762) | Baseline + v2 | 15 | 73 | None |
| [colbert.pdf](https://arxiv.org/abs/2004.12832) | v2 | 10 | 112 | None |
| [conformer.pdf](https://arxiv.org/abs/2005.08100) | v2 | 5 | 56 | None |
| [flashattention.pdf](https://arxiv.org/abs/2205.14135) | Baseline + v2 | 34 | 217 | 8 CFF Type 1 font encoding warnings (optional fontTools absent) |
| [flashattention2.pdf](https://arxiv.org/abs/2307.08691) | v2 | 14 | 80 | None |
| [lost_in_the_middle.pdf](https://arxiv.org/abs/2307.03172) | v2 | 17 | 101 | None |
| [pagedattention.pdf](https://arxiv.org/abs/2309.06180) | v2 | 16 | 145 | None |
| [rag_2020.pdf](https://arxiv.org/abs/2005.11401) | Baseline + v2 | 19 | 129 | None |
| [wav2vec2.pdf](https://arxiv.org/abs/2006.11477) | v2 | 19 | 116 | One image XObject could not be decoded because of a stream length limit |
| [whisper.pdf](https://arxiv.org/abs/2212.04356) | v2 | 27 | 209 | None |
| **Total** | | **176** | **1,238** | |

```text
bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697  attention_is_all_you_need.pdf
2e487d9b96e3c2e5e286d843e98a066adc0442faa3a6e8ddbd1771221bf9ae14  colbert.pdf
4d53a2d87f7b68b3b5a86d8e4301c5092ae61e260f9c76dab48af34faac9a81f  conformer.pdf
ca7f9fda10b90fc05dd291a3accc85e9c1a4a860b99b31928dab03ed3fcb14e4  flashattention.pdf
4aa8935dfacaf6ae8c68f772ca92f730154a0dd0e1bceeb59c7bc56c512d5868  flashattention2.pdf
f37f6fabe0fe0d8c73b67579cd115dbf14e282cfb7b7af655ae8109638402c9b  lost_in_the_middle.pdf
55b3b324d779a67c59dac2519445e3b07c14e6ff5c656fadb47a3d7b5997469e  pagedattention.pdf
23e3249e9a1e75418d82efecab0ea8c4d033b89c93742f63208d47ce01f21233  rag_2020.pdf
4a881708378143547a5c053f62090104afb49782a4c71d78441dde0a818758ff  wav2vec2.pdf
84496ed07ac01bf3d00866a1934f81173ddbba99f7488de59bb8ae3b28f0f85e  whisper.pdf
```

## Build and validate

Use Python 3.11 and `uv sync --locked`. The index uses
`sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, 180 tokens per chunk including special
tokens, 30 tokens of overlap, and exact FAISS `IndexFlatIP` with 384 dimensions.

```sh
uv run python -m research_assistant index data/papers --out data/index_v2
```

To reconstruct the original corpus independently, stage only its three exact PDF
files in a temporary directory and write to a separate output:

```sh
baseline_papers=$(mktemp -d)
cp data/papers/attention_is_all_you_need.pdf data/papers/flashattention.pdf \
  data/papers/rag_2020.pdf "$baseline_papers/"
uv run python -m research_assistant index "$baseline_papers" --out data/index_baseline_rebuilt
```

On November 26, 2026, the v2 build took **13.11 seconds** of wall time on an
Apple M5 Pro CPU with cached model weights and Hugging Face offline mode. That
includes CLI startup, ingestion, chunking, embedding and persistence. The three
files total **3,016,273 bytes** (2.88 MiB). Reloading through `VectorIndex.load()`
passed manifest and checksum validation. All ten document IDs matched their
PDF hashes. All 1,238 nonempty chunks had unique IDs and aligned with the 1,238
FAISS rows. Stored vectors were finite float32 with shape `(1238, 384)` and L2
norms between 0.99999994 and 1.0.

## Retrieval smoke tests

Run each question below with `--method hybrid` and `--method rerank`:

```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 uv run python -m research_assistant search \
  "How does PagedAttention manage KV-cache memory?" \
  --index data/index_v2 --method hybrid --top-k 3
```

Offline mode requires cached weights. The initial validation process exited
before returning search results and emitted a multiprocessing semaphore warning.
All six CLI runs completed with `OMP_NUM_THREADS=1`. The initial exit's cause
was unconfirmed. Hybrid scores are RRF scores and reranked scores are raw
cross-encoder scores. These exploratory queries illustrate retrieval behavior
and are separate from the formal evaluation set.

| Question | Method | Rank | Source | Page | Score | Passage focus |
| --- | --- | ---: | --- | ---: | ---: | --- |
| How does PagedAttention manage KV-cache memory? | Hybrid | 1 | pagedattention.pdf | 5 | 0.032522 | Scheduler and paged KV cache manager |
| | Hybrid | 2 | pagedattention.pdf | 5 | 0.031754 | Fixed-size logical blocks filled as tokens arrive |
| | Hybrid | 3 | pagedattention.pdf | 6 | 0.031498 | Block tables map logical to physical blocks and permit growth on demand |
| | Reranked | 1 | pagedattention.pdf | 5 | 8.575878 | Scheduler and paged KV cache manager |
| | Reranked | 2 | pagedattention.pdf | 2 | 8.357053 | Noncontiguous allocation, fragmentation reduction and sharing |
| | Reranked | 3 | pagedattention.pdf | 2 | 7.756890 | Paging analogy and fixed-size blocks |
| How does Whisper use weak supervision for speech recognition? | Hybrid | 1 | whisper.pdf | 9 | 0.031778 | Scaling supervised pretraining and zero-shot transfer |
| | Hybrid | 2 | whisper.pdf | 7 | 0.030415 | Speech translation results |
| | Hybrid | 3 | whisper.pdf | 21 | 0.030118 | Transcription performance table |
| | Reranked | 1 | whisper.pdf | 16 | 5.924320 | Word error rate comparison and evaluation datasets |
| | Reranked | 2 | whisper.pdf | 9 | 5.870264 | Scaling supervised pretraining and zero-shot transfer |
| | Reranked | 3 | whisper.pdf | 25 | 5.144668 | Translation performance table |
| What is the late-interaction mechanism in ColBERT? | Hybrid | 1 | colbert.pdf | 4 | 0.031754 | Separate query and document encoders |
| | Hybrid | 2 | colbert.pdf | 9 | 0.031746 | Architecture ablation setup |
| | Hybrid | 3 | colbert.pdf | 9 | 0.031545 | MaxSim compared with average similarity |
| | Reranked | 1 | colbert.pdf | 1 | 7.281985 | Independent encoding followed by fine-grained similarity |
| | Reranked | 2 | colbert.pdf | 4 | 6.501929 | Separate query and document encoders |
| | Reranked | 3 | colbert.pdf | 2 | 5.618662 | Contextual embedding sets and late interaction |

PagedAttention returned mechanism evidence under both methods. Whisper's
page 9 passage explains the broad training approach. The other hits mainly
describe outcomes and neither top three gives the web transcript collection
details. For ColBERT, reranking surfaced clearer architecture descriptions and
the hybrid list supplied additional MaxSim context. Three queries are too few
to establish an overall ranking improvement.
