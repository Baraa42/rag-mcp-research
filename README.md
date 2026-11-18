# rag-mcp-research

A production-oriented research assistant for ML papers, planned to combine
hybrid retrieval with citation-grounded answers and expose research tools
through MCP and FastAPI.

**Status:** Session 1, Step 5 — ingestion, chunking, embeddings, exact FAISS
search, persistence, and a CLI are implemented. BM25, reranking, generated
answers, MCP, and FastAPI are not implemented yet.

## Planned architecture

PDF/Markdown → text with source metadata → token-aware chunks → Sentence
Transformer embeddings → FAISS vector search + BM25 lexical search → hybrid
retrieval → cross-encoder reranking → citation-grounded answers.

The Python package will keep ingestion, retrieval, and answering logic separate
from future MCP and FastAPI interfaces. Docker deployment, evaluation, and
benchmarks will follow. No LangChain or LlamaIndex is planned.

## Requirements

- Python 3.11 (the project supports `>=3.11,<3.12`)
- `uv` and Git
- Internet access for initial dependencies and the first tokenizer/embedding
  model download; later sessions will also download reranking models

## Setup

From the repository root:

```sh
uv python install 3.11
uv sync --locked
```

`uv` creates `.venv`, installs the package in editable mode, and includes the
`dev` dependency group by default. Commit `uv.lock` to preserve the resolved
versions. Use `uv sync` when intentionally updating project configuration;
use `--locked` for setup that must not change the lockfile.

The distribution is named `rag-mcp-research`; its Python import is
`research_assistant`. Put local input papers in `data/papers/`; their contents
are ignored by Git.

## Checks

```sh
uv run pytest
uv run ruff check .
uv run python -c "import research_assistant; print(research_assistant.__file__)"
```

Ordinary tests use synthetic documents and a tiny local WordPiece tokenizer;
they do not require downloaded papers or models. The actual MiniLM tokenizer
check is opt-in (downloads tokenizer files on first use, never model weights):

```sh
RUN_TOKENIZER_INTEGRATION=1 uv run pytest tests/test_chunking.py -k actual_minilm
```

Embedding unit tests use a fake model. Run the real-model check separately:

```sh
RUN_EMBEDDING_INTEGRATION=1 uv run pytest tests/test_embeddings.py -k real_model
```

This downloads the pinned weights on first use. It checks dimensions, unit
norms, identical inputs, and a relative related/unrelated sentence comparison;
it does not impose a universal semantic-similarity threshold.

## Document ingestion

```python
from pathlib import Path
from research_assistant.ingestion import ingest_directory

pages = ingest_directory(Path("data/papers"))
for page in pages:
    print(page.source, page.page_number, len(page.text))
```

`extract_pdf(path)` returns one `DocumentPage` per nonempty physical PDF page,
with 1-based page numbers. `extract_markdown(path)` reads UTF-8 Markdown into a
single record with `page_number=None`, preserving Markdown syntax. Both trim
leading/trailing whitespace and skip whitespace-only text. Valid PDFs with no
pages and empty Markdown files return an empty list; a zero-byte PDF is invalid.

Each immutable record contains `document_id` (SHA-256 of the original file
bytes), `source`, `page_number`, and `text`. Identical bytes have the same ID
even after renaming; any byte change, including PDF metadata, changes the ID.
Direct extractors use the filename. Directory ingestion uses relative POSIX
paths, scans `.pdf` and `.md` case-insensitively, sorts paths lexicographically,
and preserves page order. Other extensions are ignored; directory symlinks are
not traversed.

Filesystem errors propagate. Invalid UTF-8, PDF parsing/extraction failures,
and encrypted PDFs raise contextual `ValueError` exceptions. Directory scanning
errors also propagate; a failed file aborts ingestion rather than returning
partial results. Strict PDF parsing deliberately rejects malformed files.

Limitations: no OCR, layout reconstruction, equation/table interpretation, or
header/footer removal. Image-only pages may produce no records. Column order,
spacing, ligatures, and mathematical notation may be imperfect. Markdown is
not rendered or split into sections. Files and extracted records are held in
memory, so large collections need future resource limits or streaming. Source
metadata supports future citations; citation generation is not implemented.

Local validation extracted 15/15 nonempty pages from Attention Is All You Need,
19/19 from RAG (2020), and 34/34 from FlashAttention. FlashAttention emitted
optional `fontTools` warnings for embedded CFF fonts; extraction completed, but
some font encodings may be imperfect. No additional dependency was added.

## Token-aware chunking

```python
from research_assistant.chunking import chunk_pages, load_tokenizer

tokenizer = load_tokenizer()
chunks = chunk_pages(pages, max_tokens=180, overlap=30, tokenizer=tokenizer)
```

The default tokenizer is `sentence-transformers/all-MiniLM-L6-v2`, pinned to
revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Loading a tokenizer does not
load an embedding model. `chunk_pages` loads it if none is supplied; reuse one
instance across calls to avoid repeated loading.

The 180-token maximum includes special tokens: MiniLM adds two, leaving 178
text tokens. This is a conservative project budget, not an experimentally
optimized size. Character offsets locate windows in the original string;
the implementation never decodes token slices. Each final substring is encoded
again without truncation and shortened if necessary to satisfy the limit.
Capitalization, accents, punctuation, and internal whitespace remain unchanged.

Consecutive windows overlap by 30 source tokens by default. Independently
tokenizing a substring that starts inside a word can change the tokenization,
so overlap is not guaranteed to equal 30 independently encoded tokens. Windows
always advance; overlap can shrink for unusually small windows. Zero overlap
is supported. Size/overlap must be integers, the text budget must be positive,
and overlap must be nonnegative and smaller than that budget. The configured
maximum cannot exceed the tokenizer's own input limit.

PDF pages are never merged. Markdown (`page_number=None`) starts a new section
at each ATX heading (`#` through `######`) outside backtick/tilde fences. Heading
text stays in the first chunk; `section` records the latest heading title.
There is no overlap across page or section boundaries. Introductory text has
`section=None`. Setext headings, nested heading paths, and the full CommonMark
grammar are not implemented.

`Chunk` is immutable and retains `document_id`, `source`, `page_number`, and
`section`, plus `text`, a zero-based `chunk_index` within each input page, and
`chunk_id`. IDs hash the tokenizer configuration, chunking settings, provenance,
index, and exact text. Mutable tokenizer padding/truncation settings are excluded
from the fingerprint. Input order is preserved; IDs are stable for identical
inputs and configuration, but can change when configuration or source changes.

Trade-offs: boundaries may split sentences or words; short final chunks remain.
Whitespace-only spans produce no chunks. Source text ignored by the tokenizer
is retained in adjacent slices, but the tokenizer itself can still normalize or
ignore characters when encoding. No PDF extraction defects are repaired here.
The implementation re-tokenizes remaining section text for clarity; very large
sections may need a more efficient streaming approach later.

Local validation: 68 pages produced 419 chunks (73 Attention, 129 RAG, 217
FlashAttention), averaging 166.38 tokens including special tokens, with a maximum
of 180. Every non-whitespace character in the ingested pages was covered.

## Sentence embeddings

```python
from research_assistant.embeddings import Embedder

embedder = Embedder(batch_size=32)
vectors = embedder.encode([chunk.text for chunk in chunks])
# vectors.shape == (len(chunks), 384); vectors.dtype == numpy.float32
```

`Embedder` loads `SentenceTransformer` once per instance and explicitly uses
CPU inference. Reuse the instance for multiple calls. Model name and revision
are defined once in `model_config.py` and shared with chunking: MiniLM at the
same pinned revision documented above. No new dependencies are required.

`encode` accepts a list of strings and preserves row ordering. Sentence
Transformers handles batches internally; `batch_size` must be a positive
integer. An empty list returns `(0, 384)` without model inference (the instance
has already loaded its model). Empty strings are valid inputs, although ingestion
and chunking do not produce empty text records. Other container types and
non-string elements raise `TypeError`.

Output shape and finite values are checked before each vector is divided by
its L2 norm. Norms accumulate in float64 for numerical stability; the returned
matrix is float32. Zero-norm, non-finite, and incorrectly shaped model output
raise `ValueError`. For these unit-length vectors, a dot product is cosine
similarity, up to floating-point rounding.

### Local CPU measurement (2026-11-18)

Apple M5 Pro, macOS arm64, PyTorch using 6 CPU threads, batch size 32. The three
papers were ingested and chunked with 180 tokens / 30 overlap. Model files were
prefetched before timing model construction. The inference timer covered one
corpus `encode` call, including tokenization, batching, conversion, validation,
and normalization; no explicit warm-up or repeated-run averaging was used.

| Measurement | Result |
| --- | --- |
| Chunks / matrix shape | 419 / `(419, 384)` |
| Data type | `float32` |
| L2 norm min / mean / max | 0.9999999404 / 1.0000000000 / 1.0000000000 |
| Initial download/cache check | 16.234 s |
| Model construction from cached files | 2.466 s |
| Corpus embedding inference | 1.824 s |
| Throughput | 229.77 chunks/s |
| Device | CPU; no CUDA or MPS used |

Sentence comparisons, measured separately from the corpus timing:

| Sentence A | Sentence B | Cosine similarity |
| --- | --- | --- |
| A cat is sleeping on the sofa. | A kitten is resting on a couch. | 0.694172 |
| A cat is sleeping on the sofa. | The stock market closed higher today. | 0.044238 |
| A cat is sleeping on the sofa. | A cat is sleeping on the sofa. | 1.000000 |

These are illustrative measurements on one machine, not performance guarantees
or a retrieval evaluation. The model truncates inputs beyond its 256-token
sequence limit; current 180-token chunks fit, but arbitrary longer strings passed
directly to `encode` can be truncated. Corpus quality still depends on PDF
extraction and chunk boundaries. Results can vary slightly across hardware and
library versions. All output vectors are held in memory; batching reduces model
working memory, not final matrix size. Weights remain in the external Hugging
Face cache and are not committed.

## Exact semantic retrieval

```sh
uv run python -m research_assistant index data/papers --out data/index
uv run python -m research_assistant search \
  "How does FlashAttention reduce memory usage?" \
  --index data/index --top-k 5
```

The current executable pipeline is PDF/Markdown ingestion → per-page/section
chunking → normalized MiniLM embeddings → exact FAISS `IndexFlatIP(384)` →
ranked passages. Search embeds only the question, never the indexed documents.
The CLI displays rank, cosine similarity, filename, physical page or Markdown
section, and the full retrieved text; it does not generate an answer. Each CLI
search invocation loads the model; reuse an `Embedder` for repeated library calls.

```python
from pathlib import Path
from research_assistant.retrieval import VectorIndex

index = VectorIndex.load(Path("data/index"))
results = index.search("Why use positional encoding?", embedder, top_k=5)
```

`VectorIndex(chunks, vectors)` preserves an immutable tuple of chunk records in
FAISS row order. `SearchResult` contains chunk/document IDs, source, page, section,
text, and score. Inputs are converted to contiguous float32 and checked for
384 dimensions, finite values, and unit norms. `top_k` must be a positive integer;
requests larger than the corpus are capped. An empty index returns no results
without query inference. Scores descend; equal-score ordering at the top-k
cutoff is not a cross-version guarantee.

### Saved format

```text
data/index/
├── vectors.faiss
├── chunks.jsonl
└── manifest.json
```

The JSONL contains one complete Chunk per FAISS row. The versioned manifest
records the shared model name/revision, dimension, chunk count, chunk size and
overlap, index type, source-document IDs, and SHA-256 hashes of both data files.
Loading verifies checksums before FAISS deserialization, then checks the actual
index type/metric/dimension, row count, vector norms/finiteness, chunk field types,
unique chunk IDs, and manifest consistency. It does not re-embed documents.
Caller-provided vectors must correspond to the supplied chunks and pinned model;
that semantic relationship cannot be inferred from vector values alone.

Saving replaces files individually using temporary files and writes the manifest
last. Interrupted or mixed writes fail validation on load; this is not a
transactional or concurrent-writer store. Load only trusted local artifacts:
checksums detect accidental corruption, not maliciously rewritten manifests.
Source files are not re-read on load; rebuild the index when sources change.
Generated files under `data/index/` are ignored by Git.

### Actual retrieval review

[Full top-three passages, scores, and manual assessments](docs/retrieval-evaluation.md)
are recorded for all four requested questions. No relevance metrics or ground
truth labels are claimed.

| Question topic | Top three: filename / page / score | Manual assessment |
| --- | --- | --- |
| Self-attention complexity | flashattention.pdf / 1 / 0.592781; attention_is_all_you_need.pdf / 6 / 0.588325; attention_is_all_you_need.pdf / 6 / 0.583313 | Ranks 1 and 3 directly address complexity; rank 2 is contextual. |
| FlashAttention memory transfers | flashattention.pdf / 22 / 0.712328; flashattention.pdf / 10 / 0.675855; flashattention.pdf / 22 / 0.613865 | Useful; the mechanism is clearest at rank 3. |
| RAG memory combination | rag_2020.pdf / 3 / 0.516974; rag_2020.pdf / 2 / 0.514233; rag_2020.pdf / 2 / 0.504367 | Relevant; rank 3 is the most direct explanation. |
| Positional encoding | flashattention.pdf / 16 / 0.476756; attention_is_all_you_need.pdf / 3 / 0.474649; flashattention.pdf / 17 / 0.461168 | Failure: none answers the question; a bibliography fragment ranks first. |

On Apple M5 Pro CPU, batch 32, document embedding took **1.706088 s**, raw FAISS
construction **0.013958 ms** (validated wrapper construction **0.416583 ms**).
Query embedding took **2.703–2.969 ms**; raw FAISS searches took **0.988084,
0.021250, 0.019916, and 0.020750 ms**, respectively. Loading/downloads were excluded.
These are single-run measurements, with first-use overhead visible in the first
search. The FAISS file is **643,629 bytes**; all three saved files total
**1,027,841 bytes**. Reloaded results matched all four in-memory queries exactly.

```sh
RUN_RETRIEVAL_INTEGRATION=1 uv run pytest tests/test_retrieval.py -k real_end_to_end
```

Offline tests use synthetic vectors/fake embedders. The opt-in test runs actual
Markdown ingestion, chunking, model inference, search, and save/load. FAISS is
imported after the embedding runtime: importing it first caused a native crash
with the installed macOS wheels; standalone integration and corpus validation
passed after this initialization-order fix.

Limitations: exact search scales linearly with corpus size; this corpus is only
419 vectors. Overlap can yield redundant hits, bibliography text can outrank
answers, and cosine scores are not relevance probabilities. There is no keyword
retrieval, reranking, metadata filtering, confidence threshold, or generated
answer. Existing PDF font/formatting warnings and chunk-boundary artifacts remain.

## Five-session roadmap (10 hours total)

| Session | Time | Scope |
| --- | --- | --- |
| 1 | ~2 hours | PDF/Markdown ingestion, chunking, Sentence Transformer embeddings, FAISS vector search |
| 2 | ~2 hours | BM25 hybrid retrieval, reranking, citation-grounded answers |
| 3 | ~2 hours | MCP server and research tools |
| 4 | ~2 hours | FastAPI, Docker, reliability, logging |
| 5 | ~2 hours | Evaluation, benchmarking, GitHub documentation |

Session 1 is complete. Next: Session 2 will add BM25 hybrid retrieval, reranking,
and citation-grounded answering separately.

Intended GitHub owner: `baraa42`. No remote repository is required for local setup.
