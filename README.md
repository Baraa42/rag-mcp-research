# rag-mcp-research

A production-oriented research assistant for ML papers combining hybrid
retrieval with citation-grounded answers and local MCP research tools.
A local FastAPI interface and CPU Docker packaging are available.

**Status:** S1–S4 complete. FastAPI reliability, CPU Docker deployment, and manual
HTTP retrieval integration were verified on November 26, 2026; see
[integration evidence and limitations](docs/integration.md). S3 MCP integration
and real-model retrieval verification are complete. Answer generation is tested
offline; live paid HTTP/MCP generation remains unverified. S5 expanded evaluation
is next. This remains a local, unauthenticated service.

## Architecture

PDF/Markdown → text with source metadata → token-aware chunks → Sentence
Transformer embeddings → FAISS vector search + BM25 lexical search → hybrid
retrieval → cross-encoder reranking → citation-grounded answers.

The Python package separates ingestion, retrieval, and answering logic from MCP
and FastAPI interfaces. Initial [offline retrieval evaluation](docs/evaluation.md)
uses 151 manual judgments across five questions; component timing measurements
are also available below. Broader evaluation and benchmarking remain S5 work.
CPU Docker packaging is included in S4. No LangChain or LlamaIndex is used.

## Source organization

```text
src/research_assistant/
├── __init__.py
├── __main__.py              # delegates to interfaces.cli
├── model_config.py
├── service.py               # reusable orchestration
├── ingest/
│   ├── __init__.py
│   ├── ingestion.py
│   └── chunking.py
├── retrieval/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── vector.py
│   ├── bm25.py
│   ├── hybrid.py
│   └── reranking.py
├── generation/
│   ├── __init__.py
│   └── answering.py
└── interfaces/
    ├── __init__.py
    ├── api.py
    ├── cli.py
    └── mcp_server.py
```

Internal imports use these subpackage paths; no legacy module aliases are provided.
Hybrid and reranked CLI searches and `ask` delegate to `ResearchService`.
The `ask` CLI checks that `OPENAI_MODEL` and `OPENAI_API_KEY` are nonempty before
loading the index or models, and closes the service on success or failure.
The service initializes generation lazily and bypasses it for empty evidence.
The MCP interface uses stdio only; the HTTP interface binds locally by default.

## Requirements

- Python 3.11 (the project supports `>=3.11,<3.12`)
- `uv` and Git
- Internet access for initial dependencies and the first tokenizer/embedding
  and reranking model downloads; `ask` also requires OpenAI API access

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
from research_assistant.ingest.ingestion import ingest_directory

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
metadata supplies physical source locations for citations in the `ask` command.

Local validation extracted 15/15 nonempty pages from Attention Is All You Need,
19/19 from RAG (2020), and 34/34 from FlashAttention. FlashAttention emitted
optional `fontTools` warnings for embedded CFF fonts; extraction completed, but
some font encodings may be imperfect. No additional dependency was added.

## Token-aware chunking

```python
from research_assistant.ingest.chunking import chunk_pages, load_tokenizer

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
from research_assistant.retrieval.embeddings import Embedder

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
from research_assistant.retrieval.vector import VectorIndex

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
answers, and cosine scores are not relevance probabilities. The vector method has no keyword matching, reranking, metadata filtering,
confidence threshold, or generated answer; BM25 is now available separately. Existing PDF font/formatting warnings and chunk-boundary artifacts remain.

## BM25 lexical retrieval

```sh
uv run python -m research_assistant search \
  "Why does the Transformer use positional encoding?" \
  --index data/index --method bm25 --top-k 5
```

`--method vector` remains the default. BM25 CLI search calls `VectorIndex.load()`,
which loads and validates the FAISS index from disk alongside the persisted
chunks. BM25 statistics are rebuilt from those chunks using
`rank_bm25.BM25Okapi` for each CLI invocation; no second corpus or persistence
format is introduced. BM25 does not instantiate the embedding model, run
inference, or download model files. Existing module imports and FAISS index
loading/validation still contribute to CLI startup time.

```python
from research_assistant.retrieval.bm25 import BM25Index

lexical = BM25Index(list(index.chunks), k1=1.5, b=0.75)
results = lexical.search("positional encoding", top_k=5)
```

BM25 sums query-term contributions using inverse document frequency, saturating
term frequency, and document-length normalization. `k1=1.5` controls term-frequency
saturation; larger values allow repeated terms to contribute more. `b=0.75`
controls length normalization; zero disables it and one applies full normalization.
Here each chunk is a BM25 document. Repeated query terms contribute repeatedly.
The library's default IDF-floor parameter (`epsilon=0.25`) is unchanged.

Documents and queries use `re.findall(r"\w+", text.casefold())`: Unicode letters,
numbers, and underscores form tokens. Punctuation and hyphens split tokens, so
`self-attention` matches `self attention`; `Q_K` stays `q_k`, `BM25` stays `bm25`,
and `C++` becomes `c`. Casefolding handles forms such as `Straße` → `strasse`.
Accents are not stripped; there is no Unicode canonical normalization, stemming,
lemmatization, stopword removal, or repair of PDF line-break hyphenation.

Results reuse `SearchResult` and original chunk metadata. Matching chunks are
sorted by descending BM25 score, then original corpus position for ties. Only
chunks sharing at least one query token are returned. Empty/tokenless queries,
empty corpora, and absent terms return no results. Invalid types, nonpositive
`top_k`, invalid parameters, and duplicate chunk IDs are rejected. Punctuation-only
chunks retain their corpus positions but cannot match a query. Zero/negative
scores are retained for matches: Okapi IDF can yield these values in small corpora.
BM25 scores are not cosine scores or probabilities; do not compare or add the raw
scores across methods.

### Observed comparison and performance

[All 40 top-five results with scores and excerpts](docs/bm25-evaluation.md) compare
both methods on the same 419 chunks. BM25 retrieved the direct positional-encoding
explanation at rank 1 (`attention_is_all_you_need.pdf`, page 6, **15.931601**).
FAISS missed it in its top five and ranked a bibliography fragment first. BM25
also ranked the direct RAG architecture explanation first and the FlashAttention
tiling explanation first. Conversely, its complexity-query winner was a noisy
repeated-text figure passage, and an unrelated RAG passage appeared in its
FlashAttention results. Four examples do not establish overall superiority.

On Apple M5 Pro CPU (2026-11-19), 30 construction measurements after warm-up
produced median **7.823 ms** (min 7.614, max 8.198). Across 200 repetitions per
question, median query latencies were **0.276, 0.212, 0.302, and 0.246 ms** in the
four-question order used in the report. Construction and query timers exclude
imports, persisted-index loading, and model operations. A separate Python
allocation trace measured approximately **2.44 MiB retained / 3.56 MiB peak**
additional memory, excluding existing chunks and untraced native allocations;
this is not total process memory.

Limitations include literal-token matching, stopword noise, vocabulary mismatch,
PDF artifacts, and redundant overlapping passages. The implementation scores the
small corpus in memory and makes no large-corpus performance claims. The BM25 method remains independent; hybrid RRF and cross-encoder reranking
are available as separate CLI modes.

## Hybrid retrieval with Reciprocal Rank Fusion

```sh
uv run python -m research_assistant search \
  "Why does the Transformer use positional encoding?" \
  --index data/index --method hybrid --top-k 5
```

Vector search remains the default. Hybrid reuses the saved `VectorIndex`, builds
BM25 statistics from the same ordered chunks, and loads one `Embedder` to encode
the question. It does not rebuild document embeddings or create a separate index.
For library usage, reuse all three existing instances:

```python
from research_assistant.retrieval.hybrid import HybridRetriever

hybrid = HybridRetriever(index, lexical, embedder, candidate_k=20, rrf_k=60)
results = hybrid.search("Why use positional encoding?", top_k=5)
```

`reciprocal_rank_fusion(rankings, top_k=5, rrf_k=60)` is independently usable.
For each chunk ID, it sums `1 / (rrf_k + rank)` across ranked lists, using 1-based
ranks and equal method weights. Raw BM25 and cosine scores are ignored: their
scales have different meanings, so simply adding them is not justified. Returned
`SearchResult.score` is an RRF score, not cosine similarity or a probability.

Chunks are deduplicated by ID. Within one list, only the first occurrence
contributes; subsequent positions retain their original ranks. Conflicting
metadata for the same ID raises an error. Metadata/text are preserved, inputs
are not mutated, and ties are broken lexicographically by chunk ID. `top_k` must
be a positive integer; `rrf_k` must be a nonnegative integer. Empty lists produce
no results. Hybrid indexes must contain exactly the same ordered Chunk records.

Each method supplies 20 candidates by default. If `top_k` exceeds `candidate_k`,
the effective depth automatically grows to `max(candidate_k, top_k)`; small corpora
or sparse lexical matches can still return fewer results. RRF cannot recover a
passage absent from both candidate lists. It loses raw score margins and can
promote broad consensus over a more directly answering passage. SHA-based ID
ordering is deterministic but has no relevance meaning. Overlapping chunks with
different IDs remain distinct, so fusion does not remove passage redundancy.

[All 60 comparison results and component ranks](docs/hybrid-evaluation.md) cover
the same four questions as the BM25 review. Hybrid ranks the complexity table
first, the FlashAttention tiling/recomputation comparison first, and the direct
RAG architecture explanation first. For positional encoding, the direct page-6
explanation is **hybrid rank 2**, RRF **0.03067916** (BM25 rank 1, vector rank 10).
The less-direct dropout passage wins at **0.03076923** (rank 5 in both lists).
This illustrates both recovery beyond vector top five and the limits of rank
fusion; four manually inspected examples do not establish general improvement.

## Cross-encoder reranking

```sh
uv run python -m research_assistant search \
  "Why does the Transformer use positional encoding?" \
  --index data/index --method rerank --top-k 5
```

This mode retrieves `max(20, top_k)` hybrid candidates, scores them with a
cross-encoder, and returns up to `top_k`. The hybrid retriever also expands its
component candidate depth as needed. Existing `vector`, `bm25`, and `hybrid`
modes are unchanged; vector remains the default. Persisted document embeddings
are reused, never recomputed.

```python
from research_assistant.retrieval.reranking import CrossEncoderReranker

reranker = CrossEncoderReranker(batch_size=16)
candidates = hybrid.search(question, top_k=20)
results = reranker.rerank(question, candidates, top_k=5)
```

The CPU model is `cross-encoder/ms-marco-MiniLM-L6-v2`, pinned to verified Hugging
Face revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. It loads once per instance;
reuse the instance across queries. The CLI creates an instance per invocation.
`CrossEncoder.predict()` jointly processes `(question, passage_text)` pairs in
batches (default 16). An explicit identity activation preserves raw relevance
logits: these scores are not probabilities and are not directly comparable to
BM25, cosine, or RRF scores.

The wrapper validates question/candidate types, nonempty text/IDs, unique IDs,
positive integer batch size/top_k, output count/shape, and finite scores. It
returns new SearchResult records, replacing only scores and preserving all
source metadata exactly. Descending score order uses original candidate position
for ties. Empty candidates return `[]` without prediction; the model was already
loaded at construction. Requests above the candidate count return all candidates.

```sh
RUN_RERANKING_INTEGRATION=1 uv run pytest tests/test_reranking.py -k real_model
```

Ordinary tests use a fake cross-encoder and do not download weights. The opt-in
check loads the real pinned model and verifies a simple relevant/unrelated
ranking and metadata preservation. Downloaded weights remain in the external
Hugging Face cache and are not committed.

Limitations: reranking cannot recover a passage missing from the hybrid candidate
pool. It adds CPU latency, is trained on MS MARCO rather than ML research papers,
and is sensitive to PDF artifacts and chunk boundaries. The question/passage pair
is limited to 512 model tokens; long inputs may be truncated. Overlapping chunks
can still dominate the output. Relevance logits are uncalibrated, may be negative,
and do not establish factual correctness. The `search` command does not generate answers; use `ask` for generation.

### Measured reranking comparison

[All 80 top-five results across four methods](docs/reranking-evaluation.md) include
scores, sources, physical pages, excerpts, and timing methodology. The direct
positional-encoding explanation moves from hybrid rank 2 to **reranked rank 1**
(raw score **3.436294**); the dropout passage falls from hybrid rank 1 to
**reranked rank 3** (**1.625410**). Results are mixed: the complexity query also
promotes a Linformer bibliography fragment to rank 3. Four examples are not a
labeled evaluation and do not establish overall improvement.

On Apple M5 Pro CPU, six PyTorch threads, batch size 16, 20 candidates, one
warm-up followed by 10 timed repetitions per question produced:

| Question | Candidate retrieval median ms | Cross-encoder predict median ms | Warm end-to-end median ms |
| --- | ---: | ---: | ---: |
| Self-attention complexity | 3.619 | 86.860 | 90.536 |
| FlashAttention transfers | 3.515 | 88.639 | 92.196 |
| RAG memory | 3.838 | 94.775 | 98.613 |
| Positional encoding | 3.722 | 94.501 | 98.218 |

Initial weight download/cache preparation took **25.096 s**; cross-encoder
initialization after prefetch took **2.732 s**, separately from all inference
measurements. Embedder initialization took **1.416 s**. Warm end-to-end includes
hybrid retrieval and reranking, but excludes imports, disk/index loading, model
initialization, and CLI output. `predict` timing includes tokenization and NumPy
conversion. Repeated model construction in separate CLI calls adds startup cost.

## Citation-grounded answers

`ask` loads the persisted index, retrieves 20 hybrid candidates, reranks them,
and sends the top 5 passages to the OpenAI Responses API. Document embeddings
are reused. Existing `search` modes remain local and do not construct an API client.

Configure credentials in your shell (do not commit them), and select a
Responses-compatible model available to your account:

```sh
export OPENAI_API_KEY='your-api-key'
export OPENAI_MODEL='your-model-id'
uv run python -m research_assistant ask \
  "Why does the Transformer use positional encoding?" --index data/index
```

There is no default generation model. Prefer a snapshot model ID when available
for reproducibility. `.env` files are ignored but are not automatically loaded.
The CLI prints the answer followed by its cited references: label, source path,
physical PDF page or Markdown section, and chunk ID.

`GroundedAnswerer(client, model=...)` accepts an injected OpenAI-shaped client;
`answer(question, passages)` accepts ordered `SearchResult` records. Frozen
`GroundedAnswer` records contain `answer` and a tuple of unique cited
`CitationSource` records in evidence order, retaining the exact chunk ID,
document ID, source, page, section, and passage text. Labels S1, S2, etc. are
stable for the same ordered evidence within an answer, not global document IDs.
Uncited passages are omitted from the returned source mapping.

Evidence is serialized deterministically as JSON in the request input, separate
from higher-priority instructions. Those instructions require evidence-only
answers, factual-claim citations, and acknowledgement of insufficient evidence;
passages and questions are treated as untrusted data, never instructions.
Empty evidence returns a fixed insufficient-evidence message without an API
request. For nonempty evidence, that exact fallback is also accepted without
citations. Otherwise at least one citation is required. Unknown or malformed
bracketed S-labels, empty output, and incomplete Responses are rejected with a
clear error rather than edited or presented as a valid answer. Repeated valid
citations produce only one source entry. API failures are surfaced by the CLI.

**Citation validity is not factual support.** Validation checks label membership,
not whether each claim follows from the cited passage or whether every factual
claim is cited. Prompt instructions and JSON separation do not prove resistance
to prompt injection. Retrieval omissions, PDF artifacts, overlapping evidence,
model hallucinations, and model-dependent abstention remain limitations. The
strict fallback/label format can reject otherwise useful responses. Deterministic
context and labels do not make LLM output deterministic. No factual-grounding
evaluation suite has been implemented.

The question and selected passages are sent to OpenAI. Each nonempty generation
uses one Responses call, with `store=False` and an default output budget of 1024 tokens
(including reasoning tokens). The CLI uses a 60-second SDK timeout and up to two
retries; retries can increase total latency. Input/output tokens incur API costs
that depend on the configured model. `GroundedAnswerer(..., max_output_tokens=1024)`
accepts a custom positive integer budget (booleans are rejected).
Questions are limited to 2,000 characters and total evidence passage text to
20,000 characters, inclusive. Oversized inputs raise `ValueError` before the API
call; nothing is silently truncated. These application-level limits count Python
string characters, not tokens; they do not guarantee token counts or API costs.
Evidence metadata and prompt instructions are not included in the text limit.
Network generation adds latency beyond retrieval and reranking; the earlier local
benchmarks do not measure `ask`. No live generation cost or latency has been
measured in this step. See the official [Responses API reference](https://developers.openai.com/api/reference/python/resources/responses/methods/create).

Offline tests inject a fake client, requiring neither credentials nor network.
The live smoke test is disabled by default and incurs API usage when enabled:

```sh
RUN_ANSWERING_INTEGRATION=1 uv run pytest tests/test_answering.py -k live_api -v
```

It requires both environment variables above and checks a simple citation
contract; it is not a factual-grounding evaluation.

## Local MCP server

Uses the official MCP Python SDK v2 `MCPServer` API (`mcp>=2,<3`, locked to
2.3.0), with no optional SDK extras. See the [official SDK documentation](https://py.sdk.modelcontextprotocol.io/).

```sh
uv run python -m research_assistant.interfaces.mcp_server --index data/index
```

The process waits for an MCP host on stdin/stdout; stdout is reserved for the
protocol and diagnostics go to stderr. There is no HTTP listener. Configure a
local host to launch this command from the repository root (or use absolute
paths). As with `ask`, answer generation needs `OPENAI_API_KEY` and `OPENAI_MODEL`
in the process environment. `.env` is not automatically loaded; optionally use
`uv run --env-file .env python -m research_assistant.interfaces.mcp_server --index data/index`.
Search alone needs neither OpenAI configuration nor paid API calls. Models may
be downloaded on first search. Answer calls with evidence can incur API charges.

### Local host example: VS Code Copilot

This native user configuration follows the [official VS Code MCP reference](https://code.visualstudio.com/docs/agents/reference/mcp-configuration)
(checked 2026-11-24). Merge it into `/Users/baraa/.copilot/mcp-config.json`
when `COPILOT_HOME` is unset; otherwise use `$COPILOT_HOME/mcp-config.json`.
Replace the absolute paths for your machine (`command -v uv` locates uv).

```json
{
  "mcpServers": {
    "researchAssistant": {
      "type": "stdio",
      "command": "/Users/baraa/.local/bin/uv",
      "args": [
        "run", "python", "-m", "research_assistant.interfaces.mcp_server",
        "--index", "/Users/baraa/Projects/rag-mcp-research/data/index"
      ],
      "cwd": "/Users/baraa/Projects/rag-mcp-research",
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "OPENAI_MODEL": "${OPENAI_MODEL}"
      },
      "tools": ["search_papers", "answer_question"]
    }
  }
}
```

`cwd` selects the uv project. Install dependencies with `uv sync --locked` there
first; the local index must exist. For answering, export credentials before
launching a fresh VS Code process from that shell:

```sh
export OPENAI_API_KEY='your-api-key'
export OPENAI_MODEL='your-model-id'
code /Users/baraa/Projects/rag-mcp-research
```

Do not store secrets in the configuration. For retrieval-only use, omit `env`.
Select the Copilot harness and manage the server through **Chat: Open Customizations →
MCP Servers**. This format does not load `.env` automatically. The example is
verified against documentation, not an executed VS Code host session. Calling
`answer_question` with evidence may incur OpenAI charges.

### Tools and lifecycle

Exactly two tools are registered:

- `search_papers(question, top_k=5, rerank=True)` returns a structured `results`
  list with 1-based `rank`, `chunk_id`, `document_id`, `source`, `page_number`,
  `section`, full `text`, and method-specific `score`. Order and metadata are
  preserved; empty searches return an empty list. Invalid top_k is not clamped.
- `answer_question(question)` returns `answer` and `sources`, each containing
  `label`, `chunk_id`, `document_id`, `source`, `page_number`, `section`, and
  evidence `text`. Citation-ID validity does not prove factual support.

The SDK lifespan loads one persisted index through one `ResearchService`, reuses
it across requests, and closes it in `finally`. Imports and tool registration do
not load models or indexes. An AnyIO lock serializes both tools; synchronous
service work runs in a worker thread without abandoning it on cancellation.
Shutdown waits for active service work before closing owned resources. Expected
input errors are reported safely; backend errors are sanitized rather than
returning provider messages, environment values, or stack traces. Our
application diagnostics use Python logging to stderr. Application code never
redirects process-wide stdout or file descriptors; third-party/native stdout
writes are not intercepted by our code. Such writes can still affect protocol
reliability (including before SDK transport startup); configure those libraries
to use stderr rather than relying on SDK-specific transport handling. Forced
process termination cannot promise cleanup, and an active inference/API request can delay shutdown.

Offline tests use the SDK's in-memory client and a fake service to verify schemas,
structured output, reuse, serialization, errors, and cleanup. No paid API calls
are made. The real stdio smoke tests launch the actual `uv run python -m` server
command through the official SDK subprocess transport. They exercise nonempty FAISS +
BM25 hybrid retrieval twice with a deterministic test embedder, plus empty-index
answering without API access. They check JSON-RPC stdout, stderr diagnostics,
service reuse, cleanup, and zero-exit EOF shutdown. Test-only Python startup
hooks inject dependencies; production code has no test mode. The tests observe
two private SDK transport helpers, so SDK upgrades may require test adjustments.
They do not validate real model quality or paid answer generation. Run them with:

```sh
uv run pytest tests/test_mcp_stdio.py -v
```

The tests require `uv` on PATH and installed project dependencies; no model
downloads or API credentials are needed. No HTTP transport, authentication, or
passage-lookup tool is exposed here.

### Opt-in real-model MCP smoke test

Run manually against the existing three-paper index (never run this script in CI):

```sh
uv run python scripts/smoke_mcp.py --index data/index
# Require cached models and prohibit Hugging Face downloads:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/smoke_mcp.py
```

The script starts the actual server through the official SDK stdio transport,
initializes MCP, checks the exact two tools, and searches for
"Why do Transformers need positional encoding?" with and without reranking.
Both queries are repeated on the same connection. The parent uses
`VectorIndex.load()` to validate the manifest, checksums, vector dimensions, and
chunk alignment, then builds a lookup of the original `Chunk` objects. This loads
the index without initializing embedding or reranking models in the parent.
It prints top-five passages and validates exact IDs, document IDs, source, text,
physical page, and section against those chunks, plus finite descending scores,
sequential ranks, unique results, and unchanged repeated results. The SDK context
managers close the session and subprocess, including on failure. A 180-second
overall timeout bounds the run. Dependencies must already be installed.

The first hybrid call includes lazy embedder initialization; the first reranked
call includes lazy cross-encoder initialization. Subsequent calls use warmed
models. Times are explicitly labeled single-run, client-observed round-trip latencies
including serialization, not rigorous inference benchmarks. Parent index validation
is excluded from MCP startup and request timings. On the local cached corpus (2026-11-23), startup + MCP
initialization took 3.178s, cold hybrid 0.204s, first reranked 0.198s, warm hybrid
0.005s, and warm reranked 0.093s. These are single-run observations, not a latency
distribution. The direct positional-encoding explanation from Attention page 6
was hybrid rank 2 (RRF 0.030478) and reranked rank 1 (raw score 2.534949).

Generation is disabled by default; the script does not forward OpenAI credentials
to the child unless explicitly enabled. To opt into a potentially paid call,
export `OPENAI_API_KEY` and `OPENAI_MODEL` and run:

```sh
uv run python scripts/smoke_mcp.py --with-answer
```

This checks returned citation labels and source metadata, not factual support.
The generation path was not executed during the retrieval-only verification.
`answer_question` has offline mocked and empty-index stdio coverage, but no live
paid API generation over MCP has been verified.
A legitimate insufficient-evidence response may have no sources. Errors fail
with diagnostics without printing API credentials or raw provider exceptions.

## Local HTTP API (S4 Step 3)

```sh
uv run python -m research_assistant.interfaces.api --index data/index
```

Uvicorn binds to `127.0.0.1:8000` with one worker and reload disabled. Interactive
OpenAPI documentation is at `http://127.0.0.1:8000/docs`. Module import and
`create_app(index_path)` do not initialize the index or models; lifespan loads
one index through `ResearchService`. Models remain lazy. Both operations share
one AnyIO lock and run in non-abandoned worker threads. Shutdown marks the
service unavailable, waits for active work, then closes it once.

```sh
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"Why use positional encoding?","top_k":5,"rerank":true}'
# Potentially paid: configure OPENAI_API_KEY and OPENAI_MODEL before server startup.
curl -X POST http://127.0.0.1:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"Why use positional encoding?"}'
```

| Endpoint | Contract |
| --- | --- |
| `GET /health` | `{"status":"ok"}`; no inference or API access |
| `GET /ready` | `{"status":"ready"}` when the persisted index loaded; 503 if unavailable |
| `POST /search` | Required nonblank string `question`; integer `top_k=5` in 1–50; boolean `rerank=true`. Returns ordered `results` with all SearchResult fields |
| `POST /answer` | Required nonblank string `question`, maximum 2,000 characters. Returns `answer` and complete citation `sources`, including labels and evidence text |

Request types are strict; unknown fields and invalid bodies return 422.
Unavailable service and queue-wait expiry return 503. OpenAI timeouts return 504;
connection failures, rate limits, and provider 5xx responses return 503; other
OpenAI API errors return 502. Unexpected backend/output failures return 500.
Messages are generic, without provider exception bodies or internal paths. A failed startup
aborts server startup rather than claiming readiness. Readiness confirms index
loading only, not warmed models, API credentials, or external API availability.
`.env` is not automatically loaded; export variables or explicitly use uv's
`--env-file .env` option. Citation validity is not proof of factual support.

Every HTTP response includes a server-generated `X-Request-ID` (client-supplied
IDs are ignored). Standard Python logging writes one JSON request record to
stderr; Uvicorn access logging is disabled by the entry point to avoid duplicate
request records. Logs contain no request bodies, headers, passage text, query
strings, or raw provider errors. Paths use matched route templates; unknown
paths are recorded as `[unmatched]`. Example record (illustrative, not a benchmark):

```json
{"request_id":"a891e9b4732d4e618f075a1b209cd638","method":"POST","path":"/search","status":200,"duration_ms":125.4,"error_category":null}
```

Error categories are `invalid_request`, `service_unavailable`, `queue_timeout`,
`upstream_timeout`, `upstream_unavailable`, `upstream_error`, and `internal_error`.
Other HTTP errors use `http_error`. Cancelled requests log `request_cancelled`;
499 denotes cancellation before a response starts, not a delivered response.

`/search` and `/answer` wait at most 30 seconds to acquire the shared service
lock. Configure this positive finite duration at startup:

```sh
uv run python -m research_assistant.interfaces.api --index data/index --queue-wait-timeout 10
```

Python callers can use `create_app(index_path, queue_wait_timeout=10)`.
This bounds **queue waiting only**, not model execution or end-to-end latency.
Workers remain non-abandoned; Python threads and native inference are not forcibly
interrupted. The existing OpenAI client retains its 60-second SDK timeout and two
retries, so total API time can exceed 60 seconds. Shutdown still waits for active
operations; waiting requests recheck availability after acquiring the lock.

This is a local, single-process API with no authentication, hard inference
timeout, bounded queue capacity, or rate limiting. Do not expose
it publicly or add workers as a workaround for serialized service access.
Active inference/API calls can delay shutdown. Offline tests mock ResearchService;
no paid HTTP answer-generation call has been verified.

## CPU Docker deployment (S4 Step 4)

Prerequisites: a running Docker Engine/Desktop, network access for dependency
installation, and a trusted persisted index built with this project. Run these
commands from the repository root. The image uses digest-pinned Python 3.11.17
and uv 0.13.5, installs locked runtime dependencies without the development group,
and runs as UID/GID `10001:10001`. Linux PyTorch uses the official CPU wheel index;
its existing 2.15.0 version is now an explicit dependency to select that source
using [uv's PyTorch configuration](https://docs.astral.sh/uv/guides/integration/pytorch/).
The build asserts that PyTorch has no CUDA runtime. No model weights are downloaded
at build time. Debian's `libgomp1` supplies OpenMP support; OS package repositories
and the package build backend are not snapshot-pinned, so this is not a claim of
bit-for-bit image reproducibility. Image digests require deliberate security updates.

```sh
docker build -t rag-mcp-research:local .
docker volume create rag-mcp-hf-cache
docker run -d --name rag-mcp-api \
  -p 127.0.0.1:8000:8000 \
  --mount "type=bind,src=$(pwd)/data/index,dst=/data/index,readonly" \
  --mount type=volume,src=rag-mcp-hf-cache,dst=/home/app/.cache/huggingface \
  rag-mcp-research:local
```

The index directory must contain `manifest.json`, `chunks.jsonl`, and
`vectors.faiss`. Startup validates their checksums and compatibility. The mount
is read-only and must be readable/traversable by UID 10001. No papers, local
indexes, credentials, caches, or benchmark outputs are copied into the image.
Only the installed application and runtime dependencies are copied from the build stage.

The fresh named volume inherits ownership from the image's cache directory.
[`HF_HOME`](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)
sets `/home/app/.cache/huggingface` as the download location; model
loading remains lazy. The first search may download embedding weights, and the
first reranked search may download cross-encoder weights. The cache persists across
container recreation. For a bind-mounted cache instead, provision a writable
directory owned by UID/GID 10001 on the host; Docker does not fix bind-mount
permissions. If overriding `HF_HOME`, mount a writable directory at that same path.

For optional paid `/answer` generation, export `OPENAI_API_KEY` and `OPENAI_MODEL`
in the calling shell, then add `-e OPENAI_API_KEY -e OPENAI_MODEL` before the image
name in `docker run`. Never pass secrets as image build arguments or copy `.env`
into the image. The application does not automatically load `.env`. Docker
administrators can inspect runtime environment variables; this is local development
credential handling, not a production secret store.

The container binds to `0.0.0.0:8000` internally; the published port above is
restricted to host loopback. Direct Python startup still defaults to `127.0.0.1`.
One Uvicorn worker owns one service. Exec-form startup forwards SIGTERM to Python;
shutdown waits for active non-abandoned operations before closing the service.
Queue waiting still defaults to 30 seconds. Override it by appending
`--index /data/index --host 0.0.0.0 --queue-wait-timeout 10` after the image name.

Smoke checks (no paid call or model download required):

```sh
# Poll for index initialization (60 attempts); fail visibly if it never succeeds.
ready=0
for attempt in $(seq 1 60); do
  if curl --fail --silent --max-time 1 http://127.0.0.1:8000/ready >/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
[ "$ready" -eq 1 ] || { docker logs rag-mcp-api; exit 1; }
curl --fail -i http://127.0.0.1:8000/health
curl --fail -i http://127.0.0.1:8000/ready
curl -i -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"test","top_k":0}'
docker logs rag-mcp-api
```

Expect 200 for health/readiness, 422 for the invalid request, `X-Request-ID` on
all three responses, and JSON request records in stderr logs. `/ready` confirms
index initialization only, not model warm-up or provider availability. After an
idle stop, expect completed application shutdown and exit code 0. Docker may
force-kill work that outlasts the stop grace period; adjust it to workload needs.
This unauthenticated single-process service must not be exposed publicly. No
Compose file is needed for this single-container setup.

Manual Docker Desktop verification on November 26, 2026 confirmed build/startup,
health/readiness 200, invalid search 422, request IDs, JSON logs, CPU embedding and
reranking, and shutdown exit code 0. The build took 52.2 seconds. See the
[compact results table, latency observations and native/Docker comparison](docs/integration.md).
These supplied manual observations supersede the earlier daemon-unavailable check;
they were not rerun in the documentation audit.

Optional real retrieval after readiness (may download weights; no OpenAI call):

```sh
for rerank in false true; do
  curl --fail -i -w '\nClient total: %{time_total} s\n' \
    http://127.0.0.1:8000/search \
    -H 'Content-Type: application/json' \
    -d "{\"question\":\"How does FlashAttention reduce GPU memory access?\",\"top_k\":3,\"rerank\":$rerank}"
done
```

Repeat to observe warmed requests. First hybrid and first reranked calls initialize
different models; a persisted disk cache does not preserve in-memory models across
restart. Individual times are not rigorous benchmarks or performance guarantees.

When finished, stop and remove the container (the named model-cache volume remains):

```sh
docker stop --timeout 120 rag-mcp-api
docker logs rag-mcp-api
docker inspect --format '{{.State.ExitCode}}' rag-mcp-api
docker rm rag-mcp-api
```

## Five-session roadmap (10 hours total)

The original budget was approximately two hours per session; this is a scope
roadmap, not a record of measured time spent.

| Session | Planned time | Status | Scope |
| --- | --- | --- | --- |
| 1 | ~2 hours | Complete | PDF/Markdown ingestion, chunking, Sentence Transformer embeddings, FAISS vector search |
| 2 | ~2 hours | Complete | BM25 hybrid retrieval, reranking, citation-grounded answers; initial pooled offline evaluation completed in S2C |
| 3 | ~2 hours | Complete | ResearchService, two-tool stdio MCP server, automated subprocess tests, real-model retrieval smoke test, documentation; live paid MCP answering remains unverified |
| 4 | ~2 hours | Complete | Local FastAPI, observability, queue-wait limits, CPU Docker deployment, manual HTTP retrieval integration and final documentation audit |
| 5 | ~2 hours | Upcoming | Broader evaluation, benchmarking, and GitHub documentation building on the existing five-question evaluation and timing measurements |

Next: S5 expanded retrieval evaluation after review. No new experiments are included in this S4 documentation audit.

Intended GitHub owner: `baraa42`. No remote repository is required for local setup.
