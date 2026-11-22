# rag-mcp-research

A production-oriented research assistant for ML papers, planned to combine
hybrid retrieval with citation-grounded answers and expose research tools
through MCP and FastAPI.

**Status:** Session 2B, Step 4 — minimal citation-grounded answer generation is
available after hybrid retrieval and cross-encoder reranking. MCP and FastAPI
are not implemented.

## Planned architecture

PDF/Markdown → text with source metadata → token-aware chunks → Sentence
Transformer embeddings → FAISS vector search + BM25 lexical search → hybrid
retrieval → cross-encoder reranking → citation-grounded answers.

The Python package will keep ingestion, retrieval, and answering logic separate
from future MCP and FastAPI interfaces. Docker deployment, evaluation, and
benchmarks will follow. No LangChain or LlamaIndex is planned.

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
    └── cli.py
```

Internal imports use these subpackage paths; no legacy module aliases are provided.
Hybrid and reranked CLI searches delegate to `ResearchService`. The `ask` CLI
retains its existing orchestration to preserve API configuration checks before
model loading, including its empty-index configuration requirement. The service
instead initializes generation lazily and bypasses it for empty evidence.
MCP and FastAPI interfaces are not implemented yet.

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

## Five-session roadmap (10 hours total)

| Session | Time | Scope |
| --- | --- | --- |
| 1 | ~2 hours | PDF/Markdown ingestion, chunking, Sentence Transformer embeddings, FAISS vector search |
| 2 | ~2 hours | BM25 hybrid retrieval, reranking, citation-grounded answers |
| 3 | ~2 hours | MCP server and research tools |
| 4 | ~2 hours | FastAPI, Docker, reliability, logging |
| 5 | ~2 hours | Evaluation, benchmarking, GitHub documentation |

Next steps will be implemented separately after reviewing answer generation.

Intended GitHub owner: `baraa42`. No remote repository is required for local setup.
