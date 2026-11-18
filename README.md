# rag-mcp-research

A production-oriented research assistant for ML papers, planned to combine
hybrid retrieval with citation-grounded answers and expose research tools
through MCP and FastAPI.

**Status:** Session 1, Step 3 — ingestion and token-aware chunking implemented.
Embeddings, retrieval, answer generation, and servers are not implemented yet.

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
- Internet access for initial dependencies and the first tokenizer download;
  later sessions will also download embedding and reranking models

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

## Five-session roadmap (10 hours total)

| Session | Time | Scope |
| --- | --- | --- |
| 1 | ~2 hours | PDF/Markdown ingestion, chunking, Sentence Transformer embeddings, FAISS vector search |
| 2 | ~2 hours | BM25 hybrid retrieval, reranking, citation-grounded answers |
| 3 | ~2 hours | MCP server and research tools |
| 4 | ~2 hours | FastAPI, Docker, reliability, logging |
| 5 | ~2 hours | Evaluation, benchmarking, GitHub documentation |

Next: Sentence Transformer embeddings will be implemented separately.

Intended GitHub owner: `baraa42`. No remote repository is required for local setup.
