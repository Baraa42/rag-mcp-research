# rag-mcp-research

A production-oriented research assistant for ML papers, planned to combine
hybrid retrieval with citation-grounded answers and expose research tools
through MCP and FastAPI.

**Status:** Session 1, Step 1 — repository setup only. Retrieval, ingestion,
answer generation, and servers are not implemented yet.

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
- Internet access for initial dependency installation; later sessions will also
  download embedding and reranking models

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

The smoke test verifies only package importability, not any ML functionality.

## Five-session roadmap (10 hours total)

| Session | Time | Scope |
| --- | --- | --- |
| 1 | ~2 hours | PDF/Markdown ingestion, chunking, Sentence Transformer embeddings, FAISS vector search |
| 2 | ~2 hours | BM25 hybrid retrieval, reranking, citation-grounded answers |
| 3 | ~2 hours | MCP server and research tools |
| 4 | ~2 hours | FastAPI, Docker, reliability, logging |
| 5 | ~2 hours | Evaluation, benchmarking, GitHub documentation |

Next: Session 1, Step 2 will implement PDF/Markdown ingestion with source
metadata and focused tests. No future component placeholder modules are present.

Intended GitHub owner: `baraa42`. No remote repository is required for local setup.
