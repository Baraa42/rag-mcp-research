"""Minimal local index/search commands; retrieval remains usable as a library."""

import argparse
import os
from pathlib import Path

from openai import OpenAIError

from research_assistant.ingest.chunking import chunk_pages
from research_assistant.ingest.ingestion import ingest_directory
from research_assistant.retrieval.bm25 import BM25Index
from research_assistant.retrieval.embeddings import Embedder
from research_assistant.retrieval.vector import VectorIndex
from research_assistant.service import ResearchService


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Index and search research papers")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("index", help="Build an exact vector index")
    build.add_argument("directory", type=Path)
    build.add_argument("--out", type=Path, required=True)
    search = commands.add_parser("search", help="Retrieve passages for a question")
    search.add_argument("question")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument(
        "--method", choices=("vector", "bm25", "hybrid", "rerank"), default="vector"
    )
    ask = commands.add_parser("ask", help="Answer using reranked evidence and citations")
    ask.add_argument("question")
    ask.add_argument("--index", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            pages = ingest_directory(args.directory)
            chunks = chunk_pages(pages)
            vectors = Embedder().encode([chunk.text for chunk in chunks])
            VectorIndex(chunks, vectors).save(args.out)
            print(
                f"Indexed {len(chunks)} chunks from {len(pages)} pages into {args.out}"
            )
        elif args.command == "ask":
            if not args.question.strip():
                raise ValueError("Provide a nonempty question")
            # Fail before index/model loading without constructing an API client.
            for name in ("OPENAI_MODEL", "OPENAI_API_KEY"):
                if not os.environ.get(name, "").strip():
                    raise ValueError(f"Set {name} to a nonempty value before using ask")
            service = ResearchService(args.index)
            try:
                answer = service.answer(args.question)
            finally:
                service.close()
            print(answer.answer)
            for source in answer.sources:
                location = (
                    f"page {source.page_number}"
                    if source.page_number is not None
                    else f"section {source.section or '(untitled)'}"
                )
                print(
                    f"[{source.label}] {source.source} | {location}"
                    f" | chunk {source.chunk_id}"
                )
        else:
            if args.top_k <= 0 or not args.question.strip():
                raise ValueError("Provide a nonempty question and positive --top-k")
            if args.method in ("hybrid", "rerank"):
                service = ResearchService(args.index)
                try:
                    results = service.search(
                        args.question, args.top_k, rerank=args.method == "rerank"
                    )
                finally:
                    service.close()
            else:
                index = VectorIndex.load(args.index)
                if args.method == "bm25":
                    results = BM25Index(list(index.chunks)).search(
                        args.question, args.top_k
                    )
                else:
                    results = (
                        index.search(args.question, Embedder(), args.top_k)
                        if index.chunks
                        else []
                    )
            if not results:
                print("No results: empty index or no matching terms.")
            for rank, result in enumerate(results, 1):
                location = (
                    f"page {result.page_number}"
                    if result.page_number is not None
                    else f"section {result.section or '(untitled)'}"
                )
                print(
                    f"\n{rank}. score={result.score:.6f} | {result.source} | {location}"
                )
                print(result.text)
    except (OSError, ValueError, TypeError, RuntimeError, OpenAIError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
