"""Minimal local index/search commands; retrieval remains usable as a library."""

import argparse
from pathlib import Path

from openai import OpenAI, OpenAIError

from research_assistant.answering import GroundedAnswerer
from research_assistant.bm25 import BM25Index
from research_assistant.chunking import chunk_pages
from research_assistant.embeddings import Embedder
from research_assistant.hybrid import HybridRetriever
from research_assistant.ingestion import ingest_directory
from research_assistant.reranking import CrossEncoderReranker
from research_assistant.retrieval import VectorIndex


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
            index = VectorIndex.load(args.index)
            # Configure generation before expensive local model initialization.
            with OpenAI(timeout=60.0, max_retries=2) as client:
                answerer = GroundedAnswerer(client)
                passages = []
                if index.chunks:
                    candidates = HybridRetriever(
                        index, BM25Index(list(index.chunks)), Embedder()
                    ).search(args.question, 20)
                    passages = CrossEncoderReranker().rerank(
                        args.question, candidates, 5
                    )
                answer = answerer.answer(args.question, passages)
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
            index = VectorIndex.load(args.index)
            if args.method == "bm25":
                results = BM25Index(list(index.chunks)).search(
                    args.question, args.top_k
                )
            elif args.method == "rerank":
                results = []
                if index.chunks:
                    candidates = HybridRetriever(
                        index, BM25Index(list(index.chunks)), Embedder()
                    ).search(args.question, max(20, args.top_k))
                    results = CrossEncoderReranker().rerank(
                        args.question, candidates, args.top_k
                    )
            elif args.method == "hybrid":
                results = (
                    HybridRetriever(
                        index, BM25Index(list(index.chunks)), Embedder()
                    ).search(args.question, args.top_k)
                    if index.chunks
                    else []
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
