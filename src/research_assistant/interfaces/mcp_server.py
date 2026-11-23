"""Local stdio MCP tools backed by one sequential ResearchService."""

import argparse
import logging
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import anyio
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from research_assistant.service import ResearchService

logger = logging.getLogger(__name__)


@dataclass
class ServerState:
    service: ResearchService
    lock: anyio.Lock


def _invoke(operation: Callable[[], Any]) -> Any:
    """Report safe errors without modifying process-wide output streams."""
    try:
        return operation()
    except ValueError as exc:
        # Only fixed validation messages are safe to expose, never user/provider data.
        safe = {
            "question must be a nonempty string",
            "top_k must be a positive integer",
            "rerank must be a boolean",
            "Set OPENAI_MODEL or provide a nonempty model name",
            "question exceeds 2000 characters",
            "Total evidence text exceeds 20000 characters",
        }
        message = (
            str(exc) if str(exc) in safe else "Research request or evidence is invalid."
        )
        logger.warning("Research request rejected: %s", message)
        raise ToolError(message) from None
    except Exception:
        logger.error("Research operation failed; backend details withheld.")
        raise ToolError(
            "Research operation failed. Check the index, model configuration and API access."
        ) from None


def create_server(index_path: Path) -> MCPServer:
    """Register tools without loading an index, models, or API credentials."""

    @asynccontextmanager
    async def lifespan(server: MCPServer) -> AsyncIterator[ServerState]:
        service = ResearchService(index_path)
        state = ServerState(service, anyio.Lock())
        try:
            yield state
        finally:
            # Wait for active non-abandoned worker calls before closing API resources.
            with anyio.CancelScope(shield=True):
                async with state.lock:
                    service.close()

    server = MCPServer("rag-mcp-research", lifespan=lifespan, log_level="WARNING")

    @server.tool()
    async def search_papers(
        question: Annotated[str, Field(strict=True)],
        ctx: Context[ServerState],
        top_k: Annotated[int, Field(strict=True)] = 5,
        rerank: Annotated[bool, Field(strict=True)] = True,
    ) -> dict[str, Any]:
        """Retrieve research passages; scores are method-specific relevance values."""
        state = ctx.request_context.lifespan_context
        async with state.lock:
            search_operation = partial(state.service.search, question, top_k, rerank)
            safe_operation = partial(_invoke, search_operation)
            results = await anyio.to_thread.run_sync(
                safe_operation,
                abandon_on_cancel=False,
            )
        return {
            "results": [
                dict(rank=rank, **asdict(result))
                for rank, result in enumerate(results, 1)
            ]
        }

    @server.tool()
    async def answer_question(
        question: Annotated[str, Field(strict=True)], ctx: Context[ServerState]
    ) -> dict[str, Any]:
        """Answer from evidence with validated source IDs, not proof of factual support."""
        state = ctx.request_context.lifespan_context
        async with state.lock:
            answer_operation = partial(state.service.answer, question)
            safe_operation = partial(_invoke, answer_operation)
            answer = await anyio.to_thread.run_sync(
                safe_operation,
                abandon_on_cancel=False,
            )
        return {"answer": answer.answer, "sources": [asdict(s) for s in answer.sources]}

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    try:
        create_server(args.index).run(transport="stdio")
    except KeyboardInterrupt:
        pass
    except Exception:
        # Startup/shutdown exceptions may include paths or provider credentials.
        parser.exit(
            1, "Error: MCP server failed; check index and service configuration.\n"
        )


if __name__ == "__main__":
    main()
