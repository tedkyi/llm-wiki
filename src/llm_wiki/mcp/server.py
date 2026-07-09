"""Wiki-agnostic MCP server: declares the tools, delegates to `adapter`.

This module deliberately knows nothing about the wiki's internals — it imports
only `adapter` and translates its plain-dict results / errors into MCP tool
responses. The tool signatures (typed params + docstrings) are the contract the
client sees; FastMCP generates the JSON-Schema from them.
"""

from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from . import adapter

Scope = Literal["wiki", "raw", "hybrid"]
Mode = Literal["hybrid", "lex", "vec"]
Grounding = Literal["strict", "augmented"]

_INSTRUCTIONS = (
    "Query an LLM-maintained personal wiki. Use `search_wiki` for RAG-style "
    "retrieval of ranked page excerpts, `read_wiki_page` to fetch a full page, "
    "`ask_wiki` to get a synthesized natural-language answer, and `wiki_status` "
    "to check backend health. All tools are read-only."
)


def build_server(
    ctx: adapter.WikiContext,
    *,
    name: str = "llm-wiki",
    host: str = "127.0.0.1",
    port: int = 8010,
    path: str = "/mcp",
) -> FastMCP:
    """Build a FastMCP server bound to one wiki project."""
    mcp = FastMCP(
        name,
        instructions=_INSTRUCTIONS,
        host=host,
        port=port,
        streamable_http_path=path,
    )

    @mcp.tool()
    def search_wiki(
        query: str,
        scope: Scope = "wiki",
        limit: int = 8,
        mode: Mode = "hybrid",
        rerank: bool = True,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """Search the wiki and return ranked hits with their full page content.

        RAG-style retrieval — no answer is synthesized, no LLM is called.

        scope: 'wiki' (curated pages), 'raw' (source document text), or 'hybrid'.
        mode: 'hybrid' (BM25+vector+rerank), 'lex' (keyword), or 'vec' (semantic).
        Returns {"hits": [{title, path, collection, score, snippet, content}], "count"}.
        """
        return adapter.do_search(
            ctx,
            query,
            scope=scope,
            limit=limit,
            mode=mode,
            rerank=rerank,
            min_score=min_score,
        )

    @mcp.tool()
    def read_wiki_page(path: str) -> dict[str, Any]:
        """Read one wiki page by its wiki-relative path (e.g. 'concepts/foo.md').

        Returns {title, path, frontmatter, body}. Errors if the page is missing
        or the path escapes the wiki root.
        """
        try:
            return adapter.do_read_page(ctx, path)
        except adapter.WikiError as e:
            raise ToolError(str(e)) from e

    @mcp.tool()
    def ask_wiki(
        question: str,
        scope: Scope = "wiki",
        grounding: Grounding = "strict",
        limit: int = 8,
        page_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Ask a natural-language question and get a synthesized answer.

        Runs the full retrieve→synthesize pipeline (costs one LLM call on the
        wiki's configured provider). Read-only: never saves a synthesis page.

        grounding: 'strict' answers only from sources; 'augmented' may add the
        model's own knowledge, labeled. page_types optionally restricts wiki-page
        hits to kinds like ['entities', 'concepts'].
        Returns {"answer", "sources": [{title, path, score}]}.
        """
        try:
            return adapter.do_ask(
                ctx,
                question,
                scope=scope,
                grounding=grounding,
                limit=limit,
                page_types=page_types,
            )
        except adapter.WikiError as e:
            raise ToolError(str(e)) from e

    @mcp.tool()
    def wiki_status() -> dict[str, Any]:
        """Report search-backend health and wiki size.

        Returns {initialized, root, backend_installed, backend_version,
        collections, page_count, source_count}.
        """
        return adapter.do_status(ctx)

    return mcp
