"""The wiki-coupled adapter — the entire interaction surface with `llm_wiki`.

Every call into the wiki lives here. The MCP tool layer (`server.py`) only ever
calls the `do_*` functions below and never imports `search`, `query`, etc. These
functions return plain, JSON-serializable dicts; no `SearchHit`, `QueryResult`,
`WikiPaths`, or other internal type ever crosses this boundary.

To point the MCP server at a different wiki backend, this is the only file that
would need to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import config as cfg
from .. import credentials, db, page_writer, search
from .. import query as query_module
from ..llm import LLMRouter


# ---------------------------------------------------------------------------
# Errors — the tool layer maps these to MCP tool errors
# ---------------------------------------------------------------------------


class WikiError(Exception):
    """Base error for adapter operations."""


class PageNotFound(WikiError):
    """Requested page does not exist."""


class PathOutsideWiki(WikiError):
    """Requested path is absolute or escapes the wiki root."""


class AskError(WikiError):
    """The query pipeline returned an error (no results, provider down, …)."""


# scope → QMD collections, mirroring query.run_query's mapping.
_SCOPE_COLLECTIONS: dict[str, list[str]] = {
    "wiki": ["llm-wiki-pages"],
    "raw": ["llm-wiki-raw"],
    "hybrid": ["llm-wiki-pages", "llm-wiki-raw"],
}

# Wiki page folders counted for `page_count`, excluding meta files. Mirrors the
# dashboard's `_count_md`.
_PAGE_FOLDERS = ("entities", "concepts", "sources", "synthesis")


# ---------------------------------------------------------------------------
# Project binding
# ---------------------------------------------------------------------------


@dataclass
class WikiContext:
    """A bound wiki project. Holds resolved paths and a lazily-built router.

    The router is only constructed on the first `do_ask` call, so read-only
    tools (`do_search`, `do_read_page`, `do_status`) work even when no LLM
    provider is configured or reachable.
    """

    paths: cfg.WikiPaths
    _config: dict | None = field(default=None, repr=False)
    _router: LLMRouter | None = field(default=None, repr=False)


def bind_project(root: Path | str | None = None) -> WikiContext:
    """Resolve and bind a single wiki project.

    With `root=None`, discovers the project from the current directory. Raises
    `WikiError` if no initialized wiki is found.
    """
    if root is None:
        found = cfg.find_wiki_root()
        if found is None:
            raise WikiError(
                "Not inside an LLM-Wiki project. Launch `wiki mcp` from a wiki "
                "directory, or pass --root."
            )
        resolved = found
    else:
        resolved = Path(root)

    paths = cfg.WikiPaths(root=resolved)
    if not paths.is_initialized():
        raise WikiError(f"No LLM-Wiki project at {resolved} (missing .wiki/config.yml).")
    return WikiContext(paths=paths)


def _load_config(ctx: WikiContext) -> dict:
    if ctx._config is None:
        ctx._config = cfg.load_config(ctx.paths)
    return ctx._config


def _build_router(ctx: WikiContext) -> LLMRouter:
    """Build (once) and cache the LLM router for this context."""
    if ctx._router is None:
        credentials.prime_environment()
        config = _load_config(ctx)
        ctx._router = LLMRouter(config.get("llm", {}))
    return ctx._router


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def _resolve_within_wiki(ctx: WikiContext, rel_path: str) -> Path:
    """Resolve `rel_path` against the wiki folder, rejecting escapes.

    Blocks absolute paths and any path (including via `..` or symlinks) that
    resolves outside `paths.wiki`.
    """
    candidate = Path(rel_path)
    if candidate.is_absolute():
        raise PathOutsideWiki("path outside wiki root")

    wiki_root = ctx.paths.wiki.resolve()
    resolved = (wiki_root / candidate).resolve()
    if not resolved.is_relative_to(wiki_root):
        raise PathOutsideWiki("path outside wiki root")
    return resolved


# ---------------------------------------------------------------------------
# Result shaping
# ---------------------------------------------------------------------------


def _hit_to_dict(hit: search.SearchHit) -> dict[str, Any]:
    return {
        "title": hit.title,
        "path": hit.path,
        "collection": hit.collection,
        "score": hit.score,
        "snippet": hit.snippet,
        "content": hit.full_content,
    }


def _count_pages(paths: cfg.WikiPaths) -> int:
    total = 0
    for sub in _PAGE_FOLDERS:
        folder = paths.wiki / sub
        if not folder.exists():
            continue
        total += sum(
            1
            for p in folder.glob("*.md")
            if not p.name.startswith(".") and not p.name.startswith("lint-report-")
        )
    return total


# ---------------------------------------------------------------------------
# Tool operations
# ---------------------------------------------------------------------------


def do_search(
    ctx: WikiContext,
    query: str,
    *,
    scope: str = "wiki",
    limit: int = 8,
    mode: str = "hybrid",
    rerank: bool = True,
    min_score: float = 0.0,
) -> dict[str, Any]:
    """RAG-style retrieval. Returns hydrated hits; makes no LLM call."""
    collections = _SCOPE_COLLECTIONS.get(scope)
    results = search.query(
        ctx.paths,
        query,
        mode=mode,
        limit=limit,
        min_score=min_score,
        collections=collections,
        hydrate=True,
        rerank=rerank,
    )
    hits = [_hit_to_dict(h) for h in results.hits]
    return {"hits": hits, "count": len(hits)}


def do_read_page(ctx: WikiContext, path: str) -> dict[str, Any]:
    """Read one wiki page by its wiki-relative path. Enforces path safety."""
    resolved = _resolve_within_wiki(ctx, path)
    parsed = page_writer.read_page(resolved)
    if parsed is None:
        raise PageNotFound(f"page not found: {path}")
    title = parsed.frontmatter.get("title") or resolved.stem
    rel = resolved.relative_to(ctx.paths.wiki.resolve()).as_posix()
    return {
        "title": title,
        "path": rel,
        "frontmatter": parsed.frontmatter,
        "body": parsed.body,
    }


def do_ask(
    ctx: WikiContext,
    question: str,
    *,
    scope: str = "wiki",
    grounding: str = "strict",
    limit: int = 8,
    page_types: list[str] | None = None,
) -> dict[str, Any]:
    """User-style query: retrieve + synthesize. Costs one LLM call. Never writes."""
    router = _build_router(ctx)
    result = query_module.run_query(
        ctx.paths,
        router,
        question,
        query_module.QueryCallbacks(),  # all no-ops; we use the return value
        scope=scope,
        grounding=grounding,
        limit=limit,
        page_types=page_types,
        save_as=None,
    )
    if result.error:
        raise AskError(result.error)
    sources = [
        {"title": h.title, "path": h.path, "score": h.score} for h in result.hits
    ]
    return {"answer": result.answer, "sources": sources}


def do_status(ctx: WikiContext) -> dict[str, Any]:
    """Backend health + wiki size. Cheap; no LLM call."""
    status = search.get_status(ctx.paths)
    stats = db.get_stats(ctx.paths.state_db)
    return {
        "initialized": ctx.paths.is_initialized(),
        "root": str(ctx.paths.root),
        "backend_installed": status.installed,
        "backend_version": status.version,
        "collections": status.collections,
        "page_count": _count_pages(ctx.paths),
        "source_count": stats.get("sources_total", 0),
    }
