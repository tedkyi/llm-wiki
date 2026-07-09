"""Unit tests for the MCP adapter — the only wiki-coupled module.

The wiki seams are monkeypatched so these are fast and hermetic. Behavior under
test: dict-shaping of results, scope→collection mapping, path-safety, lazy router
construction, and error translation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_wiki import config as cfg
from llm_wiki import page_writer, search
from llm_wiki import query as query_module
from llm_wiki.mcp import adapter


# ---------------------------------------------------------------------------
# Helpers to build real llm_wiki result objects (no mocks of the data types)
# ---------------------------------------------------------------------------


def _hit(path: str, collection: str = "llm-wiki-pages", **kw) -> search.SearchHit:
    return search.SearchHit(
        docid=kw.get("docid", "#abc"),
        path=path,
        collection=collection,
        title=kw.get("title", "Some Title"),
        score=kw.get("score", 0.9),
        snippet=kw.get("snippet", "a snippet"),
        full_content=kw.get("full_content", "full body text"),
    )


# ---------------------------------------------------------------------------
# bind_project
# ---------------------------------------------------------------------------


def test_bind_project_explicit_root(wiki_root: Path):
    ctx = adapter.bind_project(wiki_root)
    assert ctx.paths.root == wiki_root


def test_bind_project_uninitialized_raises(tmp_path: Path):
    with pytest.raises(adapter.WikiError):
        adapter.bind_project(tmp_path / "not-a-wiki")


def test_bind_project_no_project_found(monkeypatch):
    monkeypatch.setattr(cfg, "find_wiki_root", lambda start=None: None)
    with pytest.raises(adapter.WikiError):
        adapter.bind_project(None)


# ---------------------------------------------------------------------------
# do_search
# ---------------------------------------------------------------------------


def test_do_search_shapes_hits(paths, monkeypatch):
    captured = {}

    def fake_query(p, q, **kwargs):
        captured["q"] = q
        captured.update(kwargs)
        return search.SearchResults(
            query=q, hits=[_hit("concepts/a.md"), _hit("concepts/b.md")]
        )

    monkeypatch.setattr(search, "query", fake_query)
    ctx = adapter.bind_project(paths.root)

    out = adapter.do_search(ctx, "amortized", scope="wiki", limit=5)

    assert out["count"] == 2
    assert len(out["hits"]) == 2
    first = out["hits"][0]
    assert set(first) == {"title", "path", "collection", "score", "snippet", "content"}
    assert first["content"] == "full body text"
    # forwarded params
    assert captured["limit"] == 5
    assert captured["hydrate"] is True
    assert captured["collections"] == ["llm-wiki-pages"]


@pytest.mark.parametrize(
    "scope,expected",
    [
        ("wiki", ["llm-wiki-pages"]),
        ("raw", ["llm-wiki-raw"]),
        ("hybrid", ["llm-wiki-pages", "llm-wiki-raw"]),
    ],
)
def test_do_search_scope_to_collections(paths, monkeypatch, scope, expected):
    captured = {}
    monkeypatch.setattr(
        search,
        "query",
        lambda p, q, **kw: captured.update(kw) or search.SearchResults(query=q),
    )
    ctx = adapter.bind_project(paths.root)
    adapter.do_search(ctx, "x", scope=scope)
    assert captured["collections"] == expected


def test_do_search_empty(paths, monkeypatch):
    monkeypatch.setattr(
        search, "query", lambda p, q, **kw: search.SearchResults(query=q, hits=[])
    )
    ctx = adapter.bind_project(paths.root)
    out = adapter.do_search(ctx, "nothing")
    assert out == {"hits": [], "count": 0}


def test_do_search_forwards_mode_rerank_minscore(paths, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        search,
        "query",
        lambda p, q, **kw: captured.update(kw) or search.SearchResults(query=q),
    )
    ctx = adapter.bind_project(paths.root)
    adapter.do_search(ctx, "x", mode="lex", rerank=False, min_score=0.4)
    assert captured["mode"] == "lex"
    assert captured["rerank"] is False
    assert captured["min_score"] == 0.4


# ---------------------------------------------------------------------------
# do_read_page — including path safety (security-critical)
# ---------------------------------------------------------------------------


def test_do_read_page_ok(paths):
    ctx = adapter.bind_project(paths.root)
    out = adapter.do_read_page(ctx, "concepts/amortized-inference.md")
    assert out["title"] == "Amortized Inference"
    assert out["path"] == "concepts/amortized-inference.md"
    assert out["frontmatter"]["tags"] == ["ml", "inference"]
    assert "Amortized inference trades" in out["body"]


def test_do_read_page_missing(paths):
    ctx = adapter.bind_project(paths.root)
    with pytest.raises(adapter.PageNotFound):
        adapter.do_read_page(ctx, "concepts/does-not-exist.md")


@pytest.mark.parametrize(
    "bad",
    [
        "../.wiki/secret.txt",
        "../../etc/passwd",
        "concepts/../../.wiki/secret.txt",
        "..",
    ],
)
def test_do_read_page_traversal_blocked(paths, bad):
    ctx = adapter.bind_project(paths.root)
    with pytest.raises(adapter.PathOutsideWiki):
        adapter.do_read_page(ctx, bad)


def test_do_read_page_absolute_blocked(paths):
    ctx = adapter.bind_project(paths.root)
    abs_path = str(paths.internal / "secret.txt")
    with pytest.raises(adapter.PathOutsideWiki):
        adapter.do_read_page(ctx, abs_path)


def test_do_read_page_symlink_escape_blocked(paths, tmp_path):
    """A symlink inside wiki/ pointing outside the root must not be readable."""
    outside = tmp_path / "outside.md"
    outside.write_text("---\ntitle: X\n---\nsecret\n", encoding="utf-8")
    link = paths.wiki / "concepts" / "escape.md"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this platform/run")
    ctx = adapter.bind_project(paths.root)
    with pytest.raises(adapter.PathOutsideWiki):
        adapter.do_read_page(ctx, "concepts/escape.md")


# ---------------------------------------------------------------------------
# do_ask
# ---------------------------------------------------------------------------


def test_do_ask_shapes_result(paths, monkeypatch):
    captured = {}

    def fake_run_query(p, router, question, callbacks, **kwargs):
        captured.update(kwargs)
        captured["router"] = router
        return query_module.QueryResult(
            question=question,
            answer="The answer.",
            hits=[_hit("concepts/a.md", title="A", score=0.8)],
        )

    monkeypatch.setattr(query_module, "run_query", fake_run_query)
    # Don't actually build a router; hand back a sentinel.
    monkeypatch.setattr(adapter, "_build_router", lambda ctx: "ROUTER")

    ctx = adapter.bind_project(paths.root)
    out = adapter.do_ask(ctx, "what is X?", scope="wiki", grounding="augmented")

    assert out["answer"] == "The answer."
    assert out["sources"] == [{"title": "A", "path": "concepts/a.md", "score": 0.8}]
    assert captured["save_as"] is None  # never writes
    assert captured["scope"] == "wiki"
    assert captured["grounding"] == "augmented"
    assert captured["router"] == "ROUTER"


def test_do_ask_error_raises(paths, monkeypatch):
    monkeypatch.setattr(adapter, "_build_router", lambda ctx: "ROUTER")
    monkeypatch.setattr(
        query_module,
        "run_query",
        lambda *a, **k: query_module.QueryResult(
            question="q", error="No matching wiki pages found."
        ),
    )
    ctx = adapter.bind_project(paths.root)
    with pytest.raises(adapter.AskError, match="No matching"):
        adapter.do_ask(ctx, "q")


# ---------------------------------------------------------------------------
# Lazy router — search/read/status must NOT build a router
# ---------------------------------------------------------------------------


def test_router_is_lazy(paths, monkeypatch):
    calls = []
    monkeypatch.setattr(adapter, "_build_router", lambda ctx: calls.append(1) or "R")
    monkeypatch.setattr(
        search, "query", lambda p, q, **kw: search.SearchResults(query=q)
    )
    monkeypatch.setattr(
        search,
        "get_status",
        lambda p: search.BackendStatus(installed=True, version="1", collections=[]),
    )
    monkeypatch.setattr(
        adapter.db, "get_stats", lambda db_path: {"sources_total": 3}
    )

    ctx = adapter.bind_project(paths.root)
    adapter.do_search(ctx, "x")
    adapter.do_read_page(ctx, "concepts/amortized-inference.md")
    adapter.do_status(ctx)
    assert calls == []  # no router built yet

    adapter.do_ask(ctx, "q") if False else None  # (ask covered elsewhere)


# ---------------------------------------------------------------------------
# do_status
# ---------------------------------------------------------------------------


def test_do_status(paths, monkeypatch):
    monkeypatch.setattr(
        search,
        "get_status",
        lambda p: search.BackendStatus(
            installed=True, version="2.0", collections=["llm-wiki-pages"]
        ),
    )
    monkeypatch.setattr(
        adapter.db,
        "get_stats",
        lambda db_path: {"sources_total": 7, "sources_ingested": 5, "ingest_runs": 2},
    )
    ctx = adapter.bind_project(paths.root)
    out = adapter.do_status(ctx)

    assert out["initialized"] is True
    assert out["root"] == str(paths.root)
    assert out["backend_installed"] is True
    assert out["backend_version"] == "2.0"
    assert out["collections"] == ["llm-wiki-pages"]
    assert out["source_count"] == 7
    # one page created in the fixture (concepts/amortized-inference.md)
    assert out["page_count"] == 1
