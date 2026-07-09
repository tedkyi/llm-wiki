"""Protocol-level tests for the MCP server.

Thin by design: these assert the tool *contract* the client sees (names, schemas,
return shapes, error surfacing), not wiki behavior. The adapter is stubbed, so the
server is exercised over a real in-memory MCP client/server session without any
wiki, qmd, or LLM.
"""

from __future__ import annotations

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from llm_wiki.mcp import adapter, server


@pytest.fixture
def stub_adapter(monkeypatch):
    """Replace every adapter.do_* with a canned, recorded stub."""
    calls = {}

    def rec(name, ret):
        def _fn(ctx, *args, **kwargs):
            calls[name] = {"args": args, "kwargs": kwargs}
            return ret
        return _fn

    monkeypatch.setattr(
        adapter,
        "do_search",
        rec("search", {"hits": [{"title": "A", "path": "concepts/a.md",
                                 "collection": "llm-wiki-pages", "score": 0.9,
                                 "snippet": "s", "content": "c"}], "count": 1}),
    )
    monkeypatch.setattr(
        adapter,
        "do_read_page",
        rec("read", {"title": "A", "path": "concepts/a.md",
                     "frontmatter": {"title": "A"}, "body": "b"}),
    )
    monkeypatch.setattr(
        adapter,
        "do_ask",
        rec("ask", {"answer": "the answer",
                    "sources": [{"title": "A", "path": "concepts/a.md", "score": 0.9}]}),
    )
    monkeypatch.setattr(
        adapter,
        "do_status",
        rec("status", {"initialized": True, "root": "/w", "backend_installed": True,
                       "backend_version": "1", "collections": ["llm-wiki-pages"],
                       "page_count": 1, "source_count": 1}),
    )
    return calls


def _make_server():
    # ctx is irrelevant because the adapter is stubbed.
    return server.build_server(ctx=object())._mcp_server


async def _client():
    return create_connected_server_and_client_session(_make_server())


async def test_lists_exactly_four_tools(stub_adapter):
    async with await _client() as session:
        result = await session.list_tools()
    names = {t.name for t in result.tools}
    assert names == {"search_wiki", "read_wiki_page", "ask_wiki", "wiki_status"}
    for tool in result.tools:
        assert tool.description  # docstring surfaced
        assert tool.inputSchema is not None


async def test_search_wiki_schema_params(stub_adapter):
    async with await _client() as session:
        result = await session.list_tools()
    tool = next(t for t in result.tools if t.name == "search_wiki")
    props = tool.inputSchema["properties"]
    assert "query" in props
    for optional in ("scope", "limit", "mode", "rerank", "min_score"):
        assert optional in props
    assert tool.inputSchema.get("required") == ["query"]


async def test_call_search_wiki(stub_adapter):
    async with await _client() as session:
        res = await session.call_tool("search_wiki", {"query": "amortized", "limit": 3})
    assert not res.isError
    assert res.structuredContent["count"] == 1
    # arguments were forwarded to the adapter
    assert stub_adapter["search"]["kwargs"]["limit"] == 3


async def test_call_wiki_status(stub_adapter):
    async with await _client() as session:
        res = await session.call_tool("wiki_status", {})
    assert not res.isError
    assert res.structuredContent["page_count"] == 1


async def test_call_ask_wiki(stub_adapter):
    async with await _client() as session:
        res = await session.call_tool("ask_wiki", {"question": "what is X?"})
    assert not res.isError
    assert res.structuredContent["answer"] == "the answer"


async def test_read_page_error_surfaces(stub_adapter, monkeypatch):
    def boom(ctx, path):
        raise adapter.PathOutsideWiki("path outside wiki root")

    monkeypatch.setattr(adapter, "do_read_page", boom)
    async with await _client() as session:
        res = await session.call_tool("read_wiki_page", {"path": "../secret"})
    assert res.isError
    text = " ".join(
        block.text for block in res.content if getattr(block, "text", None)
    )
    assert "outside wiki root" in text
