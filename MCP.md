# LLM-Wiki — MCP Server

Expose your wiki to **other LLM sessions** over the [Model Context Protocol](https://modelcontextprotocol.io). Run one command and any MCP-capable client — Claude Desktop, Claude Code, Cursor, or your own agent — can search the wiki, read pages, and ask it synthesized questions.

The server is **read-only**: it never ingests, edits, or deletes anything. It's the same wiki the CLI and web UI use, reached through a different door.

---

## Table of Contents

- [Two ways to use it](#two-ways-to-use-it)
- [Quick start](#quick-start)
- [The tools](#the-tools)
- [Command reference (`wiki mcp`)](#command-reference-wiki-mcp)
- [Connecting a client](#connecting-a-client)
  - [Claude Code](#claude-code)
  - [Cursor](#cursor)
  - [Claude Desktop](#claude-desktop)
  - [A custom Python client](#a-custom-python-client)
- [Security](#security)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)

---

## Two ways to use it

The four tools split cleanly into two mental models:

1. **The wiki as a document store (RAG).** Another agent calls `search_wiki` to pull ranked, hydrated page content and `read_wiki_page` to fetch whole pages, then reasons over that context itself. No answer is synthesized on the wiki's side and no LLM is called — it's fast retrieval.
2. **The wiki as a teammate.** The agent calls `ask_wiki` with a natural-language question and gets back a cited, synthesized answer — exactly what the CLI's `wiki query` or the web UI produces. This runs the full retrieve→synthesize pipeline and costs one LLM call on the wiki's configured provider.

`wiki_status` supports both: a cheap health/size check a client can run before it starts.

---

## Quick start

From inside your wiki project (or point at one with `--root`):

```bash
wiki mcp
```

```
┌────────────────────────── 🔌 MCP ───────────────────────────┐
│ LLM-Wiki MCP server starting…                               │
│                                                             │
│   URL: http://127.0.0.1:8010/mcp                            │
│   Transport: streamable-http                                │
│   Tools: search_wiki, read_wiki_page, ask_wiki, wiki_status │
│   Project: /Users/you/Documents/LLM-Wiki                    │
│                                                             │
│ Press Ctrl+C to stop.                                       │
└─────────────────────────────────────────────────────────────┘
```

The server now speaks MCP over **Streamable HTTP** at `http://127.0.0.1:8010/mcp`. Point a client at that URL (see [Connecting a client](#connecting-a-client)).

Run it in its own terminal — like `wiki serve`, it stays in the foreground until you `Ctrl+C`. You can run the MCP server and the web UI at the same time; they use different ports (8010 vs 8000).

---

## The tools

Each tool returns structured JSON. On failure (a missing page, a path escape, no search results, an unreachable provider) the tool returns an MCP error carrying a plain-text message.

### `search_wiki`

RAG-style retrieval. **No LLM call.**

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `query` | string | *(required)* | The search text |
| `scope` | `wiki` \| `raw` \| `hybrid` | `wiki` | `wiki` = curated pages, `raw` = source-document text, `hybrid` = both |
| `limit` | int | `8` | Max hits |
| `mode` | `hybrid` \| `lex` \| `vec` | `hybrid` | `hybrid` = BM25+vector+rerank, `lex` = keyword, `vec` = semantic |
| `rerank` | bool | `true` | LLM reranking (hybrid mode only) |
| `min_score` | float | `0.0` | Drop hits below this score |

Returns:

```json
{
  "hits": [
    {
      "title": "Amortized Inference",
      "path": "concepts/amortized-inference.md",
      "collection": "llm-wiki-pages",
      "score": 0.93,
      "snippet": "…context around the match…",
      "content": "…full page markdown…"
    }
  ],
  "count": 1
}
```

### `read_wiki_page`

Fetch one page by its **wiki-relative** path (e.g. the `path` from a `search_wiki` hit).

| Parameter | Type | Notes |
|---|---|---|
| `path` | string | Relative to the `wiki/` folder, e.g. `concepts/foo.md` |

Returns `{ "title", "path", "frontmatter": {…}, "body": "…" }`. Errors with `page not found: <path>` if it doesn't exist, or `path outside wiki root` if the path is absolute or escapes the wiki folder.

### `ask_wiki`

Full retrieve→synthesize pipeline. **Costs one LLM call** on the wiki's configured provider. Read-only — never saves a synthesis page.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `question` | string | *(required)* | Natural-language question |
| `scope` | `wiki` \| `raw` \| `hybrid` | `wiki` | Same meaning as `search_wiki` |
| `grounding` | `strict` \| `augmented` | `strict` | `strict` = answer only from sources; `augmented` = may add the model's own knowledge, labeled |
| `limit` | int | `8` | Hits retrieved before synthesis |
| `page_types` | list of string \| null | `null` | Restrict wiki-page hits to kinds, e.g. `["entities", "concepts"]` |

Returns:

```json
{
  "answer": "## Amortized Inference\n\nAmortized inference is …",
  "sources": [
    { "title": "Amortized Inference", "path": "concepts/amortized-inference.md", "score": 0.93 }
  ]
}
```

### `wiki_status`

Cheap health + size check. No LLM call. Returns:

```json
{
  "initialized": true,
  "root": "/Users/you/Documents/LLM-Wiki",
  "backend_installed": true,
  "backend_version": "qmd 2.5.3",
  "collections": ["llm-wiki-pages", "llm-wiki-raw"],
  "page_count": 3236,
  "source_count": 407
}
```

---

## Command reference (`wiki mcp`)

```bash
wiki mcp [--host H] [--port N] [--path P] [--root DIR]
```

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address. A non-loopback address exposes the wiki to your network **with no authentication** (you'll get a warning). |
| `--port`, `-p` | `8010` | Port to listen on (distinct from the web UI's 8000). |
| `--path` | `/mcp` | HTTP path the MCP endpoint is mounted at. |
| `--root` | *(auto-discover)* | Wiki project root. Defaults to walking up from the current directory, like every other `wiki` command. |

The server binds to **one** wiki project chosen at startup.

---

## Connecting a client

The server uses **Streamable HTTP**, so any client that supports an HTTP/remote MCP transport can connect to `http://127.0.0.1:8010/mcp`. Start `wiki mcp` first, then configure your client. (Client config formats change over time — if one of these looks stale, check your client's current MCP docs and use the URL above.)

### Claude Code

Add it with the CLI:

```bash
claude mcp add --transport http llm-wiki http://127.0.0.1:8010/mcp
```

…or commit it to a project by writing `.mcp.json` in the repo root:

```json
{
  "mcpServers": {
    "llm-wiki": {
      "type": "http",
      "url": "http://127.0.0.1:8010/mcp"
    }
  }
}
```

Then ask Claude Code things like *"use llm-wiki to find what my notes say about amortized inference"* or *"ask the wiki to compare RAG and fine-tuning."*

### Cursor

Add to `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global):

```json
{
  "mcpServers": {
    "llm-wiki": {
      "url": "http://127.0.0.1:8010/mcp"
    }
  }
}
```

### Claude Desktop

Claude Desktop's config launches **local (stdio)** servers, so point it at the HTTP server through the [`mcp-remote`](https://www.npmjs.com/package/mcp-remote) bridge. Edit `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "llm-wiki": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://127.0.0.1:8010/mcp"]
    }
  }
}
```

Restart Claude Desktop; the four tools appear under the wiki's name.

### A custom Python client

Using the official `mcp` SDK (already a dependency of this project):

```python
import anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    async with streamablehttp_client("http://127.0.0.1:8010/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print([t.name for t in (await session.list_tools()).tools])

            res = await session.call_tool(
                "search_wiki", {"query": "amortized inference", "limit": 3}
            )
            print(res.structuredContent)

            ans = await session.call_tool(
                "ask_wiki", {"question": "What is amortized inference?"}
            )
            print(ans.structuredContent["answer"])

anyio.run(main)
```

---

## Security

- **Localhost by default.** The server binds to `127.0.0.1`, so only processes on your machine can reach it. Passing a non-loopback `--host` exposes the wiki to your network **with no authentication** — the server prints a warning when you do.
- **Read-only.** There is deliberately no ingest, no page editing, and no synthesis-page saving over MCP. The worst a connected client can do is read.
- **Path-safe.** `read_wiki_page` resolves paths against the `wiki/` folder and rejects anything absolute or escaping it (including via `..` or symlinks) — a client cannot read arbitrary files on disk.
- **`ask_wiki` sends content to your configured provider.** If you've routed `query_synthesis` to a cloud model, an `ask_wiki` call sends retrieved wiki excerpts to that provider, same as `wiki query`. On the default all-local setup nothing leaves your machine. Check routing with `wiki providers list`.

---

## Requirements

`wiki mcp` runs in the same environment as the rest of the CLI, so the tools need what they always need:

- **`qmd` and `node` on `PATH`** — `search_wiki` and `ask_wiki` (and the `collections`/`backend_version` fields of `wiki_status`) go through QMD. If they're missing you'll get a clear `qmd command was not found` error; `wiki_status` still reports `backend_installed: false`.
- **A reachable LLM provider** — only `ask_wiki` needs one (for synthesis). `search_wiki`, `read_wiki_page`, and `wiki_status` work even with no provider running, because the server builds the LLM router lazily on the first `ask_wiki` call.

See the main [README](./README.md#requirements) for installing QMD and a provider.

---

## Troubleshooting

### `Error executing tool search_wiki: The 'qmd' command was not found on PATH`

QMD isn't visible to the process running `wiki mcp`. Install it (`npm install -g @tobilu/qmd`) and launch `wiki mcp` from a shell where `qmd --version` works.

### `ask_wiki` errors with a provider / connection message

The synthesis provider isn't reachable. If you're on the default local setup, make sure Ollama is running and the model is pulled (`ollama list`). Check routing with `wiki providers list` and reachability with `wiki providers test <provider>`. `search_wiki` will still work regardless.

### `address already in use` on startup

Another process holds port 8010 (perhaps an earlier `wiki mcp`). Pick another with `--port`, or stop the other process.

### The client connects but lists no tools

Make sure the client's URL includes the path — `http://127.0.0.1:8010/mcp`, not just the host and port. Change the mount path with `--path` if needed.

### `Not inside an LLM-Wiki project`

You launched `wiki mcp` outside a wiki folder and didn't pass `--root`. Either `cd` into your wiki or run `wiki mcp --root /path/to/wiki`.

---

## See also

- [README.md](./README.md) — architecture and internals
- [USAGE.md](./USAGE.md) — day-to-day CLI and web UI guide
- [PROVIDERS.md](./PROVIDERS.md) — configuring local/cloud LLM providers
