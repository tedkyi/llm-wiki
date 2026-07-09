"""MCP server for LLM-Wiki.

Exposes the wiki to other LLM sessions over the Model Context Protocol. Two
layers, deliberately decoupled:

- `adapter` — the ONLY module that touches wiki internals (`search`, `query`,
  `page_writer`, `config`, `db`, `llm`). It converts everything to plain,
  JSON-serializable dicts. Rewiring the server to a different wiki means editing
  only this file.
- `server` — wiki-agnostic. Declares the MCP tools and delegates to `adapter`.
"""
