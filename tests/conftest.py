"""Shared fixtures for the LLM-Wiki test suite.

The MCP tests never touch qmd, Ollama, or a real LLM — the wiki seams
(`search.query`, `query.run_query`, `page_writer.read_page`, `search.get_status`,
`db.get_stats`) are monkeypatched. These fixtures build a throwaway on-disk wiki
just complete enough for `bind_project` / path-safety logic to work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_wiki import config as cfg


SAMPLE_PAGE = """\
---
title: Amortized Inference
tags: [ml, inference]
---

Amortized inference trades upfront training cost for cheap per-query inference.
"""


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """A minimally-initialized wiki project on disk."""
    root = tmp_path / "mywiki"
    (root / cfg.INTERNAL_DIR).mkdir(parents=True)
    # `is_initialized()` only requires the config file to exist.
    (root / cfg.INTERNAL_DIR / cfg.CONFIG_FILE).write_text(
        "version: 1\n", encoding="utf-8"
    )
    for sub in ("entities", "concepts", "sources", "synthesis"):
        (root / cfg.WIKI_DIR / sub).mkdir(parents=True)
    # One real page so read_wiki_page has something to return.
    (root / cfg.WIKI_DIR / "concepts" / "amortized-inference.md").write_text(
        SAMPLE_PAGE, encoding="utf-8"
    )
    # A file OUTSIDE the wiki/ folder, used to prove traversal is blocked.
    (root / cfg.INTERNAL_DIR / "secret.txt").write_text("top secret", encoding="utf-8")
    return root


@pytest.fixture
def paths(wiki_root: Path) -> cfg.WikiPaths:
    return cfg.WikiPaths(root=wiki_root)
