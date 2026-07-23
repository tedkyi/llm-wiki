"""Unit tests for the source-page assembly in the ingest pipeline.

These exercise `_assemble_source_page` directly (no LLM/backends), locking in the
fix for the duplicate-header-block bug: an LLM emits a valid single-frontmatter
page, but its `file_path: ["D:\\Data\\x.pdf"]` is invalid YAML (bad backslash
escapes), which used to make the page look frontmatter-less and get a second,
synthesized header stacked on top.
"""

from __future__ import annotations

from llm_wiki import ingest_llm
from llm_wiki import page_writer


WIN_PATH = r"D:\Data\papers\Steering Language Models With Activation Engineering 2308.10248v5.pdf"

# A clean, correct single-frontmatter page — exactly what the LLM produces —
# except the file_path is a double-quoted Windows path (invalid YAML escapes).
LLM_OUTPUT_DOUBLE_QUOTED_WIN_PATH = f'''---
title: "Steering Language Models With Activation Engineering"
type: source
tags: [activation-engineering, llm-steering, interpretability]
created: 2026-07-15
updated: 2026-07-15
file_path: ["{WIN_PATH}"]
confidence: high
---

# Steering Language Models With Activation Engineering

## Summary

The paper introduces activation engineering.
'''


def _pre_h1(body: str) -> str:
    """Everything in a body before its first H1 line."""
    out = []
    for line in body.splitlines():
        if line.startswith("# "):
            break
        out.append(line)
    return "\n".join(out)


def test_double_quoted_windows_path_does_not_duplicate_header() -> None:
    md = ingest_llm._assemble_source_page(
        LLM_OUTPUT_DOUBLE_QUOTED_WIN_PATH,
        source_title="Steering Language Models With Activation Engineering",
        tags=["activation-engineering"],
        today="2026-07-15",
        source_relpath=WIN_PATH,
        existing_source_page=None,
    )
    parsed = page_writer.parse_page(md)

    # Exactly one frontmatter block: the body starts at the H1, with no stray
    # header ahead of it.
    assert parsed.frontmatter, "frontmatter should have been recovered, not dropped"
    assert parsed.body.lstrip().startswith("# "), (
        f"body should start at the H1, not a duplicate header. pre-H1 was:\n"
        f"{_pre_h1(parsed.body)!r}"
    )
    # The tell-tale of the old bug was `type: source` (and the title) appearing
    # twice in the rendered document.
    assert md.count("type: source") == 1


def test_recovered_frontmatter_is_preserved() -> None:
    md = ingest_llm._assemble_source_page(
        LLM_OUTPUT_DOUBLE_QUOTED_WIN_PATH,
        source_title="RAW EXTRACTED TITLE",  # would be used only as a fallback
        tags=["ignored"],
        today="2026-07-15",
        source_relpath=WIN_PATH,
        existing_source_page=None,
    )
    parsed = page_writer.parse_page(md)
    # The LLM's own frontmatter survived (title cased nicely, confidence kept) —
    # i.e. we did NOT fall back to the synthesized minimal frontmatter.
    assert parsed.frontmatter["title"] == "Steering Language Models With Activation Engineering"
    assert parsed.frontmatter["confidence"] == "high"


def test_file_path_is_authoritative_not_from_llm() -> None:
    # LLM claims a bogus/relative path; the pipeline must overwrite it with the
    # source record's relpath.
    llm_output = LLM_OUTPUT_DOUBLE_QUOTED_WIN_PATH.replace(
        f'["{WIN_PATH}"]', '["some/wrong/path.pdf"]'
    )
    md = ingest_llm._assemble_source_page(
        llm_output,
        source_title="T",
        tags=[],
        today="2026-07-15",
        source_relpath=WIN_PATH,
        existing_source_page=None,
    )
    parsed = page_writer.parse_page(md)
    assert parsed.frontmatter["file_path"] == [WIN_PATH]


def test_merge_accumulates_file_paths() -> None:
    existing = page_writer.ParsedPage(
        frontmatter={"title": "T", "type": "source", "file_path": [r"D:\old\v1.pdf"]},
        body="# T\n\nold body\n",
    )
    md = ingest_llm._assemble_source_page(
        LLM_OUTPUT_DOUBLE_QUOTED_WIN_PATH,
        source_title="T",
        tags=[],
        today="2026-07-16",
        source_relpath=WIN_PATH,
        existing_source_page=existing,
    )
    parsed = page_writer.parse_page(md)
    assert parsed.frontmatter["file_path"] == [r"D:\old\v1.pdf", WIN_PATH]


def test_truly_missing_frontmatter_is_synthesized_without_duplication() -> None:
    # No frontmatter at all in the LLM output → synthesize one, and the body must
    # not be re-stacked (invariant 1).
    llm_output = "# Some Title\n\n## Summary\n\nbody text\n"
    md = ingest_llm._assemble_source_page(
        llm_output,
        source_title="Some Title",
        tags=["t1"],
        today="2026-07-15",
        source_relpath=WIN_PATH,
        existing_source_page=None,
    )
    parsed = page_writer.parse_page(md)
    assert parsed.frontmatter["title"] == "Some Title"
    assert parsed.frontmatter["type"] == "source"
    assert parsed.frontmatter["file_path"] == [WIN_PATH]
    assert parsed.body.lstrip().startswith("# Some Title")
    assert md.count("## Summary") == 1


# ---------------------------------------------------------------------------
# page_writer.parse_page recovery (the underlying enabler)
# ---------------------------------------------------------------------------


def test_parse_page_recovers_double_quoted_windows_path() -> None:
    parsed = page_writer.parse_page(LLM_OUTPUT_DOUBLE_QUOTED_WIN_PATH)
    assert parsed.frontmatter, "should recover instead of returning empty"
    assert parsed.frontmatter["file_path"] == [WIN_PATH]
    assert parsed.body.lstrip().startswith("# ")
