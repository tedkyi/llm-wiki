"""Page I/O utilities: frontmatter parsing, page writing, index/log updates.

This module handles the structural bookkeeping that does NOT need the LLM:
- Writing validated markdown pages to disk
- Parsing and updating YAML frontmatter
- Stripping LLM response noise (code fences, preamble)
- Rebuilding index.md from the current wiki/ contents
- Appending new entries to log.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from . import config as cfg


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


@dataclass
class ParsedPage:
    """A wiki page split into frontmatter and body."""

    frontmatter: dict[str, Any] = field(default_factory=dict)
    body: str = ""

    def to_markdown(self) -> str:
        """Serialize back to a markdown string with YAML frontmatter."""
        if not self.frontmatter:
            return self.body
        fm_yaml = yaml.safe_dump(
            self.frontmatter,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        ).strip()
        return f"---\n{fm_yaml}\n---\n\n{self.body.strip()}\n"


def parse_page(content: str) -> ParsedPage:
    """Split a markdown string into frontmatter + body."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, re.DOTALL)
    if not match:
        return ParsedPage(frontmatter={}, body=content)
    fm_text = match.group(1)
    body = match.group(2)
    try:
        fm = yaml.safe_load(fm_text) or {}
        if not isinstance(fm, dict):
            fm = {}
    except yaml.YAMLError:
        fm = {}
    return ParsedPage(frontmatter=fm, body=body)


def read_page(path: Path) -> ParsedPage | None:
    """Read and parse a wiki page from disk. Returns None if missing."""
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return parse_page(content)


# ---------------------------------------------------------------------------
# LLM response cleanup
# ---------------------------------------------------------------------------


def strip_llm_noise(text: str) -> str:
    """Remove common LLM response artifacts from raw output.

    - Leading/trailing ```markdown or ``` fences
    - Explanatory preamble like 'Here is the page:'
    - Trailing commentary after the last frontmatter block
    """
    text = text.strip()

    # Remove outer code fences
    fence_match = re.match(
        r"^```(?:markdown|md)?\s*\n(.*?)\n```\s*$", text, re.DOTALL
    )
    if fence_match:
        text = fence_match.group(1).strip()

    # Remove common preambles (only if they appear before the first ---)
    preamble_patterns = [
        r"^Here (?:is|'s) the (?:updated |new )?(?:markdown )?page:?\s*\n+",
        r"^Here (?:is|'s) the (?:updated |new )?page:?\s*\n+",
        r"^Sure[,.]?\s*here.*?:\s*\n+",
        r"^Okay[,.]?\s*here.*?:\s*\n+",
    ]
    for pattern in preamble_patterns:
        text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)

    return text.strip()


# ---------------------------------------------------------------------------
# Wikilink extraction
# ---------------------------------------------------------------------------


WIKILINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]*)?\]\]")


def extract_wikilinks(content: str) -> list[str]:
    """Find all [[wikilinks]] in a page. Returns the link targets (no brackets).

    E.g. '[[karpathy]] and [[entities/openai]]' → ['karpathy', 'entities/openai']
    """
    return [m.group(1).strip() for m in WIKILINK_RE.finditer(content)]


# ---------------------------------------------------------------------------
# Page writing
# ---------------------------------------------------------------------------


def write_page(path: Path, content: str) -> None:
    """Write a page to disk, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure trailing newline
    if not content.endswith("\n"):
        content += "\n"
    path.write_text(content, encoding="utf-8")


def ensure_frontmatter_fields(page: ParsedPage, required: dict[str, Any]) -> ParsedPage:
    """Ensure the page has these frontmatter fields, adding defaults if missing.

    Does not overwrite existing values.
    """
    for key, default_value in required.items():
        if key not in page.frontmatter:
            page.frontmatter[key] = default_value
    return page


def add_source_to_frontmatter(
    page: ParsedPage, source_slug: str, today: str
) -> ParsedPage:
    """Add a source reference to the page's frontmatter `sources` list
    and update the `updated` date. Idempotent.
    """
    sources = page.frontmatter.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    # Normalize: strip any stale .md suffixes from existing entries
    sources = [s.removesuffix(".md") if isinstance(s, str) else s for s in sources]
    ref = f"sources/{source_slug}"
    if ref not in sources:
        sources.append(ref)
    page.frontmatter["sources"] = sources
    page.frontmatter["updated"] = today
    return page


# ---------------------------------------------------------------------------
# index.md rebuild
# ---------------------------------------------------------------------------


INDEX_HEADER = """---
title: "Wiki Index"
type: index
updated: {today}
---

# Wiki Index

> This file is auto-maintained by the LLM-Wiki agent. It lists every wiki
> page organized by category. Rebuilt after every ingest.

"""


def _list_pages_in(directory: Path) -> list[tuple[str, str]]:
    """Return a sorted list of (slug, title) tuples for every .md page in
    the directory, skipping dotfiles.
    """
    if not directory.exists():
        return []
    out = []
    for page_path in sorted(directory.glob("*.md")):
        if page_path.name.startswith("."):
            continue
        parsed = read_page(page_path)
        title = page_path.stem
        if parsed and isinstance(parsed.frontmatter.get("title"), str):
            title = parsed.frontmatter["title"]
        out.append((page_path.stem, title))
    return out


def rebuild_index(paths: cfg.WikiPaths, today: str) -> None:
    """Rebuild wiki/index.md from the current wiki/ contents."""
    sources = _list_pages_in(paths.wiki / "sources")
    entities = _list_pages_in(paths.wiki / "entities")
    concepts = _list_pages_in(paths.wiki / "concepts")
    synthesis = _list_pages_in(paths.wiki / "synthesis")

    lines = [INDEX_HEADER.format(today=today)]

    def _section(title: str, pages: list[tuple[str, str]], subdir: str) -> None:
        lines.append(f"## {title}\n")
        if not pages:
            lines.append(f"*No {title.lower()} pages yet.*\n")
        else:
            for slug, page_title in pages:
                lines.append(f"- [[{subdir}/{slug}|{page_title}]]")
            lines.append("")
        lines.append("")

    _section("Sources", sources, "sources")
    _section("Entities", entities, "entities")
    _section("Concepts", concepts, "concepts")
    _section("Synthesis", synthesis, "synthesis")

    lines.append("---\n")
    lines.append(
        f"**Stats:** {len(sources)} sources · {len(entities)} entities · "
        f"{len(concepts)} concepts · {len(synthesis)} synthesis pages\n"
    )

    paths.index.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# log.md append
# ---------------------------------------------------------------------------


def append_log_entry(
    paths: cfg.WikiPaths,
    today: str,
    action: str,
    title: str,
    bullets: list[str],
) -> None:
    """Append a new entry to log.md.

    Format:
        ## [YYYY-MM-DD] action | title
        - bullet 1
        - bullet 2
    """
    if not paths.log.exists():
        paths.log.parent.mkdir(parents=True, exist_ok=True)
        paths.log.write_text(
            '---\ntitle: "Wiki Log"\ntype: log\n---\n\n# Wiki Log\n\n',
            encoding="utf-8",
        )

    existing = paths.log.read_text(encoding="utf-8")
    if not existing.endswith("\n"):
        existing += "\n"
    if not existing.endswith("\n\n"):
        existing += "\n"

    entry_lines = [f"## [{today}] {action} | {title}", ""]
    for bullet in bullets:
        entry_lines.append(f"- {bullet}")
    entry_lines.append("")

    paths.log.write_text(existing + "\n".join(entry_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def today_iso() -> str:
    """YYYY-MM-DD for the current local date."""
    return date.today().isoformat()
