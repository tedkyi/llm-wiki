"""Slug generation for wiki page names.

The goal is canonical, kebab-case, ASCII-only slugs that stay stable across
ingest runs. Uses a small alias table to catch common name variations (e.g.
"Andrej Karpathy" vs "A. Karpathy" vs "karpathy, andrej" all → "karpathy").
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

# Honorifics and title prefixes to strip when canonicalizing names
_HONORIFICS = {
    "dr", "dr.", "prof", "prof.", "mr", "mr.", "mrs", "mrs.", "ms", "ms.",
    "sir", "lord", "lady", "phd", "md",
}

# Common "corporate" suffixes to strip from organization names
_CORP_SUFFIXES = {
    "inc", "inc.", "llc", "ltd", "ltd.", "gmbh", "corp", "corp.",
    "co", "co.", "company", "corporation", "limited",
}


def _strip_accents(text: str) -> str:
    """Decompose accented characters to ASCII (Karpathy → Karpathy, é → e)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def slugify(text: str, max_length: int = 60) -> str:
    """Convert arbitrary text to a kebab-case ASCII slug.

    Examples:
        'Andrej Karpathy' → 'andrej-karpathy'
        'Retrieval-Augmented Generation' → 'retrieval-augmented-generation'
        'GPT-4' → 'gpt-4'
        'C++' → 'c'  (punctuation dropped)
    """
    if not text:
        return ""
    text = _strip_accents(text).lower()
    # Replace anything that's not alphanumeric with a hyphen
    text = re.sub(r"[^a-z0-9]+", "-", text)
    # Collapse multiple hyphens
    text = re.sub(r"-+", "-", text)
    # Strip leading/trailing hyphens
    text = text.strip("-")
    # Truncate to max length without cutting mid-word
    if len(text) > max_length:
        text = text[:max_length].rsplit("-", 1)[0]
    return text or "untitled"


_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")


def extract_arxiv_id(text: str) -> str | None:
    """Pull a base arXiv ID (no version suffix) out of a filename or path.

    'Foo Bar 2405.15071v2.pdf' -> '2405.15071'. Deterministic and independent
    of any LLM extraction, so it's a reliable way to recognize that two
    differently-named files are different versions of the same preprint.
    """
    m = _ARXIV_ID_RE.search(text)
    return m.group(1) if m else None


def canonical_name(text: str, kind: str = "any") -> str:
    """Normalize a name for alias matching — aggressive, lossy.

    Differs from slugify: this is *only* for comparing whether two names
    refer to the same thing. It's not for filesystem use.
    """
    text = _strip_accents(text).lower()
    # Remove punctuation entirely
    text = re.sub(r"[^\w\s]", " ", text)
    # Split into tokens
    tokens = text.split()
    # Strip honorifics
    tokens = [t for t in tokens if t not in _HONORIFICS]
    # For person names: drop single-letter tokens (middle initials) and re-sort
    # so "Karpathy, Andrej" == "Andrej Karpathy"
    if kind == "person":
        tokens = [t for t in tokens if len(t) > 1]
        tokens = sorted(tokens)
    # For org names: drop corporate suffixes
    elif kind == "organization":
        tokens = [t for t in tokens if t not in _CORP_SUFFIXES]
    return " ".join(tokens)


def find_existing_slug(
    name: str,
    kind: str,
    search_dirs: list[Path],
) -> str | None:
    """Check if an existing wiki page matches this name via canonical comparison.

    Args:
        name: The raw name from LLM extraction (e.g. "Andrej Karpathy").
        kind: 'person' | 'organization' | 'concept' | 'any' — drives normalization.
        search_dirs: Which directories to scan (e.g. [wiki/entities, wiki/concepts]).

    Returns:
        The existing slug (e.g. 'karpathy') if a match is found, else None.
    """
    target_canonical = canonical_name(name, kind=kind)
    if not target_canonical:
        return None

    for directory in search_dirs:
        if not directory.exists():
            continue
        for page in directory.glob("*.md"):
            # Read the page title from frontmatter, fall back to filename
            try:
                content = page.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            # Extract title from YAML frontmatter
            title = None
            fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
            if fm_match:
                title_match = re.search(
                    r'^title:\s*["\']?(.+?)["\']?\s*$',
                    fm_match.group(1),
                    re.MULTILINE,
                )
                if title_match:
                    title = title_match.group(1).strip()

            # Fall back to filename stem
            if not title:
                title = page.stem.replace("-", " ").title()

            if canonical_name(title, kind=kind) == target_canonical:
                return page.stem

    return None
