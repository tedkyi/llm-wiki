"""Unit tests for slugify — pure, deterministic slug/name logic. No backends."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_wiki import slugify


# ---------------------------------------------------------------------------
# slugify()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Andrej Karpathy", "andrej-karpathy"),
        ("Retrieval-Augmented Generation", "retrieval-augmented-generation"),
        ("GPT-4", "gpt-4"),
        ("C++", "c"),                       # punctuation dropped
        ("  spaced  out  ", "spaced-out"),
        ("Multiple---hyphens", "multiple-hyphens"),  # collapsed
        ("café résumé", "cafe-resume"),     # accents stripped
    ],
)
def test_slugify_basic(text: str, expected: str) -> None:
    assert slugify.slugify(text) == expected


def test_slugify_empty_returns_empty() -> None:
    # An empty input short-circuits to "" before the untitled fallback.
    assert slugify.slugify("") == ""


def test_slugify_all_punctuation_returns_untitled() -> None:
    # Non-empty but nothing survives normalization → the "untitled" fallback.
    assert slugify.slugify("+++") == "untitled"


def test_slugify_truncates_on_word_boundary() -> None:
    text = "one two three four five six seven eight nine ten eleven twelve"
    result = slugify.slugify(text, max_length=20)
    assert len(result) <= 20
    # Truncation happens at a hyphen, never mid-word.
    assert not result.endswith("-")
    assert text.split()[0] in result


def test_strip_accents() -> None:
    assert slugify._strip_accents("Kärpâthy") == "Karpathy"
    assert slugify._strip_accents("naïve café") == "naive cafe"


# ---------------------------------------------------------------------------
# extract_arxiv_id()
# ---------------------------------------------------------------------------


def test_extract_arxiv_id_with_version_suffix() -> None:
    assert slugify.extract_arxiv_id("Foo Bar 2405.15071v2.pdf") == "2405.15071"


def test_extract_arxiv_id_without_version() -> None:
    assert slugify.extract_arxiv_id("paper-2405.15071.pdf") == "2405.15071"


def test_extract_arxiv_id_five_digit() -> None:
    assert slugify.extract_arxiv_id("1234.56789") == "1234.56789"


def test_extract_arxiv_id_no_match() -> None:
    assert slugify.extract_arxiv_id("just-a-regular-file.pdf") is None


# ---------------------------------------------------------------------------
# canonical_name()
# ---------------------------------------------------------------------------


def test_canonical_name_strips_honorifics() -> None:
    assert slugify.canonical_name("Dr. Jane Smith") == slugify.canonical_name("Jane Smith")


def test_canonical_name_person_reorders_and_drops_initials() -> None:
    # "Karpathy, Andrej" == "Andrej Karpathy" == "Andrej K. Karpathy" for people.
    a = slugify.canonical_name("Karpathy, Andrej", kind="person")
    b = slugify.canonical_name("Andrej Karpathy", kind="person")
    c = slugify.canonical_name("Andrej K. Karpathy", kind="person")
    assert a == b == c


def test_canonical_name_org_drops_corp_suffix() -> None:
    assert slugify.canonical_name("OpenAI Inc.", kind="organization") == (
        slugify.canonical_name("OpenAI", kind="organization")
    )


def test_canonical_name_any_keeps_order_and_short_tokens() -> None:
    # kind="any" doesn't reorder or drop single-letter tokens.
    result = slugify.canonical_name("A B C", kind="any")
    assert result == "a b c"


# ---------------------------------------------------------------------------
# find_existing_slug()
# ---------------------------------------------------------------------------


def _write_page(directory: Path, stem: str, title: str | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if title is not None:
        body = f"---\ntitle: {title}\n---\n\nBody.\n"
    else:
        body = "Body without frontmatter.\n"
    (directory / f"{stem}.md").write_text(body, encoding="utf-8")


def test_find_existing_slug_matches_via_frontmatter_title(tmp_path: Path) -> None:
    entities = tmp_path / "entities"
    _write_page(entities, "karpathy", "Andrej Karpathy")
    # "Karpathy, Andrej" canonicalizes (person) to the same tokens as the title.
    match = slugify.find_existing_slug("Karpathy, Andrej", "person", [entities])
    assert match == "karpathy"


def test_find_existing_slug_matches_via_filename_fallback(tmp_path: Path) -> None:
    entities = tmp_path / "entities"
    # No frontmatter title — comparison falls back to the humanized stem.
    _write_page(entities, "andrej-karpathy", None)
    match = slugify.find_existing_slug("Andrej Karpathy", "person", [entities])
    assert match == "andrej-karpathy"


def test_find_existing_slug_no_match_returns_none(tmp_path: Path) -> None:
    entities = tmp_path / "entities"
    _write_page(entities, "karpathy", "Andrej Karpathy")
    assert slugify.find_existing_slug("Yann LeCun", "person", [entities]) is None


def test_find_existing_slug_tolerates_missing_dir(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    assert slugify.find_existing_slug("Anyone", "person", [missing]) is None


def test_find_existing_slug_empty_name_returns_none(tmp_path: Path) -> None:
    entities = tmp_path / "entities"
    _write_page(entities, "karpathy", "Andrej Karpathy")
    assert slugify.find_existing_slug("", "person", [entities]) is None
