"""Unit tests for page_writer — frontmatter/wikilink/index logic. No LLM/backends."""

from __future__ import annotations

from llm_wiki import config as cfg
from llm_wiki import page_writer


# ---------------------------------------------------------------------------
# parse_page / to_markdown
# ---------------------------------------------------------------------------


def test_parse_page_round_trip() -> None:
    content = "---\ntitle: Foo\ntype: concept\n---\n\nBody text here.\n"
    parsed = page_writer.parse_page(content)
    assert parsed.frontmatter == {"title": "Foo", "type": "concept"}
    assert parsed.body.strip() == "Body text here."
    # Re-serializing preserves both frontmatter and body.
    out = parsed.to_markdown()
    reparsed = page_writer.parse_page(out)
    assert reparsed.frontmatter == parsed.frontmatter
    assert reparsed.body.strip() == "Body text here."


def test_parse_page_no_frontmatter_passthrough() -> None:
    parsed = page_writer.parse_page("Just a body, no frontmatter.")
    assert parsed.frontmatter == {}
    assert parsed.body == "Just a body, no frontmatter."


def test_parse_page_malformed_yaml_yields_empty_frontmatter() -> None:
    # Unbalanced brackets → YAMLError, swallowed to an empty dict (never raises).
    content = "---\ntitle: [unclosed\n---\n\nBody.\n"
    parsed = page_writer.parse_page(content)
    assert parsed.frontmatter == {}
    assert parsed.body.strip() == "Body."


def test_parse_page_non_dict_frontmatter_yields_empty() -> None:
    # A bare scalar between the fences is not a dict → coerced to {}.
    content = "---\njust a string\n---\n\nBody.\n"
    parsed = page_writer.parse_page(content)
    assert parsed.frontmatter == {}


def test_to_markdown_without_frontmatter_returns_body() -> None:
    page = page_writer.ParsedPage(frontmatter={}, body="hello")
    assert page.to_markdown() == "hello"


# ---------------------------------------------------------------------------
# strip_llm_noise
# ---------------------------------------------------------------------------


def test_strip_llm_noise_removes_markdown_fence() -> None:
    text = "```markdown\n---\ntitle: X\n---\n\nBody.\n```"
    assert page_writer.strip_llm_noise(text) == "---\ntitle: X\n---\n\nBody."


def test_strip_llm_noise_removes_plain_fence() -> None:
    text = "```\nhello\n```"
    assert page_writer.strip_llm_noise(text) == "hello"


def test_strip_llm_noise_removes_preamble() -> None:
    text = "Here is the page:\n\n---\ntitle: X\n---\n\nBody."
    result = page_writer.strip_llm_noise(text)
    assert result.startswith("---")
    assert "Here is the page" not in result


def test_strip_llm_noise_preamble_case_insensitive() -> None:
    text = "SURE, here is the updated page:\n\n# Heading"
    assert page_writer.strip_llm_noise(text) == "# Heading"


def test_strip_llm_noise_leaves_clean_text() -> None:
    text = "---\ntitle: X\n---\n\nBody."
    assert page_writer.strip_llm_noise(text) == text


# ---------------------------------------------------------------------------
# extract_wikilinks
# ---------------------------------------------------------------------------


def test_extract_wikilinks_plain() -> None:
    assert page_writer.extract_wikilinks("see [[karpathy]] here") == ["karpathy"]


def test_extract_wikilinks_strips_alias() -> None:
    assert page_writer.extract_wikilinks("[[karpathy|Andrej]]") == ["karpathy"]


def test_extract_wikilinks_path_style_and_multiple() -> None:
    content = "[[karpathy]] and [[entities/openai]] and [[concepts/rag|RAG]]"
    assert page_writer.extract_wikilinks(content) == [
        "karpathy",
        "entities/openai",
        "concepts/rag",
    ]


def test_extract_wikilinks_trims_whitespace() -> None:
    assert page_writer.extract_wikilinks("[[  spaced  ]]") == ["spaced"]


def test_extract_wikilinks_none_present() -> None:
    assert page_writer.extract_wikilinks("no links at all") == []


# ---------------------------------------------------------------------------
# add_source_to_frontmatter / ensure_frontmatter_fields
# ---------------------------------------------------------------------------


def test_add_source_appends_and_sets_updated() -> None:
    page = page_writer.ParsedPage(frontmatter={"title": "X"}, body="b")
    page_writer.add_source_to_frontmatter(page, "my-paper", "2026-07-15")
    assert page.frontmatter["sources"] == ["sources/my-paper"]
    assert page.frontmatter["updated"] == "2026-07-15"


def test_add_source_is_idempotent() -> None:
    page = page_writer.ParsedPage(frontmatter={"sources": ["sources/my-paper"]}, body="b")
    page_writer.add_source_to_frontmatter(page, "my-paper", "2026-07-15")
    assert page.frontmatter["sources"] == ["sources/my-paper"]


def test_add_source_strips_stale_md_suffix() -> None:
    page = page_writer.ParsedPage(
        frontmatter={"sources": ["sources/old.md"]}, body="b"
    )
    page_writer.add_source_to_frontmatter(page, "new", "2026-07-15")
    assert page.frontmatter["sources"] == ["sources/old", "sources/new"]


def test_add_source_tolerates_non_list_sources() -> None:
    page = page_writer.ParsedPage(frontmatter={"sources": "oops"}, body="b")
    page_writer.add_source_to_frontmatter(page, "new", "2026-07-15")
    assert page.frontmatter["sources"] == ["sources/new"]


def test_ensure_frontmatter_fields_adds_missing_only() -> None:
    page = page_writer.ParsedPage(frontmatter={"title": "Keep"}, body="b")
    page_writer.ensure_frontmatter_fields(page, {"title": "Overwrite?", "type": "concept"})
    assert page.frontmatter["title"] == "Keep"        # not overwritten
    assert page.frontmatter["type"] == "concept"      # added


# ---------------------------------------------------------------------------
# rebuild_index
# ---------------------------------------------------------------------------


def test_rebuild_index_lists_pages_and_counts(paths: cfg.WikiPaths) -> None:
    # The `paths`/`wiki_root` fixture already has concepts/amortized-inference.md.
    (paths.wiki / "entities" / "openai.md").write_text(
        "---\ntitle: OpenAI\n---\n\nBody.\n", encoding="utf-8"
    )
    (paths.wiki / "sources" / "paper.md").write_text(
        "---\ntitle: A Paper\n---\n\nBody.\n", encoding="utf-8"
    )

    page_writer.rebuild_index(paths, "2026-07-15")

    index_text = paths.index.read_text(encoding="utf-8")
    # Titles from frontmatter and slug-based wikilinks are present.
    assert "[[entities/openai|OpenAI]]" in index_text
    assert "[[sources/paper|A Paper]]" in index_text
    assert "[[concepts/amortized-inference|Amortized Inference]]" in index_text
    # Stats line reflects the counts (1 source, 1 entity, 1 concept, 0 synthesis).
    assert "1 sources · 1 entities · 1 concepts · 0 synthesis pages" in index_text
