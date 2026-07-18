"""Unit tests for the pure helpers in ingest_llm.py — JSON extraction, schema
parsing, excerpt trimming, slug resolution, and result types. No LLM calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_wiki import config as cfg
from llm_wiki import ingest_llm


# ---------------------------------------------------------------------------
# _extract_json_object — brace matching, fence stripping
# ---------------------------------------------------------------------------


def test_extract_json_object_plain() -> None:
    assert ingest_llm._extract_json_object('{"a": 1}') == '{"a": 1}'


def test_extract_json_object_strips_prose_around() -> None:
    text = 'Here is the JSON:\n{"a": 1}\nHope that helps!'
    assert ingest_llm._extract_json_object(text) == '{"a": 1}'


def test_extract_json_object_strips_code_fence() -> None:
    text = '```json\n{"a": 1}\n```'
    assert ingest_llm._extract_json_object(text).strip() == '{"a": 1}'


def test_extract_json_object_nested_braces() -> None:
    text = '{"outer": {"inner": 2}, "b": 3}'
    assert ingest_llm._extract_json_object(text) == text


def test_extract_json_object_ignores_braces_in_strings() -> None:
    # A closing brace inside a string literal must not end the object early.
    text = '{"note": "a } brace", "x": 1}'
    assert ingest_llm._extract_json_object(text) == text


def test_extract_json_object_handles_escaped_quote() -> None:
    text = r'{"q": "say \"hi\"", "n": 1}'
    assert ingest_llm._extract_json_object(text) == text


def test_extract_json_object_no_brace_returns_text() -> None:
    assert ingest_llm._extract_json_object("no json here") == "no json here"


# ---------------------------------------------------------------------------
# _parse_extraction — JSON → Pydantic
# ---------------------------------------------------------------------------


def test_parse_extraction_valid() -> None:
    raw = """{
      "title": "A Paper",
      "source_slug": "a-paper",
      "summary": "It solves X.",
      "key_takeaways": ["one", "two"],
      "entities": [{"name": "OpenAI", "slug": "openai", "type": "organization", "description": "d"}],
      "concepts": [{"name": "RAG", "slug": "rag", "type": "concept", "description": "d"}],
      "tags": ["ml"]
    }"""
    ex = ingest_llm._parse_extraction(raw)
    assert ex.title == "A Paper"
    assert ex.source_slug == "a-paper"
    assert ex.entities[0].slug == "openai"
    assert ex.concepts[0].name == "RAG"


def test_parse_extraction_tolerates_surrounding_prose() -> None:
    raw = 'Sure!\n{"title": "T", "source_slug": "t", "summary": "s"}\nDone.'
    ex = ingest_llm._parse_extraction(raw)
    assert ex.title == "T"
    assert ex.entities == []      # defaults for omitted lists


def test_parse_extraction_invalid_json_raises_valueerror() -> None:
    with pytest.raises(ValueError, match="Invalid JSON"):
        ingest_llm._parse_extraction("{not valid json")


def test_parse_extraction_schema_violation_raises_valueerror() -> None:
    # Missing required fields (title/source_slug/summary).
    with pytest.raises(ValueError, match="didn't match expected schema"):
        ingest_llm._parse_extraction('{"tags": ["x"]}')


# ---------------------------------------------------------------------------
# _build_excerpt
# ---------------------------------------------------------------------------


def test_build_excerpt_short_text_unchanged() -> None:
    assert ingest_llm._build_excerpt("short", max_chars=100) == "short"


def test_build_excerpt_truncates_long_text() -> None:
    text = "x" * 500
    out = ingest_llm._build_excerpt(text, max_chars=100)
    assert out.startswith("x" * 100)
    assert out.endswith("[... truncated ...]")


# ---------------------------------------------------------------------------
# _resolve_slug
# ---------------------------------------------------------------------------


def test_resolve_slug_new_uses_llm_suggestion(paths: cfg.WikiPaths) -> None:
    slug, exists = ingest_llm._resolve_slug("Yann LeCun", "entity", paths, "lecun")
    assert slug == "lecun"
    assert exists is False


def test_resolve_slug_falls_back_to_slugified_name(paths: cfg.WikiPaths) -> None:
    # Empty LLM suggestion → slugify the name.
    slug, exists = ingest_llm._resolve_slug("Geoffrey Hinton", "entity", paths, "")
    assert slug == "geoffrey-hinton"
    assert exists is False


def test_resolve_slug_matches_existing_page(paths: cfg.WikiPaths) -> None:
    entities = paths.wiki / "entities"
    entities.mkdir(parents=True, exist_ok=True)
    (entities / "karpathy.md").write_text(
        "---\ntitle: Andrej Karpathy\n---\n\nBody.\n", encoding="utf-8"
    )
    slug, exists = ingest_llm._resolve_slug("Andrej Karpathy", "entity", paths, "andrej")
    assert slug == "karpathy"
    assert exists is True


# ---------------------------------------------------------------------------
# IngestResult.ok
# ---------------------------------------------------------------------------


def test_ingest_result_ok_true_when_clean() -> None:
    r = ingest_llm.IngestResult(source_id=1, source_title="T", source_slug="t")
    assert r.ok is True


def test_ingest_result_ok_false_on_error() -> None:
    r = ingest_llm.IngestResult(source_id=1, source_title="T", source_slug="t", error="boom")
    assert r.ok is False


def test_ingest_result_ok_false_when_skipped() -> None:
    r = ingest_llm.IngestResult(source_id=1, source_title="T", source_slug="t", skipped=True)
    assert r.ok is False
