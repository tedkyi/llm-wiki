"""Unit tests for parsers.base — pure text normalization/hashing. No backends."""

from __future__ import annotations

from pathlib import Path

from llm_wiki.parsers import base


# ---------------------------------------------------------------------------
# normalize_text()
# ---------------------------------------------------------------------------


def test_normalize_text_empty() -> None:
    assert base.normalize_text("") == ""


def test_normalize_text_collapses_line_endings() -> None:
    assert base.normalize_text("a\r\nb\rc") == "a\nb\nc"


def test_normalize_text_collapses_intra_line_whitespace() -> None:
    assert base.normalize_text("foo   \t  bar") == "foo bar"


def test_normalize_text_collapses_excess_blank_lines() -> None:
    text = "a\n\n\n\n\nb"
    # 3+ blank lines collapse to exactly one blank line (two \n).
    assert base.normalize_text(text) == "a\n\nb"


def test_normalize_text_strips_leading_trailing() -> None:
    assert base.normalize_text("\n\n  hi  \n\n") == "hi"


def test_normalize_text_is_stable_across_equivalent_inputs() -> None:
    a = base.normalize_text("Hello   world\r\n\r\n\r\nline")
    b = base.normalize_text("Hello world\n\nline")
    assert a == b
    # Hash stability follows directly from normalized-text equality.
    assert base.compute_hash(a) == base.compute_hash(b)


# ---------------------------------------------------------------------------
# compute_hash()
# ---------------------------------------------------------------------------


def test_compute_hash_is_deterministic() -> None:
    assert base.compute_hash("hello") == base.compute_hash("hello")


def test_compute_hash_is_sensitive_to_content() -> None:
    assert base.compute_hash("hello") != base.compute_hash("world")


def test_compute_hash_is_sha256_hex() -> None:
    h = base.compute_hash("hello")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# fallback_title_from_path()
# ---------------------------------------------------------------------------


def test_fallback_title_humanizes_separators() -> None:
    assert base.fallback_title_from_path(Path("some_doc-v2.pdf")) == "Some Doc V2"


def test_fallback_title_preserves_short_acronyms() -> None:
    # All-caps tokens of length <= 4 are preserved as-is.
    assert base.fallback_title_from_path(Path("NASA report.pdf")) == "NASA Report"


def test_fallback_title_capitalizes_long_uppercase() -> None:
    # 5+ char uppercase tokens are not treated as acronyms.
    assert base.fallback_title_from_path(Path("HELLOWORLD.txt")) == "Helloworld"


# ---------------------------------------------------------------------------
# ParsedDocument properties
# ---------------------------------------------------------------------------


def _doc(text: str) -> base.ParsedDocument:
    return base.ParsedDocument(
        source_path=Path("x.txt"),
        file_type="txt",
        title="X",
        text=text,
        content_hash=base.compute_hash(text),
        bytes=len(text),
    )


def test_parsed_document_word_count() -> None:
    assert _doc("one two three").word_count == 3
    assert _doc("").word_count == 0


def test_parsed_document_text_length() -> None:
    assert _doc("abcde").text_length == 5


def test_parsed_document_is_empty_below_threshold() -> None:
    # is_empty is True when fewer than 10 words were extracted.
    assert _doc("just a few words here").is_empty is True


def test_parsed_document_is_not_empty_above_threshold() -> None:
    assert _doc(" ".join(["word"] * 10)).is_empty is False
