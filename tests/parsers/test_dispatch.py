"""Unit tests for parsers.__init__ — extension dispatch. No heavy parsers invoked."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_wiki import parsers
from llm_wiki.parsers.base import ParserError


@pytest.mark.parametrize(
    "name, supported",
    [
        ("a.md", True),
        ("a.markdown", True),
        ("a.txt", True),
        ("a.pdf", True),
        ("a.docx", True),
        ("a.html", True),
        ("a.htm", True),
        ("a.PDF", True),      # case-insensitive
        ("a.TXT", True),
        ("a.png", False),
        ("a.epub", False),
        ("noext", False),
    ],
)
def test_is_supported(name: str, supported: bool) -> None:
    assert parsers.is_supported(Path(name)) is supported


def test_supported_extensions_membership() -> None:
    assert ".md" in parsers.SUPPORTED_EXTENSIONS
    assert ".png" not in parsers.SUPPORTED_EXTENSIONS


def test_parse_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ParserError, match="File not found"):
        parsers.parse(tmp_path / "nope.txt")


def test_parse_directory_raises(tmp_path: Path) -> None:
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(ParserError, match="Not a regular file"):
        parsers.parse(d)


def test_parse_unsupported_extension_raises(tmp_path: Path) -> None:
    f = tmp_path / "image.png"
    f.write_bytes(b"\x89PNG")
    with pytest.raises(ParserError, match="Unsupported file type"):
        parsers.parse(f)


def test_parse_text_file_dispatches(tmp_path: Path) -> None:
    # A real .txt round-trip proves the dispatcher reaches the text parser
    # without needing any heavy PDF/DOCX/HTML dependency.
    f = tmp_path / "note.txt"
    f.write_text("Hello world, this is a plain note.\n", encoding="utf-8")
    doc = parsers.parse(f)
    assert doc.file_type in ("txt", "text", "md")
    assert "Hello world" in doc.text
