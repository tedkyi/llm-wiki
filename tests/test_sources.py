"""Tests for the source-storage layer: path references vs copies.

These exercise `ingest_raw` end-to-end against a throwaway on-disk wiki (real
files, real SQLite) but never touch Ollama/QMD/an LLM — `add_file` only parses
and records. Sources default to being *referenced* in place; `--copy` (copy=True)
falls back to copying into raw/.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_wiki import config as cfg
from llm_wiki import db
from llm_wiki import ingest_raw


TEXT_A = "Qwen3 is a large language model family developed by Alibaba Cloud for complex reasoning tasks.\n"
TEXT_B = "Retrieval augmented generation grounds a language model's answers in documents fetched at query time.\n"


@pytest.fixture
def paths(tmp_path: Path) -> cfg.WikiPaths:
    """A wiki root with an initialized state DB, but no raw/ copies."""
    root = tmp_path / "mywiki"
    p = cfg.WikiPaths(root=root)
    db.init_db(p.state_db)
    return p


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# source_path resolver
# ---------------------------------------------------------------------------


def test_source_path_resolves_relative_under_root(paths: cfg.WikiPaths):
    got = ingest_raw.source_path(paths, "raw/foo.pdf")
    assert got == paths.root / "raw" / "foo.pdf"


def test_source_path_returns_absolute_as_is(paths: cfg.WikiPaths, tmp_path: Path):
    abs_path = (tmp_path / "elsewhere" / "bar.pdf").resolve()
    got = ingest_raw.source_path(paths, str(abs_path))
    assert got.is_absolute()
    assert got == abs_path


# ---------------------------------------------------------------------------
# add_file: reference (default) vs copy
# ---------------------------------------------------------------------------


def test_add_file_references_in_place_by_default(paths: cfg.WikiPaths, tmp_path: Path):
    src = _write(tmp_path / "papers" / "note.txt", TEXT_A)

    outcome = ingest_raw.add_file(paths, src)  # copy defaults to False

    assert outcome.result == ingest_raw.AddResult.ADDED
    # Stored as the file's absolute path...
    assert Path(outcome.relpath).is_absolute()
    assert Path(outcome.relpath) == src.resolve()
    # ...and no copy was made under raw/.
    assert not paths.raw.exists() or not any(paths.raw.iterdir())
    # A raw-text mirror is written, keyed on the source id.
    mirror = paths.raw_text / f"{outcome.source_id}-{ingest_raw._mirror_slug(src)}.md"
    assert mirror.exists()


def test_add_file_copy_true_copies_into_raw(paths: cfg.WikiPaths, tmp_path: Path):
    src = _write(tmp_path / "papers" / "note.txt", TEXT_A)

    outcome = ingest_raw.add_file(paths, src, copy=True)

    assert outcome.result == ingest_raw.AddResult.ADDED
    # Stored relative to the project root, and the copy lives under raw/.
    assert not Path(outcome.relpath).is_absolute()
    copied = ingest_raw.source_path(paths, outcome.relpath)
    assert copied.exists()
    assert ingest_raw._is_inside_raw(copied, paths.raw)
    # The original is untouched by a copy-add.
    assert src.exists()


# ---------------------------------------------------------------------------
# remove_source: delete copies, keep references
# ---------------------------------------------------------------------------


def test_remove_keeps_referenced_file(paths: cfg.WikiPaths, tmp_path: Path):
    src = _write(tmp_path / "papers" / "ref.txt", TEXT_A)
    outcome = ingest_raw.add_file(paths, src)  # referenced

    ok, _ = ingest_raw.remove_source(paths, outcome.source_id)

    assert ok
    assert src.exists(), "a referenced original must never be deleted on remove"
    # The mirror is cleaned up though.
    mirror = paths.raw_text / f"{outcome.source_id}-{ingest_raw._mirror_slug(src)}.md"
    assert not mirror.exists()


def test_remove_deletes_copied_file(paths: cfg.WikiPaths, tmp_path: Path):
    src = _write(tmp_path / "papers" / "copy.txt", TEXT_A)
    outcome = ingest_raw.add_file(paths, src, copy=True)
    copied = ingest_raw.source_path(paths, outcome.relpath)
    assert copied.exists()

    ok, _ = ingest_raw.remove_source(paths, outcome.source_id)

    assert ok
    assert not copied.exists(), "a raw/ copy should be deleted on remove"
    assert src.exists(), "the external original the copy came from is left alone"


# ---------------------------------------------------------------------------
# Mirror naming: no collision across identical basenames
# ---------------------------------------------------------------------------


def test_same_basename_different_dirs_get_distinct_mirrors(
    paths: cfg.WikiPaths, tmp_path: Path
):
    f1 = _write(tmp_path / "dir1" / "notes.txt", TEXT_A)
    f2 = _write(tmp_path / "dir2" / "notes.txt", TEXT_B)  # different content → no dedupe

    o1 = ingest_raw.add_file(paths, f1)
    o2 = ingest_raw.add_file(paths, f2)

    assert o1.result == ingest_raw.AddResult.ADDED
    assert o2.result == ingest_raw.AddResult.ADDED
    m1 = paths.raw_text / f"{o1.source_id}-{ingest_raw._mirror_slug(f1)}.md"
    m2 = paths.raw_text / f"{o2.source_id}-{ingest_raw._mirror_slug(f2)}.md"
    assert m1 != m2
    assert m1.exists() and m2.exists()
    # Each mirror holds its own file's text, not the other's.
    assert "Qwen3" in m1.read_text(encoding="utf-8")
    assert "Retrieval" in m2.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Pasted paths wrapped in quotes (Explorer "Copy as path")
# ---------------------------------------------------------------------------


def test_strip_path_quotes():
    assert ingest_raw.strip_path_quotes('"D:/a b/c.pdf"') == "D:/a b/c.pdf"
    assert ingest_raw.strip_path_quotes("'D:/a b/c.pdf'") == "D:/a b/c.pdf"
    assert ingest_raw.strip_path_quotes('  "D:/x.pdf"  ') == "D:/x.pdf"
    # No surrounding pair → unchanged (mismatched / interior quotes preserved).
    assert ingest_raw.strip_path_quotes("D:/x.pdf") == "D:/x.pdf"
    assert ingest_raw.strip_path_quotes('"D:/x.pdf') == '"D:/x.pdf'


def test_add_file_via_quoted_path_resolves(paths: cfg.WikiPaths, tmp_path: Path):
    src = _write(tmp_path / "papers" / "quoted note.txt", TEXT_A)
    quoted = f'"{src}"'
    outcome = ingest_raw.add_file(paths, Path(ingest_raw.strip_path_quotes(quoted)))
    assert outcome.result == ingest_raw.AddResult.ADDED
    assert Path(outcome.relpath) == src.resolve()


# ---------------------------------------------------------------------------
# Legacy mirror cleanup (self-healing rename on write)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Re-adding a tracked path: unchanged (no-op) vs changed (update in place)
# ---------------------------------------------------------------------------


def _status(paths: cfg.WikiPaths, source_id: int) -> str:
    return ingest_raw.get_source(paths, source_id)["status"]


def test_readd_unchanged_is_deduped(paths: cfg.WikiPaths, tmp_path: Path):
    src = _write(tmp_path / "papers" / "p.txt", TEXT_A)
    first = ingest_raw.add_file(paths, src)
    assert first.result == ingest_raw.AddResult.ADDED

    again = ingest_raw.add_file(paths, src)

    assert again.result == ingest_raw.AddResult.DEDUPED
    assert again.source_id == first.source_id
    assert len(ingest_raw.list_sources(paths)) == 1


def test_readd_changed_updates_in_place_and_marks_pending(
    paths: cfg.WikiPaths, tmp_path: Path
):
    src = _write(tmp_path / "papers" / "p.txt", TEXT_A)
    first = ingest_raw.add_file(paths, src)
    # Pretend it was already ingested, so we can see it get re-queued.
    with db.connect(paths.state_db) as conn:
        conn.execute(
            "UPDATE sources SET status = 'ingested', last_ingested = '2020-01-01' WHERE id = ?",
            (first.source_id,),
        )
    old_hash = first.content_hash

    # Same path, new content.
    _write(src, TEXT_B)
    out = ingest_raw.add_file(paths, src)

    # Updated in place — same row, no crash, no duplicate.
    assert out.result == ingest_raw.AddResult.UPDATED
    assert out.source_id == first.source_id
    assert len(ingest_raw.list_sources(paths)) == 1
    # Hash refreshed and re-queued for ingest.
    row = ingest_raw.get_source(paths, first.source_id)
    assert row["content_hash"] != old_hash
    assert row["status"] == "pending"
    assert row["last_ingested"] is None
    # Mirror reflects the new content.
    mirror = paths.raw_text / f"{first.source_id}-{ingest_raw._mirror_slug(src)}.md"
    assert "Retrieval" in mirror.read_text(encoding="utf-8")


def test_write_text_mirror_removes_legacy_name(paths: cfg.WikiPaths, tmp_path: Path):
    from llm_wiki import parsers

    src = _write(tmp_path / "papers" / "legacy.txt", TEXT_A)
    parsed = parsers.parse(src)
    # Simulate a pre-existing legacy (`<slug>.md`) mirror from before id-keying.
    legacy = ingest_raw._legacy_mirror_path(paths, src)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("stale", encoding="utf-8")

    mirror = ingest_raw.write_text_mirror(paths, 7, src, parsed)

    assert mirror.exists()
    assert mirror.name == f"7-{ingest_raw._mirror_slug(src)}.md"
    assert not legacy.exists(), "writing the id-keyed mirror should remove the legacy one"
