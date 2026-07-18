"""Unit tests for the pure QMD-output parsing helpers in search.py. No qmd
subprocess is ever invoked — these operate on canned JSON strings and dicts.
"""

from __future__ import annotations

from pathlib import Path

from llm_wiki import config as cfg
from llm_wiki import search


# ---------------------------------------------------------------------------
# _parse_qmd_json
# ---------------------------------------------------------------------------


def test_parse_qmd_json_plain_array() -> None:
    assert search._parse_qmd_json('[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]


def test_parse_qmd_json_empty_string() -> None:
    assert search._parse_qmd_json("   ") == []


def test_parse_qmd_json_results_wrapper() -> None:
    assert search._parse_qmd_json('{"results": [{"x": 1}]}') == [{"x": 1}]


def test_parse_qmd_json_hits_wrapper() -> None:
    assert search._parse_qmd_json('{"hits": [{"y": 2}]}') == [{"y": 2}]


def test_parse_qmd_json_recovers_array_amid_noise() -> None:
    stdout = "loading model...\n[{\"a\": 1}]\nDone.\n"
    assert search._parse_qmd_json(stdout) == [{"a": 1}]


def test_parse_qmd_json_unparseable_returns_empty() -> None:
    assert search._parse_qmd_json("not json at all") == []


def test_parse_qmd_json_object_without_list_key_returns_empty() -> None:
    assert search._parse_qmd_json('{"status": "ok"}') == []


# ---------------------------------------------------------------------------
# _hit_from_dict — tolerant field names
# ---------------------------------------------------------------------------


def test_hit_from_dict_canonical_fields() -> None:
    hit = search._hit_from_dict(
        {"docid": "#1", "path": "concepts/rag.md", "collection": "llm-wiki-pages",
         "title": "RAG", "score": 0.9, "snippet": "snip"}
    )
    assert hit.docid == "#1"
    assert hit.path == "concepts/rag.md"
    assert hit.title == "RAG"
    assert hit.score == 0.9


def test_hit_from_dict_alternate_field_names() -> None:
    hit = search._hit_from_dict(
        {"id": "abc", "filepath": "a.md", "col": "c", "preview": "pv"}
    )
    assert hit.docid == "abc"
    assert hit.path == "a.md"
    assert hit.collection == "c"
    assert hit.snippet == "pv"


def test_hit_from_dict_missing_fields_default_safely() -> None:
    hit = search._hit_from_dict({})
    assert hit.docid == ""
    assert hit.path == ""
    assert hit.score == 0.0


# ---------------------------------------------------------------------------
# SearchHit / SearchResults
# ---------------------------------------------------------------------------


def test_search_hit_full_path() -> None:
    hit = search.SearchHit(
        docid="#1", path="concepts/rag.md", collection="llm-wiki-pages",
        title="RAG", score=0.5,
    )
    assert hit.full_path == "llm-wiki-pages/concepts/rag.md"


def test_search_results_len_and_iter() -> None:
    hits = [
        search.SearchHit(docid="1", path="a.md", collection="c", title="A", score=0.1),
        search.SearchHit(docid="2", path="b.md", collection="c", title="B", score=0.2),
    ]
    results = search.SearchResults(query="q", hits=hits)
    assert len(results) == 2
    assert [h.docid for h in results] == ["1", "2"]


# ---------------------------------------------------------------------------
# _normalize_hit — qmd:// URI and absolute-path → collection mapping
# ---------------------------------------------------------------------------


def test_normalize_hit_qmd_uri(tmp_path: Path) -> None:
    paths = cfg.WikiPaths(root=tmp_path)
    hit = search.SearchHit(
        docid="1", path="qmd://llm-wiki-pages/concepts/rag", collection="",
        title="RAG", score=0.5,
    )
    search._normalize_hit(paths, hit)
    assert hit.collection == "llm-wiki-pages"
    assert hit.path == "concepts/rag"


def test_normalize_hit_absolute_path_maps_to_wiki_collection(tmp_path: Path) -> None:
    paths = cfg.WikiPaths(root=tmp_path)
    page = paths.wiki / "concepts" / "rag.md"
    page.parent.mkdir(parents=True)
    page.write_text("x", encoding="utf-8")

    hit = search.SearchHit(
        docid="1", path=str(page), collection="", title="RAG", score=0.5,
    )
    search._normalize_hit(paths, hit)
    assert hit.collection == "llm-wiki-pages"
    assert hit.path == "concepts/rag.md"
    assert hit.abs_path == str(page)


def test_normalize_hit_absolute_path_maps_raw_text(tmp_path: Path) -> None:
    paths = cfg.WikiPaths(root=tmp_path)
    mirror = paths.raw_text / "1-note.md"
    mirror.parent.mkdir(parents=True)
    mirror.write_text("x", encoding="utf-8")

    hit = search.SearchHit(
        docid="1", path=str(mirror), collection="", title="Note", score=0.5,
    )
    search._normalize_hit(paths, hit)
    assert hit.collection == "llm-wiki-raw"
    assert hit.path == "1-note.md"


def test_normalize_hit_relative_path_left_untouched(tmp_path: Path) -> None:
    paths = cfg.WikiPaths(root=tmp_path)
    hit = search.SearchHit(
        docid="1", path="concepts/rag.md", collection="llm-wiki-pages",
        title="RAG", score=0.5,
    )
    search._normalize_hit(paths, hit)
    # Already a plain relative path — nothing changes.
    assert hit.path == "concepts/rag.md"
    assert hit.abs_path == ""
