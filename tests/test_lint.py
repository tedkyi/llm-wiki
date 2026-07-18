"""Unit tests for the fast (non-LLM) lint checks. The deep contradiction check
is LLM-backed and deliberately not tested here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_wiki import config as cfg
from llm_wiki import lint
from llm_wiki import page_writer


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write(paths: cfg.WikiPaths, relpath: str, body: str, **frontmatter) -> None:
    """Write a wiki page under wiki/<relpath>."""
    fm = page_writer.ParsedPage(frontmatter=dict(frontmatter), body=body)
    dest = paths.wiki / relpath
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(fm.to_markdown(), encoding="utf-8")


# ---------------------------------------------------------------------------
# _normalize_link
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("entities/qwen", "entities/qwen"),
        ("entities/qwen.md", "entities/qwen"),
        ("qmd://llm-wiki-pages/entities/qwen", "entities/qwen"),
        ("/qmd://llm-wiki-pages/entities/qwen", "entities/qwen"),
        ("/entities/qwen", "entities/qwen"),
        ("sources/notes|Q Notes", "sources/notes"),
        ("", ""),
    ],
)
def test_normalize_link(raw: str, expected: str) -> None:
    assert lint._normalize_link(raw) == expected


# ---------------------------------------------------------------------------
# check_broken_wikilinks
# ---------------------------------------------------------------------------


def test_broken_wikilink_genuinely_missing_is_error(paths: cfg.WikiPaths) -> None:
    _write(paths, "concepts/a.md", "links to [[totally-missing]]", title="A", type="concept")
    inv = lint._build_inventory(paths)
    issues = lint.check_broken_wikilinks(inv)
    missing = [i for i in issues if "totally-missing" in i.message]
    assert missing and missing[0].severity == lint.Severity.ERROR
    assert missing[0].fixable is False


def test_broken_wikilink_bare_basename_is_fixable_warning(paths: cfg.WikiPaths) -> None:
    # The fixture already has concepts/amortized-inference.md. A bare-basename
    # link resolves to it unambiguously → fixable warning.
    _write(paths, "entities/x.md", "see [[amortized-inference]]", title="X", type="entity")
    inv = lint._build_inventory(paths)
    issues = lint.check_broken_wikilinks(inv)
    fixable = [i for i in issues if i.fixable and i.page == "entities/x.md"]
    assert fixable
    assert fixable[0].severity == lint.Severity.WARNING
    assert fixable[0].context["new_target"] == "concepts/amortized-inference"


def test_broken_wikilink_ambiguous_is_error(paths: cfg.WikiPaths) -> None:
    # Same basename in two subdirs → ambiguous, cannot auto-fix.
    _write(paths, "entities/dup.md", "e", title="E", type="entity")
    _write(paths, "concepts/dup.md", "c", title="C", type="concept")
    _write(paths, "synthesis/s.md", "link [[dup]]", title="S", type="synthesis")
    inv = lint._build_inventory(paths)
    issues = lint.check_broken_wikilinks(inv)
    ambiguous = [i for i in issues if "ambiguous" in i.message]
    assert ambiguous and ambiguous[0].severity == lint.Severity.ERROR
    assert ambiguous[0].fixable is False


def test_valid_wikilink_produces_no_issue(paths: cfg.WikiPaths) -> None:
    _write(paths, "entities/x.md", "see [[concepts/amortized-inference]]",
           title="X", type="entity")
    inv = lint._build_inventory(paths)
    issues = lint.check_broken_wikilinks(inv)
    assert not [i for i in issues if i.page == "entities/x.md"]


# ---------------------------------------------------------------------------
# check_orphan_pages
# ---------------------------------------------------------------------------


def test_orphan_entity_flagged(paths: cfg.WikiPaths) -> None:
    _write(paths, "entities/lonely.md", "nobody links here", title="Lonely", type="entity")
    inv = lint._build_inventory(paths)
    issues = lint.check_orphan_pages(inv)
    assert any(i.page == "entities/lonely.md" for i in issues)


def test_linked_entity_not_orphan(paths: cfg.WikiPaths) -> None:
    _write(paths, "entities/popular.md", "content", title="Popular", type="entity")
    _write(paths, "concepts/ref.md", "see [[entities/popular]]", title="Ref", type="concept")
    inv = lint._build_inventory(paths)
    issues = lint.check_orphan_pages(inv)
    assert not any(i.page == "entities/popular.md" for i in issues)


def test_source_pages_exempt_from_orphan_check(paths: cfg.WikiPaths) -> None:
    _write(paths, "sources/unlinked.md", "raw notes", title="Src", type="source")
    inv = lint._build_inventory(paths)
    issues = lint.check_orphan_pages(inv)
    assert not any(i.page == "sources/unlinked.md" for i in issues)


# ---------------------------------------------------------------------------
# check_frontmatter
# ---------------------------------------------------------------------------


def test_frontmatter_missing_fields_warns(paths: cfg.WikiPaths) -> None:
    # Entity requires title/type/created/updated; only title present.
    _write(paths, "entities/thin.md", "body", title="Thin")
    inv = lint._build_inventory(paths)
    issues = lint.check_frontmatter(inv)
    thin = [i for i in issues if i.page == "entities/thin.md"]
    assert thin and thin[0].check == lint.CheckId.INVALID_FRONTMATTER


def test_frontmatter_absent_is_error(paths: cfg.WikiPaths) -> None:
    # No frontmatter at all.
    dest = paths.wiki / "entities" / "raw.md"
    dest.write_text("no frontmatter here\n", encoding="utf-8")
    inv = lint._build_inventory(paths)
    issues = lint.check_frontmatter(inv)
    raw = [i for i in issues if i.page == "entities/raw.md"]
    assert raw and raw[0].check == lint.CheckId.MISSING_FRONTMATTER
    assert raw[0].severity == lint.Severity.ERROR


# ---------------------------------------------------------------------------
# check_malformed_wikilinks
# ---------------------------------------------------------------------------


def test_malformed_wikilink_with_md_suffix_is_fixable(paths: cfg.WikiPaths) -> None:
    _write(paths, "concepts/a.md", "link to [[entities/qwen.md]]", title="A", type="concept")
    inv = lint._build_inventory(paths)
    issues = lint.check_malformed_wikilinks(inv, paths)
    mal = [i for i in issues if i.page == "concepts/a.md"]
    assert mal and mal[0].fixable
    assert mal[0].context["new_target"] == "entities/qwen"


def test_clean_wikilink_not_malformed(paths: cfg.WikiPaths) -> None:
    _write(paths, "concepts/a.md", "link to [[entities/qwen]]", title="A", type="concept")
    inv = lint._build_inventory(paths)
    issues = lint.check_malformed_wikilinks(inv, paths)
    assert not [i for i in issues if i.page == "concepts/a.md"]


# ---------------------------------------------------------------------------
# check_missing_concepts (threshold)
# ---------------------------------------------------------------------------


def test_missing_concept_flagged_at_threshold(paths: cfg.WikiPaths) -> None:
    # Three distinct pages link to the same nonexistent target.
    for i in range(3):
        _write(paths, f"concepts/p{i}.md", "mentions [[transformers]]",
               title=f"P{i}", type="concept")
    inv = lint._build_inventory(paths)
    issues = lint.check_missing_concepts(inv, threshold=3)
    assert any(i.context.get("target") == "transformers" for i in issues)


def test_missing_concept_below_threshold_not_flagged(paths: cfg.WikiPaths) -> None:
    for i in range(2):
        _write(paths, f"concepts/p{i}.md", "mentions [[transformers]]",
               title=f"P{i}", type="concept")
    inv = lint._build_inventory(paths)
    issues = lint.check_missing_concepts(inv, threshold=3)
    assert not any(i.context.get("target") == "transformers" for i in issues)


# ---------------------------------------------------------------------------
# check_stale_source_refs
# ---------------------------------------------------------------------------


def test_stale_source_ref_flagged(paths: cfg.WikiPaths) -> None:
    _write(paths, "concepts/a.md", "body", title="A", type="concept",
           sources=["sources/deleted-paper"])
    inv = lint._build_inventory(paths)
    issues = lint.check_stale_source_refs(inv, paths)
    assert any("deleted-paper" in i.message for i in issues)


def test_present_source_ref_not_stale(paths: cfg.WikiPaths) -> None:
    _write(paths, "sources/real-paper.md", "content", title="Real", type="source")
    _write(paths, "concepts/a.md", "body", title="A", type="concept",
           sources=["sources/real-paper"])
    inv = lint._build_inventory(paths)
    issues = lint.check_stale_source_refs(inv, paths)
    assert not any("real-paper" in i.message for i in issues)


# ---------------------------------------------------------------------------
# check_duplicate_header_blocks
# ---------------------------------------------------------------------------


# A duplicate frontmatter block fenced by tripled hyphens, ahead of the H1.
_DUP_HYPHENS = (
    "---\n"
    'title: "Dup"\n'
    "type: source\n"
    "---\n"
    "\n"
    "# Real Title\n"
    "\n"
    "## Summary\n"
    "body\n"
)

# Same artifact but fenced with tripled backticks (a common variant).
_DUP_BACKTICKS = (
    "```yaml\n"
    'title: "Dup"\n'
    "```\n"
    "\n"
    "# Real Title\n"
    "\n"
    "body\n"
)


def test_duplicate_header_block_hyphen_fence_flagged(paths: cfg.WikiPaths) -> None:
    _write(paths, "sources/a.md", _DUP_HYPHENS, title="Real", type="source")
    inv = lint._build_inventory(paths)
    issues = lint.check_duplicate_header_blocks(inv)
    dup = [i for i in issues if i.page == "sources/a.md"]
    assert dup and dup[0].check == lint.CheckId.DUPLICATE_HEADER_BLOCK
    assert dup[0].fixable is True


def test_duplicate_header_block_backtick_fence_flagged(paths: cfg.WikiPaths) -> None:
    _write(paths, "sources/b.md", _DUP_BACKTICKS, title="Real", type="source")
    inv = lint._build_inventory(paths)
    issues = lint.check_duplicate_header_blocks(inv)
    assert any(i.page == "sources/b.md" for i in issues)


def test_clean_page_no_duplicate_header(paths: cfg.WikiPaths) -> None:
    _write(paths, "sources/c.md", "# Real Title\n\n## Summary\nbody\n",
           title="Real", type="source")
    inv = lint._build_inventory(paths)
    issues = lint.check_duplicate_header_blocks(inv)
    assert not any(i.page == "sources/c.md" for i in issues)


def test_pre_h1_content_without_fence_is_flagged(paths: cfg.WikiPaths) -> None:
    # A well-formed body starts at its H1, so ANY non-whitespace before it is
    # junk — fenced or not. (Widened from the original fence-only detection.)
    _write(paths, "sources/d.md", "some stray text\n\n# Real Title\n\nbody\n",
           title="Real", type="source")
    inv = lint._build_inventory(paths)
    issues = lint.check_duplicate_header_blocks(inv)
    assert any(i.page == "sources/d.md" for i in issues)


def test_page_without_h1_not_flagged(paths: cfg.WikiPaths) -> None:
    # No H1 to anchor on — we must not strip the whole body.
    _write(paths, "sources/e.md", "---\ntitle: dup\n---\n\nno heading here\n",
           title="Real", type="source")
    inv = lint._build_inventory(paths)
    issues = lint.check_duplicate_header_blocks(inv)
    assert not any(i.page == "sources/e.md" for i in issues)


def test_stray_text_before_h1_flagged(paths: cfg.WikiPaths) -> None:
    # A non-fence fragment before the H1 (e.g. a truncated word) is still junk.
    _write(paths, "sources/f.md", "remov\n# Real Title\n\nbody\n",
           title="Real", type="source")
    inv = lint._build_inventory(paths)
    issues = lint.check_duplicate_header_blocks(inv)
    stray = [i for i in issues if i.page == "sources/f.md"]
    assert stray and stray[0].fixable is True
    assert "stray text" in stray[0].message


def test_apply_fix_strips_duplicate_header_block() -> None:
    parsed = page_writer.ParsedPage(
        frontmatter={},
        body='---\ntitle: "Dup"\n---\n\n# Real Title\n\nbody\n',
    )
    fix = lint.LintIssue(
        check=lint.CheckId.DUPLICATE_HEADER_BLOCK,
        severity=lint.Severity.WARNING,
        page="sources/a.md",
        message="",
        fixable=True,
        context={"location": "body", "strip_before_h1": True},
    )
    changed = lint._apply_fixes_to_page(parsed, [fix])
    assert changed is True
    assert parsed.body.lstrip().startswith("# Real Title")
    assert "Dup" not in parsed.body


# ---------------------------------------------------------------------------
# check_llm_artifact_body
# ---------------------------------------------------------------------------


def test_llm_refusal_body_flagged_not_fixable(paths: cfg.WikiPaths) -> None:
    # No H1, body is a refusal message (the entities/zhang.md pattern).
    _write(paths, "entities/z.md",
           "The write requires permission — please approve it.\n",
           title="Z", type="entity")
    inv = lint._build_inventory(paths)
    issues = lint.check_llm_artifact_body(inv)
    hit = [i for i in issues if i.page == "entities/z.md"]
    assert hit and hit[0].check == lint.CheckId.LLM_ARTIFACT_BODY
    assert hit[0].fixable is False


def test_llm_meta_commentary_before_h1_flagged(paths: cfg.WikiPaths) -> None:
    # Assistant chatter ahead of the title (the entities/deepmind.md pattern).
    _write(paths, "entities/d.md",
           "I'll just output the updated page directly.\n\n# Real Title\n\nbody\n",
           title="D", type="entity")
    inv = lint._build_inventory(paths)
    issues = lint.check_llm_artifact_body(inv)
    assert any(i.page == "entities/d.md" for i in issues)


def test_llm_artifact_still_flagged_after_h1_inserted(paths: cfg.WikiPaths) -> None:
    # After a missing-H1 fix inserts a title, the refusal text sits just below
    # it — the check must still see it (so --fix doesn't mask the problem).
    _write(paths, "entities/z.md",
           "# Z\n\nThe write requires permission — please approve it.\n",
           title="Z", type="entity")
    inv = lint._build_inventory(paths)
    issues = lint.check_llm_artifact_body(inv)
    assert any(i.page == "entities/z.md" for i in issues)


def test_prose_first_page_not_llm_artifact(paths: cfg.WikiPaths) -> None:
    # A legitimate prose-first page (no H1) must NOT trip the artifact check.
    _write(paths, "entities/g.md",
           "Google is a technology company that developed the LaMDA models.\n",
           title="Google", type="entity")
    inv = lint._build_inventory(paths)
    issues = lint.check_llm_artifact_body(inv)
    assert not any(i.page == "entities/g.md" for i in issues)


# ---------------------------------------------------------------------------
# check_missing_h1
# ---------------------------------------------------------------------------


def test_missing_h1_flagged_and_fixable(paths: cfg.WikiPaths) -> None:
    _write(paths, "entities/g.md", "Google is a technology company.\n",
           title="Google", type="entity")
    inv = lint._build_inventory(paths)
    issues = lint.check_missing_h1(inv)
    hit = [i for i in issues if i.page == "entities/g.md"]
    assert hit and hit[0].fixable is True
    assert hit[0].context["insert_h1_title"] == "Google"


def test_missing_h1_without_title_not_fixable(paths: cfg.WikiPaths) -> None:
    dest = paths.wiki / "entities" / "notitle.md"
    dest.write_text("just prose, no frontmatter, no heading\n", encoding="utf-8")
    inv = lint._build_inventory(paths)
    issues = lint.check_missing_h1(inv)
    hit = [i for i in issues if i.page == "entities/notitle.md"]
    assert hit and hit[0].fixable is False


def test_page_with_h1_not_missing(paths: cfg.WikiPaths) -> None:
    _write(paths, "entities/g.md", "# Google\n\nGoogle is a company.\n",
           title="Google", type="entity")
    inv = lint._build_inventory(paths)
    issues = lint.check_missing_h1(inv)
    assert not any(i.page == "entities/g.md" for i in issues)


def test_apply_fix_inserts_h1_from_title() -> None:
    parsed = page_writer.ParsedPage(
        frontmatter={"title": "Google"}, body="Google is a company.\n"
    )
    fix = lint.LintIssue(
        check=lint.CheckId.MISSING_H1,
        severity=lint.Severity.WARNING,
        page="entities/g.md",
        message="",
        fixable=True,
        context={"insert_h1_title": "Google"},
    )
    changed = lint._apply_fixes_to_page(parsed, [fix])
    assert changed is True
    assert parsed.body.startswith("# Google\n")
    assert "Google is a company." in parsed.body


# ---------------------------------------------------------------------------
# _apply_fixes_to_page
# ---------------------------------------------------------------------------


def test_apply_fixes_rewrites_body_wikilink() -> None:
    parsed = page_writer.ParsedPage(frontmatter={}, body="see [[old-target]] now")
    fix = lint.LintIssue(
        check=lint.CheckId.MALFORMED_WIKILINK,
        severity=lint.Severity.WARNING,
        page="concepts/a.md",
        message="",
        fixable=True,
        context={"old_target": "old-target", "new_target": "concepts/new", "location": "body"},
    )
    changed = lint._apply_fixes_to_page(parsed, [fix])
    assert changed is True
    assert "[[concepts/new]]" in parsed.body
    assert "old-target" not in parsed.body


def test_apply_fixes_frontmatter_replacement() -> None:
    parsed = page_writer.ParsedPage(
        frontmatter={"sources": ["sources/old.md"]}, body="b"
    )
    fix = lint.LintIssue(
        check=lint.CheckId.MALFORMED_WIKILINK,
        severity=lint.Severity.WARNING,
        page="concepts/a.md",
        message="",
        fixable=True,
        context={
            "old_target": "sources/old.md",
            "new_target": "sources/old",
            "location": "frontmatter",
            "field": "sources",
        },
    )
    changed = lint._apply_fixes_to_page(parsed, [fix])
    assert changed is True
    assert parsed.frontmatter["sources"] == ["sources/old"]


def test_apply_fixes_noop_when_not_fixable() -> None:
    parsed = page_writer.ParsedPage(frontmatter={}, body="unchanged")
    fix = lint.LintIssue(
        check=lint.CheckId.BROKEN_WIKILINK,
        severity=lint.Severity.ERROR,
        page="concepts/a.md",
        message="",
        fixable=False,
    )
    assert lint._apply_fixes_to_page(parsed, [fix]) is False
    assert parsed.body == "unchanged"


# ---------------------------------------------------------------------------
# LintReport.health_score
# ---------------------------------------------------------------------------


def test_health_score_no_pages_is_100() -> None:
    assert lint.LintReport(pages_checked=0).health_score == 100


def test_health_score_clean_is_100() -> None:
    assert lint.LintReport(pages_checked=5).health_score == 100


def test_health_score_drops_with_errors() -> None:
    report = lint.LintReport(
        pages_checked=5,
        issues=[
            lint.LintIssue(
                check=lint.CheckId.BROKEN_WIKILINK,
                severity=lint.Severity.ERROR,
                page="concepts/a.md",
                message="",
            )
        ],
    )
    assert report.health_score < 100


def test_health_score_errors_weigh_more_than_infos() -> None:
    def _report(sev: lint.Severity) -> lint.LintReport:
        return lint.LintReport(
            pages_checked=5,
            issues=[
                lint.LintIssue(check=lint.CheckId.BROKEN_WIKILINK, severity=sev,
                               page="p", message="")
            ],
        )

    error_score = _report(lint.Severity.ERROR).health_score
    info_score = _report(lint.Severity.INFO).health_score
    assert error_score < info_score
