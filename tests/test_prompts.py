"""Unit tests for prompts.py — deterministic ChatMessage builders. No LLM."""

from __future__ import annotations

from llm_wiki import prompts
from llm_wiki.llm import ChatMessage


def _roles(messages: list[ChatMessage]) -> list[str]:
    return [m.role for m in messages]


# ---------------------------------------------------------------------------
# build_extraction_messages
# ---------------------------------------------------------------------------


def test_build_extraction_messages_structure() -> None:
    msgs = prompts.build_extraction_messages("My Title", "The body text.")
    assert _roles(msgs) == ["system", "user"]
    assert msgs[0].content == prompts.SYSTEM_PROMPT
    assert "My Title" in msgs[1].content
    assert "The body text." in msgs[1].content
    assert prompts.EXTRACTION_INSTRUCTIONS in msgs[1].content


def test_build_extraction_retry_includes_bad_response() -> None:
    msgs = prompts.build_extraction_retry_messages("T", "body", "BROKEN OUTPUT")
    assert _roles(msgs) == ["system", "user", "assistant", "user"]
    assert msgs[2].content == "BROKEN OUTPUT"
    assert "not valid JSON" in msgs[1].content


def test_build_extraction_retry_truncates_bad_response() -> None:
    msgs = prompts.build_extraction_retry_messages("T", "body", "x" * 5000)
    assert len(msgs[2].content) == 2000    # bad_response[:2000]


# ---------------------------------------------------------------------------
# build_draft_page_messages
# ---------------------------------------------------------------------------


def test_build_draft_entity_page_uses_entity_template() -> None:
    msgs = prompts.build_draft_page_messages(
        kind="entity", name="OpenAI", source_title="A Paper", source_slug="a-paper",
        description="An AI lab.", excerpts="Some excerpt.",
        related=["concepts/rag"], today="2026-07-15",
    )
    assert _roles(msgs) == ["system", "user"]
    user = msgs[1].content
    assert "wiki entity page for 'OpenAI'" in user
    assert "type: entity" in user
    assert "[[concepts/rag]]" in user             # related link rendered
    assert "sources/a-paper" in user
    # Context block (excerpts) leads the user message for prefix caching.
    assert user.index("Some excerpt.") < user.index("Draft a wiki entity page")


def test_build_draft_concept_page_uses_concept_template() -> None:
    msgs = prompts.build_draft_page_messages(
        kind="concept", name="RAG", source_title="A Paper", source_slug="a-paper",
        description="A technique.", excerpts="Excerpt.",
        related=[], today="2026-07-15",
    )
    user = msgs[1].content
    assert "wiki concept page for 'RAG'" in user
    assert "type: concept" in user
    assert "(none)" in user                        # empty related list


# ---------------------------------------------------------------------------
# build_merge_page_messages
# ---------------------------------------------------------------------------


def test_build_merge_page_messages_embeds_existing_content() -> None:
    msgs = prompts.build_merge_page_messages(
        name="OpenAI", existing_content="# OpenAI\n\nOld body.",
        source_title="New Paper", source_slug="new-paper",
        description="More info.", excerpts="Excerpt.", today="2026-07-15",
    )
    assert _roles(msgs) == ["system", "user"]
    user = msgs[1].content
    assert "Old body." in user
    assert "sources/new-paper" in user


# ---------------------------------------------------------------------------
# build_source_page_messages
# ---------------------------------------------------------------------------


def test_build_source_page_messages_renders_lists() -> None:
    msgs = prompts.build_source_page_messages(
        source_title="A Paper", source_slug="a-paper",
        file_path="/abs/a.pdf", file_type="pdf",
        summary="It solves X.", key_takeaways=["First point", "Second point"],
        tags=["ml", "nlp"], entity_slugs=["openai"], concept_slugs=["rag"],
        today="2026-07-15",
    )
    user = msgs[1].content
    assert "- First point" in user
    assert "- Second point" in user
    assert "[[entities/openai]]" in user
    assert "[[concepts/rag]]" in user
    assert "ml, nlp" in user


def test_build_source_page_messages_empty_related_lists() -> None:
    msgs = prompts.build_source_page_messages(
        source_title="A", source_slug="a", file_path="a.txt", file_type="txt",
        summary="s", key_takeaways=[], tags=[], entity_slugs=[], concept_slugs=[],
        today="2026-07-15",
    )
    assert "(none)" in msgs[1].content


# ---------------------------------------------------------------------------
# build_merge_source_page_messages — same-file vs new-file branch
# ---------------------------------------------------------------------------


def test_build_merge_source_page_same_file_branch() -> None:
    msgs = prompts.build_merge_source_page_messages(
        existing_content="# A\n\nOld.", file_path="/abs/a.pdf", file_type="pdf",
        summary="s", key_takeaways=["k"], entity_slugs=["openai"], concept_slugs=[],
        today="2026-07-15", is_same_file=True,
    )
    user = msgs[1].content
    assert "REFRESHED SOURCE FILE" in user
    assert "re-ingest of the same file" in user
    assert "[[entities/openai]]" in user


def test_build_merge_source_page_new_file_branch() -> None:
    msgs = prompts.build_merge_source_page_messages(
        existing_content="# A\n\nOld.", file_path="/abs/a-v2.pdf", file_type="pdf",
        summary="s", key_takeaways=["k"], entity_slugs=[], concept_slugs=["rag"],
        today="2026-07-15", is_same_file=False,
    )
    user = msgs[1].content
    assert "NEW SOURCE FILE" in user
    assert "another file recognized as the same document" in user
    assert "[[concepts/rag]]" in user
