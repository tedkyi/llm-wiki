"""Unit tests for the pure helpers in llm.py — option resolution, think-tag
handling, prompt flattening, client dispatch, and LLMRouter routing. None of
these touch Ollama, LiteLLM, or a CLI subprocess.
"""

from __future__ import annotations

import json

import pytest

from llm_wiki import llm


# ---------------------------------------------------------------------------
# resolve_llm_options — precedence & reserved-key exclusion
# ---------------------------------------------------------------------------


def test_resolve_options_excludes_reserved_keys() -> None:
    profile = {"type": "ollama-native", "model": "m", "host": "h", "temperature": 0.6}
    opts = llm.resolve_llm_options(profile)
    assert opts == {"temperature": 0.6}
    assert "type" not in opts and "model" not in opts and "host" not in opts


def test_resolve_options_task_overrides_base() -> None:
    profile = {
        "temperature": 0.6,
        "num_predict": 8192,
        "tasks": {"ingest_extract": {"num_predict": -1}},
    }
    opts = llm.resolve_llm_options(profile, task="ingest_extract")
    assert opts["temperature"] == 0.6      # base retained
    assert opts["num_predict"] == -1       # task override wins


def test_resolve_options_explicit_override_wins_and_skips_none() -> None:
    profile = {"temperature": 0.6, "tasks": {"t": {"temperature": 0.4}}}
    opts = llm.resolve_llm_options(profile, task="t", temperature=1.0, top_p=None)
    assert opts["temperature"] == 1.0      # explicit override beats task
    assert "top_p" not in opts             # None overrides are dropped


def test_resolve_options_unknown_task_uses_base_only() -> None:
    profile = {"temperature": 0.6, "tasks": {"other": {"temperature": 0.1}}}
    opts = llm.resolve_llm_options(profile, task="missing")
    assert opts == {"temperature": 0.6}


# ---------------------------------------------------------------------------
# _apply_qwen_think_tag
# ---------------------------------------------------------------------------


def test_apply_qwen_think_tag_appends_think() -> None:
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    out = llm._apply_qwen_think_tag(msgs, thinking=True)
    assert out[-1]["content"] == "hi\n\n/think"


def test_apply_qwen_think_tag_appends_no_think_to_last_user() -> None:
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    out = llm._apply_qwen_think_tag(msgs, thinking=False)
    assert out[2]["content"] == "second\n\n/no_think"
    assert out[0]["content"] == "first"   # only the last user turn is tagged


# ---------------------------------------------------------------------------
# _strip_thinking_text
# ---------------------------------------------------------------------------


def test_strip_thinking_text_removes_block() -> None:
    assert llm._strip_thinking_text("<think>reasoning</think>answer") == "answer"


def test_strip_thinking_text_no_tag_passthrough() -> None:
    assert llm._strip_thinking_text("plain answer") == "plain answer"


def test_strip_thinking_text_multiline() -> None:
    text = "<think>\nline1\nline2\n</think>\nfinal"
    assert llm._strip_thinking_text(text) == "final"


# ---------------------------------------------------------------------------
# _ThinkStripper — stateful streaming
# ---------------------------------------------------------------------------


def test_think_stripper_single_chunk() -> None:
    s = llm._ThinkStripper()
    assert s.feed("<think>hidden</think>visible") == "visible"


def test_think_stripper_span_across_chunks() -> None:
    s = llm._ThinkStripper()
    # The <think> span is split across three feed() calls.
    out = ""
    out += s.feed("before <think>hid")
    out += s.feed("den still hidden")
    out += s.feed("</think>after")
    assert out == "before after"


def test_think_stripper_visible_before_and_after() -> None:
    s = llm._ThinkStripper()
    assert s.feed("A<think>x</think>B<think>y</think>C") == "ABC"


def test_think_stripper_no_tags_passthrough() -> None:
    s = llm._ThinkStripper()
    assert s.feed("just text") == "just text"


# ---------------------------------------------------------------------------
# _messages_to_prompt / _stdin_events
# ---------------------------------------------------------------------------


def test_messages_to_prompt_flattens_roles() -> None:
    msgs = [
        llm.ChatMessage(role="system", content="SYS"),
        llm.ChatMessage(role="user", content="Q"),
        llm.ChatMessage(role="assistant", content="A"),
    ]
    prompt = llm._messages_to_prompt(msgs)
    assert "SYS" in prompt and "Q" in prompt
    assert "[Your previous reply]\nA" in prompt


def test_stdin_events_folds_system_into_first_user() -> None:
    msgs = [
        llm.ChatMessage(role="system", content="SYS"),
        llm.ChatMessage(role="user", content="hello"),
    ]
    lines = [json.loads(ln) for ln in llm.ClaudeCliClient._stdin_events(msgs).splitlines()]
    assert len(lines) == 1
    text = lines[0]["message"]["content"][0]["text"]
    assert text == "SYS\n\nhello"
    assert lines[0]["type"] == "user"


def test_stdin_events_system_only_becomes_user_turn() -> None:
    msgs = [llm.ChatMessage(role="system", content="just system")]
    lines = [json.loads(ln) for ln in llm.ClaudeCliClient._stdin_events(msgs).splitlines()]
    assert len(lines) == 1
    assert lines[0]["type"] == "user"
    assert lines[0]["message"]["content"][0]["text"] == "just system"


def test_stdin_events_preamble_only_on_first_user() -> None:
    msgs = [
        llm.ChatMessage(role="system", content="SYS"),
        llm.ChatMessage(role="user", content="one"),
        llm.ChatMessage(role="assistant", content="reply"),
        llm.ChatMessage(role="user", content="two"),
    ]
    lines = [json.loads(ln) for ln in llm.ClaudeCliClient._stdin_events(msgs).splitlines()]
    first_user = lines[0]["message"]["content"][0]["text"]
    last_user = lines[2]["message"]["content"][0]["text"]
    assert first_user.startswith("SYS")
    assert last_user == "two"           # preamble not repeated


# ---------------------------------------------------------------------------
# make_client — dispatch on profile type (no network in constructors)
# ---------------------------------------------------------------------------


def test_make_client_ollama_native() -> None:
    client = llm.make_client({"type": "ollama-native", "model": "m", "host": "http://h"})
    assert isinstance(client, llm.OllamaClient)
    client.close()


def test_make_client_litellm() -> None:
    client = llm.make_client({"type": "litellm", "model": "openai/gpt-4o"})
    assert isinstance(client, llm.LiteLLMClient)


def test_make_client_defaults_to_ollama_native() -> None:
    client = llm.make_client({"model": "m"})
    assert isinstance(client, llm.OllamaClient)
    client.close()


def test_make_client_unknown_type_raises() -> None:
    with pytest.raises(llm.LLMError, match="Unknown provider type"):
        llm.make_client({"type": "bogus", "model": "m"})


# ---------------------------------------------------------------------------
# LLMRouter — routing/resolution (no clients built unless requested)
# ---------------------------------------------------------------------------


def _router_cfg() -> dict:
    return {
        "default_provider": "ollama",
        "task_providers": {"ingest_extract": "cloud", "query_intent": "ollama"},
        "providers": {
            "ollama": {
                "type": "ollama-native", "model": "qwen3", "thinking": True,
                "temperature": 0.6, "tasks": {"query_intent": {"temperature": 0.2}},
            },
            "cloud": {"type": "litellm", "model": "openai/gpt-4o", "temperature": 1.0},
        },
    }


def test_router_provider_for_task_and_default() -> None:
    r = llm.LLMRouter(_router_cfg())
    assert r.provider_for("ingest_extract") == "cloud"
    assert r.provider_for("query_intent") == "ollama"
    assert r.provider_for("unmapped_task") == "ollama"   # falls back to default
    assert r.provider_for(None) == "ollama"


def test_router_options_for_applies_task_override() -> None:
    r = llm.LLMRouter(_router_cfg())
    opts = r.options_for("query_intent")
    assert opts["temperature"] == 0.2       # task override on the ollama profile


def test_router_model_and_thinking_default() -> None:
    r = llm.LLMRouter(_router_cfg())
    assert r.model_for("ingest_extract") == "openai/gpt-4o"
    assert r.thinking_default("query_intent") is True
    assert r.thinking_default("ingest_extract") is False   # cloud profile has no thinking


def test_router_override_forces_single_provider() -> None:
    r = llm.LLMRouter(_router_cfg(), override="cloud")
    assert r.provider_for("query_intent") == "cloud"
    assert r.provider_for("ingest_extract") == "cloud"


def test_router_unknown_override_raises() -> None:
    with pytest.raises(llm.LLMError, match="Unknown provider"):
        llm.LLMRouter(_router_cfg(), override="nope")


def test_router_unknown_provider_profile_raises() -> None:
    cfg = {
        "default_provider": "ghost",
        "task_providers": {},
        "providers": {},
    }
    r = llm.LLMRouter(cfg)
    with pytest.raises(llm.LLMError, match="not configured"):
        r.options_for("query_intent")
