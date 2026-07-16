"""Unit tests for config — paths, provider-profile building, config load/save.
No LLM/backends touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_wiki import config as cfg


# ---------------------------------------------------------------------------
# WikiPaths
# ---------------------------------------------------------------------------


def test_wikipaths_properties_resolve_under_root(tmp_path: Path) -> None:
    p = cfg.WikiPaths(root=tmp_path)
    assert p.raw == tmp_path / "raw"
    assert p.raw_text == tmp_path / "raw-text"
    assert p.wiki == tmp_path / "wiki"
    assert p.internal == tmp_path / ".wiki"
    assert p.index == tmp_path / "wiki" / "index.md"
    assert p.config_file == tmp_path / ".wiki" / "config.yml"
    assert p.state_db == tmp_path / ".wiki" / "state.sqlite"


def test_is_initialized_false_before_config(tmp_path: Path) -> None:
    assert cfg.WikiPaths(root=tmp_path).is_initialized() is False


def test_is_initialized_true_after_config(paths: cfg.WikiPaths) -> None:
    # The `wiki_root` fixture writes .wiki/config.yml.
    assert paths.is_initialized() is True


# ---------------------------------------------------------------------------
# build_provider_profile
# ---------------------------------------------------------------------------


def test_build_provider_profile_applies_prefix() -> None:
    profile = cfg.build_provider_profile("openai", "gpt-4o")
    assert profile["type"] == "litellm"
    assert profile["model"] == "openai/gpt-4o"
    assert profile["tasks"] == {}


def test_build_provider_profile_does_not_double_prefix() -> None:
    profile = cfg.build_provider_profile("openai", "openai/gpt-4o")
    assert profile["model"] == "openai/gpt-4o"


def test_build_provider_profile_stores_endpoint() -> None:
    profile = cfg.build_provider_profile(
        "openai-compatible", "my-model", endpoint="http://localhost:8000/v1"
    )
    assert profile["api_base"] == "http://localhost:8000/v1"
    assert profile["model"] == "hosted_vllm/my-model"


def test_build_provider_profile_native_ollama_uses_host_field() -> None:
    profile = cfg.build_provider_profile(
        "ollama-native", "qwen3", endpoint="http://localhost:11434"
    )
    assert profile["type"] == "ollama-native"
    assert profile["host"] == "http://localhost:11434"


def test_build_provider_profile_stores_api_key_env() -> None:
    profile = cfg.build_provider_profile("openai", "gpt-4o", api_key_env="MY_KEY")
    assert profile["api_key_env"] == "MY_KEY"


def test_build_provider_profile_cli_provider_allows_blank_model_and_isolates() -> None:
    profile = cfg.build_provider_profile("claude-code", "")
    assert profile["type"] == "claude-code"
    assert profile["isolated"] is True


def test_build_provider_profile_unknown_preset_raises() -> None:
    with pytest.raises(ValueError, match="Unknown provider type"):
        cfg.build_provider_profile("nonexistent", "model")


def test_build_provider_profile_missing_model_raises_for_non_cli() -> None:
    with pytest.raises(ValueError, match="Model is required"):
        cfg.build_provider_profile("openai", "")


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


def test_deep_merge_nested_without_clobbering_siblings() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 20, "z": 30}}
    result = cfg._deep_merge(base, override)
    assert result == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"x": 1}}
    override = {"a": {"y": 2}}
    cfg._deep_merge(base, override)
    assert base == {"a": {"x": 1}}
    assert override == {"a": {"y": 2}}


def test_deep_merge_scalar_override_replaces_dict() -> None:
    result = cfg._deep_merge({"a": {"x": 1}}, {"a": 5})
    assert result == {"a": 5}


# ---------------------------------------------------------------------------
# _normalize_llm_config
# ---------------------------------------------------------------------------


def test_normalize_llm_config_upgrades_legacy_flat_block() -> None:
    legacy = {
        "llm": {
            "provider": "ollama",
            "model": "qwen3",
            "host": "http://localhost:11434",
            "thinking": True,
            "tasks": {"query_intent": {"temperature": 0.5}},
        }
    }
    out = cfg._normalize_llm_config(legacy)
    llm = out["llm"]
    assert "providers" in llm
    assert llm["default_provider"] == "ollama"
    profile = llm["providers"]["ollama"]
    assert profile["type"] == "ollama-native"
    assert profile["model"] == "qwen3"
    assert profile["tasks"] == {"query_intent": {"temperature": 0.5}}
    # Every task is routed to the upgraded provider.
    assert set(llm["task_providers"].keys()) == set(cfg.LLM_TASKS)


def test_normalize_llm_config_leaves_new_shape_untouched() -> None:
    already = {"llm": {"providers": {"ollama": {}}, "default_provider": "ollama"}}
    assert cfg._normalize_llm_config(already) is already


def test_normalize_llm_config_no_llm_block_untouched() -> None:
    d = {"search": {"backend": "qmd"}}
    assert cfg._normalize_llm_config(d) is d


# ---------------------------------------------------------------------------
# load_config / save_config
# ---------------------------------------------------------------------------


def test_load_config_returns_defaults_when_no_file(tmp_path: Path) -> None:
    p = cfg.WikiPaths(root=tmp_path)
    loaded = cfg.load_config(p)
    assert loaded["version"] == cfg.DEFAULT_CONFIG["version"]
    assert loaded["search"]["backend"] == "qmd"


def test_load_config_merges_partial_override(tmp_path: Path) -> None:
    p = cfg.WikiPaths(root=tmp_path)
    p.internal.mkdir(parents=True)
    # Override only one nested key; unrelated defaults must survive the merge.
    p.config_file.write_text(
        "search:\n  rerank: false\n", encoding="utf-8"
    )
    loaded = cfg.load_config(p)
    assert loaded["search"]["rerank"] is False          # overridden
    assert loaded["search"]["backend"] == "qmd"          # default preserved
    assert loaded["ingest"]["interactive"] is True       # unrelated default preserved


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    p = cfg.WikiPaths(root=tmp_path)
    config = dict(cfg.DEFAULT_CONFIG)
    cfg.save_config(p, config)
    assert p.config_file.exists()
    loaded = cfg.load_config(p)
    assert loaded["version"] == config["version"]
    assert loaded["search"] == config["search"]


# ---------------------------------------------------------------------------
# find_wiki_root
# ---------------------------------------------------------------------------


def test_find_wiki_root_walks_upward(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / cfg.INTERNAL_DIR).mkdir(parents=True)
    (root / cfg.INTERNAL_DIR / cfg.CONFIG_FILE).write_text("version: 1\n", encoding="utf-8")
    nested = root / "a" / "b" / "c"
    nested.mkdir(parents=True)
    assert cfg.find_wiki_root(nested) == root


def test_find_wiki_root_returns_none_when_absent(tmp_path: Path) -> None:
    assert cfg.find_wiki_root(tmp_path) is None
