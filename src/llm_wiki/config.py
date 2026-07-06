"""Path constants and config loading for an LLM-Wiki project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Folder layout (relative to a wiki project root)
# ---------------------------------------------------------------------------

RAW_DIR = "raw"
RAW_TEXT_DIR = "raw-text"
WIKI_DIR = "wiki"
SCHEMA_DIR = "schema"
INTERNAL_DIR = ".wiki"

WIKI_SUBDIRS = ("sources", "entities", "concepts", "synthesis")

INDEX_FILE = "index.md"
LOG_FILE = "log.md"
AGENTS_FILE = "AGENTS.md"
CONFIG_FILE = "config.yml"
STATE_DB = "state.sqlite"

OBSIDIAN_DIR = ".obsidian"


@dataclass
class WikiPaths:
    """Resolved absolute paths for a wiki project."""

    root: Path

    @property
    def raw(self) -> Path:
        return self.root / RAW_DIR

    @property
    def raw_text(self) -> Path:
        """Plain-text mirrors of raw/ sources — this is what search indexes.

        raw/ holds original binaries (PDF, DOCX, …) which the QMD search
        backend cannot parse; raw-text/ holds the parser-extracted markdown
        for each source so full-document search works on real text.
        """
        return self.root / RAW_TEXT_DIR

    @property
    def wiki(self) -> Path:
        return self.root / WIKI_DIR

    @property
    def schema(self) -> Path:
        return self.root / SCHEMA_DIR

    @property
    def internal(self) -> Path:
        return self.root / INTERNAL_DIR

    @property
    def index(self) -> Path:
        return self.wiki / INDEX_FILE

    @property
    def log(self) -> Path:
        return self.wiki / LOG_FILE

    @property
    def agents(self) -> Path:
        return self.schema / AGENTS_FILE

    @property
    def config_file(self) -> Path:
        return self.internal / CONFIG_FILE

    @property
    def state_db(self) -> Path:
        return self.internal / STATE_DB

    @property
    def logs(self) -> Path:
        return self.internal / "logs"

    @property
    def obsidian(self) -> Path:
        return self.wiki / OBSIDIAN_DIR

    def is_initialized(self) -> bool:
        """A folder counts as initialized if its config file exists."""
        return self.config_file.exists()


# ---------------------------------------------------------------------------
# Default config (written on `wiki init`)
# ---------------------------------------------------------------------------

# The pipeline tasks that request an LLM completion. Each is routed to a
# provider via `llm.task_providers` and resolves its sampling settings from
# that provider's profile.
# Ordered by owning command (ingest → query → lint) and pipeline stage, so the
# settings UI and generated config.yml group tasks the way the user thinks about
# them. Names follow a `command_stage` convention for the same reason.
LLM_TASKS = (
    "ingest_extract",
    "ingest_extract_retry",
    "ingest_draft",
    "query_intent",
    "query_chitchat",
    "query_synthesis",
    "lint_deep",
)

# Keys on a provider profile that are structural, not sampling options. Anything
# else at a profile's top level (and under its `tasks.<task>`) is passed through
# to the client as a sampling parameter in that provider's own vocabulary.
PROVIDER_RESERVED_KEYS = frozenset(
    {"type", "model", "host", "api_base", "api_key_env", "isolated", "command",
     "thinking", "tasks", "timeout"}
)

# Friendly provider "type" presets for the settings UI. Each maps a
# protocol/vendor the user picks to how a profile is built: the internal client
# `type`, the LiteLLM model-string prefix, and which endpoint field (if any) is
# relevant. This lets users add any OpenAI-compatible or vendor provider by name
# without knowing LiteLLM's prefix conventions. `endpoint_field` is None for
# vendors reached at a fixed cloud URL. Order defines the dropdown order.
PROVIDER_TYPE_PRESETS: dict = {
    "ollama-native": {
        "label": "Ollama (native)",
        "type": "ollama-native",
        "model_prefix": "",
        "endpoint_field": "host",
        "endpoint_label": "Host",
        "endpoint_placeholder": "http://localhost:11434",
    },
    "ollama": {
        "label": "Ollama (via LiteLLM)",
        "type": "litellm",
        "model_prefix": "ollama/",
        "endpoint_field": "api_base",
        "endpoint_label": "API base",
        "endpoint_placeholder": "http://localhost:11434",
    },
    "openai-compatible": {
        "label": "OpenAI-compatible (vLLM, LocalAI, …)",
        "type": "litellm",
        "model_prefix": "hosted_vllm/",
        "endpoint_field": "api_base",
        "endpoint_label": "API base",
        "endpoint_placeholder": "http://localhost:8000/v1",
    },
    "lm-studio": {
        "label": "LM Studio",
        "type": "litellm",
        "model_prefix": "lm_studio/",
        "endpoint_field": "api_base",
        "endpoint_label": "API base",
        "endpoint_placeholder": "http://localhost:1234/v1",
    },
    "openai": {
        "label": "OpenAI",
        "type": "litellm",
        "model_prefix": "openai/",
        "endpoint_field": None,
        "endpoint_label": "",
        "endpoint_placeholder": "",
    },
    "anthropic": {
        "label": "Anthropic",
        "type": "litellm",
        "model_prefix": "anthropic/",
        "endpoint_field": None,
        "endpoint_label": "",
        "endpoint_placeholder": "",
    },
    "gemini": {
        "label": "Google Gemini",
        "type": "litellm",
        "model_prefix": "gemini/",
        "endpoint_field": None,
        "endpoint_label": "",
        "endpoint_placeholder": "",
    },
    "claude-code": {
        "label": "Claude Code (no API key)",
        "type": "claude-code",
        "model_prefix": "",  # model passes straight to `claude --model`
        "endpoint_field": None,
        "endpoint_label": "",
        "endpoint_placeholder": "",
    },
    "codex": {
        "label": "Codex (no API key)",
        "type": "codex",
        "model_prefix": "",  # model passes straight to `codex exec -m`
        "endpoint_field": None,
        "endpoint_label": "",
        "endpoint_placeholder": "",
    },
    "custom": {
        "label": "Custom (full LiteLLM model string)",
        "type": "litellm",
        "model_prefix": "",  # user supplies the prefix themselves
        "endpoint_field": "api_base",
        "endpoint_label": "API base (optional)",
        "endpoint_placeholder": "",
    },
}

# Provider types that run a local agent CLI (no API key, no HTTP endpoint).
CLI_PROVIDER_TYPES = frozenset({"claude-code", "codex"})


def build_provider_profile(
    preset_id: str, model: str, endpoint: str = "", api_key_env: str = ""
) -> dict:
    """Construct a provider profile dict from a UI preset + user inputs.

    Applies the preset's LiteLLM model-string prefix (unless the user already
    typed it) and stores the endpoint under the right field. An optional
    `api_key_env` names the environment variable to read the API key from (for
    OpenAI-compatible / secured endpoints that need their own key). Raises
    ValueError on an unknown preset or empty model.
    """
    preset = PROVIDER_TYPE_PRESETS.get(preset_id)
    if preset is None:
        raise ValueError(f"Unknown provider type '{preset_id}'.")
    ptype = preset["type"]
    model = (model or "").strip()
    # Agent-CLI providers may leave the model blank (use the CLI's default).
    if not model and ptype not in CLI_PROVIDER_TYPES:
        raise ValueError("Model is required.")
    prefix = preset["model_prefix"]
    if prefix and not model.startswith(prefix):
        model = prefix + model
    profile: dict = {"type": ptype, "model": model, "tasks": {}}
    field = preset["endpoint_field"]
    if field and endpoint.strip():
        profile[field] = endpoint.strip()
    if api_key_env.strip():
        profile["api_key_env"] = api_key_env.strip()
    if ptype in CLI_PROVIDER_TYPES:
        # Default to an isolated run (no personal MCP/CLAUDE.md/slash commands).
        profile["isolated"] = True
    return profile

DEFAULT_CONFIG: dict = {
    "version": 1,
    "llm": {
        # Fallback provider for tasks missing from task_providers and for
        # ad-hoc calls that don't name a task.
        "default_provider": "ollama",
        # Which provider profile handles each pipeline task.
        "task_providers": {task: "ollama" for task in LLM_TASKS},
        # Named, self-contained provider profiles. Each carries its model,
        # connection info, sampling defaults, and per-task overrides in that
        # provider's own parameter vocabulary.
        "providers": {
            "ollama": {
                # Existing native httpx path (Ollama /api/chat).
                "type": "ollama-native",
                "model": "qwen3.6:35b",
                "host": "http://localhost:11434",
                # Qwen3 thinking mode — useful for synthesis/lint, slower for
                # routine ops.
                "thinking": True,
                # Base sampling defaults (Ollama option names), used unless a
                # task below overrides them.
                "temperature": 0.6,
                "top_k": 400,
                "top_p": 0.99,
                "min_p": 0.0,
                "num_ctx": 131072,
                "num_predict": 8192,
                # Sparse per-task overrides, merged over the base settings above.
                "tasks": {
                    "ingest_extract": {"num_predict": -1},
                    "ingest_extract_retry": {"temperature": 0.6, "num_predict": -1},
                    "query_intent": {"temperature": 0.6},
                    "query_chitchat": {"temperature": 1.0},
                    "lint_deep": {"temperature": 0.6},
                },
            },
            "ollama-litellm": {
                # Same Ollama server, reached through LiteLLM's OpenAI-compatible
                # path — kept alongside the native profile for A/B comparison.
                "type": "litellm",
                "model": "ollama/qwen3.6:35b",
                "api_base": "http://localhost:11434",
                "thinking": True,
                "temperature": 0.6,
                "top_k": 400,
                "top_p": 0.99,
                "min_p": 0.0,
                "num_ctx": 131072,
                "num_predict": 8192,
                "tasks": {
                    "ingest_extract": {"num_predict": -1},
                    "ingest_extract_retry": {"temperature": 0.6, "num_predict": -1},
                    "query_intent": {"temperature": 0.6},
                    "query_chitchat": {"temperature": 1.0},
                    "lint_deep": {"temperature": 0.6},
                },
            },
            # Example cloud profiles — not wired/tested yet, but selectable.
            # API keys are resolved per-vendor from the environment/keyring
            # (see credentials.py), never stored here.
            "anthropic": {
                "type": "litellm",
                "model": "anthropic/claude-opus-4-8",
                # OpenAI/Anthropic vocabulary (max_tokens, not num_predict).
                "temperature": 1.0,
                "max_tokens": 8192,
                "tasks": {},
            },
            "openai": {
                "type": "litellm",
                "model": "openai/gpt-4o",
                "temperature": 1.0,
                "max_tokens": 8192,
                "tasks": {},
            },
        },
    },
    "search": {
        "backend": "qmd",
        "rerank": True,
    },
    "ingest": {
        "interactive": True,
        "auto_update_index": True,
        "auto_update_log": True,
        # Sources longer than this are truncated before being sent to the LLM
        "max_source_chars": 100_000,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` onto `base`, without mutating either."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _normalize_llm_config(loaded: dict) -> dict:
    """Upgrade a legacy flat `llm` block to the provider-profile schema.

    Pre-multi-provider configs stored model/host/sampling/tasks directly under
    `llm` with a top-level `provider` string. Detect that shape (no `providers`
    key) and wrap it into a single `ollama` profile so existing `.wiki/config.yml`
    files keep working without a manual edit. Returns the loaded dict unchanged
    if it's already in the new shape (or has no `llm` block).
    """
    llm = loaded.get("llm")
    if not isinstance(llm, dict) or "providers" in llm:
        return loaded

    llm = dict(llm)
    legacy_provider = llm.pop("provider", "ollama") or "ollama"
    tasks = llm.pop("tasks", {})
    # Everything left on the flat block (model, host, thinking, sampling) becomes
    # the profile body.
    profile = {"type": "ollama-native", **llm, "tasks": tasks}

    loaded = dict(loaded)
    loaded["llm"] = {
        "default_provider": legacy_provider,
        "task_providers": {task: legacy_provider for task in LLM_TASKS},
        "providers": {legacy_provider: profile},
    }
    return loaded


def load_config(paths: WikiPaths) -> dict:
    """Load the wiki's config.yml, falling back to defaults for missing keys.

    Nested dicts (e.g. `llm`, `llm.providers`) are merged key-by-key so a partial
    override in config.yml doesn't drop unrelated defaults. Legacy flat `llm`
    blocks are upgraded to the provider-profile schema before merging.
    """
    if not paths.config_file.exists():
        return dict(DEFAULT_CONFIG)
    with paths.config_file.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    loaded = _normalize_llm_config(loaded)
    return _deep_merge(DEFAULT_CONFIG, loaded)


def save_config(paths: WikiPaths, config: dict) -> None:
    """Write config to disk, creating the internal directory if needed."""
    paths.internal.mkdir(parents=True, exist_ok=True)
    with paths.config_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)


def find_wiki_root(start: Path | None = None) -> Path | None:
    """Walk upward from `start` looking for a `.wiki/config.yml`. Returns the
    project root if found, else None.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / INTERNAL_DIR / CONFIG_FILE).exists():
            return candidate
    return None
