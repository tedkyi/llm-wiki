"""Path constants and config loading for an LLM-Wiki project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Folder layout (relative to a wiki project root)
# ---------------------------------------------------------------------------

RAW_DIR = "raw"
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

DEFAULT_CONFIG: dict = {
    "version": 1,
    "llm": {
        "provider": "ollama",
        "model": "qwen3:14b",
        "host": "http://localhost:11434",
        # Qwen3 thinking mode — useful for synthesis/lint, slower for routine ops
        "thinking": True,
        # Base sampling defaults, used unless a task below overrides them
        "temperature": 0.3,
        "top_k": 40,
        "top_p": 0.9,
        "min_p": 0.0,
        "num_ctx": 131072,
        "num_predict": 8192,
        # Sparse per-task overrides, merged over the base settings above
        "tasks": {
            "extraction": {"num_predict": -1},
            "extraction_retry": {"temperature": 0.2, "num_predict": -1},
            "draft": {},
            "synthesis": {},
            "intent_classify": {"temperature": 0.0},
            "chitchat": {"temperature": 0.7},
            "contradiction": {"temperature": 0.2},
        },
    },
    "search": {
        "backend": "qmd",
        "rerank": True,
        "gpu": "auto",  # "auto" | "cuda" | "vulkan" | "metal" | "false"
    },
    "ingest": {
        "interactive": True,
        "auto_update_index": True,
        "auto_update_log": True,
        # Sources longer than this are truncated before being sent to the LLM
        "max_source_chars": 60_000,
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


def load_config(paths: WikiPaths) -> dict:
    """Load the wiki's config.yml, falling back to defaults for missing keys.

    Nested dicts (e.g. `llm`, `llm.tasks`) are merged key-by-key so a partial
    override in config.yml doesn't drop unrelated defaults.
    """
    if not paths.config_file.exists():
        return dict(DEFAULT_CONFIG)
    with paths.config_file.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
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
