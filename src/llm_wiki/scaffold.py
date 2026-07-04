"""Scaffolding logic for `wiki init` — creates the folder structure, copies
templates, configures Obsidian, and initializes the state DB.
"""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path

from . import config as cfg
from . import db


class ScaffoldError(Exception):
    """Raised when scaffolding cannot proceed (e.g. existing wiki, no perms)."""


def _read_template(name: str) -> str:
    """Read a template file shipped inside the package."""
    files = resources.files("llm_wiki.templates")
    return (files / name).read_text(encoding="utf-8")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def scaffold(root: Path, force: bool = False) -> cfg.WikiPaths:
    """Create a fresh LLM-Wiki project at `root`.

    Layout produced:
        root/
            raw/                       (empty, gitkeep)
            wiki/
                .obsidian/
                    app.json
                    graph.json
                index.md
                log.md
                sources/, entities/, concepts/, synthesis/   (gitkeep each)
            schema/
                AGENTS.md
            .wiki/
                config.yml
                state.sqlite

    Raises ScaffoldError if `wiki/` already exists and `force` is False.
    """
    root = root.resolve()
    paths = cfg.WikiPaths(root=root)

    # Refuse to clobber an existing wiki unless explicitly forced
    if paths.wiki.exists() and any(paths.wiki.iterdir()) and not force:
        raise ScaffoldError(
            f"Wiki folder already exists and is not empty: {paths.wiki}\n"
            f"Use --force to scaffold anyway (may overwrite files)."
        )

    if paths.is_initialized() and not force:
        raise ScaffoldError(
            f"This folder is already an LLM-Wiki project (found "
            f"{paths.config_file.relative_to(root)}).\n"
            f"Use --force to re-scaffold."
        )

    # 1. Top-level folders
    root.mkdir(parents=True, exist_ok=True)
    paths.raw.mkdir(parents=True, exist_ok=True)
    paths.raw_text.mkdir(parents=True, exist_ok=True)
    paths.wiki.mkdir(parents=True, exist_ok=True)
    paths.schema.mkdir(parents=True, exist_ok=True)
    paths.internal.mkdir(parents=True, exist_ok=True)

    # 2. Empty subdirectories with .gitkeep so git tracks them
    (paths.raw / ".gitkeep").touch()
    (paths.raw_text / ".gitkeep").touch()
    for sub in cfg.WIKI_SUBDIRS:
        d = paths.wiki / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / ".gitkeep").touch()

    # 3. Wiki seed files (index.md, log.md)
    _write(paths.index, _read_template("index.md"))
    _write(paths.log, _read_template("log.md"))

    # 4. Schema (AGENTS.md)
    _write(paths.agents, _read_template("AGENTS.md"))

    # 5. Obsidian config
    paths.obsidian.mkdir(parents=True, exist_ok=True)
    _write(paths.obsidian / "app.json", _read_template("obsidian_app.json"))
    _write(paths.obsidian / "graph.json", _read_template("obsidian_graph.json"))

    # 6. Internal config and state DB
    cfg.save_config(paths, dict(cfg.DEFAULT_CONFIG))
    db.init_db(paths.state_db)

    return paths
