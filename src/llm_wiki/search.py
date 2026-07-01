"""QMD search backend wrapper.

QMD (https://github.com/tobi/qmd) is a local search engine for markdown
files with hybrid BM25 + vector + LLM reranking. We use it as a subprocess:
  - `qmd collection add <path> --name <name>` — register a directory
  - `qmd update` — reindex everything
  - `qmd embed` — generate vector embeddings
  - `qmd query "<q>" --json -n <N>` — hybrid search with rerank
  - `qmd search "<q>" --json -n <N>` — BM25 only
  - `qmd vsearch "<q>" --json -n <N>` — vector only

QMD keeps its index in a SQLite file. We point it at a per-wiki index stored
inside .wiki/qmd.sqlite so each project has its own search state.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from . import config as cfg

QMD_TIMEOUT_SHORT = 30.0      # for quick commands (status, collection list)
QMD_TIMEOUT_MEDIUM = 600.0    # for query (after models are loaded)
QMD_TIMEOUT_LONG = 1800.0     # for update + embed (30 minutes)

# ---------------------------------------------------------------------------
# Background embed scheduler
# ---------------------------------------------------------------------------
# Only one `qmd embed` subprocess runs at a time (loading a 300M GGUF model
# concurrently would just fight over GPU VRAM / CPU cache). If a second
# ingest finishes while embed is running, we set _embed_pending so the
# worker loops and picks up the new docs immediately after finishing.

_embed_lock = threading.Lock()
_embed_pending: bool = False
_embed_thread: threading.Thread | None = None


class SearchBackendError(Exception):
    """Raised when the QMD backend fails in a user-recoverable way."""


class QmdNotInstalled(SearchBackendError):
    """The `qmd` binary isn't on PATH."""


@dataclass
class SearchHit:
    """A single search result from QMD."""

    docid: str               # e.g. '#a1b2c3'
    path: str                # relative path inside the QMD collection
    collection: str          # collection name
    title: str               # extracted title
    score: float             # final score, 0.0 - 1.0 (approximately)
    snippet: str = ""        # context around match
    context: str = ""        # collection context set by `qmd context add`
    full_content: str = ""   # only populated after we read the file

    @property
    def full_path(self) -> str:
        """Collection-relative path with collection prefix."""
        return f"{self.collection}/{self.path}"


@dataclass
class SearchResults:
    """Grouped results from a single query."""

    query: str
    hits: list[SearchHit] = field(default_factory=list)
    search_mode: str = "query"   # 'query' | 'search' | 'vsearch'

    def __len__(self) -> int:
        return len(self.hits)

    def __iter__(self):
        return iter(self.hits)


# ---------------------------------------------------------------------------
# QMD binary discovery & invocation
# ---------------------------------------------------------------------------


def _find_qmd() -> str:
    """Locate the qmd binary. Raises QmdNotInstalled if not found."""
    qmd_path = shutil.which("qmd")
    if not qmd_path:
        raise QmdNotInstalled(
            "The 'qmd' command was not found on PATH.\n"
            "Install it with: npm install -g @tobilu/qmd"
        )
    return qmd_path


def _qmd_env(paths: cfg.WikiPaths) -> dict[str, str]:
    """Build the env for qmd subprocess calls.

    QMD uses XDG_CACHE_HOME to locate its default index. We override it to
    keep each wiki's index isolated inside .wiki/qmd-cache/ .

    QMD_LLAMA_GPU controls llama.cpp GPU offloading: "auto" (default), "cuda",
    "vulkan", "metal", or "false" for CPU-only. We read it from search.gpu in
    config, but the user can always override via environment variable.
    """
    env = os.environ.copy()
    cache_dir = paths.internal / "qmd-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    env["XDG_CACHE_HOME"] = str(cache_dir)
    if "QMD_LLAMA_GPU" not in env:
        config = cfg.load_config(paths)
        gpu = config.get("search", {}).get("gpu", "auto")
        env["QMD_LLAMA_GPU"] = str(gpu)
    return env


def _run_qmd(
    paths: cfg.WikiPaths,
    args: list[str],
    *,
    timeout: float = QMD_TIMEOUT_SHORT,
    check: bool = True,
    stdin_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the qmd binary with a wiki-scoped env. Returns the CompletedProcess.

    Raises:
        QmdNotInstalled: qmd binary missing.
        SearchBackendError: non-zero exit (if `check`).
    """
    qmd_bin = _find_qmd()
    env = _qmd_env(paths)
    try:
        result = subprocess.run(
            [qmd_bin, *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            input=stdin_text,
        )
    except FileNotFoundError as e:
        raise QmdNotInstalled(f"qmd binary not runnable: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise SearchBackendError(
            f"qmd {' '.join(args)} timed out after {timeout}s"
        ) from e

    if check and result.returncode != 0:
        err = result.stderr.strip() or result.stdout.strip() or "(no output)"
        raise SearchBackendError(
            f"qmd {' '.join(args)} failed ({result.returncode}):\n{err}"
        )
    return result


# ---------------------------------------------------------------------------
# Setup & maintenance
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """True if qmd is installed and runnable."""
    try:
        _find_qmd()
        return True
    except QmdNotInstalled:
        return False


def get_version() -> str | None:
    """Return the qmd version string, or None if not available."""
    if not is_available():
        return None
    try:
        result = subprocess.run(
            [_find_qmd(), "--version"], capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def ensure_collections(paths: cfg.WikiPaths) -> list[str]:
    """Make sure the wiki/ and raw/ collections exist in QMD.

    Returns the list of collection names that are registered after the call.
    Safe to re-run — `qmd collection add` is idempotent by name.
    """
    added: list[str] = []

    # 1. Check what's already there
    try:
        result = _run_qmd(paths, ["collection", "list"], check=False)
        existing_text = result.stdout + result.stderr
    except SearchBackendError:
        existing_text = ""

    # 2. wiki collection (markdown only)
    if "llm-wiki-pages" not in existing_text:
        _run_qmd(
            paths,
            [
                "collection", "add", str(paths.wiki),
                "--name", "llm-wiki-pages",
                "--mask", "**/*.md",
            ],
            timeout=QMD_TIMEOUT_SHORT,
        )
        added.append("llm-wiki-pages")

    # 3. raw collection (everything the parsers support)
    if "llm-wiki-raw" not in existing_text:
        _run_qmd(
            paths,
            [
                "collection", "add", str(paths.raw),
                "--name", "llm-wiki-raw",
                "--mask", "**/*.{md,txt,pdf,docx,html,htm}",
            ],
            timeout=QMD_TIMEOUT_SHORT,
        )
        added.append("llm-wiki-raw")

    return added


def update_fts(paths: cfg.WikiPaths) -> None:
    """Rebuild the BM25 full-text index only. Fast (seconds, no model load).

    Call this immediately after an ingest so keyword search works right away,
    then call schedule_embed() to generate vector embeddings in background.
    """
    ensure_collections(paths)
    _run_qmd(paths, ["update"], timeout=QMD_TIMEOUT_LONG)


def _embed_worker(paths: cfg.WikiPaths) -> None:
    """Thread target: run embed, then re-run if more docs arrived during it."""
    global _embed_pending, _embed_thread
    while True:
        with _embed_lock:
            _embed_pending = False
        try:
            _run_qmd(paths, ["embed"], timeout=QMD_TIMEOUT_LONG)
        except SearchBackendError:
            pass
        with _embed_lock:
            if not _embed_pending:
                _embed_thread = None
                return
            # Another ingest finished while we were embedding — loop once more
            # to pick up its new docs without starting a second model load.


def schedule_embed(paths: cfg.WikiPaths) -> bool:
    """Schedule a background vector embedding pass.

    Returns True if a new worker thread was started, False if one was already
    running (the new docs will be picked up when it loops).

    The worker thread is non-daemon so the process stays alive until embeddings
    finish. Callers that want to wait can call wait_for_embed().
    """
    global _embed_pending, _embed_thread
    with _embed_lock:
        _embed_pending = True
        if _embed_thread is not None and _embed_thread.is_alive():
            return False
        _embed_thread = threading.Thread(
            target=_embed_worker, args=(paths,), daemon=False, name="qmd-embed"
        )
        _embed_thread.start()
        return True


def wait_for_embed(timeout: float | None = None) -> bool:
    """Block until the background embed thread finishes (or timeout expires).

    Returns True if no embed is running or it completed within the timeout.
    """
    with _embed_lock:
        thread = _embed_thread
    if thread is None or not thread.is_alive():
        return True
    thread.join(timeout=timeout)
    return not thread.is_alive()


def update_index(paths: cfg.WikiPaths, embed: bool = True) -> None:
    """Rebuild FTS index and optionally embed, both synchronously.

    Use this for the explicit `wiki reindex` command. For post-ingest
    auto-updates prefer update_fts() + schedule_embed() so BM25 is
    immediately available and embedding runs in background.
    """
    ensure_collections(paths)
    _run_qmd(paths, ["update"], timeout=QMD_TIMEOUT_LONG)
    if embed:
        _run_qmd(paths, ["embed"], timeout=QMD_TIMEOUT_LONG)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def _parse_qmd_json(stdout: str) -> list[dict]:
    """QMD's --json output is a JSON array of result objects. Be lenient
    with extra lines or stray stderr mixed in.
    """
    stdout = stdout.strip()
    if not stdout:
        return []
    # QMD emits a single JSON document
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        # Try to find the first [...] block
        start = stdout.find("[")
        end = stdout.rfind("]")
        if start == -1 or end == -1 or end < start:
            return []
        try:
            data = json.loads(stdout[start : end + 1])
        except json.JSONDecodeError:
            return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Some versions wrap in {"results": [...]}
        for key in ("results", "hits", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def _hit_from_dict(raw: dict) -> SearchHit:
    """Build a SearchHit from a QMD JSON result entry (tolerant of field names)."""
    return SearchHit(
        docid=str(raw.get("docid") or raw.get("id") or ""),
        path=str(raw.get("path") or raw.get("filepath") or raw.get("file") or ""),
        collection=str(raw.get("collection") or raw.get("col") or ""),
        title=str(raw.get("title") or ""),
        score=float(raw.get("score") or 0.0),
        snippet=str(raw.get("snippet") or raw.get("preview") or ""),
        context=str(raw.get("context") or ""),
    )


def _read_full_content(
    paths: cfg.WikiPaths, hit: SearchHit, max_chars: int = 8000
) -> str:
    """Look up the original file for a hit and return its (truncated) text."""
    # QMD returns 'path' relative to the collection. We know the collection
    # roots from ensure_collections — try both.
    candidates: list[Path] = []
    if hit.collection == "llm-wiki-pages" or not hit.collection:
        candidates.append(paths.wiki / hit.path)
    if hit.collection == "llm-wiki-raw" or not hit.collection:
        candidates.append(paths.raw / hit.path)
    # Also try interpreting the path as relative to the project root
    candidates.append(paths.root / hit.path)

    for c in candidates:
        if c.exists() and c.is_file():
            try:
                content = c.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[... truncated ...]"
            return content
    return ""


def query(
    paths: cfg.WikiPaths,
    question: str,
    *,
    mode: str = "hybrid",         # 'hybrid' | 'lex' | 'vec'
    limit: int = 8,
    min_score: float = 0.0,
    collections: Iterable[str] | None = None,
    hydrate: bool = True,         # read full content of each hit
    rerank: bool = True,
) -> SearchResults:
    """Execute a query via QMD and return structured results.

    Args:
        paths: Wiki project paths.
        question: The user's query text.
        mode: 'hybrid' uses `qmd query` (BM25+vector+rerank), 'lex' uses
              `qmd search` (BM25 only), 'vec' uses `qmd vsearch`.
        limit: Maximum number of hits to return.
        min_score: Drop hits with score below this threshold.
        collections: Optional list of collection names to restrict search to.
                     Defaults to all.
        hydrate: If True, read the full content of each hit into hit.full_content.
        rerank: Only meaningful for mode='hybrid'. If False, passes --no-rerank.

    Returns:
        SearchResults with up to `limit` hits.
    """
    mode_to_subcmd = {"hybrid": "query", "lex": "search", "vec": "vsearch"}
    if mode not in mode_to_subcmd:
        raise ValueError(f"Invalid mode '{mode}'. Use one of {list(mode_to_subcmd)}")
    subcmd = mode_to_subcmd[mode]

    args = [subcmd, question, "--json", "-n", str(limit)]
    if min_score > 0:
        args.extend(["--min-score", str(min_score)])
    if collections:
        for col in collections:
            args.extend(["-c", col])
    if mode == "hybrid" and not rerank:
        args.append("--no-rerank")

    result = _run_qmd(
        paths, args, timeout=QMD_TIMEOUT_MEDIUM, check=False
    )

    # QMD may return non-zero on "no results" — inspect before raising
    if result.returncode != 0:
        combined = (result.stderr + result.stdout).lower()
        if "no results" in combined or "empty" in combined:
            return SearchResults(query=question, search_mode=subcmd, hits=[])
        err = result.stderr.strip() or result.stdout.strip() or "(no output)"
        raise SearchBackendError(f"qmd {subcmd} failed: {err}")

    raw_hits = _parse_qmd_json(result.stdout)
    hits: list[SearchHit] = []
    for raw in raw_hits:
        hit = _hit_from_dict(raw)
        if hit.score < min_score:
            continue
        if hydrate:
            hit.full_content = _read_full_content(paths, hit)
        hits.append(hit)

    return SearchResults(query=question, search_mode=subcmd, hits=hits)


# ---------------------------------------------------------------------------
# Status / diagnostics
# ---------------------------------------------------------------------------


@dataclass
class BackendStatus:
    installed: bool
    version: str | None = None
    collections: list[str] = field(default_factory=list)
    error: str | None = None


def get_status(paths: cfg.WikiPaths) -> BackendStatus:
    """Inspect the QMD backend for the status command."""
    if not is_available():
        return BackendStatus(
            installed=False,
            error="qmd not installed. Run: npm install -g @tobilu/qmd",
        )

    status = BackendStatus(installed=True, version=get_version())
    try:
        result = _run_qmd(paths, ["collection", "list"], check=False)
        text = result.stdout + result.stderr
        # Tolerant parsing — we just want collection names
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("llm-wiki-"):
                # Extract the collection name (first word)
                name = line.split()[0]
                if name not in status.collections:
                    status.collections.append(name)
    except SearchBackendError as e:
        status.error = str(e)
    return status
