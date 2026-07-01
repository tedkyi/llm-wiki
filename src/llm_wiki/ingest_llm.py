"""LLM ingest pipeline — the orchestrator for `wiki ingest`.

Three-pass flow per source:
    1. EXTRACT — JSON: summary, entities, concepts, takeaways
    2. DRAFT  — one wiki page per entity and concept (or MERGE if exists)
    3. SOURCE — the sources/<slug>.md summary page

After all passes: index.md is rebuilt and log.md is appended.

Transactional: pages are staged and only committed to wiki/ on success.
The DB source status flips to 'ingested' only after a full successful run.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field, ValidationError

from . import config as cfg
from . import db
from . import page_writer
from . import parsers
from . import prompts
from . import slugify
from .llm import (
    LLMError,
    ModelNotFound,
    OllamaClient,
    OllamaNotRunning,
)


MAX_SOURCE_CHARS = 60_000   # ~15K tokens — leaves headroom for prompt + output
EXCERPT_CHARS = 4000        # how much of the source we include in draft prompts


# ---------------------------------------------------------------------------
# Pydantic models for Pass 1 JSON validation
# ---------------------------------------------------------------------------


class ExtractedEntity(BaseModel):
    name: str
    slug: str
    type: str = "entity"
    description: str


class ExtractedConcept(BaseModel):
    name: str
    slug: str
    type: str = "concept"
    description: str


class Extraction(BaseModel):
    title: str
    source_slug: str
    summary: str
    key_takeaways: list[str] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    concepts: list[ExtractedConcept] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class PageChange:
    slug: str
    path: str       # relative to wiki root, e.g. 'entities/karpathy.md'
    kind: str       # 'entity' | 'concept' | 'source'
    operation: str  # 'created' | 'updated'


@dataclass
class IngestResult:
    source_id: int
    source_title: str
    source_slug: str
    pages_created: int = 0
    pages_updated: int = 0
    changes: list[PageChange] = field(default_factory=list)
    error: str | None = None
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and not self.skipped


# ---------------------------------------------------------------------------
# Progress callback interface
# ---------------------------------------------------------------------------


class IngestCallbacks:
    """Hooks the CLI provides to render progress during ingest.

    All methods have default no-ops. The CLI subclasses to add rich output.
    """

    def on_start(self, source_id: int, source_title: str, file_path: str) -> None: ...

    def on_verbose_log_path(self, log_path: str) -> None: ...

    def on_parsing(self) -> None: ...

    def on_extracting(self) -> None: ...

    def on_extracted(self, extraction: Extraction) -> None: ...

    def on_extraction_failed(self, error: str) -> None: ...

    def ask_confirm(self, extraction: Extraction) -> bool:
        """Interactive confirmation before writing pages. Default: yes."""
        return True

    def on_drafting_page(self, kind: str, slug: str, operation: str) -> None: ...

    def on_stream_chunk(self, chunk: str) -> None: ...

    def on_page_written(self, page: PageChange) -> None: ...

    def on_finalizing(self) -> None: ...

    def on_complete(self, result: IngestResult) -> None: ...

    def on_error(self, error: str) -> None: ...


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> str:
    """Find the first top-level {...} block in text. Robust to extra prose."""
    text = text.strip()
    # Strip possible markdown fences
    if text.startswith("```"):
        lines = text.split("\n", 1)
        if len(lines) == 2:
            text = lines[1]
        if text.rstrip().endswith("```"):
            text = text.rsplit("```", 1)[0]

    start = text.find("{")
    if start == -1:
        return text
    # Find matching closing brace
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _parse_extraction(raw: str) -> Extraction:
    """Parse the JSON from Pass 1, raising ValueError on failure."""
    json_str = _extract_json_object(raw)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM: {e}") from e
    try:
        return Extraction(**data)
    except ValidationError as e:
        raise ValueError(f"JSON didn't match expected schema: {e}") from e


def _build_excerpt(text: str, max_chars: int = EXCERPT_CHARS) -> str:
    """Return a trimmed snippet of the source text suitable for draft prompts."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated ...]"


def _resolve_slug(
    name: str,
    kind: str,
    paths: cfg.WikiPaths,
    llm_suggested_slug: str,
) -> tuple[str, bool]:
    """Resolve the canonical slug for an entity/concept.

    Returns (slug, exists) where `exists` is True if we're updating an
    existing page vs creating a new one.
    """
    if kind == "entity":
        search_dirs = [paths.wiki / "entities"]
    else:
        search_dirs = [paths.wiki / "concepts"]

    # Determine entity type for canonical_name (fuzzy match)
    match_kind = "any"
    # We don't know if it's person/org here without more signal, so use 'any'

    existing = slugify.find_existing_slug(name, kind=match_kind, search_dirs=search_dirs)
    if existing:
        return existing, True

    # No existing match — sanitize the LLM's suggestion
    raw_slug = llm_suggested_slug or name
    clean = slugify.slugify(raw_slug)
    if not clean:
        clean = slugify.slugify(name) or "untitled"
    return clean, False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Verbose diagnostic logger
# ---------------------------------------------------------------------------


class _IngestLogger:
    """Writes a verbose diagnostic log for a single ingest run.

    Flushes after every write so the file is readable even if the process
    is killed mid-generation (useful for diagnosing hanging Ollama calls).
    """

    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = log_path.open("w", encoding="utf-8", buffering=1)

    @staticmethod
    def _ts() -> str:
        return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3] + "Z"

    def section(self, title: str) -> None:
        sep = "=" * 60
        self._f.write(f"\n{sep}\n[{self._ts()}] {title}\n{sep}\n")
        self._f.flush()

    def info(self, text: str) -> None:
        self._f.write(f"[{self._ts()}] {text}\n")
        self._f.flush()

    def raw(self, label: str, content: str) -> None:
        self._f.write(f"\n--- {label} ---\n{content}\n--- END {label} ---\n")
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.flush()
            self._f.close()

    def __del__(self) -> None:
        self.close()


class _NoopLogger:
    """Drop-in no-op when verbose_log=False."""

    def section(self, title: str) -> None: ...
    def info(self, text: str) -> None: ...
    def raw(self, label: str, content: str) -> None: ...
    def close(self) -> None: ...


def _write_failure_log(
    paths: cfg.WikiPaths,
    source_id: int,
    source_title: str,
    step: str,
    error: str,
) -> Path:
    """Write a minimal failure log for any ingest error (always-on, no flag needed)."""
    paths.logs.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_title)[:60]
    log_path = paths.logs / f"fail-{source_id}-{safe_title}-{ts}.log"
    lines = [
        f"Ingest failure log",
        f"Timestamp   : {ts}",
        f"Source ID   : {source_id}",
        f"Source title: {source_title}",
        f"Failed step : {step}",
        f"",
        f"Error:",
        error,
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path


def _write_extraction_log(
    paths: cfg.WikiPaths,
    source_title: str,
    initial_response: str,
    initial_error: str,
    retry_response: str | None = None,
    retry_error: str | None = None,
) -> Path:
    """Write a diagnostic log for an extraction failure and return the log path."""
    paths.logs.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_title)[:60]
    log_path = paths.logs / f"extraction-{safe_title}-{ts}.log"

    lines = [
        f"Extraction failure log",
        f"Timestamp : {ts}",
        f"Source    : {source_title}",
        f"",
        f"=== ATTEMPT 1 ERROR ===",
        initial_error,
        f"",
        f"=== ATTEMPT 1 RAW RESPONSE ===",
        initial_response,
    ]
    if retry_response is not None:
        lines += [
            f"",
            f"=== RETRY ERROR ===",
            retry_error or "",
            f"",
            f"=== RETRY RAW RESPONSE ===",
            retry_response,
        ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path


def _mark_source_status(
    paths: cfg.WikiPaths, source_id: int, status: str, last_ingested: str | None = None
) -> None:
    with db.connect(paths.state_db) as conn:
        if last_ingested:
            conn.execute(
                "UPDATE sources SET status = ?, last_ingested = ? WHERE id = ?",
                (status, last_ingested, source_id),
            )
        else:
            conn.execute(
                "UPDATE sources SET status = ? WHERE id = ?", (status, source_id)
            )


def _record_ingest_run(
    paths: cfg.WikiPaths,
    source_id: int,
    started: str,
    mode: str,
    pages_created: int,
    pages_updated: int,
    error: str | None,
) -> None:
    finished = _now_iso()
    with db.connect(paths.state_db) as conn:
        conn.execute(
            """
            INSERT INTO ingest_runs
                (started_at, finished_at, source_id, mode, pages_created,
                 pages_updated, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (started, finished, source_id, mode, pages_created, pages_updated, error),
        )


def _record_source_pages(
    paths: cfg.WikiPaths, source_id: int, changes: list[PageChange], at: str
) -> None:
    with db.connect(paths.state_db) as conn:
        for change in changes:
            conn.execute(
                """
                INSERT INTO source_pages (source_id, wiki_path, operation, at)
                VALUES (?, ?, ?, ?)
                """,
                (source_id, change.path, change.operation, at),
            )


def _auto_discover_pending(paths: cfg.WikiPaths) -> int:
    """Scan raw/ for files not yet tracked in the DB and register them.

    Returns the number of newly discovered files.
    """
    from . import ingest_raw

    with db.connect(paths.state_db) as conn:
        rows = conn.execute("SELECT relpath FROM sources").fetchall()
        tracked = {row["relpath"] for row in rows}

    discovered = 0
    if not paths.raw.exists():
        return 0

    for file_path in paths.raw.rglob("*"):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue
        if not parsers.is_supported(file_path):
            continue
        try:
            relpath = str(file_path.relative_to(paths.root))
        except ValueError:
            continue
        if relpath in tracked:
            continue
        outcome = ingest_raw.add_file(paths, file_path, copy=False)
        if outcome.result == ingest_raw.AddResult.ADDED:
            discovered += 1
    return discovered


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _stream_with_retry(
    client: OllamaClient,
    messages: list,
    callbacks: "IngestCallbacks",
    logger: "_IngestLogger | _NoopLogger",
    label: str,
) -> tuple[str, float]:
    """Stream an LLM response, retrying once after 5 s on any LLMError.

    Returns (full_content, elapsed_seconds_of_final_successful_call).
    """
    for attempt in range(2):
        if attempt == 1:
            logger.info(f"LLM call failed for {label}; sleeping 5s before retry…")
            time.sleep(5)
            logger.info(f"Retrying {label}…")
        _t0 = datetime.now(timezone.utc)
        try:
            full = ""
            gen = client.chat_stream(messages, thinking=False, temperature=0.3)
            try:
                while True:
                    chunk = next(gen)
                    callbacks.on_stream_chunk(chunk)
                    full += chunk
            except StopIteration as stop:
                if stop.value:
                    full = stop.value
            elapsed = (datetime.now(timezone.utc) - _t0).total_seconds()
            if not full.strip():
                raise LLMError(f"Empty response from Ollama for {label}")
            return full, elapsed
        except LLMError:
            elapsed = (datetime.now(timezone.utc) - _t0).total_seconds()
            logger.info(f"LLM call failed after {elapsed:.1f}s")
            if attempt == 1:
                raise
    raise AssertionError("unreachable")


def ingest_source(
    paths: cfg.WikiPaths,
    source_id: int,
    client: OllamaClient,
    callbacks: IngestCallbacks,
    *,
    mode: str = "interactive",
    thinking_for_extraction: bool = True,
    verbose_log: bool = True,
) -> IngestResult:
    """Run the full 3-pass ingest pipeline on a single source."""
    started = _now_iso()

    # Set up verbose logger — created before any early returns so all paths can close it
    if verbose_log:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _log_path = paths.logs / f"ingest-{source_id}-{ts}.log"
        logger: _IngestLogger | _NoopLogger = _IngestLogger(_log_path)
        callbacks.on_verbose_log_path(str(_log_path))
    else:
        logger = _NoopLogger()

    # 1. Load the source row
    with db.connect(paths.state_db) as conn:
        row = conn.execute(
            "SELECT * FROM sources WHERE id = ?", (source_id,)
        ).fetchone()
        if row is None:
            result = IngestResult(
                source_id=source_id,
                source_title="?",
                source_slug="?",
                error=f"No source with id {source_id}",
            )
            callbacks.on_error(result.error)
            logger.close()
            return result
        source_row = dict(row)

    file_path = paths.root / source_row["relpath"]
    logger.section(f"INGEST SOURCE #{source_id}")
    logger.info(f"relpath : {source_row['relpath']}")
    logger.info(f"started : {started}")
    callbacks.on_start(source_id, source_row["relpath"], str(file_path))

    # 2. Parse the source
    callbacks.on_parsing()
    try:
        parsed = parsers.parse(file_path)
    except parsers.ParserError as e:
        result = IngestResult(
            source_id=source_id,
            source_title=source_row["relpath"],
            source_slug="?",
            error=f"Parse failed: {e}",
        )
        logger.info(f"ERROR (parse): {e}")
        logger.close()
        _mark_source_status(paths, source_id, "error")
        _record_ingest_run(paths, source_id, started, mode, 0, 0, result.error)
        callbacks.on_error(result.error)
        return result

    # Truncate very long sources
    source_text = parsed.text
    if len(source_text) > MAX_SOURCE_CHARS:
        original_chars = len(source_text)
        source_text = source_text[:MAX_SOURCE_CHARS] + "\n\n[... truncated ...]"
        logger.info(f"TRUNCATED: {original_chars} chars → {MAX_SOURCE_CHARS} (dropped {original_chars - MAX_SOURCE_CHARS} chars)")

    logger.info(f"parsed  : title={parsed.title!r}, file_type={parsed.file_type}, chars={len(parsed.text)}")

    # 3. Pass 1 — extraction
    callbacks.on_extracting()
    extraction_messages = prompts.build_extraction_messages(parsed.title, source_text)
    logger.section("PASS 1 — EXTRACTION")
    logger.info(f"LLM call start: thinking={thinking_for_extraction}, json_mode=True, temperature=0.3")
    _t0 = datetime.now(timezone.utc)
    try:
        raw_response = client.chat(
            extraction_messages,
            thinking=thinking_for_extraction,
            json_mode=True,
            temperature=0.3,
            num_predict=None,
        )
    except (OllamaNotRunning, ModelNotFound) as e:
        logger.info(f"ERROR (Ollama unreachable): {e}")
        logger.close()
        result = IngestResult(
            source_id=source_id,
            source_title=parsed.title,
            source_slug="?",
            error=str(e),
        )
        _write_failure_log(paths, source_id, parsed.title, "extraction (Ollama unreachable)", str(e))
        callbacks.on_error(result.error)
        # Don't mark as error — user needs to fix Ollama, then retry
        return result
    except LLMError as e:
        # Retry once for transient errors (e.g. timeout)
        logger.info(f"ERROR (LLM): {e} — retrying once…")
        logger.section("PASS 1 — EXTRACTION RETRY")
        logger.info("Sleeping 5s before retry…")
        time.sleep(5)
        logger.info(f"LLM call start: thinking={thinking_for_extraction}, json_mode=True, temperature=0.3")
        _t0 = datetime.now(timezone.utc)
        try:
            raw_response = client.chat(
                extraction_messages,
                thinking=thinking_for_extraction,
                json_mode=True,
                temperature=0.3,
                num_predict=None,
            )
        except LLMError as e2:
            logger.info(f"ERROR (LLM): {e2}")
            logger.close()
            result = IngestResult(
                source_id=source_id,
                source_title=parsed.title,
                source_slug="?",
                error=f"LLM error: {e2}",
            )
            _write_failure_log(paths, source_id, parsed.title, "extraction (LLM error)", str(e2))
            _mark_source_status(paths, source_id, "error")
            _record_ingest_run(paths, source_id, started, mode, 0, 0, result.error)
            callbacks.on_error(result.error)
            return result
    _elapsed = (datetime.now(timezone.utc) - _t0).total_seconds()
    logger.info(f"LLM call end: {_elapsed:.1f}s, {len(raw_response)} chars")
    logger.raw("EXTRACTION RESPONSE", raw_response)

    try:
        extraction = _parse_extraction(raw_response)
        logger.info(f"extraction parsed OK: {len(extraction.entities)} entities, {len(extraction.concepts)} concepts")
    except ValueError as e:
        # Retry once with explicit correction
        initial_response = raw_response
        initial_error = str(e)
        logger.info(f"parse failed: {initial_error}")
        callbacks.on_extraction_failed(initial_error)
        retry_response: str | None = None
        logger.section("PASS 1 — EXTRACTION RETRY")
        logger.info("Sleeping 5s before retry…")
        time.sleep(5)
        logger.info("LLM call start: thinking=False, json_mode=True, temperature=0.2")
        _t0 = datetime.now(timezone.utc)
        try:
            retry_messages = prompts.build_extraction_retry_messages(
                parsed.title, source_text, raw_response
            )
            retry_response = client.chat(
                retry_messages,
                thinking=False,  # retry without thinking, faster
                json_mode=True,
                temperature=0.2,
                num_predict=None,
            )
            _elapsed = (datetime.now(timezone.utc) - _t0).total_seconds()
            logger.info(f"LLM call end: {_elapsed:.1f}s, {len(retry_response)} chars")
            logger.raw("EXTRACTION RETRY RESPONSE", retry_response)
            extraction = _parse_extraction(retry_response)
            logger.info(f"retry parsed OK: {len(extraction.entities)} entities, {len(extraction.concepts)} concepts")
        except (ValueError, LLMError) as e2:
            _elapsed = (datetime.now(timezone.utc) - _t0).total_seconds()
            logger.info(f"ERROR after {_elapsed:.1f}s: {e2}")
            if retry_response:
                logger.raw("EXTRACTION RETRY RESPONSE", retry_response)
            log_path = _write_extraction_log(
                paths,
                source_title=parsed.title,
                initial_response=initial_response,
                initial_error=initial_error,
                retry_response=retry_response,
                retry_error=str(e2),
            )
            logger.close()
            result = IngestResult(
                source_id=source_id,
                source_title=parsed.title,
                source_slug="?",
                error=f"Extraction failed after retry: {e2} (log: {log_path})",
            )
            _mark_source_status(paths, source_id, "error")
            _record_ingest_run(paths, source_id, started, mode, 0, 0, result.error)
            callbacks.on_error(result.error)
            return result

    # Sanitize the source slug
    source_slug = slugify.slugify(extraction.source_slug or extraction.title)
    extraction.source_slug = source_slug

    callbacks.on_extracted(extraction)

    # 4. Interactive confirmation gate
    if mode == "interactive":
        if not callbacks.ask_confirm(extraction):
            result = IngestResult(
                source_id=source_id,
                source_title=parsed.title,
                source_slug=source_slug,
                skipped=True,
            )
            callbacks.on_complete(result)
            return result

    # 5. Resolve slugs for all entities and concepts (dedupe against existing)
    today = page_writer.today_iso()
    entity_plans: list[tuple[ExtractedEntity, str, bool]] = []  # (item, slug, exists)
    for ent in extraction.entities:
        slug, exists = _resolve_slug(ent.name, "entity", paths, ent.slug)
        entity_plans.append((ent, slug, exists))

    concept_plans: list[tuple[ExtractedConcept, str, bool]] = []
    for con in extraction.concepts:
        slug, exists = _resolve_slug(con.name, "concept", paths, con.slug)
        concept_plans.append((con, slug, exists))

    # 6. Staging directory for transactional writes
    staging = Path(tempfile.mkdtemp(prefix="llm-wiki-ingest-"))
    try:
        staged_files: list[tuple[Path, Path, PageChange]] = []  # (staged, final, change)

        # 6a. Build the "related" list for each page (used in draft prompts)
        all_entity_slugs = [s for _, s, _ in entity_plans]
        all_concept_slugs = [s for _, s, _ in concept_plans]

        def _related_for(exclude_slug: str, exclude_kind: str) -> list[str]:
            rel: list[str] = []
            for s in all_entity_slugs:
                if not (exclude_kind == "entity" and s == exclude_slug):
                    rel.append(f"entities/{s}")
            for s in all_concept_slugs:
                if not (exclude_kind == "concept" and s == exclude_slug):
                    rel.append(f"concepts/{s}")
            return rel

        excerpt = _build_excerpt(parsed.text)

        # 6b. Draft/merge each entity page
        for ent, slug, exists in entity_plans:
            operation = "updated" if exists else "created"
            callbacks.on_drafting_page("entity", slug, operation)

            final_path = paths.wiki / "entities" / f"{slug}.md"
            staged_path = staging / f"entities__{slug}.md"

            try:
                if exists:
                    existing_page = page_writer.read_page(final_path)
                    existing_content = (
                        existing_page.to_markdown() if existing_page else ""
                    )
                    messages = prompts.build_merge_page_messages(
                        name=ent.name,
                        existing_content=existing_content,
                        source_title=parsed.title,
                        source_slug=source_slug,
                        description=ent.description,
                        excerpts=excerpt,
                        today=today,
                    )
                else:
                    messages = prompts.build_draft_page_messages(
                        kind="entity",
                        name=ent.name,
                        source_title=parsed.title,
                        source_slug=source_slug,
                        description=ent.description,
                        excerpts=excerpt,
                        related=_related_for(slug, "entity"),
                        today=today,
                    )

                # Stream the response
                logger.section(f"PASS 2 — ENTITY {slug} ({operation})")
                logger.info("Sleeping 1s before LLM call…")
                time.sleep(1)
                logger.info("LLM call start: thinking=False, temperature=0.3")
                full, _elapsed = _stream_with_retry(client, messages, callbacks, logger, f"entity {slug}")
                logger.info(f"LLM call end: {_elapsed:.1f}s, {len(full)} chars")
                logger.raw(f"ENTITY {slug}", full)

                content = page_writer.strip_llm_noise(full)

                # Validate frontmatter presence; if missing, wrap it
                parsed_page = page_writer.parse_page(content)
                if not parsed_page.frontmatter:
                    # LLM forgot frontmatter — synthesize minimal one
                    parsed_page.frontmatter = {
                        "title": ent.name,
                        "type": "entity",
                        "tags": extraction.tags[:3],
                        "created": today,
                        "updated": today,
                        "sources": [f"sources/{source_slug}"],
                        "confidence": "medium",
                    }
                    parsed_page.body = content
                # Always ensure source is in sources list
                page_writer.add_source_to_frontmatter(parsed_page, source_slug, today)
                content = parsed_page.to_markdown()

                staged_path.write_text(content, encoding="utf-8")
                change = PageChange(
                    slug=slug,
                    path=f"entities/{slug}.md",
                    kind="entity",
                    operation=operation,
                )
                staged_files.append((staged_path, final_path, change))
                callbacks.on_page_written(change)

            except LLMError as e:
                logger.info(f"ERROR: {e}")
                logger.close()
                result = IngestResult(
                    source_id=source_id,
                    source_title=parsed.title,
                    source_slug=source_slug,
                    error=f"Failed drafting entity '{slug}': {e}",
                )
                _write_failure_log(paths, source_id, parsed.title, f"entity page '{slug}'", str(e))
                _mark_source_status(paths, source_id, "error")
                _record_ingest_run(paths, source_id, started, mode, 0, 0, result.error)
                callbacks.on_error(result.error)
                return result

        # 6c. Draft/merge each concept page
        for con, slug, exists in concept_plans:
            operation = "updated" if exists else "created"
            callbacks.on_drafting_page("concept", slug, operation)

            final_path = paths.wiki / "concepts" / f"{slug}.md"
            staged_path = staging / f"concepts__{slug}.md"

            try:
                if exists:
                    existing_page = page_writer.read_page(final_path)
                    existing_content = (
                        existing_page.to_markdown() if existing_page else ""
                    )
                    messages = prompts.build_merge_page_messages(
                        name=con.name,
                        existing_content=existing_content,
                        source_title=parsed.title,
                        source_slug=source_slug,
                        description=con.description,
                        excerpts=excerpt,
                        today=today,
                    )
                else:
                    messages = prompts.build_draft_page_messages(
                        kind="concept",
                        name=con.name,
                        source_title=parsed.title,
                        source_slug=source_slug,
                        description=con.description,
                        excerpts=excerpt,
                        related=_related_for(slug, "concept"),
                        today=today,
                    )

                logger.section(f"PASS 2 — CONCEPT {slug} ({operation})")
                logger.info("Sleeping 1s before LLM call…")
                time.sleep(1)
                logger.info("LLM call start: thinking=False, temperature=0.3")
                full, _elapsed = _stream_with_retry(client, messages, callbacks, logger, f"concept {slug}")
                logger.info(f"LLM call end: {_elapsed:.1f}s, {len(full)} chars")
                logger.raw(f"CONCEPT {slug}", full)

                content = page_writer.strip_llm_noise(full)

                parsed_page = page_writer.parse_page(content)
                if not parsed_page.frontmatter:
                    parsed_page.frontmatter = {
                        "title": con.name,
                        "type": "concept",
                        "tags": extraction.tags[:3],
                        "created": today,
                        "updated": today,
                        "sources": [f"sources/{source_slug}"],
                        "confidence": "medium",
                    }
                    parsed_page.body = content
                page_writer.add_source_to_frontmatter(parsed_page, source_slug, today)
                content = parsed_page.to_markdown()

                staged_path.write_text(content, encoding="utf-8")
                change = PageChange(
                    slug=slug,
                    path=f"concepts/{slug}.md",
                    kind="concept",
                    operation=operation,
                )
                staged_files.append((staged_path, final_path, change))
                callbacks.on_page_written(change)

            except LLMError as e:
                logger.info(f"ERROR: {e}")
                logger.close()
                result = IngestResult(
                    source_id=source_id,
                    source_title=parsed.title,
                    source_slug=source_slug,
                    error=f"Failed drafting concept '{slug}': {e}",
                )
                _write_failure_log(paths, source_id, parsed.title, f"concept page '{slug}'", str(e))
                _mark_source_status(paths, source_id, "error")
                _record_ingest_run(paths, source_id, started, mode, 0, 0, result.error)
                callbacks.on_error(result.error)
                return result

        # 6d. Pass 3 — source summary page
        callbacks.on_drafting_page("source", source_slug, "created")
        source_final = paths.wiki / "sources" / f"{source_slug}.md"
        source_staged = staging / f"sources__{source_slug}.md"

        try:
            messages = prompts.build_source_page_messages(
                source_title=parsed.title,
                source_slug=source_slug,
                file_path=source_row["relpath"],
                file_type=parsed.file_type,
                summary=extraction.summary,
                key_takeaways=extraction.key_takeaways,
                tags=extraction.tags,
                entity_slugs=[s for _, s, _ in entity_plans],
                concept_slugs=[s for _, s, _ in concept_plans],
                today=today,
            )

            logger.section(f"PASS 3 — SOURCE PAGE {source_slug}")
            logger.info("Sleeping 1s before LLM call…")
            time.sleep(1)
            logger.info("LLM call start: thinking=False, temperature=0.3")
            full, _elapsed = _stream_with_retry(client, messages, callbacks, logger, f"source page {source_slug}")
            logger.info(f"LLM call end: {_elapsed:.1f}s, {len(full)} chars")
            logger.raw(f"SOURCE PAGE {source_slug}", full)

            content = page_writer.strip_llm_noise(full)
            parsed_page = page_writer.parse_page(content)
            if not parsed_page.frontmatter:
                parsed_page.frontmatter = {
                    "title": parsed.title,
                    "type": "source",
                    "tags": extraction.tags,
                    "created": today,
                    "updated": today,
                    "file_path": source_row["relpath"],
                    "file_type": parsed.file_type,
                }
                parsed_page.body = content
            content = parsed_page.to_markdown()

            source_staged.write_text(content, encoding="utf-8")
            source_operation = "updated" if source_final.exists() else "created"
            change = PageChange(
                slug=source_slug,
                path=f"sources/{source_slug}.md",
                kind="source",
                operation=source_operation,
            )
            staged_files.append((source_staged, source_final, change))
            callbacks.on_page_written(change)

        except LLMError as e:
            logger.info(f"ERROR: {e}")
            logger.close()
            result = IngestResult(
                source_id=source_id,
                source_title=parsed.title,
                source_slug=source_slug,
                error=f"Failed drafting source page: {e}",
            )
            _write_failure_log(paths, source_id, parsed.title, "source page", str(e))
            _mark_source_status(paths, source_id, "error")
            _record_ingest_run(paths, source_id, started, mode, 0, 0, result.error)
            callbacks.on_error(result.error)
            return result

        # 7. Commit: move staged files to final locations
        callbacks.on_finalizing()
        pages_created = 0
        pages_updated = 0
        changes: list[PageChange] = []
        for staged, final, change in staged_files:
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged, final)
            changes.append(change)
            if change.operation == "created":
                pages_created += 1
            else:
                pages_updated += 1

        # 8. Rebuild index.md and append to log.md
        page_writer.rebuild_index(paths, today)
        log_bullets = [
            f"{c.operation}: [[{c.path.replace('.md', '')}]]" for c in changes
        ]
        page_writer.append_log_entry(
            paths, today, "ingest", parsed.title, log_bullets
        )

        # 9. Record in DB
        _record_source_pages(paths, source_id, changes, _now_iso())
        _mark_source_status(paths, source_id, "ingested", last_ingested=_now_iso())
        _record_ingest_run(
            paths,
            source_id,
            started,
            mode,
            pages_created,
            pages_updated,
            error=None,
        )

        result = IngestResult(
            source_id=source_id,
            source_title=parsed.title,
            source_slug=source_slug,
            pages_created=pages_created,
            pages_updated=pages_updated,
            changes=changes,
        )
        logger.section("COMPLETE")
        logger.info(f"pages_created={pages_created}, pages_updated={pages_updated}")
        logger.close()
        callbacks.on_complete(result)
        return result

    finally:
        shutil.rmtree(staging, ignore_errors=True)


def ingest_pending(
    paths: cfg.WikiPaths,
    client: OllamaClient,
    callbacks_factory: Callable[[], IngestCallbacks],
    *,
    mode: str = "interactive",
    auto_discover: bool = True,
    thinking_for_extraction: bool = True,
    verbose_log: bool = True,
) -> list[IngestResult]:
    """Ingest all pending sources in the DB.

    Args:
        paths: Wiki project paths.
        client: An active Ollama client.
        callbacks_factory: Called once per source to get a fresh callback object.
        mode: 'interactive' | 'batch'.
        auto_discover: If True, scan raw/ for untracked files first.
        thinking_for_extraction: Whether to use Qwen3's thinking mode in Pass 1.

    Returns:
        A list of IngestResult, one per source attempted.
    """
    if auto_discover:
        _auto_discover_pending(paths)

    with db.connect(paths.state_db) as conn:
        rows = conn.execute(
            "SELECT id FROM sources WHERE status = 'pending' ORDER BY id ASC"
        ).fetchall()
        pending_ids = [row["id"] for row in rows]

    results: list[IngestResult] = []
    for sid in pending_ids:
        cb = callbacks_factory()
        result = ingest_source(
            paths,
            sid,
            client,
            cb,
            mode=mode,
            thinking_for_extraction=thinking_for_extraction,
            verbose_log=verbose_log,
        )
        results.append(result)
        if result.error and "Ollama" in result.error:
            # Stop the batch if Ollama is unreachable — no point continuing
            break

    return results
