"""Raw ingest: copy files into raw/, parse them, register in the state DB.

This is the "plumbing" layer. No LLM calls happen here — that's Stage 3's
`wiki ingest` command. This module just ensures every file in `raw/` has
a corresponding row in the `sources` table with a content hash so dedupe
and provenance work downstream.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable

from . import config as cfg
from . import db
from . import parsers
from . import slugify


class AddResult(str, Enum):
    ADDED = "added"
    UPDATED = "updated"          # already-tracked path whose content changed
    DEDUPED = "deduped"
    SKIPPED_EMPTY = "skipped_empty"
    SKIPPED_UNSUPPORTED = "skipped_unsupported"
    ERROR = "error"


@dataclass
class AddOutcome:
    """Result of attempting to add a single file."""

    result: AddResult
    source_path: Path              # Where the file ended up (or original on error)
    relpath: str                   # Relative to project root
    title: str | None = None
    file_type: str | None = None
    bytes: int = 0
    word_count: int = 0
    content_hash: str | None = None
    source_id: int | None = None   # Row ID in sources table
    message: str = ""              # Human-friendly explanation

    @property
    def ok(self) -> bool:
        return self.result == AddResult.ADDED

    @property
    def is_warning(self) -> bool:
        return self.result in {
            AddResult.DEDUPED,
            AddResult.SKIPPED_EMPTY,
            AddResult.SKIPPED_UNSUPPORTED,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _unique_destination(dest_dir: Path, filename: str) -> Path:
    """If `dest_dir/filename` already exists, append `-1`, `-2`, ... before the
    extension until we find an unused path.
    """
    target = dest_dir / filename
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    counter = 1
    while True:
        candidate = dest_dir / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _is_inside_raw(path: Path, raw_dir: Path) -> bool:
    """True if `path` is already inside the project's raw/ directory."""
    try:
        path.resolve().relative_to(raw_dir.resolve())
        return True
    except ValueError:
        return False


def source_path(paths: cfg.WikiPaths, relpath: str) -> Path:
    """Resolve a source row's stored path to an absolute filesystem path.

    A source's location is stored either as a project-root-relative path (files
    copied into raw/, plus all legacy rows) or as an absolute path (files
    *referenced* in place — the default for `wiki add`). This helper centralizes
    that distinction so callers never blindly do ``paths.root / relpath``, which
    silently mishandles absolute paths (fragile on Windows in particular).
    """
    p = Path(relpath)
    return p if p.is_absolute() else paths.root / p


# ---------------------------------------------------------------------------
# Text mirrors: raw-text/<source-id>-<original-filename>.md
# ---------------------------------------------------------------------------
# The search backend (QMD) only understands plain text / markdown, so for
# every source in raw/ we persist the parser-extracted text as a markdown
# mirror in raw-text/. That mirror — not the original binary — is what gets
# indexed for full-document search.


def _mirror_slug(source_file: Path) -> str:
    """Slugified original filename (extension included, so 'foo.pdf' and
    'foo.txt' stay distinct: 'foo-pdf' / 'foo-txt')."""
    return slugify.slugify(source_file.name, max_length=120)


def _legacy_mirror_path(paths: cfg.WikiPaths, source_file: Path) -> Path:
    """The pre-source-id mirror name (`<slug>.md`). Kept only so touching a
    source can clean up its old mirror after the id-based rename."""
    return paths.raw_text / f"{_mirror_slug(source_file)}.md"


def text_mirror_path(
    paths: cfg.WikiPaths, source_id: int, source_file: Path
) -> Path:
    """Return the raw-text/ mirror path for a source.

    Keyed on the source id (`<id>-<slug>.md`) so two referenced files that share
    a basename but live in different directories don't collide on one mirror.
    Slugs contain only [a-z0-9-], which QMD's internal path slugging leaves
    untouched — this keeps QMD's reported document paths mappable back to the
    file on disk. The original filename is recorded in the mirror's frontmatter
    (source_file).
    """
    return paths.raw_text / f"{source_id}-{_mirror_slug(source_file)}.md"


def write_text_mirror(
    paths: cfg.WikiPaths, source_id: int, source_file: Path, parsed
) -> Path:
    """Write the extracted text of a parsed source to its raw-text/ mirror.

    The mirror gets minimal frontmatter (title + provenance) and an H1 so
    QMD can extract a meaningful title for search results. Any legacy
    (`<slug>.md`) mirror for this source is removed so the search index doesn't
    end up with two copies of the same content after the id-based rename.
    """
    mirror = text_mirror_path(paths, source_id, source_file)
    mirror.parent.mkdir(parents=True, exist_ok=True)
    try:
        source_rel = str(source_file.relative_to(paths.root)).replace("\\", "/")
    except ValueError:
        source_rel = str(source_file).replace("\\", "/")
    title = (parsed.title or source_file.stem).strip()
    safe_title = title.replace('"', "'")
    content = (
        "---\n"
        f'title: "{safe_title}"\n'
        f'source_file: "{source_rel}"\n'
        f"content_hash: {parsed.content_hash}\n"
        "---\n"
        "\n"
        f"# {title}\n"
        "\n"
        f"{parsed.text}\n"
    )
    mirror.write_text(content, encoding="utf-8")
    legacy = _legacy_mirror_path(paths, source_file)
    if legacy != mirror and legacy.exists():
        try:
            legacy.unlink()
        except OSError:
            pass
    return mirror


def delete_text_mirror(
    paths: cfg.WikiPaths, source_id: int, source_file: Path
) -> bool:
    """Remove the raw-text/ mirror(s) for a source, if present.

    Removes both the id-based name and any leftover legacy (`<slug>.md`) mirror.
    """
    deleted = False
    for mirror in (
        text_mirror_path(paths, source_id, source_file),
        _legacy_mirror_path(paths, source_file),
    ):
        if mirror.exists():
            try:
                mirror.unlink()
                deleted = True
            except OSError:
                pass
    return deleted


@dataclass
class BackfillStats:
    written: int = 0
    skipped_existing: int = 0
    missing_file: int = 0
    parse_errors: list[str] = field(default_factory=list)


def backfill_text_mirrors(
    paths: cfg.WikiPaths,
    *,
    force: bool = False,
    progress=None,  # optional callable(relpath: str, index: int, total: int)
) -> BackfillStats:
    """Ensure every tracked source has a raw-text/ mirror.

    Re-parses sources from raw/ (no LLM involved) and writes any missing
    mirrors. With force=True, existing mirrors are rewritten too. Used by
    `wiki reindex --full` to migrate wikis created before mirrors existed.
    """
    stats = BackfillStats()
    rows = list_sources(paths)
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        source_file = source_path(paths, row["relpath"])
        if progress is not None:
            progress(row["relpath"], i, total)
        if not source_file.exists():
            stats.missing_file += 1
            continue
        if not force and text_mirror_path(paths, row["id"], source_file).exists():
            stats.skipped_existing += 1
            continue
        try:
            parsed = parsers.parse(source_file)
        except parsers.ParserError as e:
            stats.parse_errors.append(f"{row['relpath']}: {e}")
            continue
        write_text_mirror(paths, row["id"], source_file, parsed)
        stats.written += 1
    return stats


def add_file(
    paths: cfg.WikiPaths,
    source: Path,
    *,
    copy: bool = False,
) -> AddOutcome:
    """Register a source file with the wiki.

    By default the file is *referenced in place* — its absolute path is stored
    and no copy is made (saving disk for content that already lives on the
    machine). Pass ``copy=True`` to fall back to the old behavior of copying the
    file into raw/ (used by ``wiki add --copy`` and by web uploads, whose bytes
    have no external path to reference).

    Args:
        paths: Resolved wiki project paths.
        source: The file to add (can be anywhere on disk).
        copy: If True, copy the file into raw/ and register the copy. If False
              (default), reference the file where it is.

    Returns:
        AddOutcome describing what happened. Check `.result` for the variant.
    """
    source = source.expanduser().resolve()

    # Guard: file must exist
    if not source.exists() or not source.is_file():
        return AddOutcome(
            result=AddResult.ERROR,
            source_path=source,
            relpath=str(source),
            message=f"File not found: {source}",
        )

    # Guard: file type must be supported
    if not parsers.is_supported(source):
        return AddOutcome(
            result=AddResult.SKIPPED_UNSUPPORTED,
            source_path=source,
            relpath=str(source),
            message=f"Unsupported file type: {source.suffix or '(no extension)'}",
        )

    # Determine the final destination in raw/
    if copy and not _is_inside_raw(source, paths.raw):
        paths.raw.mkdir(parents=True, exist_ok=True)
        dest = _unique_destination(paths.raw, source.name)
        try:
            shutil.copy2(source, dest)
        except OSError as e:
            return AddOutcome(
                result=AddResult.ERROR,
                source_path=source,
                relpath=str(source),
                message=f"Failed to copy into raw/: {e}",
            )
        final_path = dest
    else:
        final_path = source

    # Parse
    try:
        parsed = parsers.parse(final_path)
    except parsers.ParserError as e:
        return AddOutcome(
            result=AddResult.ERROR,
            source_path=final_path,
            relpath=str(final_path.relative_to(paths.root)) if _is_inside_raw(final_path, paths.raw) else str(final_path),
            message=f"Parse failed: {e}",
        )

    try:
        relpath = str(final_path.relative_to(paths.root))
    except ValueError:
        relpath = str(final_path)

    with db.connect(paths.state_db) as conn:
        # 1. Is this exact path already tracked? If so, a re-add is either a
        #    no-op (unchanged) or an in-place update (content changed).
        by_path = conn.execute(
            "SELECT id, content_hash FROM sources WHERE relpath = ?",
            (relpath,),
        ).fetchone()

        if by_path is not None:
            if by_path["content_hash"] == parsed.content_hash:
                return AddOutcome(
                    result=AddResult.DEDUPED,
                    source_path=final_path,
                    relpath=relpath,
                    title=parsed.title,
                    file_type=parsed.file_type,
                    bytes=parsed.bytes,
                    word_count=parsed.word_count,
                    content_hash=parsed.content_hash,
                    source_id=by_path["id"],
                    message=f"Already tracked as #{by_path['id']} (unchanged)",
                )
            # Same path, new content: refresh the row and re-queue for ingest.
            conn.execute(
                """
                UPDATE sources
                   SET content_hash = ?, file_type = ?, bytes = ?,
                       status = 'pending', last_ingested = NULL
                 WHERE id = ?
                """,
                (parsed.content_hash, parsed.file_type, parsed.bytes, by_path["id"]),
            )
            source_id = by_path["id"]
            result_kind = AddResult.UPDATED
            message = (
                f"Source already tracked as #{source_id}; content changed — "
                f"marked for re-ingestion (run wiki ingest)."
            )
        else:
            # 2. Not tracked by path — dedupe identical content added elsewhere.
            existing = conn.execute(
                "SELECT id, relpath FROM sources WHERE content_hash = ?",
                (parsed.content_hash,),
            ).fetchone()

            if existing is not None:
                # If we just copied this in, remove the duplicate copy.
                if copy and final_path.exists() and final_path != source:
                    try:
                        final_path.unlink()
                    except OSError:
                        pass
                return AddOutcome(
                    result=AddResult.DEDUPED,
                    source_path=final_path,
                    relpath=relpath,
                    title=parsed.title,
                    file_type=parsed.file_type,
                    bytes=parsed.bytes,
                    word_count=parsed.word_count,
                    content_hash=parsed.content_hash,
                    source_id=existing["id"],
                    message=f"Already tracked as #{existing['id']}: {existing['relpath']}",
                )

            # 3. Genuinely new source.
            if parsed.is_empty:
                # We still register it so the user sees it in `wiki sources list`
                # with SKIPPED_EMPTY status, but flag it loudly.
                status = "error"
                message = (
                    f"Extracted only {parsed.word_count} words — likely a scanned "
                    f"PDF or empty file. OCR not yet supported."
                )
                result_kind = AddResult.SKIPPED_EMPTY
            else:
                status = "pending"
                message = f"Added as #{'?'}: {parsed.title}"
                result_kind = AddResult.ADDED

            cur = conn.execute(
                """
                INSERT INTO sources (relpath, content_hash, file_type, bytes, added_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    relpath,
                    parsed.content_hash,
                    parsed.file_type,
                    parsed.bytes,
                    _now_iso(),
                    status,
                ),
            )
            source_id = cur.lastrowid
            if "#'?'" in message:
                message = message.replace("#'?'", f"#{source_id}")

    # Persist the extracted text so the search backend has real content to
    # index (it can't parse PDFs/DOCX). A content update rewrites the mirror too.
    if result_kind in (AddResult.ADDED, AddResult.UPDATED):
        try:
            write_text_mirror(paths, source_id, final_path, parsed)
        except OSError:
            pass  # mirror is best-effort; backfill can recreate it

    return AddOutcome(
        result=result_kind,
        source_path=final_path,
        relpath=relpath,
        title=parsed.title,
        file_type=parsed.file_type,
        bytes=parsed.bytes,
        word_count=parsed.word_count,
        content_hash=parsed.content_hash,
        source_id=source_id,
        message=message,
    )


def strip_path_quotes(text: str) -> str:
    """Strip one pair of surrounding single/double quotes from a pasted path.

    Windows Explorer's "Copy as path" wraps the path in double quotes, and
    Windows filenames can't contain quotes anyway — so removing a matched
    surrounding pair makes copy-pasted quoted paths just work.
    """
    s = text.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    return s


def iter_addable_files(root: Path, recursive: bool) -> Iterable[Path]:
    """Yield every supported file at or under `root`.

    If `root` is a file, yield it (if supported). If it's a directory, walk
    it (recursively if requested) and yield every supported file found.
    """
    root = root.expanduser().resolve()
    if not root.exists():
        return
    if root.is_file():
        if parsers.is_supported(root):
            yield root
        return
    if root.is_dir():
        iterator = root.rglob("*") if recursive else root.iterdir()
        for child in iterator:
            if child.is_file() and not child.name.startswith(".") and parsers.is_supported(child):
                yield child


def list_sources(
    paths: cfg.WikiPaths, status_filter: str | None = None
) -> list[dict]:
    """Return all tracked sources as a list of dicts, ordered by id."""
    query = "SELECT * FROM sources"
    params: tuple = ()
    if status_filter:
        query += " WHERE status = ?"
        params = (status_filter,)
    query += " ORDER BY id ASC"

    with db.connect(paths.state_db) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def summarize_sources(paths: cfg.WikiPaths) -> list[dict]:
    """Return per-status counts and id ranges for all tracked sources."""
    query = (
        "SELECT status, COUNT(*) AS count, MIN(id) AS min_id, MAX(id) AS max_id "
        "FROM sources GROUP BY status ORDER BY status ASC"
    )
    with db.connect(paths.state_db) as conn:
        rows = conn.execute(query).fetchall()
        return [dict(row) for row in rows]


def get_source(paths: cfg.WikiPaths, source_id: int) -> dict | None:
    """Fetch a single source row by id."""
    with db.connect(paths.state_db) as conn:
        row = conn.execute(
            "SELECT * FROM sources WHERE id = ?", (source_id,)
        ).fetchone()
        return dict(row) if row else None


def mark_source_pending(
    paths: cfg.WikiPaths, source_id: int
) -> tuple[bool, str]:
    """Reset a source's status to 'pending' so it can be re-ingested.

    Doesn't touch any wiki pages that were created from this source —
    re-ingestion will use the merge path on existing pages, or create
    new ones with new content. Useful when:

    - You blocked an earlier ingest mid-flight
    - You want Qwen to re-process the source with updated prompts
    - The previous extraction was poor quality and you want to try again
    """
    row = get_source(paths, source_id)
    if row is None:
        return False, f"No source with id {source_id}"

    file_path = source_path(paths, row["relpath"])
    if not file_path.exists():
        return False, f"Source file no longer on disk: {row['relpath']}"

    with db.connect(paths.state_db) as conn:
        conn.execute(
            "UPDATE sources SET status = 'pending', last_ingested = NULL WHERE id = ?",
            (source_id,),
        )
        conn.commit()

    return True, f"Marked #{source_id} ({row['relpath']}) as pending"


def get_source_pages(paths: cfg.WikiPaths, source_id: int) -> list[str]:
    """Wiki page paths (e.g. 'sources/foo.md') that this source contributed to."""
    with db.connect(paths.state_db) as conn:
        rows = conn.execute(
            "SELECT DISTINCT wiki_path FROM source_pages WHERE source_id = ? ORDER BY wiki_path",
            (source_id,),
        ).fetchall()
    return [r["wiki_path"] for r in rows]


def remove_source(
    paths: cfg.WikiPaths, source_id: int, delete_file: bool = True
) -> tuple[bool, str]:
    """Remove a source from tracking.

    Only *copied* files (those living under raw/) are deleted from disk;
    *referenced* files are always left in place. Returns (success, message).
    """
    row = get_source(paths, source_id)
    if row is None:
        return False, f"No source with id {source_id}"

    file_path = source_path(paths, row["relpath"])

    with db.connect(paths.state_db) as conn:
        # Cascade: remove source_pages rows first (no FK CASCADE by default)
        conn.execute("DELETE FROM source_pages WHERE source_id = ?", (source_id,))
        conn.execute("DELETE FROM ingest_runs WHERE source_id = ?", (source_id,))
        conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))

    delete_text_mirror(paths, source_id, file_path)

    deleted_file = False
    if delete_file and file_path.exists() and _is_inside_raw(file_path, paths.raw):
        try:
            file_path.unlink()
            deleted_file = True
        except OSError as e:
            return True, f"Removed #{source_id} from DB but failed to delete file: {e}"

    msg = f"Removed #{source_id} ({row['relpath']})"
    if deleted_file:
        msg += " — file deleted from raw/"
    elif delete_file:
        msg += " — referenced file left in place"
    return True, msg
