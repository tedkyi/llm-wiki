"""Command-line interface for LLM-Wiki.

Stage 1 commands:
    wiki init [PATH]         Scaffold a new wiki project.
    wiki status              Show the current wiki's stats and config.
    wiki version             Show version.

Stage 2 commands:
    wiki add <path> [-r]     Copy a file or folder into raw/, parse, track.
    wiki sources list        List all tracked sources.
    wiki sources summary     Show source counts and id ranges grouped by status.
    wiki sources show <id>   Show details for one source (with text preview).
    wiki sources rm <id>     Remove a source from tracking.

Later stages add: ingest, query, lint, serve.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from . import config as cfg
from . import credentials
from . import db
from . import ingest_llm
from . import ingest_raw
from . import lint as lint_module
from . import query as query_module
from . import scaffold
from . import search
from .llm import (
    LLMError,
    LLMRouter,
    ModelNotFound,
    OllamaNotRunning,
    make_client,
)

def _force_utf8_console() -> None:
    """Make stdout/stderr encode as UTF-8 for every ``wiki`` command.

    On Windows the console's default code page (e.g. cp1252) can't encode emoji
    or many Unicode characters, so printing them — especially when output is
    redirected to a file or pipe — raises UnicodeEncodeError and crashes the
    command. Reconfiguring to UTF-8 with ``errors="replace"`` makes output safe
    regardless of how the command is launched. Encoding-only: it does not affect
    program logic or file I/O.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # e.g. stream replaced by a test harness
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


_force_utf8_console()


app = typer.Typer(
    name="wiki",
    help="LLM-Wiki — an LLM-maintained personal knowledge base.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

sources_app = typer.Typer(
    name="sources",
    help="Manage tracked source files in raw/.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
app.add_typer(sources_app, name="sources")

providers_app = typer.Typer(
    name="providers",
    help="Manage LLM inference providers and their API keys.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
app.add_typer(providers_app, name="providers")

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hint(text: str) -> None:
    """Print a one-liner tip in a subdued style."""
    console.print(f"[dim]💡 {text}[/dim]")


def _err(text: str) -> None:
    console.print(f"[bold red]✗[/bold red] {text}")


def _ok(text: str) -> None:
    console.print(f"[bold green]✓[/bold green] {text}")


def _warn(text: str) -> None:
    console.print(f"[bold yellow]![/bold yellow] {text}")


def _resolve_root_or_die() -> cfg.WikiPaths:
    """Find the wiki project root from the cwd, or exit with a helpful error."""
    root = cfg.find_wiki_root()
    if root is None:
        _err("Not inside an LLM-Wiki project.")
        _hint("Run [bold]wiki init[/bold] in an empty folder to create one.")
        raise typer.Exit(code=1)
    return cfg.WikiPaths(root=root)


def _report_routing(router: LLMRouter, tasks: tuple[str, ...]) -> None:
    """Print which provider/model handles each of a command's tasks.

    Resolves each task through the router's own routing (so a --provider
    override or per-task mapping is reflected). Collapses to one line when every
    task resolves to the same provider.
    """
    rows = [(t, router.provider_for(t), router.model_for(t)) for t in tasks]
    distinct = {(p, m) for _, p, m in rows}
    if len(distinct) == 1:
        provider, model = next(iter(distinct))
        _ok(f"Provider ready · [bold]{provider}[/bold] · {model}")
        return
    _ok("Providers ready:")
    for task, provider, model in rows:
        console.print(f"    [dim]{task:16}[/dim] [bold]{provider}[/bold] · {model}")


def _build_router(
    paths: cfg.WikiPaths,
    *,
    provider: str | None = None,
    tasks: tuple[str, ...] = (),
    fail_hint: str | None = None,
) -> LLMRouter:
    """Build an LLMRouter from config and verify its providers are reachable.

    `provider` (the `--provider` flag) forces every task onto one profile.
    `tasks` are the pipeline tasks this command will run — reported (with their
    resolved provider/model) once the check passes. Loads vendor API keys into
    the environment first. Exits with a helpful message on any error.
    """
    credentials.prime_environment()
    config = cfg.load_config(paths)
    llm_cfg = config.get("llm", {})
    try:
        router = LLMRouter(llm_cfg, override=provider)
    except LLMError as e:
        _err(str(e))
        raise typer.Exit(code=1)

    console.print()
    console.print("[dim]Checking inference providers…[/dim]")
    try:
        router.ensure_ready()
    except (OllamaNotRunning, ModelNotFound) as e:
        _err(str(e))
        if fail_hint:
            _hint(fail_hint)
        raise typer.Exit(code=1)
    except LLMError as e:
        _err(f"LLM check failed: {e}")
        if fail_hint:
            _hint(fail_hint)
        raise typer.Exit(code=1)

    if tasks:
        _report_routing(router, tasks)
    return router


def _format_bytes(n: int) -> str:
    """Human-readable byte count: 1234 → '1.2 KB'."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def _status_style(status: str) -> str:
    return {
        "pending": "yellow",
        "ingested": "green",
        "error": "red",
        "skipped": "dim",
    }.get(status, "white")


# ---------------------------------------------------------------------------
# Stage 1 commands
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Show the LLM-Wiki version."""
    console.print(f"llm-wiki [bold cyan]{__version__}[/bold cyan]")


@app.command()
def init(
    path: Path = typer.Argument(
        Path("."),
        help="Folder to scaffold the wiki in. Defaults to the current directory.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite an existing wiki folder. Use with care.",
    ),
) -> None:
    """Scaffold a new LLM-Wiki project at PATH."""
    target = path.expanduser().resolve()

    console.print(
        Panel.fit(
            f"[bold]Initializing LLM-Wiki[/bold]\n[dim]{target}[/dim]",
            border_style="cyan",
        )
    )

    try:
        paths = scaffold.scaffold(target, force=force)
    except scaffold.ScaffoldError as e:
        _err(str(e))
        raise typer.Exit(code=1)
    except OSError as e:
        _err(f"Failed to create files: {e}")
        raise typer.Exit(code=1)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("raw/", "Drop your source documents here (PDF, MD, HTML, DOCX, TXT)")
    table.add_row("wiki/", "LLM-maintained markdown — open this in Obsidian")
    table.add_row("schema/AGENTS.md", "The rules file. Edit it as your conventions evolve.")
    table.add_row(".wiki/", "Internal state (config, SQLite). Git-ignored by default.")

    console.print()
    _ok(f"Wiki initialized at [bold]{paths.root}[/bold]")
    console.print(table)
    console.print()

    _hint(
        f"Open the vault in Obsidian: [bold]open {paths.wiki}[/bold]  "
        f"(then 'Open folder as vault')"
    )
    _hint("Add your first source with [bold]wiki add <file>[/bold]")
    _hint("Check the status anytime with [bold]wiki status[/bold]")


@app.command()
def status() -> None:
    """Show the current wiki's stats, paths, and config."""
    paths = _resolve_root_or_die()
    config = cfg.load_config(paths)
    stats = db.get_stats(paths.state_db)

    def _count_md(folder: Path) -> int:
        if not folder.exists():
            return 0
        return sum(1 for p in folder.glob("*.md") if not p.name.startswith("."))

    sources_pages = _count_md(paths.wiki / "sources")
    entities_pages = _count_md(paths.wiki / "entities")
    concepts_pages = _count_md(paths.wiki / "concepts")
    synthesis_pages = _count_md(paths.wiki / "synthesis")

    raw_files = (
        sum(1 for p in paths.raw.rglob("*") if p.is_file() and not p.name.startswith("."))
        if paths.raw.exists()
        else 0
    )

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]LLM-Wiki[/bold cyan]  [dim]@ {paths.root}[/dim]",
            border_style="cyan",
        )
    )

    table = Table(title="Project", show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", width=22)
    table.add_column()
    table.add_row("Raw sources (files)", str(raw_files))
    table.add_row("Sources tracked (DB)", str(stats["sources_total"]))
    table.add_row("Sources ingested", str(stats["sources_ingested"]))
    table.add_row("Ingest runs", str(stats["ingest_runs"]))
    console.print(table)

    pages_table = Table(title="Wiki pages", show_header=False, box=None, padding=(0, 2))
    pages_table.add_column(style="dim", width=22)
    pages_table.add_column()
    pages_table.add_row("sources/", str(sources_pages))
    pages_table.add_row("entities/", str(entities_pages))
    pages_table.add_row("concepts/", str(concepts_pages))
    pages_table.add_row("synthesis/", str(synthesis_pages))
    pages_table.add_row(
        "[bold]total[/bold]",
        f"[bold]{sources_pages + entities_pages + concepts_pages + synthesis_pages}[/bold]",
    )
    console.print(pages_table)

    cfg_table = Table(title="Config", show_header=False, box=None, padding=(0, 2))
    cfg_table.add_column(style="dim", width=22)
    cfg_table.add_column()
    llm = config.get("llm", {})
    search_cfg = config.get("search", {})
    ingest_cfg = config.get("ingest", {})
    providers = llm.get("providers", {})
    default_provider = llm.get("default_provider")

    # Each configured provider profile: model + (host/api_base) + vendor key.
    cfg_table.add_row("Providers", "")
    for name, profile in providers.items():
        model = profile.get("model", "?")
        endpoint = profile.get("host") or profile.get("api_base") or ""
        if credentials.profile_needs_key(profile):
            has = credentials.profile_has_key(profile)
            src = f"${profile['api_key_env']}" if profile.get("api_key_env") else "key"
            key_note = f"[green]{src} set[/green]" if has else f"[red]{src} missing[/red]"
        else:
            key_note = "[dim]local[/dim]"
        endpoint_note = f" · {endpoint}" if endpoint else ""
        cfg_table.add_row(f"  {name}", f"{model}{endpoint_note} · {key_note}")

    # Task → provider routing, grouped by provider (resolve each task the way
    # the router would: its mapping, falling back to default_provider).
    task_providers = llm.get("task_providers", {})
    groups: dict[str, list[str]] = {}
    for task in cfg.LLM_TASKS:
        prov = task_providers.get(task) or default_provider or "?"
        groups.setdefault(prov, []).append(task)
    routing_lines = "\n".join(
        f"[bold]{prov}[/bold]: {', '.join(tasks)}" for prov, tasks in groups.items()
    )
    cfg_table.add_row("Task routing", routing_lines)
    cfg_table.add_row("Max source chars", str(ingest_cfg.get("max_source_chars", "?")))
    cfg_table.add_row("Search backend", search_cfg.get("backend", "?"))
    cfg_table.add_row("Reranking", "on" if search_cfg.get("rerank") else "off")
    # Show QMD binary availability
    if search.is_available():
        version = search.get_version() or "installed"
        cfg_table.add_row("QMD binary", f"[green]{version}[/green]")
    else:
        cfg_table.add_row("QMD binary", "[red]not installed[/red]")
    console.print(cfg_table)

    console.print()
    if stats["sources_total"] == 0:
        _hint("No sources yet. Add one with [bold]wiki add <file>[/bold]")
    elif stats["sources_ingested"] == 0:
        _hint(
            f"{stats['sources_total']} source(s) tracked but not ingested. "
            f"Stage 3 will add [bold]wiki ingest[/bold] to process them with the LLM."
        )


# ---------------------------------------------------------------------------
# Stage 2 commands
# ---------------------------------------------------------------------------


@app.command()
def add(
    path: Path = typer.Argument(
        ...,
        help="File or folder to add. Can be anywhere on disk; it gets copied into raw/.",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="When PATH is a folder, walk it recursively.",
    ),
) -> None:
    """Copy one or more files into raw/, parse them, and register in the DB.

    Accepts a single file or a folder. Skips unsupported types and deduplicates
    by content hash. No LLM calls happen here — that's `wiki ingest` (Stage 3).
    """
    paths = _resolve_root_or_die()
    source = path.expanduser().resolve()

    if not source.exists():
        _err(f"Not found: {source}")
        raise typer.Exit(code=1)

    files_to_add = list(ingest_raw.iter_addable_files(source, recursive=recursive))

    if not files_to_add:
        if source.is_file():
            _err(f"Unsupported file type: {source.suffix or '(no extension)'}")
            _hint("Supported: .md, .txt, .pdf, .docx, .html, .htm")
        else:
            _warn(f"No supported files found in {source}")
            if not recursive and source.is_dir():
                _hint("Use [bold]--recursive[/bold] (or [bold]-r[/bold]) to walk subdirectories.")
        raise typer.Exit(code=1)

    console.print()
    console.print(
        f"Adding [bold]{len(files_to_add)}[/bold] file(s) to "
        f"[dim]{paths.raw}[/dim]"
    )
    console.print()

    added = deduped = skipped = errored = 0

    for file_path in files_to_add:
        outcome = ingest_raw.add_file(paths, file_path)

        if outcome.result == ingest_raw.AddResult.ADDED:
            added += 1
            console.print(
                f"  [green]+[/green] [bold]#{outcome.source_id}[/bold] "
                f"[cyan]{outcome.relpath}[/cyan]  "
                f"[dim]{outcome.title}[/dim]"
            )
            console.print(
                f"      [dim]{outcome.file_type} · {_format_bytes(outcome.bytes)} · "
                f"{outcome.word_count:,} words[/dim]"
            )
        elif outcome.result == ingest_raw.AddResult.DEDUPED:
            deduped += 1
            console.print(
                f"  [yellow]≈[/yellow] [dim]{file_path.name}[/dim]  "
                f"[dim]→ {outcome.message}[/dim]"
            )
        elif outcome.result == ingest_raw.AddResult.SKIPPED_EMPTY:
            skipped += 1
            console.print(
                f"  [yellow]![/yellow] [dim]{file_path.name}[/dim]  "
                f"[yellow]{outcome.message}[/yellow]"
            )
        elif outcome.result == ingest_raw.AddResult.SKIPPED_UNSUPPORTED:
            skipped += 1
            console.print(
                f"  [dim]-[/dim] [dim]{file_path.name}  ({outcome.message})[/dim]"
            )
        else:
            errored += 1
            console.print(
                f"  [red]✗[/red] [dim]{file_path.name}[/dim]  "
                f"[red]{outcome.message}[/red]"
            )

    console.print()
    summary_parts = []
    if added:
        summary_parts.append(f"[green]{added} added[/green]")
    if deduped:
        summary_parts.append(f"[yellow]{deduped} deduped[/yellow]")
    if skipped:
        summary_parts.append(f"[yellow]{skipped} skipped[/yellow]")
    if errored:
        summary_parts.append(f"[red]{errored} errors[/red]")
    console.print("  " + " · ".join(summary_parts))
    console.print()

    if added:
        _hint("See everything with [bold]wiki sources list[/bold]")


@sources_app.command("list")
def sources_list_cmd(
    status_filter: str = typer.Option(
        None,
        "--status",
        "-s",
        help="Only show sources with this status (pending|ingested|error).",
    ),
    tail: int = typer.Option(
        None,
        "--tail",
        "-n",
        help="Only show the last N sources.",
    ),
) -> None:
    """List all tracked sources."""
    paths = _resolve_root_or_die()
    rows = ingest_raw.list_sources(paths, status_filter=status_filter)

    if not rows:
        console.print()
        if status_filter:
            _warn(f"No sources with status '{status_filter}'")
        else:
            _warn("No sources tracked yet.")
            _hint("Add one with [bold]wiki add <file>[/bold]")
        return

    total = len(rows)
    if tail is not None:
        rows = rows[-tail:]

    table = Table(
        title=f"Sources ({len(rows)} of {total})" if tail is not None else f"Sources ({total})",
        show_header=True,
        header_style="bold",
        row_styles=["", "dim"],
    )
    table.add_column("#", justify="right", style="cyan", width=5)
    table.add_column("Type", width=6)
    table.add_column("Size", justify="right", width=9)
    table.add_column("Added", width=10)
    table.add_column("Status", width=9)
    table.add_column("Path", overflow="fold")

    for row in rows:
        added_short = row["added_at"][:10] if row["added_at"] else ""
        status = row["status"]
        status_styled = f"[{_status_style(status)}]{status}[/{_status_style(status)}]"
        table.add_row(
            str(row["id"]),
            row["file_type"],
            _format_bytes(row["bytes"]),
            added_short,
            status_styled,
            row["relpath"],
        )

    console.print()
    console.print(table)
    console.print()


@sources_app.command("summary")
def sources_summary_cmd() -> None:
    """Show source counts and id ranges grouped by status."""
    paths = _resolve_root_or_die()
    rows = ingest_raw.summarize_sources(paths)

    if not rows:
        console.print()
        _warn("No sources tracked yet.")
        _hint("Add one with [bold]wiki add <file>[/bold]")
        return

    table = Table(
        title="Sources by status",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Status", width=9)
    table.add_column("Count", justify="right", width=7)
    table.add_column("Min ID", justify="right", width=7)
    table.add_column("Max ID", justify="right", width=7)

    total = 0
    for row in rows:
        status = row["status"]
        status_styled = f"[{_status_style(status)}]{status}[/{_status_style(status)}]"
        table.add_row(
            status_styled,
            str(row["count"]),
            str(row["min_id"]),
            str(row["max_id"]),
        )
        total += row["count"]

    console.print()
    console.print(table)
    console.print(f"  [dim]total: {total}[/dim]")
    console.print()


@sources_app.command("show")
def sources_show_cmd(
    source_id: int = typer.Argument(..., help="The source ID (from `wiki sources list`)."),
    preview_chars: int = typer.Option(
        800,
        "--preview",
        "-p",
        help="Number of characters of parsed text to preview.",
    ),
) -> None:
    """Show details for one source, including a text preview."""
    paths = _resolve_root_or_die()
    row = ingest_raw.get_source(paths, source_id)
    if row is None:
        _err(f"No source with id {source_id}")
        raise typer.Exit(code=1)

    # Re-parse the file to get title and a preview — we don't store parsed text
    # in the DB to keep it small. This is cheap (local file).
    from . import parsers

    file_path = paths.root / row["relpath"]
    if not file_path.exists():
        _err(f"Source file missing from disk: {file_path}")
        raise typer.Exit(code=1)

    try:
        parsed = parsers.parse(file_path)
    except parsers.ParserError as e:
        _err(f"Parse failed: {e}")
        raise typer.Exit(code=1)

    console.print()
    console.print(
        Panel.fit(
            f"[bold]#{row['id']}[/bold]  [cyan]{parsed.title}[/cyan]",
            border_style="cyan",
        )
    )

    meta_table = Table(show_header=False, box=None, padding=(0, 2))
    meta_table.add_column(style="dim", width=16)
    meta_table.add_column()
    meta_table.add_row("Path", row["relpath"])
    meta_table.add_row("Type", parsed.file_type)
    meta_table.add_row("Size", _format_bytes(row["bytes"]))
    meta_table.add_row("Words", f"{parsed.word_count:,}")
    meta_table.add_row("Added", row["added_at"])
    meta_table.add_row("Status", f"[{_status_style(row['status'])}]{row['status']}[/{_status_style(row['status'])}]")
    meta_table.add_row("Hash", row["content_hash"][:16] + "…")
    if row["last_ingested"]:
        meta_table.add_row("Last ingested", row["last_ingested"])
    for k, v in parsed.metadata.items():
        meta_table.add_row(k, str(v)[:80])
    console.print(meta_table)

    console.print()
    console.print("[dim]── text preview ──[/dim]")
    preview = parsed.text[:preview_chars]
    if len(parsed.text) > preview_chars:
        preview += f"\n\n[dim]… ({len(parsed.text) - preview_chars:,} more characters)[/dim]"
    console.print(preview)
    console.print()


@sources_app.command("rm")
def sources_rm_cmd(
    source_id: int = typer.Argument(..., help="The source ID to remove."),
    keep_file: bool = typer.Option(
        False,
        "--keep-file",
        help="Only remove from tracking; don't delete the file from raw/.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt.",
    ),
) -> None:
    """Remove a source from tracking (and optionally delete the file)."""
    paths = _resolve_root_or_die()
    row = ingest_raw.get_source(paths, source_id)
    if row is None:
        _err(f"No source with id {source_id}")
        raise typer.Exit(code=1)

    if not yes:
        action = "remove from tracking" if keep_file else "remove from tracking AND delete file"
        confirm = typer.confirm(
            f"About to {action}: #{source_id} {row['relpath']}. Proceed?"
        )
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(code=0)

    ok, msg = ingest_raw.remove_source(paths, source_id, delete_file=not keep_file)
    if ok:
        _ok(msg)
    else:
        _err(msg)
        raise typer.Exit(code=1)


@providers_app.command("list")
def providers_list() -> None:
    """List configured provider profiles, task routing, and API-key status."""
    paths = _resolve_root_or_die()
    credentials.prime_environment()
    config = cfg.load_config(paths)
    llm = config.get("llm", {})
    providers = llm.get("providers", {})
    default_provider = llm.get("default_provider", "?")
    task_providers = llm.get("task_providers", {})

    if not providers:
        _err("No providers configured in .wiki/config.yml.")
        raise typer.Exit(code=1)

    table = Table(title="Inference providers", box=None, padding=(0, 2))
    table.add_column("Provider", style="bold")
    table.add_column("Type", style="dim")
    table.add_column("Model")
    table.add_column("Endpoint", style="dim")
    table.add_column("API key")
    for name, profile in providers.items():
        model = profile.get("model", "?")
        endpoint = profile.get("host") or profile.get("api_base") or "-"
        if credentials.profile_needs_key(profile):
            has = credentials.profile_has_key(profile)
            src = f"${profile['api_key_env']}" if profile.get("api_key_env") else "set"
            if profile.get("api_key_env"):
                key_note = f"[green]{src}[/green]" if has else f"[red]{src} (unset)[/red]"
            else:
                key_note = "[green]set[/green]" if has else "[red]missing[/red]"
        else:
            key_note = "[dim]not needed[/dim]"
        table.add_row(name, profile.get("type", "?"), model, endpoint, key_note)
    console.print()
    console.print(table)

    # Task routing — resolve each task the way the router would.
    console.print()
    routing = Table(title="Task -> provider routing", box=None, padding=(0, 2))
    routing.add_column("Task", style="dim")
    routing.add_column("Provider")
    for task in cfg.LLM_TASKS:
        routing.add_row(task, task_providers.get(task) or default_provider or "?")
    console.print(routing)


def _key_handle(name: str) -> str:
    """Map a CLI key name to its keyring handle.

    A known vendor slug (anthropic, openai, …) maps to its canonical env-var
    name; anything else is treated as a literal env-var handle (e.g. a
    profile's `api_key_env` like OPENROUTER_API_KEY).
    """
    name = name.strip()
    if name in credentials.VENDOR_ENV_VARS:
        return credentials.env_var_for(name)
    return name


@providers_app.command("set-key")
def providers_set_key(
    name: str = typer.Argument(
        ..., help="Vendor slug (anthropic, openai) or an env-var name (e.g. OPENROUTER_API_KEY)."
    ),
) -> None:
    """Store an API key in the OS keyring (prompted, hidden input)."""
    handle = _key_handle(name)
    key = typer.prompt(f"API key for '{handle}'", hide_input=True)
    if not key.strip():
        _err("No key entered.")
        raise typer.Exit(code=1)
    try:
        credentials.set_named_key(handle, key.strip())
    except RuntimeError as e:
        _err(str(e))
        raise typer.Exit(code=1)
    _ok(f"Stored API key '{handle}' in the OS keyring.")


@providers_app.command("rm-key")
def providers_rm_key(
    name: str = typer.Argument(
        ..., help="Vendor slug or env-var name whose stored key to remove."
    ),
) -> None:
    """Remove a stored API key from the keyring."""
    handle = _key_handle(name)
    credentials.delete_named_key(handle)
    _ok(f"Removed any stored API key '{handle}'.")


@providers_app.command("test")
def providers_test(
    provider: str = typer.Argument(..., help="Provider profile name to test (from `wiki providers list`)."),
) -> None:
    """Check that a provider profile is reachable / authenticated."""
    paths = _resolve_root_or_die()
    credentials.prime_environment()
    config = cfg.load_config(paths)
    providers = config.get("llm", {}).get("providers", {})
    if provider not in providers:
        _err(f"Unknown provider '{provider}'. Configured: {', '.join(providers) or '(none)'}.")
        raise typer.Exit(code=1)
    client = make_client(providers[provider])
    console.print()
    console.print(f"[dim]Testing '{provider}' ({client.model or 'default'})…[/dim]")
    try:
        message = client.probe()
    except (OllamaNotRunning, ModelNotFound) as e:
        _err(str(e))
        raise typer.Exit(code=1)
    except LLMError as e:
        _err(f"Provider check failed: {e}")
        raise typer.Exit(code=1)
    finally:
        client.close()
    _ok(f"Provider '{provider}': {message}")


@app.command()
def reingest(
    source_id: int = typer.Argument(..., help="The source ID to reset (from `wiki sources list`)."),
) -> None:
    """Reset a source to 'pending' so it will be re-ingested on the next `wiki ingest` run."""
    paths = _resolve_root_or_die()
    ok, msg = ingest_raw.mark_source_pending(paths, source_id)
    if ok:
        _ok(msg)
        _hint("Run [bold]wiki ingest[/bold] to process it.")
    else:
        _err(msg)
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Stage 3 — LLM ingest
# ---------------------------------------------------------------------------


class CliIngestCallbacks(ingest_llm.IngestCallbacks):
    """Rich terminal rendering of ingest progress."""

    def __init__(self, mode: str = "interactive") -> None:
        self.mode = mode
        self._stream_active = False
        self._stream_char_count = 0

    def on_start(self, source_id: int, source_title: str, file_path: str) -> None:
        console.print()
        console.rule(f"[bold cyan]Source #{source_id}[/bold cyan]  {source_title}")
        console.print(f"[dim]{file_path}[/dim]")

    def on_parsing(self) -> None:
        console.print("[dim]  parsing…[/dim]")

    def on_extracting(self) -> None:
        console.print("[dim]  extracting entities and concepts (thinking mode)…[/dim]")

    def on_extracted(self, extraction: ingest_llm.Extraction) -> None:
        console.print()
        console.print(f"[bold]Title:[/bold] {extraction.title}")
        console.print(f"[bold]Slug:[/bold]  [cyan]{extraction.source_slug}[/cyan]")
        console.print()
        console.print(f"[bold]Summary:[/bold]")
        console.print(f"  [dim]{extraction.summary}[/dim]")
        console.print()

        if extraction.key_takeaways:
            console.print("[bold]Key takeaways:[/bold]")
            for t in extraction.key_takeaways:
                console.print(f"  [dim]•[/dim] {t}")
            console.print()

        if extraction.entities:
            console.print(f"[bold]Entities[/bold] ({len(extraction.entities)}):")
            for e in extraction.entities:
                console.print(
                    f"  [green]+[/green] [cyan]{e.slug}[/cyan] [dim]({e.type})[/dim]  {e.name}"
                )
            console.print()

        if extraction.concepts:
            console.print(f"[bold]Concepts[/bold] ({len(extraction.concepts)}):")
            for c in extraction.concepts:
                console.print(
                    f"  [green]+[/green] [cyan]{c.slug}[/cyan]  {c.name}"
                )
            console.print()

        if extraction.tags:
            console.print(f"[bold]Tags:[/bold] [dim]{', '.join(extraction.tags)}[/dim]")
            console.print()

    def on_verbose_log_path(self, log_path: str) -> None:
        console.print(f"[dim]  verbose log → {log_path}[/dim]")

    def on_extraction_failed(self, error: str) -> None:
        _warn(f"Extraction returned bad JSON. Retrying: {error[:200]}")

    def ask_confirm(self, extraction: ingest_llm.Extraction) -> bool:
        if self.mode == "batch":
            return True
        total_pages = len(extraction.entities) + len(extraction.concepts) + 1
        return typer.confirm(
            f"File these? Will create/update ~{total_pages} wiki pages.",
            default=True,
        )

    def on_drafting_page(self, kind: str, slug: str, operation: str) -> None:
        console.print()
        op_color = "green" if operation == "created" else "yellow"
        console.print(
            f"[{op_color}]{operation}[/{op_color}] [dim]{kind}[/dim] "
            f"[cyan]{slug}[/cyan]"
        )
        console.print("[dim]┄[/dim]" * 60)
        self._stream_active = True
        self._stream_char_count = 0

    def on_stream_chunk(self, chunk: str) -> None:
        if self._stream_active:
            # markup=False: raw LLM text (e.g. [[wikilinks]]) is not rich markup —
            # otherwise rich parses [..] as style tags and drops the contents.
            console.print(chunk, end="", style="dim", highlight=False, markup=False)
            self._stream_char_count += len(chunk)

    def on_page_written(self, page: ingest_llm.PageChange) -> None:
        if self._stream_active:
            console.print()
            console.print("[dim]┄[/dim]" * 60)
            self._stream_active = False

    def on_finalizing(self) -> None:
        console.print()
        console.print("[dim]finalizing (committing files, rebuilding index, appending log)…[/dim]")

    def on_complete(self, result: ingest_llm.IngestResult) -> None:
        console.print()
        if result.skipped:
            _warn(f"Skipped by user: {result.source_title}")
            return
        _ok(
            f"Ingested [bold]{result.source_title}[/bold] — "
            f"{result.pages_created} created, {result.pages_updated} updated"
        )

    def on_error(self, error: str) -> None:
        console.print()
        _err(error)


@app.command()
def ingest(
    source_id: Optional[int] = typer.Argument(
        None,
        help="Specific source ID to ingest. If omitted, processes all pending sources.",
    ),
    batch: bool = typer.Option(
        False,
        "--batch",
        help="Skip the interactive confirmation prompt for each source.",
    ),
    no_discover: bool = typer.Option(
        False,
        "--no-discover",
        help="Don't auto-scan raw/ for untracked files before ingesting.",
    ),
    no_thinking: bool = typer.Option(
        False,
        "--no-thinking",
        help="Disable Qwen3 thinking mode in Pass 1 (faster, slightly lower quality).",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Force a specific provider profile for this run (overrides per-task routing).",
    ),
    verbose_log: bool = typer.Option(
        True,
        "--verbose-log/--no-verbose-log",
        help="Write a full diagnostic log for each source to .wiki/logs/ (raw LLM responses, timings, errors). On by default.",
    ),
) -> None:
    """Ingest pending sources: extract entities/concepts, write wiki pages.

    The pipeline runs three LLM passes per source:
      1. Extraction (thinking mode) — structured JSON
      2. Page drafting — one page per entity/concept with streaming output
      3. Source summary page

    Then rebuilds index.md and appends to log.md.
    """
    paths = _resolve_root_or_die()
    config = cfg.load_config(paths)
    ingest_cfg = config.get("ingest", {})
    max_source_chars = ingest_cfg.get("max_source_chars", ingest_llm.MAX_SOURCE_CHARS)

    router = _build_router(
        paths,
        provider=provider,
        tasks=("ingest_extract", "ingest_extract_retry", "ingest_draft"),
    )

    mode = "batch" if batch else "interactive"
    thinking = not no_thinking

    try:
        if source_id is not None:
            # Single source
            cb = CliIngestCallbacks(mode=mode)
            result = ingest_llm.ingest_source(
                paths,
                source_id,
                router,
                cb,
                max_source_chars=max_source_chars,
                mode=mode,
                thinking_for_extraction=thinking,
                verbose_log=verbose_log,
            )
            results = [result]
        else:
            # All pending (with auto-discovery)
            results = ingest_llm.ingest_pending(
                paths,
                router,
                lambda: CliIngestCallbacks(mode=mode),
                max_source_chars=max_source_chars,
                mode=mode,
                auto_discover=not no_discover,
                thinking_for_extraction=thinking,
                verbose_log=verbose_log,
            )
    finally:
        router.close()

    if not results:
        console.print()
        _warn("No pending sources to ingest.")
        _hint("Add sources with [bold]wiki add <file>[/bold] first.")
        return

    console.print()
    console.rule("[bold]Ingest summary[/bold]")
    ok_count = sum(1 for r in results if r.ok)
    skipped_count = sum(1 for r in results if r.skipped)
    error_count = sum(1 for r in results if r.error)
    total_created = sum(r.pages_created for r in results)
    total_updated = sum(r.pages_updated for r in results)

    parts = []
    if ok_count:
        parts.append(f"[green]{ok_count} ingested[/green]")
    if skipped_count:
        parts.append(f"[yellow]{skipped_count} skipped[/yellow]")
    if error_count:
        parts.append(f"[red]{error_count} errors[/red]")
    console.print("  " + " · ".join(parts))
    console.print(
        f"  [dim]total pages: {total_created} created, {total_updated} updated[/dim]"
    )
    console.print()

    if ok_count > 0:
        # Rebuild FTS immediately (fast), then embed in background.
        # BM25 keyword search is usable right away; vector search catches up
        # while hints are shown. The embed thread is non-daemon so the process
        # stays alive until it finishes — no orphaned GPU work.
        if search.is_available():
            console.print()
            console.print("[dim]Updating search index…[/dim]")
            try:
                search.update_fts(paths)
                search.schedule_embed(paths)
                _ok("BM25 index ready; vector embeddings building in background")
            except search.SearchBackendError as e:
                _warn(f"Search index update failed: {e}")
                _hint("You can retry manually later — ingest is already committed.")
        else:
            _hint(
                "qmd not installed — search index not updated. "
                "Install: [bold]npm install -g @tobilu/qmd[/bold]"
            )

        console.print()
        _hint("Open the vault in Obsidian and check the graph view!")
        _hint("Run [bold]wiki status[/bold] to see updated page counts.")
        _hint("Ask a question with [bold]wiki query \"<your question>\"[/bold]")


# ---------------------------------------------------------------------------
# Stage 4 — query
# ---------------------------------------------------------------------------


class CliQueryCallbacks(query_module.QueryCallbacks):
    """Rich terminal rendering of query progress."""

    def __init__(self) -> None:
        self._stream_active = False

    def on_start(self, question: str, mode: str) -> None:
        console.print()
        console.rule(f"[bold cyan]Query[/bold cyan]")
        console.print(f"[bold]Q:[/bold] {question}")
        console.print(f"[dim]mode: {mode}[/dim]")
        console.print()

    def on_classifying_intent(self) -> None:
        console.print("[dim]  understanding question…[/dim]")

    def on_intent_classified(self, intent: str) -> None:
        if intent == "chitchat":
            console.print("[dim]  → casual question, no search needed[/dim]")
        else:
            console.print("[dim]  → wiki question, searching…[/dim]")

    def on_chitchat_reply(self, reply: str) -> None:
        console.print()
        console.print("[dim]" + "─" * 72 + "[/dim]")
        console.print(reply, markup=False)  # raw LLM text, not rich markup
        console.print("[dim]" + "─" * 72 + "[/dim]")
        console.print()

    def on_searching(self) -> None:
        console.print("[dim]  searching wiki (BM25 + vector + rerank)…[/dim]")

    def on_search_done(self, results) -> None:
        if not len(results):
            return
        console.print(f"[dim]  found {len(results)} relevant page(s):[/dim]")
        for i, hit in enumerate(results.hits, start=1):
            score_color = (
                "green" if hit.score > 0.7 else "yellow" if hit.score > 0.4 else "dim"
            )
            path = hit.full_path or hit.path
            title = hit.title or path
            console.print(
                f"    [dim]{i}.[/dim] [{score_color}]{hit.score:.2f}[/{score_color}] "
                f"[cyan]{path}[/cyan]  [dim]{title[:60]}[/dim]"
            )

    def on_no_results(self) -> None:
        console.print()
        _warn("No matching wiki pages found.")

    def on_synthesizing(self) -> None:
        console.print()
        console.print("[dim]  synthesizing answer…[/dim]")
        console.print("[dim]" + "─" * 72 + "[/dim]")
        self._stream_active = True

    def on_stream_chunk(self, chunk: str) -> None:
        if self._stream_active:
            # markup=False so [[wikilinks]] aren't parsed as rich style tags.
            console.print(chunk, end="", highlight=False, markup=False)

    def on_saved(self, saved_path: str) -> None:
        if self._stream_active:
            console.print()
            self._stream_active = False
        console.print()
        _ok(f"Saved answer as [cyan]{saved_path}[/cyan]")

    def on_complete(self, result) -> None:
        if self._stream_active:
            console.print()
            console.print("[dim]" + "─" * 72 + "[/dim]")
            self._stream_active = False
        console.print()

    def on_error(self, error: str) -> None:
        if self._stream_active:
            console.print()
            self._stream_active = False
        console.print()
        _err(error)


@app.command()
def query(
    question: str = typer.Argument(..., help="The question to ask the wiki."),
    mode: str = typer.Option(
        "hybrid",
        "--mode",
        help="Search mode: hybrid | lex | vec",
    ),
    lex: bool = typer.Option(
        False,
        "--lex",
        help="Shortcut for --mode lex (BM25 only, fastest).",
    ),
    vec: bool = typer.Option(
        False,
        "--vec",
        help="Shortcut for --mode vec (vector only).",
    ),
    limit: int = typer.Option(
        8, "--limit", "-n", help="Max number of search hits to consider."
    ),
    min_score: float = typer.Option(
        0.0, "--min-score", help="Drop hits below this score."
    ),
    no_rerank: bool = typer.Option(
        False, "--no-rerank", help="Skip LLM reranking in hybrid mode."
    ),
    save_as: Optional[str] = typer.Option(
        None,
        "--save-as",
        help="Save the answer as wiki/synthesis/<slug>.md and update the index.",
    ),
    scope: str = typer.Option(
        "wiki",
        "--scope",
        help="Search scope: wiki (LLM-summarized pages), raw (extracted source text), or hybrid (both).",
    ),
    types: Optional[str] = typer.Option(
        None,
        "--types",
        help=(
            "Comma-separated wiki page types to search: entities, concepts, "
            "sources, synthesis. Default: all types."
        ),
    ),
    intent_classify: bool = typer.Option(
        False,
        "--intent-classify/--no-intent-classify",
        help="Classify intent before searching so pure chitchat ('hi', 'thanks') "
        "skips retrieval. Off by default — it's slow and rarely worth the wait.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Force a specific provider profile for this query (overrides per-task routing).",
    ),
    grounding: str = typer.Option(
        "strict",
        "--grounding",
        help="strict: answer only from wiki sources. augmented: also draw on "
        "the model's own knowledge, clearly labeling what isn't source-backed.",
    ),
) -> None:
    """Ask a question: search the wiki, synthesize an answer with citations.

    The query pipeline runs:
      1. QMD search (BM25 + vector + rerank by default)
      2. Load the top N matching pages
      3. Qwen3 synthesizes a cited answer, streamed to the terminal
      4. Optionally save the answer as a synthesis page
    """
    paths = _resolve_root_or_die()

    # Resolve search mode shortcuts
    if lex:
        mode = "lex"
    elif vec:
        mode = "vec"
    if mode not in ("hybrid", "lex", "vec"):
        _err(f"Invalid mode '{mode}'. Use hybrid, lex, or vec.")
        raise typer.Exit(code=1)

    if scope not in ("wiki", "raw", "hybrid"):
        _err(f"Invalid scope '{scope}'. Use wiki, raw, or hybrid.")
        raise typer.Exit(code=1)

    if grounding not in ("strict", "augmented"):
        _err(f"Invalid grounding '{grounding}'. Use strict or augmented.")
        raise typer.Exit(code=1)

    page_types: list[str] | None = None
    if types:
        page_types = [t.strip().lower() for t in types.split(",") if t.strip()]
        invalid = [t for t in page_types if t not in cfg.WIKI_SUBDIRS]
        if invalid:
            _err(
                f"Invalid page type(s): {', '.join(invalid)}. "
                f"Valid types: {', '.join(cfg.WIKI_SUBDIRS)}"
            )
            raise typer.Exit(code=1)

    # Sanity checks
    if not search.is_available():
        _err("qmd is not installed.")
        _hint("Install it with: [bold]npm install -g @tobilu/qmd[/bold]")
        raise typer.Exit(code=1)

    # Warn if wiki is empty
    wiki_pages = sum(
        1
        for sub in ("entities", "concepts", "sources", "synthesis")
        for _ in (paths.wiki / sub).glob("*.md")
        if (paths.wiki / sub).exists()
    )
    if wiki_pages == 0:
        _err("Wiki has no pages yet.")
        _hint("Run [bold]wiki ingest[/bold] first to create pages.")
        raise typer.Exit(code=1)

    # Build the provider router and verify reachability
    router = _build_router(
        paths,
        provider=provider,
        tasks=("query_intent", "query_chitchat", "query_synthesis"),
    )

    callbacks = CliQueryCallbacks()
    try:
        result = query_module.run_query(
            paths,
            router,
            question,
            callbacks,
            mode=mode,
            limit=limit,
            min_score=min_score,
            rerank=not no_rerank,
            save_as=save_as,
            scope=scope,
            page_types=page_types,
            classify_intent_first=intent_classify,
            grounding=grounding,
        )
    finally:
        router.close()

    if result.error and not result.answer:
        raise typer.Exit(code=1)


@app.command()
def reindex(
    full: bool = typer.Option(
        False,
        "--full",
        help=(
            "Nuke the QMD index and rebuild from scratch: backfill missing "
            "raw-text mirrors, delete index.sqlite (models are kept), "
            "re-register collections, then update + embed."
        ),
    ),
) -> None:
    """Force a rebuild of the QMD search index.

    Normally indexing runs automatically after `wiki ingest`, so you only
    need this if the index gets out of sync (e.g. you edited wiki pages
    manually). Use --full to migrate wikis whose index predates raw-text/
    mirrors — the old index contains embedded PDF binary garbage that only
    a rebuild removes.
    """
    paths = _resolve_root_or_die()
    if not search.is_available():
        _err("qmd is not installed.")
        _hint("Install it with: [bold]npm install -g @tobilu/qmd[/bold]")
        raise typer.Exit(code=1)

    embed_timeout = search.QMD_TIMEOUT_LONG

    if full:
        # 1. Make sure every tracked source has an extracted-text mirror,
        #    since the rebuilt raw collection indexes raw-text/ only.
        console.print()
        console.print("[dim]Backfilling raw-text mirrors from tracked sources…[/dim]")

        from rich.progress import Progress

        with Progress(console=console, transient=True) as prog:
            task = prog.add_task("Parsing sources", total=None)

            def _on_progress(relpath: str, i: int, total: int) -> None:
                prog.update(task, total=total, completed=i, description=f"Parsing {relpath}")

            stats = ingest_raw.backfill_text_mirrors(paths, progress=_on_progress)

        _ok(
            f"Mirrors: {stats.written} written, "
            f"{stats.skipped_existing} already present, "
            f"{stats.missing_file} missing source files"
        )
        for err in stats.parse_errors:
            _warn(f"Parse failed: {err}")

        # 2. Drop the old index (keeps downloaded GGUF models).
        console.print("[dim]Deleting old QMD index…[/dim]")
        if search.reset_index(paths):
            _ok("Old index deleted")
        else:
            _ok("No existing index found — starting fresh")

        # A from-scratch embed touches every chunk; give it more headroom.
        # `qmd embed` is incremental, so a timeout doesn't lose progress —
        # re-running reindex continues where it stopped.
        embed_timeout = 7200.0

    console.print()
    console.print("[dim]Rebuilding search index (this may take a while)…[/dim]")
    try:
        search.update_index(paths, embed=True, embed_timeout=embed_timeout)
        _ok("Search index rebuilt (BM25 + vector embeddings)")
    except search.SearchBackendError as e:
        _err(f"Index rebuild failed: {e}")
        if full:
            _hint("Progress is kept — re-run [bold]wiki reindex[/bold] to continue embedding.")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Stage 5 — lint
# ---------------------------------------------------------------------------


_SEVERITY_STYLE = {
    lint_module.Severity.ERROR: ("bold red", "✗"),
    lint_module.Severity.WARNING: ("yellow", "!"),
    lint_module.Severity.INFO: ("cyan", "i"),
}


def _render_lint_report_terminal(report: lint_module.LintReport) -> None:
    """Pretty-print a LintReport to the terminal using Rich."""
    console.print()
    # Summary panel
    score = report.health_score
    score_color = "green" if score >= 80 else "yellow" if score >= 50 else "red"
    summary_lines = [
        f"[bold]Health score:[/bold] [{score_color}]{score}/100[/{score_color}]",
        f"[bold]Pages checked:[/bold] {report.pages_checked}",
        f"[bold]Duration:[/bold] {report.duration_seconds:.2f}s",
        "",
        f"  [red]{len(report.errors)} errors[/red]"
        f" · [yellow]{len(report.warnings)} warnings[/yellow]"
        f" · [cyan]{len(report.infos)} infos[/cyan]",
    ]
    if report.auto_fixed:
        summary_lines.append(f"  [green]auto-fixed: {report.auto_fixed}[/green]")
    console.print(
        Panel.fit("\n".join(summary_lines), title="Lint Report", border_style="cyan")
    )

    if not report.issues:
        console.print()
        _ok("No issues found. Your wiki is in good shape!")
        return

    # Group by severity → display
    def _render_group(
        title: str, issues: list[lint_module.LintIssue], color: str
    ) -> None:
        if not issues:
            return
        console.print()
        console.rule(f"[bold {color}]{title} ({len(issues)})[/bold {color}]")

        # Group by page for cleaner reading
        by_page: dict[str, list[lint_module.LintIssue]] = {}
        for issue in issues:
            by_page.setdefault(issue.page, []).append(issue)

        for page, page_issues in by_page.items():
            console.print(f"\n  [cyan]{page}[/cyan]")
            for issue in page_issues:
                style, glyph = _SEVERITY_STYLE[issue.severity]
                # Escape any brackets in the message since wikilinks look
                # like Rich markup tags
                from rich.markup import escape as _escape
                safe_msg = _escape(issue.message)
                safe_suggestion = _escape(issue.suggestion) if issue.suggestion else ""
                console.print(
                    f"    [{style}]{glyph}[/{style}] "
                    f"[dim]{issue.check.value}:[/dim] {safe_msg}"
                )
                if safe_suggestion:
                    console.print(f"      [dim]→ {safe_suggestion}[/dim]")
                if issue.fixable:
                    console.print(
                        f"      [green dim]✓ auto-fixable[/green dim]"
                    )

    _render_group("Errors", report.errors, "red")
    _render_group("Warnings", report.warnings, "yellow")
    _render_group("Info", report.infos, "cyan")


@app.command()
def lint(
    deep: bool = typer.Option(
        False,
        "--deep",
        help="Run LLM-powered contradiction detection (slower, requires Ollama).",
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Auto-fix trivial issues (malformed wikilinks, noise in sources).",
    ),
    save: bool = typer.Option(
        False,
        "--save",
        help="Save the report as wiki/synthesis/lint-report-YYYY-MM-DD.md",
    ),
    max_pairs: int = typer.Option(
        10,
        "--max-pairs",
        help="Max page pairs to check in --deep mode.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Force a specific provider profile for --deep checks.",
    ),
) -> None:
    """Lint the wiki for broken links, orphans, missing pages, and more.

    Fast checks (default) run entirely in Python and finish in seconds.
    Use --deep to also scan page pairs for contradictions using Qwen3.
    """
    paths = _resolve_root_or_die()

    # If --deep, build the provider router and verify reachability up front
    router: Optional[LLMRouter] = None
    if deep:
        router = _build_router(
            paths,
            provider=provider,
            tasks=("lint_deep",),
            fail_hint="Fast-only lint still works without an LLM — omit --deep.",
        )

    try:
        console.print()
        if deep:
            console.print("[dim]Running fast checks + contradiction detection…[/dim]")
        else:
            console.print("[dim]Running fast checks…[/dim]")

        report = lint_module.run_lint(paths, deep=deep, router=router)
    finally:
        if router is not None:
            router.close()

    # Auto-fix before displaying
    if fix:
        fixed_count = lint_module.apply_fixes(paths, report.issues)
        report.auto_fixed = fixed_count
        if fixed_count > 0:
            # Rebuild inventory + re-run checks so the report reflects post-fix state
            report = lint_module.run_lint(paths, deep=False, router=None)
            report.auto_fixed = fixed_count

    _render_lint_report_terminal(report)

    # Save as a synthesis page
    if save:
        today = lint_module.page_writer.today_iso()  # reuse helper
        slug = f"lint-report-{today}"
        target_path = paths.wiki / "synthesis" / f"{slug}.md"
        content = lint_module.render_report_markdown(report, paths)
        lint_module.page_writer.write_page(target_path, content)
        # Rebuild index so the new page shows up
        lint_module.page_writer.rebuild_index(paths, today)
        console.print()
        _ok(f"Saved report to [cyan]synthesis/{slug}.md[/cyan]")

    # Exit code: 1 if there are errors, 0 otherwise (for CI use)
    if report.errors:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Stage 6 — web UI
# ---------------------------------------------------------------------------


@app.command()
def serve(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Bind address. Default 127.0.0.1 (localhost only).",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to listen on.",
    ),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Don't auto-open the browser on startup.",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Auto-reload on code changes (development only).",
    ),
) -> None:
    """Start the LLM-Wiki web UI on localhost.

    The UI provides a dashboard, source browser, ingest interface, query
    interface, and lint dashboard. It runs locally and binds to 127.0.0.1
    by default — no external access, no auth required.
    """
    paths = _resolve_root_or_die()

    # Verify the project looks healthy before launching
    if not paths.wiki.exists():
        _err(f"Wiki folder not found at {paths.wiki}")
        _hint("Run `wiki init` first to scaffold the project.")
        raise typer.Exit(code=1)

    # Lazy import — avoid loading FastAPI/uvicorn unless we're actually serving
    try:
        import uvicorn

        from .webapp.main import create_app
    except ImportError as e:
        _err(f"Web UI dependencies not installed: {e}")
        _hint("Install with: uv pip install -e .")
        raise typer.Exit(code=1)

    app_instance = create_app(paths)

    url = f"http://{host}:{port}"
    console.print()
    console.print(
        Panel.fit(
            f"[bold]LLM-Wiki[/bold] web UI starting…\n\n"
            f"  URL: [bold cyan]{url}[/bold cyan]\n"
            f"  Project: [dim]{paths.root}[/dim]\n\n"
            f"[dim]Press Ctrl+C to stop.[/dim]",
            title="🚀 Serve",
            border_style="cyan",
        )
    )

    # Open browser shortly after the server starts
    if not no_browser:
        import threading
        import time
        import webbrowser

        def _open_browser() -> None:
            time.sleep(1.2)  # let uvicorn finish binding
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=_open_browser, daemon=True).start()

    # Run uvicorn. We pass the app instance directly (not a string) so the
    # paths injection survives. Reload mode requires a string import path,
    # so it's not supported here — that's fine, the wiki is small and a
    # restart is instant.
    if reload:
        _hint("--reload not supported with paths injection; ignoring.")

    try:
        uvicorn.run(
            app_instance,
            host=host,
            port=port,
            log_level="warning",  # quiet — we don't need every request logged
            access_log=False,
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


@app.command()
def mcp(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Bind address. Default 127.0.0.1 (localhost only). A non-loopback "
        "address exposes the wiki to your network with no authentication.",
    ),
    port: int = typer.Option(
        8010,
        "--port",
        "-p",
        help="Port to listen on. Default 8010 (distinct from the web UI's 8000).",
    ),
    path: str = typer.Option(
        "/mcp",
        "--path",
        help="HTTP path the MCP endpoint is mounted at.",
    ),
    root: Optional[str] = typer.Option(
        None,
        "--root",
        help="Wiki project root. Defaults to auto-discovery from the current dir.",
    ),
) -> None:
    """Serve the wiki over MCP (Model Context Protocol) via Streamable HTTP.

    Lets other LLM sessions (Claude Desktop/Code, Cursor, custom agents) query
    this wiki with four read-only tools: search_wiki, read_wiki_page, ask_wiki,
    and wiki_status. Binds to localhost by default.
    """
    # Lazy import — keep the MCP SDK off the path for every other command.
    try:
        from .mcp import adapter, server
    except ImportError as e:
        _err(f"MCP dependencies not installed: {e}")
        _hint("Install with: uv pip install -e .")
        raise typer.Exit(code=1)

    try:
        ctx = adapter.bind_project(root)
    except adapter.WikiError as e:
        _err(str(e))
        _hint("Run `wiki init` in an empty folder to create a project.")
        raise typer.Exit(code=1)

    url = f"http://{host}:{port}{path}"
    console.print()
    console.print(
        Panel.fit(
            f"[bold]LLM-Wiki[/bold] MCP server starting…\n\n"
            f"  URL: [bold cyan]{url}[/bold cyan]\n"
            f"  Transport: [dim]streamable-http[/dim]\n"
            f"  Tools: [dim]search_wiki, read_wiki_page, ask_wiki, wiki_status[/dim]\n"
            f"  Project: [dim]{ctx.paths.root}[/dim]\n\n"
            f"[dim]Press Ctrl+C to stop.[/dim]",
            title="🔌 MCP",
            border_style="cyan",
        )
    )
    if host not in ("127.0.0.1", "localhost", "::1"):
        _warn(f"Binding to {host} — the wiki is reachable from the network with no auth.")

    mcp_server = server.build_server(ctx, host=host, port=port, path=path)
    try:
        mcp_server.run(transport="streamable-http")
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


def main() -> None:
    """Entry point used by the `wiki` console script."""
    app()


if __name__ == "__main__":
    main()
