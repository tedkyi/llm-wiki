"""FastAPI app factory for the LLM-Wiki web UI.

The app is bound to a single wiki project at creation time — the WikiPaths
are injected via `create_app()` so routes can access the current project
without re-resolving it on every request.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import __version__
from .. import config as cfg


def _get_template_dir() -> Path:
    """Locate the webapp/templates directory packaged with llm_wiki."""
    return Path(__file__).parent / "templates"


def _get_static_dir() -> Path:
    """Locate the webapp/static directory packaged with llm_wiki."""
    return Path(__file__).parent / "static"


def create_app(paths: cfg.WikiPaths) -> FastAPI:
    """Build a FastAPI app bound to the given wiki project.

    Args:
        paths: Resolved wiki project paths. All routes will use these
               instead of re-walking the filesystem per request.

    Returns:
        A FastAPI application ready to be served with uvicorn.
    """
    app = FastAPI(
        title="LLM-Wiki",
        version=__version__,
        description="Local LLM-maintained personal wiki",
        docs_url=None,  # disable /docs in prod — this is a personal tool
        redoc_url=None,
    )

    # Stash the paths on the app state so routes can read it via request.app.state
    app.state.wiki_paths = paths
    app.state.version = __version__

    # Templates
    template_dir = _get_template_dir()
    templates = Jinja2Templates(directory=str(template_dir))
    app.state.templates = templates

    # Static files
    static_dir = _get_static_dir()
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Routes — import lazily to avoid circular dependencies
    from .routes import dashboard, graph, ingest, lint, query, settings, sources

    app.include_router(dashboard.router)
    app.include_router(sources.router)
    app.include_router(graph.router)
    app.include_router(lint.router)
    app.include_router(query.router)
    app.include_router(ingest.router)
    app.include_router(settings.router)

    return app
