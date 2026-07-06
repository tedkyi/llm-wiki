"""Settings route — configure LLM providers, task routing, and API keys.

  GET  /settings                    — render the settings form
  POST /settings                    — save provider profiles + routing to config.yml
  POST /settings/key                — store a vendor API key in the OS keyring
  POST /settings/key/delete         — remove a vendor API key
  POST /settings/test/{provider}    — check a provider profile is reachable (JSON)

Non-secret settings live in `.wiki/config.yml`; API keys are per-vendor and
stored in the OS keyring (never written to config).
"""

from __future__ import annotations

import json
import re
import shutil

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ... import config as cfg
from ... import credentials
from ...llm import LLMError, ModelNotFound, OllamaNotRunning, make_client

router = APIRouter()

# Profile keys handled by dedicated form fields; everything else (sampling,
# thinking, per-task overrides) is edited via the "advanced" JSON textarea.
_DEDICATED_KEYS = ("type", "model", "host", "api_base", "api_key_env")

_NAME_RE = re.compile(r"[^a-z0-9_-]+")


def _slug_name(name: str) -> str:
    """Sanitize a provider name into a safe config key / URL token."""
    return _NAME_RE.sub("-", (name or "").strip().lower()).strip("-")


def _settings_context(paths: cfg.WikiPaths, **extra) -> dict:
    config = cfg.load_config(paths)
    llm = config.get("llm", {})
    providers = llm.get("providers", {})
    task_providers = llm.get("task_providers", {})
    default_provider = llm.get("default_provider", "")

    profile_views = []
    for name, profile in providers.items():
        advanced = {k: v for k, v in profile.items() if k not in _DEDICATED_KEYS}
        vendor = credentials.vendor_for_model(profile.get("model", ""))
        ptype = profile.get("type", "")
        is_cli = ptype in cfg.CLI_PROVIDER_TYPES
        api_key_env = profile.get("api_key_env", "")
        # Key mode: "cli" (agent login), "env", "vendor", or "none".
        if is_cli:
            key_mode = "cli"
        elif api_key_env:
            key_mode = "env"
        elif credentials.needs_key(vendor):
            key_mode = "vendor"
        else:
            key_mode = "none"
        # For agent-CLI providers, resolve the binary path (the analog of a URL).
        cli_path = shutil.which("claude") if ptype == "claude-code" else None
        profile_views.append(
            {
                "name": name,
                "type": ptype,
                "is_cli": is_cli,
                "has_endpoint": not is_cli,
                "cli_path": cli_path,
                "isolated": bool(profile.get("isolated", True)),
                "model": profile.get("model", ""),
                "endpoint": profile.get("host") or profile.get("api_base") or "",
                "endpoint_label": "Host" if ptype == "ollama-native" else "API base",
                "api_key_env": api_key_env,
                "advanced_json": json.dumps(advanced, indent=2, ensure_ascii=False),
                "vendor": vendor or "",
                "key_mode": key_mode,
                "has_key": credentials.profile_has_key(profile),
            }
        )

    # Distinct keyring handles across all profiles (vendor default or named
    # api_key_env), so the key-management section shows each key once.
    key_handles: dict[str, dict] = {}
    for name, profile in providers.items():
        handle = credentials.key_handle_for_profile(profile)
        if not handle or handle in key_handles:
            continue
        label = f"${handle}" if profile.get("api_key_env") else (
            credentials.vendor_for_model(profile.get("model", "")) or handle
        )
        key_handles[handle] = {
            "handle": handle,
            "label": label,
            "has_key": credentials.has_named_key(handle),
        }

    ctx = {
        "page": "settings",
        "providers": profile_views,
        "provider_names": list(providers.keys()),
        "key_handles": list(key_handles.values()),
        "default_provider": default_provider,
        "tasks": [(t, task_providers.get(t, "")) for t in cfg.LLM_TASKS],
        "type_presets": [
            {
                "id": pid,
                "label": p["label"],
                "endpoint_field": p["endpoint_field"],
                "endpoint_label": p["endpoint_label"],
                "endpoint_placeholder": p["endpoint_placeholder"],
                "cli": p["type"] in cfg.CLI_PROVIDER_TYPES,
            }
            for pid, p in cfg.PROVIDER_TYPE_PRESETS.items()
        ],
    }
    ctx.update(extra)
    return ctx


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(
    request: Request, saved: int = 0, error: str | None = None
) -> HTMLResponse:
    """Render the provider settings form."""
    paths: cfg.WikiPaths = request.app.state.wiki_paths
    ctx = _settings_context(paths, saved=bool(saved), error=error)
    return request.app.state.templates.TemplateResponse(request, "settings.html", ctx)


@router.post("/settings")
async def save_settings(request: Request) -> RedirectResponse:
    """Persist default provider, task routing, and profile edits to config.yml."""
    paths: cfg.WikiPaths = request.app.state.wiki_paths
    form = await request.form()
    config = cfg.load_config(paths)
    llm = config.setdefault("llm", {})
    providers = llm.get("providers", {})

    # Default provider
    default_provider = form.get("default_provider")
    if default_provider:
        llm["default_provider"] = default_provider

    # Task → provider routing
    task_providers = llm.setdefault("task_providers", {})
    for task in cfg.LLM_TASKS:
        value = form.get(f"task__{task}")
        if value:
            task_providers[task] = value

    # Per-provider: model, endpoint, and advanced (sampling/thinking/tasks) JSON
    errors: list[str] = []
    for name, profile in providers.items():
        model = form.get(f"model__{name}")
        if model is not None:
            profile["model"] = model.strip()
        # Isolation toggle for agent-CLI providers (checkbox: absent = off).
        if profile.get("type") in cfg.CLI_PROVIDER_TYPES:
            profile["isolated"] = f"iso__{name}" in form
        endpoint = form.get(f"endpoint__{name}")
        if endpoint is not None:
            field = "host" if profile.get("type") == "ollama-native" else "api_base"
            endpoint = endpoint.strip()
            if endpoint:
                profile[field] = endpoint
            else:
                profile.pop(field, None)
        key_env = form.get(f"keyenv__{name}")
        if key_env is not None:
            key_env = key_env.strip()
            if key_env:
                profile["api_key_env"] = key_env
            else:
                profile.pop("api_key_env", None)
        advanced = form.get(f"advanced__{name}")
        if advanced is not None and advanced.strip():
            try:
                parsed = json.loads(advanced)
                if not isinstance(parsed, dict):
                    raise ValueError("must be a JSON object")
            except (json.JSONDecodeError, ValueError) as e:
                errors.append(f"{name}: invalid advanced settings ({e})")
                continue
            # Replace all non-dedicated keys with the edited set.
            for key in [k for k in profile if k not in _DEDICATED_KEYS]:
                del profile[key]
            profile.update(parsed)

    if errors:
        return RedirectResponse(
            url=f"/settings?error={'; '.join(errors)}", status_code=303
        )

    cfg.save_config(paths, config)
    return RedirectResponse(url="/settings?saved=1", status_code=303)


@router.post("/settings/providers/add")
async def add_provider(
    request: Request,
    name: str = Form(...),
    type: str = Form(...),
    model: str = Form(...),
    endpoint: str = Form(""),
    api_key_env: str = Form(""),
) -> RedirectResponse:
    """Add a new provider profile from the UI presets."""
    paths: cfg.WikiPaths = request.app.state.wiki_paths
    slug = _slug_name(name)
    if not slug:
        return RedirectResponse(url="/settings?error=Provider+name+is+required", status_code=303)

    config = cfg.load_config(paths)
    providers = config.setdefault("llm", {}).setdefault("providers", {})
    if slug in providers:
        return RedirectResponse(
            url=f"/settings?error=A+provider+named+'{slug}'+already+exists", status_code=303
        )
    try:
        profile = cfg.build_provider_profile(type, model, endpoint, api_key_env)
    except ValueError as e:
        return RedirectResponse(url=f"/settings?error={e}", status_code=303)

    providers[slug] = profile
    cfg.save_config(paths, config)
    return RedirectResponse(url="/settings?saved=1", status_code=303)


@router.post("/settings/providers/delete")
async def delete_provider(
    request: Request, name: str = Form(...)
) -> RedirectResponse:
    """Remove a provider profile, reassigning anything that referenced it."""
    paths: cfg.WikiPaths = request.app.state.wiki_paths
    config = cfg.load_config(paths)
    llm = config.setdefault("llm", {})
    providers = llm.get("providers", {})
    if name not in providers:
        return RedirectResponse(url=f"/settings?error=No+provider+'{name}'", status_code=303)
    if len(providers) <= 1:
        return RedirectResponse(
            url="/settings?error=Cannot+remove+the+last+provider", status_code=303
        )

    del providers[name]
    remaining = list(providers.keys())
    fallback = remaining[0]

    # Repoint the default and any tasks that referenced the deleted provider.
    if llm.get("default_provider") == name:
        llm["default_provider"] = fallback
    new_default = llm.get("default_provider", fallback)
    task_providers = llm.get("task_providers", {})
    for task, prov in list(task_providers.items()):
        if prov == name:
            task_providers[task] = new_default

    cfg.save_config(paths, config)
    return RedirectResponse(url="/settings?saved=1", status_code=303)


@router.post("/settings/key")
async def set_key(
    handle: str = Form(...), api_key: str = Form(...)
) -> RedirectResponse:
    """Store an API key in the OS keyring under a handle (env-var name)."""
    if not api_key.strip():
        return RedirectResponse(url="/settings?error=empty+key", status_code=303)
    try:
        credentials.set_named_key(handle, api_key.strip())
    except RuntimeError as e:
        return RedirectResponse(url=f"/settings?error={e}", status_code=303)
    return RedirectResponse(url="/settings?saved=1", status_code=303)


@router.post("/settings/key/delete")
async def delete_key(handle: str = Form(...)) -> RedirectResponse:
    """Remove an API key from the keyring by handle."""
    credentials.delete_named_key(handle)
    return RedirectResponse(url="/settings?saved=1", status_code=303)


@router.post("/settings/test/{provider}")
async def test_provider(request: Request, provider: str) -> JSONResponse:
    """Check that a provider profile is reachable/authenticated."""
    paths: cfg.WikiPaths = request.app.state.wiki_paths
    credentials.prime_environment()
    providers = cfg.load_config(paths).get("llm", {}).get("providers", {})
    if provider not in providers:
        return JSONResponse({"ok": False, "message": f"Unknown provider '{provider}'."})
    client = make_client(providers[provider])
    try:
        # probe() does the most meaningful available check per provider type
        # (Ollama reachability, key presence, or a live agent-CLI auth round-trip).
        message = client.probe()
    except (OllamaNotRunning, ModelNotFound, LLMError) as e:
        return JSONResponse({"ok": False, "message": str(e)})
    finally:
        client.close()
    return JSONResponse({"ok": True, "message": message})
