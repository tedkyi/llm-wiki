"""Per-vendor API-key management for cloud inference providers.

Keys are keyed by *vendor* (anthropic, openai, …), not by provider profile —
LiteLLM reads them from the environment by model-string prefix, so two profiles
pointing at the same vendor share one key. Resolution order on read:

1. Process environment (populated from a gitignored `.env` via python-dotenv,
   or set directly in the shell).
2. OS keyring (Windows Credential Manager / macOS Keychain / Linux Secret
   Service). On startup any keyring-stored key whose env var is unset is copied
   into `os.environ` so LiteLLM finds it without an explicit `api_key` argument.

Local providers (ollama, vllm) need no key. On headless boxes without a keyring
backend, set the env var / use `.env` instead — `set_key` degrades gracefully.
"""

from __future__ import annotations

import os

import keyring
from keyring.errors import KeyringError
from dotenv import load_dotenv

# Service name under which keys are stored in the OS keyring.
_KEYRING_SERVICE = "llm-wiki"

# Vendors we know how to authenticate, mapped to the env var LiteLLM reads.
# The key of this dict is the vendor slug used everywhere (CLI, keyring, UI).
VENDOR_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xai": "XAI_API_KEY",
}

# Vendors that run locally and never need a key.
LOCAL_VENDORS = frozenset({"ollama", "ollama_chat", "vllm", "lm_studio", "hosted_vllm"})

_dotenv_loaded = False


def vendor_for_model(model: str) -> str | None:
    """Derive the vendor slug from a LiteLLM model string.

    `anthropic/claude-opus-4-8` -> `anthropic`; a bare `gpt-4o` (no prefix)
    can't be attributed, so returns None.
    """
    if not model:
        return None
    prefix = model.split("/", 1)[0] if "/" in model else ""
    return prefix or None


def needs_key(vendor: str | None) -> bool:
    """Whether a vendor requires an API key (cloud vendors do, local ones don't)."""
    if not vendor:
        return False
    return vendor not in LOCAL_VENDORS


def env_var_for(vendor: str) -> str:
    """Canonical env-var / keyring-account name for a vendor.

    This name is the single handle a key is stored and looked up under, so a
    vendor's key and a profile's own `api_key_env` are resolved the same way.
    """
    return VENDOR_ENV_VARS.get(vendor, f"LLM_WIKI_{vendor.upper()}_API_KEY")


def load_dotenv_once() -> None:
    """Load a `.env` file into the environment exactly once per process."""
    global _dotenv_loaded
    if not _dotenv_loaded:
        load_dotenv()
        _dotenv_loaded = True


# ---------------------------------------------------------------------------
# Named keys — the keyring is keyed by env-var name, so a vendor default and a
# user-specified `api_key_env` are stored/looked up through one mechanism.
# ---------------------------------------------------------------------------


def get_named_key(name: str | None) -> str | None:
    """Resolve a key by handle: environment variable first, then keyring."""
    if not name:
        return None
    load_dotenv_once()
    env_val = os.environ.get(name)
    if env_val:
        return env_val
    try:
        return keyring.get_password(_KEYRING_SERVICE, name)
    except KeyringError:
        return None


def has_named_key(name: str | None) -> bool:
    """True if a key is available under this handle (env or keyring)."""
    return bool(get_named_key(name))


def set_named_key(name: str, key: str) -> None:
    """Store a key in the keyring under `name` and export it for this process.

    Raises RuntimeError with an actionable message when no keyring backend is
    available (e.g. a headless server) so callers can point the user at `.env`.
    """
    try:
        keyring.set_password(_KEYRING_SERVICE, name, key)
    except KeyringError as e:
        raise RuntimeError(
            f"No OS keyring backend is available to store '{name}'. "
            f"Set {name} in your environment or a gitignored .env file instead."
        ) from e
    os.environ[name] = key


def delete_named_key(name: str) -> None:
    """Remove a key by handle from the keyring (and this process's env)."""
    try:
        keyring.delete_password(_KEYRING_SERVICE, name)
    except KeyringError:
        pass
    os.environ.pop(name, None)


# ---------------------------------------------------------------------------
# Vendor + profile convenience wrappers over the named-key layer.
# ---------------------------------------------------------------------------


def get_key(vendor: str | None) -> str | None:
    """Return a vendor's API key (via its canonical env-var handle)."""
    return get_named_key(env_var_for(vendor)) if vendor else None


def has_key(vendor: str | None) -> bool:
    """True if a key is available for the vendor from either source."""
    return bool(get_key(vendor))


def set_key(vendor: str, key: str) -> None:
    """Persist an API key for a vendor under its canonical env-var handle."""
    set_named_key(env_var_for(vendor), key)


def delete_key(vendor: str) -> None:
    """Remove a vendor's API key from the keyring (and this process's env)."""
    delete_named_key(env_var_for(vendor))


def resolve_profile_key(profile: dict) -> str | None:
    """Resolve a provider profile's API key.

    A profile may name its own env var via `api_key_env` (shared freely across
    profiles, survives key rotation); otherwise the key is resolved per-vendor
    from the model prefix. Both check the environment first, then the keyring
    under the same handle.
    """
    env_name = profile.get("api_key_env")
    if env_name:
        return get_named_key(env_name)
    return get_key(vendor_for_model(profile.get("model", "")))


def profile_needs_key(profile: dict) -> bool:
    """Whether a profile requires an API key (explicit env var, or cloud vendor)."""
    from .config import CLI_PROVIDER_TYPES

    if profile.get("type") in CLI_PROVIDER_TYPES:
        return False  # agent CLIs authenticate via their own login, no key
    if profile.get("api_key_env"):
        return True
    return needs_key(vendor_for_model(profile.get("model", "")))


def profile_has_key(profile: dict) -> bool:
    """True if a profile's API key is available from its configured source."""
    return bool(resolve_profile_key(profile))


def key_handle_for_profile(profile: dict) -> str | None:
    """The env-var/keyring handle a profile's key is stored under, or None."""
    from .config import CLI_PROVIDER_TYPES

    if profile.get("type") in CLI_PROVIDER_TYPES:
        return None
    env_name = profile.get("api_key_env")
    if env_name:
        return env_name
    vendor = vendor_for_model(profile.get("model", ""))
    return env_var_for(vendor) if needs_key(vendor) else None


def prime_environment() -> None:
    """Load `.env` and copy any keyring-stored vendor keys into `os.environ`.

    Call once at process startup (CLI and web) so LiteLLM can read vendor keys
    from the environment without an explicit `api_key` argument. Env vars that
    are already set win — the keyring only fills gaps. Profiles with a named
    `api_key_env` are resolved on demand via `get_named_key`, so they don't need
    priming here.
    """
    load_dotenv_once()
    for env_var in VENDOR_ENV_VARS.values():
        if os.environ.get(env_var):
            continue
        try:
            stored = keyring.get_password(_KEYRING_SERVICE, env_var)
        except KeyringError:
            stored = None
        if stored:
            os.environ[env_var] = stored
