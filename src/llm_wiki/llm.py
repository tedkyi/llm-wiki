"""LLM clients and provider routing.

Every completion in the app goes through a client exposing a single shared
surface — `chat` / `chat_stream` / `ensure_ready` / `list_models` — so callers
stay provider-agnostic. Two implementations exist:

- `OllamaClient` — the original hand-rolled httpx wrapper over Ollama's native
  `/api/chat` (Qwen3 `/think` inline tags, `<think>` stripping, streaming).
- `LiteLLMClient` — wraps `litellm.completion`, reaching any LiteLLM-supported
  backend (Ollama via its OpenAI path, vLLM, Anthropic, OpenAI, …).

Provider *profiles* live in config under `llm.providers.<name>`; a `make_client`
factory turns a profile into the right client, and `LLMRouter` maps each pipeline
task to a provider (per `llm.task_providers`) and hands back the client plus that
profile's resolved sampling options.

This module is intentionally stateless per call — higher layers (`ingest_llm.py`)
orchestrate multi-pass pipelines.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from typing import Generator

import httpx

from .config import PROVIDER_RESERVED_KEYS
from . import credentials

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.6:35b"

# Suppress the console-window flash when spawning a CLI subprocess on Windows.
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
DEFAULT_TIMEOUT = 300.0  # 5 minutes — thinking-mode extraction can be slow

_OLLAMA_VENDORS = ("ollama", "ollama_chat")


def resolve_llm_options(profile: dict, task: str | None = None, **overrides) -> dict:
    """Build a sampling-options dict from a provider profile.

    Precedence (lowest to highest): the profile's non-reserved top-level keys
    (its sampling defaults, in that provider's own parameter vocabulary), then
    `profile.tasks.<task>`, then any explicit non-None keyword overrides.
    Reserved keys (`type`, `model`, `host`, …) are excluded — see
    `config.PROVIDER_RESERVED_KEYS`.
    """
    options = {k: v for k, v in profile.items() if k not in PROVIDER_RESERVED_KEYS}
    if task:
        task_cfg = profile.get("tasks", {}).get(task, {})
        options.update(task_cfg)
    options.update({k: v for k, v in overrides.items() if v is not None})
    return options


class LLMError(Exception):
    """Raised when an LLM call fails in a user-recoverable way."""


class OllamaNotRunning(LLMError):
    """A local inference server (Ollama) is not reachable."""


class ModelNotFound(LLMError):
    """The requested model isn't available on the backend."""


@dataclass
class ChatMessage:
    role: str  # 'system' | 'user' | 'assistant'
    content: str


# ---------------------------------------------------------------------------
# Shared message / thinking helpers
# ---------------------------------------------------------------------------


def _to_wire(messages: list[ChatMessage]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _apply_qwen_think_tag(messages: list[dict], thinking: bool) -> list[dict]:
    """Append Qwen3's `/think` or `/no_think` tag to the last user message.

    This is the *native* Ollama convention only — the LiteLLM path uses the
    `think` boolean parameter instead (the inline tag is not honored there).
    """
    tag = "\n\n/think" if thinking else "\n\n/no_think"
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            messages[i] = {**messages[i], "content": messages[i]["content"] + tag}
            break
    return messages


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking_text(text: str) -> str:
    """Remove `<think>...</think>` blocks from a completed response."""
    if "<think>" not in text:
        return text
    return _THINK_RE.sub("", text).strip()


class _ThinkStripper:
    """Stateful stripper of `<think>...</think>` spans across streamed chunks."""

    def __init__(self) -> None:
        self._in = False

    def feed(self, chunk: str) -> str:
        visible = ""
        i = 0
        while i < len(chunk):
            if not self._in:
                start = chunk.find("<think>", i)
                if start == -1:
                    visible += chunk[i:]
                    break
                visible += chunk[i:start]
                self._in = True
                i = start + len("<think>")
            else:
                end = chunk.find("</think>", i)
                if end == -1:
                    break  # rest of chunk is thinking
                self._in = False
                i = end + len("</think>")
        return visible


# ---------------------------------------------------------------------------
# Ollama health helpers (shared by the native client and the LiteLLM checker)
# ---------------------------------------------------------------------------


def _ollama_ping(host: str, timeout: float = 5.0) -> bool:
    try:
        r = httpx.get(f"{host}/api/tags", timeout=timeout)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def _ollama_models(host: str, timeout: float = 5.0) -> list[str]:
    try:
        r = httpx.get(f"{host}/api/tags", timeout=timeout)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]
    except httpx.ConnectError as e:
        raise OllamaNotRunning(
            f"Cannot connect to Ollama at {host}. Is the Ollama app running?"
        ) from e
    except httpx.HTTPError as e:
        raise LLMError(f"Ollama error: {e}") from e


def _ensure_ollama_model(host: str, model: str) -> None:
    if not _ollama_ping(host):
        raise OllamaNotRunning(
            f"Ollama isn't reachable at {host}.\n"
            f"Start it by opening the Ollama app, or run `ollama serve`."
        )
    models = _ollama_models(host)
    # Match by exact name or prefix (ollama may return "qwen3:14b-q4_K_M").
    if not any(m == model or m.startswith(model) for m in models):
        raise ModelNotFound(
            f"Model '{model}' not found in Ollama.\n"
            f"Available: {', '.join(models) if models else '(none)'}\n"
            f"Pull it with: ollama pull {model}"
        )


# ---------------------------------------------------------------------------
# Native Ollama client
# ---------------------------------------------------------------------------


class OllamaClient:
    """Minimal, synchronous Ollama client tuned for the LLM-Wiki use case."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # -- health -------------------------------------------------------------

    def ping(self) -> bool:
        """True if Ollama is reachable. Does not check model availability."""
        return _ollama_ping(self.host)

    def list_models(self) -> list[str]:
        """List models available in this Ollama instance."""
        return _ollama_models(self.host)

    def ensure_ready(self) -> None:
        """Verify Ollama is running and the configured model is available."""
        _ensure_ollama_model(self.host, self.model)

    def probe(self) -> str:
        """Verify reachability for the settings 'Test' button; return a message."""
        self.ensure_ready()
        return f"Reachable at {self.host} · model '{self.model}' available."

    # -- chat (non-streaming) ----------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        thinking: bool = False,
        json_mode: bool = False,
        **options,
    ) -> str:
        """Non-streaming chat. Returns the full assistant message content.

        Thinking mode is controlled via Qwen3's `/think` / `/no_think` inline
        tags. `**options` are Ollama sampling options (temperature, top_k,
        num_ctx, num_predict, …) passed straight through.
        """
        payload_messages = _apply_qwen_think_tag(_to_wire(messages), thinking)
        payload = {
            "model": self.model,
            "messages": payload_messages,
            "stream": False,
            "options": {k: v for k, v in options.items() if v is not None},
        }
        if json_mode:
            payload["format"] = "json"

        try:
            r = self._client.post(f"{self.host}/api/chat", json=payload)
            r.raise_for_status()
        except httpx.ConnectError as e:
            raise OllamaNotRunning(f"Cannot connect to Ollama at {self.host}.") from e
        except httpx.HTTPStatusError as e:
            body = e.response.text
            if "not found" in body.lower() or e.response.status_code == 404:
                raise ModelNotFound(
                    f"Model '{self.model}' not found. "
                    f"Pull it with: ollama pull {self.model}"
                ) from e
            raise LLMError(f"Ollama error {e.response.status_code}: {body}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama request failed: {e}") from e

        data = r.json()
        content = data.get("message", {}).get("content", "")
        return _strip_thinking_text(content)

    # -- chat (streaming) ---------------------------------------------------

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        thinking: bool = False,
        **options,
    ) -> Generator[str, None, str]:
        """Streaming chat. Yields visible content chunks; returns the full text.

        Use `yield from` to forward chunks while capturing the final result:

            full = yield from client.chat_stream(messages)
        """
        payload_messages = _apply_qwen_think_tag(_to_wire(messages), thinking)
        payload = {
            "model": self.model,
            "messages": payload_messages,
            "stream": True,
            "options": {k: v for k, v in options.items() if v is not None},
        }

        full_content: list[str] = []
        stripper = _ThinkStripper()

        try:
            with self._client.stream(
                "POST", f"{self.host}/api/chat", json=payload
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        visible = stripper.feed(chunk)
                        if visible:
                            full_content.append(visible)
                            yield visible
                    if data.get("done"):
                        break
        except httpx.ConnectError as e:
            raise OllamaNotRunning(f"Cannot connect to Ollama at {self.host}.") from e
        except httpx.HTTPStatusError as e:
            body = e.response.read().decode(errors="replace") if e.response else ""
            raise LLMError(f"Ollama error {e.response.status_code}: {body}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama streaming failed: {e}") from e

        return "".join(full_content)


# ---------------------------------------------------------------------------
# LiteLLM client (Ollama-OpenAI, vLLM, Anthropic, OpenAI, …)
# ---------------------------------------------------------------------------

_litellm = None


def _get_litellm():
    """Import and configure litellm lazily (keeps native-only runs fast)."""
    global _litellm
    if _litellm is None:
        import litellm

        litellm.telemetry = False  # no outbound telemetry calls
        litellm.drop_params = True  # drop params a given provider doesn't support
        litellm.suppress_debug_info = True
        _litellm = litellm
    return _litellm


class LiteLLMClient:
    """Provider-agnostic client backed by `litellm.completion`.

    Exposes the same surface as `OllamaClient`. Sampling `**options` are
    forwarded to litellm as-is (each provider profile uses its own vocabulary,
    e.g. Ollama's `num_predict` vs OpenAI's `max_tokens`); unsupported params are
    dropped by litellm. Thinking is toggled via the backend's native mechanism:
    Ollama's `think` boolean, or `reasoning_effort` elsewhere.
    """

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key_env: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.model = model
        self.api_base = api_base.rstrip("/") if api_base else None
        self.api_key_env = api_key_env or None
        self.timeout = timeout
        self.vendor = credentials.vendor_for_model(model)

    def close(self) -> None:
        pass

    def __enter__(self) -> "LiteLLMClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # -- health -------------------------------------------------------------

    def list_models(self) -> list[str]:
        if self.vendor in _OLLAMA_VENDORS and self.api_base:
            return _ollama_models(self.api_base)
        return [self.model]

    def ensure_ready(self) -> None:
        """Best-effort readiness check without spending a paid completion."""
        if self.vendor in _OLLAMA_VENDORS:
            host = self.api_base or DEFAULT_HOST
            bare_model = self.model.split("/", 1)[1] if "/" in self.model else self.model
            _ensure_ollama_model(host, bare_model)
        elif self.api_key_env:
            if not credentials.get_named_key(self.api_key_env):
                raise LLMError(
                    f"No key found for '{self.api_key_env}'. Set it in your "
                    f"environment / .env file, or store it with "
                    f"`wiki providers set-key {self.api_key_env}`."
                )
        elif credentials.needs_key(self.vendor):
            if not credentials.has_key(self.vendor):
                raise LLMError(
                    f"No API key configured for '{self.vendor}'.\n"
                    f"Set one with `wiki providers set-key {self.vendor}` "
                    f"or add it to a .env file."
                )
        elif self.api_base:
            # Local OpenAI-compatible server (vLLM / LM Studio): light ping.
            try:
                httpx.get(f"{self.api_base}/models", timeout=5.0)
            except httpx.HTTPError as e:
                raise OllamaNotRunning(
                    f"Cannot reach the inference server at {self.api_base}."
                ) from e

    def probe(self) -> str:
        """Verify reachability/auth for the settings 'Test' button."""
        self.ensure_ready()
        where = f" at {self.api_base}" if self.api_base else ""
        return f"Ready · {self.model}{where}."

    # -- request assembly ---------------------------------------------------

    def _thinking_kwargs(self, thinking: bool) -> dict:
        if self.vendor in _OLLAMA_VENDORS:
            return {"think": bool(thinking)}
        # Portable fallback: explicitly disable reasoning when unwanted; when
        # wanted, leave the model/provider default in place.
        return {} if thinking else {"reasoning_effort": "none"}

    def _common_kwargs(self, options: dict) -> dict:
        kwargs = {k: v for k, v in options.items() if v is not None}
        if self.api_base:
            kwargs["api_base"] = self.api_base
        # A profile-named env var overrides LiteLLM's per-vendor key lookup, so
        # pass the key explicitly when one is configured.
        if self.api_key_env:
            key = credentials.get_named_key(self.api_key_env)
            if key:
                kwargs["api_key"] = key
        kwargs["timeout"] = self.timeout
        return kwargs

    def _translate_error(self, e: Exception) -> LLMError:
        litellm = _get_litellm()
        if isinstance(e, litellm.AuthenticationError):
            return LLMError(
                f"Authentication failed for '{self.vendor}'. Check the API key."
            )
        if isinstance(e, litellm.NotFoundError):
            return ModelNotFound(f"Model '{self.model}' not found for this provider.")
        if isinstance(e, (litellm.APIConnectionError, litellm.Timeout)):
            target = self.api_base or self.vendor or "the provider"
            return OllamaNotRunning(f"Cannot reach {target}: {e}")
        return LLMError(f"{self.vendor or 'LLM'} request failed: {e}")

    # -- chat (non-streaming) ----------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        thinking: bool = False,
        json_mode: bool = False,
        **options,
    ) -> str:
        litellm = _get_litellm()
        kwargs = self._common_kwargs(options)
        kwargs.update(self._thinking_kwargs(thinking))
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = litellm.completion(
                model=self.model,
                messages=_to_wire(messages),
                stream=False,
                **kwargs,
            )
        except litellm.LITELLM_EXCEPTION_TYPES as e:  # type: ignore[attr-defined]
            raise self._translate_error(e) from e
        except Exception as e:  # pragma: no cover - defensive
            raise LLMError(f"{self.vendor or 'LLM'} request failed: {e}") from e
        content = resp.choices[0].message.content or ""
        return _strip_thinking_text(content)

    # -- chat (streaming) ---------------------------------------------------

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        thinking: bool = False,
        **options,
    ) -> Generator[str, None, str]:
        litellm = _get_litellm()
        kwargs = self._common_kwargs(options)
        kwargs.update(self._thinking_kwargs(thinking))

        full_content: list[str] = []
        stripper = _ThinkStripper()
        try:
            stream = litellm.completion(
                model=self.model,
                messages=_to_wire(messages),
                stream=True,
                **kwargs,
            )
            for part in stream:
                delta = part.choices[0].delta
                chunk = getattr(delta, "content", None) or ""
                if chunk:
                    visible = stripper.feed(chunk)
                    if visible:
                        full_content.append(visible)
                        yield visible
        except litellm.LITELLM_EXCEPTION_TYPES as e:  # type: ignore[attr-defined]
            raise self._translate_error(e) from e
        except Exception as e:  # pragma: no cover - defensive
            raise LLMError(f"{self.vendor or 'LLM'} streaming failed: {e}") from e

        return "".join(full_content)


# ---------------------------------------------------------------------------
# Local agent-CLI transport (Claude Code / Codex) — no API key, rides the CLI's
# own OAuth login. Spawns the CLI in headless stream-json mode and reads events.
# ---------------------------------------------------------------------------


class _AuthError(LLMError):
    """The agent CLI is installed but not logged in."""


# Substrings (case-insensitive) in CLI stderr/result that mean "not logged in".
_CLI_AUTH_PATTERNS = ("log in", "login", "unauthenticated", "not authenticated",
                      "authentication", "oauth", "invalid api key", "please run `claude`",
                      "401", "unauthorized", "missing bearer")


def _resolve_codex_binary() -> str | None:
    """Locate the `codex` executable, which is not always on PATH.

    On Windows the OpenAI Codex CLI installs a versioned binary under
    %LOCALAPPDATA%\\OpenAI\\Codex\\bin\\<hash>\\codex.exe rather than on PATH.
    """
    found = shutil.which("codex")
    if found:
        return found
    base = os.environ.get("LOCALAPPDATA")
    if base:
        cands = [
            c for c in glob.glob(os.path.join(base, "OpenAI", "Codex", "bin", "*", "codex.exe"))
            if os.path.isfile(c)
        ]
        if cands:
            return max(cands, key=os.path.getmtime)  # newest install
    return None


def resolve_cli_binary(ptype: str, command: str | None = None) -> str | None:
    """Resolve the executable path for an agent-CLI provider profile.

    An explicit `command` (a path or a name on PATH) wins; otherwise fall back
    to the per-type discovery.
    """
    if command:
        return command if os.path.isfile(command) else shutil.which(command)
    if ptype == "claude-code":
        return shutil.which("claude")
    if ptype == "codex":
        return _resolve_codex_binary()
    return None


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Flatten a conversation into a single prompt string (for `codex exec`)."""
    parts: list[str] = []
    for m in messages:
        if m.role == "assistant":
            parts.append(f"[Your previous reply]\n{m.content}")
        else:  # system + user both contribute plain text; system leads the list
            parts.append(m.content)
    return "\n\n".join(parts)


class ClaudeCliClient:
    """Drives a locally installed `claude` CLI as an LLM provider (no API key).

    Spawns `claude -p --output-format stream-json --input-format stream-json`,
    writes the conversation to stdin (one JSON event per line), and reads the
    streamed events back. Auth is whatever `claude login` already set up — this
    client never sees a key. `thinking`, `json_mode`, and sampling options are
    accepted for interface compatibility but ignored (the CLI has no flags for
    them); structured tasks rely on prompt wording + the ingest JSON parser.

    Codex will be a sibling with the same shape but a different arg list and
    event schema.
    """

    BINARY = "claude"

    def __init__(
        self,
        model: str = "",
        isolated: bool = True,
        command: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.model = (model or "").strip()
        self.isolated = isolated
        self.timeout = timeout
        self.binary = resolve_cli_binary("claude-code", command)

    def close(self) -> None:
        pass

    def __enter__(self) -> "ClaudeCliClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # -- health -------------------------------------------------------------

    def ensure_ready(self) -> None:
        """Cheap pre-run check: the CLI is installed. Auth is verified lazily."""
        if not self.binary:
            raise ModelNotFound(
                f"Claude Code CLI ('{self.BINARY}') not found on PATH.\n"
                f"Install it, then run `{self.BINARY}` once to log in."
            )

    def probe(self) -> str:
        """'Test' button: confirm the CLI is found AND logged in via a tiny call."""
        self.ensure_ready()
        # A 1-token round-trip is the only real way to verify the OAuth session.
        self.chat([ChatMessage(role="user", content="Reply with the single word: ok")])
        model = self.model or "(CLI default)"
        return f"Found {self.BINARY} at {self.binary} · authenticated · model {model}."

    def list_models(self) -> list[str]:
        return [self.model] if self.model else []

    # -- request assembly ---------------------------------------------------

    def _command(self) -> list[str]:
        cmd = [
            self.binary, "-p",
            "--output-format", "stream-json",
            "--input-format", "stream-json",
            "--verbose", "--include-partial-messages",
        ]
        if self.model:
            cmd += ["--model", self.model]
        if self.isolated:
            # Don't inherit the user's MCP servers, skills/slash commands, or
            # personal settings (CLAUDE.md etc.) into wiki-generation runs.
            cmd += ["--strict-mcp-config", "--disable-slash-commands",
                    "--setting-sources", ""]
        return cmd

    @staticmethod
    def _stdin_events(messages: list[ChatMessage]) -> str:
        """Serialize the conversation as stream-json input.

        System messages are folded into a preamble on the first user turn
        (portable across CLI versions that lack a system-prompt flag).
        """
        preamble = "\n\n".join(m.content for m in messages if m.role == "system")
        lines: list[str] = []
        used_preamble = False
        for m in messages:
            if m.role == "system":
                continue
            content = m.content
            role = "assistant" if m.role == "assistant" else "user"
            if role == "user" and preamble and not used_preamble:
                content = f"{preamble}\n\n{content}"
                used_preamble = True
            lines.append(json.dumps({
                "type": role,
                "message": {"role": role, "content": [{"type": "text", "text": content}]},
            }))
        if preamble and not used_preamble:  # no user turn — send preamble as one
            lines.insert(0, json.dumps({
                "type": "user",
                "message": {"role": "user", "content": [{"type": "text", "text": preamble}]},
            }))
        return "\n".join(lines) + "\n"

    # -- chat ---------------------------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        thinking: bool = False,
        json_mode: bool = False,
        **options,
    ) -> str:
        return "".join(self._run(messages))

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        thinking: bool = False,
        **options,
    ) -> Generator[str, None, str]:
        parts: list[str] = []
        for chunk in self._run(messages):
            parts.append(chunk)
            yield chunk
        return "".join(parts)

    def _run(self, messages: list[ChatMessage]) -> Generator[str, None, None]:
        """Spawn the CLI, stream text out; raise a clear error on failure/auth."""
        self.ensure_ready()
        proc = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=tempfile.gettempdir() if self.isolated else None,
            creationflags=_CREATE_NO_WINDOW,
        )
        # Watchdog so a hung CLI can't block forever.
        timer = threading.Timer(self.timeout, proc.kill)
        timer.start()
        try:
            proc.stdin.write(self._stdin_events(messages))
            proc.stdin.close()

            emitted = False
            result_text: str | None = None
            is_error = False
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "stream_event":
                    inner = event.get("event", {})
                    if inner.get("type") == "content_block_delta":
                        delta = inner.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                emitted = True
                                yield text
                elif etype == "result":
                    is_error = bool(event.get("is_error"))
                    result_text = event.get("result")
            stderr = proc.stderr.read()
            proc.wait()
        finally:
            timer.cancel()
            if proc.poll() is None:
                proc.kill()

        # Fallback: no token deltas seen (older CLI / no partial support) — emit
        # the authoritative final text from the result event as one chunk.
        if not emitted and result_text:
            yield result_text

        if proc.returncode not in (0, None) or is_error:
            detail = (stderr or result_text or "").strip()
            low = detail.lower()
            if any(p in low for p in _CLI_AUTH_PATTERNS):
                raise _AuthError(
                    f"{self.BINARY} is not logged in. Run `{self.BINARY}` in a "
                    f"terminal and complete `/login`, then retry."
                )
            raise LLMError(
                f"{self.BINARY} CLI failed (exit {proc.returncode}): "
                f"{detail[:300] or 'no output'}"
            )
        if not emitted and not result_text:
            raise LLMError(
                f"{self.BINARY} produced no output. "
                f"Check `{self.BINARY}` runs in a terminal."
            )


class CodexCliClient:
    """Drives the local OpenAI `codex` CLI (`codex exec`) as a provider (no key).

    Runs `codex exec --json`, feeding the flattened prompt on stdin and reading
    JSONL events: `item.completed` with item type `agent_message` carries the
    answer (item type `reasoning` is the model's private thinking — ignored),
    and `-o` writes the authoritative final message. Auth is whatever
    `codex login` set up. Like the Claude client, `thinking`/`json_mode`/sampling
    are accepted but ignored. Runs read-only in a temp dir so it never edits the
    user's files.
    """

    def __init__(
        self,
        model: str = "",
        isolated: bool = True,
        command: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.model = (model or "").strip()
        self.isolated = isolated
        self.timeout = timeout
        self.binary = resolve_cli_binary("codex", command)

    def close(self) -> None:
        pass

    def __enter__(self) -> "CodexCliClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # -- health -------------------------------------------------------------

    def ensure_ready(self) -> None:
        if not self.binary:
            raise ModelNotFound(
                "OpenAI Codex CLI ('codex') not found.\n"
                "Install it, then run `codex login` once to authenticate."
            )

    def probe(self) -> str:
        """'Test' button: confirm codex is found AND logged in via a tiny call."""
        self.ensure_ready()
        self.chat([ChatMessage(role="user", content="Reply with the single word: ok")])
        model = self.model or "(CLI default)"
        return f"Found codex at {self.binary} · authenticated · model {model}."

    def list_models(self) -> list[str]:
        return [self.model] if self.model else []

    # -- request assembly ---------------------------------------------------

    def _command(self, last_file: str) -> list[str]:
        cmd = [
            self.binary, "exec", "--json",
            "--skip-git-repo-check",  # wiki root / temp dir isn't a git repo
            "--ephemeral",            # don't persist session files
            "-s", "read-only",        # never let the agent modify files
            "-o", last_file,          # final message written here (authoritative)
        ]
        if self.model:
            cmd += ["-m", self.model]
        return cmd

    # -- chat ---------------------------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        thinking: bool = False,
        json_mode: bool = False,
        **options,
    ) -> str:
        return "".join(self._run(messages))

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        thinking: bool = False,
        **options,
    ) -> Generator[str, None, str]:
        parts: list[str] = []
        for chunk in self._run(messages):
            parts.append(chunk)
            yield chunk
        return "".join(parts)

    def _run(self, messages: list[ChatMessage]) -> Generator[str, None, None]:
        self.ensure_ready()
        fd, last_path = tempfile.mkstemp(suffix="-codex-last.txt")
        os.close(fd)
        proc = subprocess.Popen(
            self._command(last_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=tempfile.gettempdir() if self.isolated else None,
            creationflags=_CREATE_NO_WINDOW,
        )
        timer = threading.Timer(self.timeout, proc.kill)
        timer.start()
        emitted = False
        errors: list[str] = []
        stderr = ""
        try:
            proc.stdin.write(_messages_to_prompt(messages))
            proc.stdin.close()
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue  # e.g. the "Reading prompt from stdin..." banner
                etype = event.get("type")
                if etype == "item.completed":
                    item = event.get("item", {})
                    itype = item.get("type")
                    if itype == "agent_message":
                        text = item.get("text", "")
                        if text:
                            emitted = True
                            yield text
                    elif itype == "error":
                        errors.append(item.get("message", ""))
                elif etype == "error":
                    errors.append(event.get("message", ""))
            stderr = proc.stderr.read()
            proc.wait()
        finally:
            timer.cancel()
            if proc.poll() is None:
                proc.kill()

        # Authoritative final text (fallback if no agent_message streamed).
        last_text = ""
        try:
            with open(last_path, encoding="utf-8", errors="replace") as f:
                last_text = f.read().strip()
        except OSError:
            pass
        finally:
            try:
                os.unlink(last_path)
            except OSError:
                pass
        if not emitted and last_text:
            emitted = True
            yield last_text

        # A run that produced text succeeded, even if it logged transient
        # reconnect errors. Only diagnose failure when we got nothing.
        if not emitted:
            detail = (" ".join(e for e in errors if e) or stderr or "").strip()
            low = detail.lower()
            if any(p in low for p in _CLI_AUTH_PATTERNS):
                raise _AuthError(
                    "codex is not logged in (or its session expired). "
                    "Run `codex login`, then retry."
                )
            raise LLMError(
                f"codex exec failed (exit {proc.returncode}): {detail[:300] or 'no output'}"
            )


# ---------------------------------------------------------------------------
# Factory + router
# ---------------------------------------------------------------------------


def make_client(profile: dict, *, timeout: float = DEFAULT_TIMEOUT):
    """Build the right client for a provider profile, dispatching on `type`."""
    ptype = profile.get("type", "ollama-native")
    model = profile.get("model", DEFAULT_MODEL)
    if ptype == "claude-code":
        return ClaudeCliClient(
            model=profile.get("model", ""),
            isolated=profile.get("isolated", True),
            command=profile.get("command"),
            timeout=timeout,
        )
    if ptype == "codex":
        return CodexCliClient(
            model=profile.get("model", ""),
            isolated=profile.get("isolated", True),
            command=profile.get("command"),
            timeout=timeout,
        )
    if ptype == "ollama-native":
        return OllamaClient(
            host=profile.get("host", DEFAULT_HOST), model=model, timeout=timeout
        )
    if ptype == "litellm":
        return LiteLLMClient(
            model=model,
            api_base=profile.get("api_base"),
            api_key_env=profile.get("api_key_env"),
            timeout=timeout,
        )
    raise LLMError(
        f"Unknown provider type '{ptype}'. Use 'ollama-native' or 'litellm'."
    )


class LLMRouter:
    """Routes pipeline tasks to provider clients per `llm.task_providers`.

    Built once per run from the `llm` config block. `client_for(task)` returns
    the client for that task's provider (cached per provider name), and
    `options_for(task)` resolves that provider profile's sampling options. An
    `override` forces every task onto one provider (the CLI `--provider` flag /
    web provider picker).
    """

    def __init__(
        self,
        llm_cfg: dict,
        *,
        override: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self._providers: dict = llm_cfg.get("providers", {})
        self._task_providers: dict = llm_cfg.get("task_providers", {})
        self._default = llm_cfg.get("default_provider") or next(
            iter(self._providers), None
        )
        self._timeout = timeout
        self._clients: dict = {}
        if override and override not in self._providers:
            raise LLMError(
                f"Unknown provider '{override}'. "
                f"Configured: {', '.join(self._providers) or '(none)'}."
            )
        self._override = override

    # -- resolution ---------------------------------------------------------

    def provider_for(self, task: str | None = None) -> str:
        if self._override:
            return self._override
        if task and task in self._task_providers:
            return self._task_providers[task]
        if not self._default:
            raise LLMError("No default_provider configured and no task mapping.")
        return self._default

    def _profile(self, name: str) -> dict:
        if name not in self._providers:
            raise LLMError(
                f"Provider '{name}' is not configured. "
                f"Available: {', '.join(self._providers) or '(none)'}."
            )
        return self._providers[name]

    def _client_by_name(self, name: str):
        if name not in self._clients:
            self._clients[name] = make_client(self._profile(name), timeout=self._timeout)
        return self._clients[name]

    # -- public API ---------------------------------------------------------

    def client_for(self, task: str | None = None):
        return self._client_by_name(self.provider_for(task))

    def options_for(self, task: str | None = None, **overrides) -> dict:
        name = self.provider_for(task)
        return resolve_llm_options(self._profile(name), task=task, **overrides)

    def model_for(self, task: str | None = None) -> str:
        """The model string of the provider that will handle this task."""
        return self._profile(self.provider_for(task)).get("model", "?")

    def thinking_default(self, task: str | None = None, fallback: bool = False) -> bool:
        return bool(self._profile(self.provider_for(task)).get("thinking", fallback))

    def ensure_ready(self) -> None:
        """Check each distinct provider this run may use."""
        if self._override:
            names = {self._override}
        else:
            names = set(self._task_providers.values())
            if self._default:
                names.add(self._default)
        for name in filter(None, names):
            self._client_by_name(name).ensure_ready()

    def close(self) -> None:
        for client in self._clients.values():
            client.close()
        self._clients.clear()

    def __enter__(self) -> "LLMRouter":
        return self

    def __exit__(self, *args) -> None:
        self.close()
