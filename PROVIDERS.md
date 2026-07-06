# LLM-Wiki — Providers & Models

LLM-Wiki runs **fully local by default** (Ollama + Qwen3), but the LLM layer is pluggable. You can point any pipeline task at a different backend — another local server (vLLM, LM Studio), a cloud API (OpenAI, Anthropic, Gemini, and other LiteLLM vendors), or a local agent CLI (Claude Code, Codex) — and mix them per task.

This page is the full reference. For the day-to-day commands see [USAGE.md](./USAGE.md#configuring-llm-providers); for the big picture see [README.md](./README.md).

---

## Table of Contents

- [How routing works](#how-routing-works)
- [The config file](#the-config-file)
- [Provider profiles](#provider-profiles)
- [Provider types & model strings](#provider-types--model-strings)
- [API keys](#api-keys)
- [Per-task routing](#per-task-routing)
- [Managing providers](#managing-providers)
- [Worked examples](#worked-examples)
- [Sampling parameters](#sampling-parameters)
- [Troubleshooting](#troubleshooting)

---

## How routing works

Every LLM call the pipeline makes belongs to one of these **tasks**:

Task names follow a `command_stage` convention so it's clear which command each one belongs to.

| Task | When it runs |
|---|---|
| `ingest_extract` | Reading a source, pulling out entities/concepts/takeaways (thinking mode) |
| `ingest_extract_retry` | The automatic retry if extraction fails |
| `ingest_draft` | Writing/merging an entity or concept page |
| `query_intent` | Deciding whether a query is a real question or chitchat (opt-in; off by default) |
| `query_chitchat` | Replying to "hi"/"thanks" without retrieval |
| `query_synthesis` | Composing the cited answer to a query |
| `lint_deep` | `wiki lint --deep` contradiction checks |

Each task is routed to a named **provider profile**. The router resolves a task like this:

1. `llm.task_providers.<task>` if set →
2. otherwise `llm.default_provider`.

So you can send everything to one provider (the default), or fan tasks out across several — e.g. run cheap/fast work locally and reserve a cloud model for `query_synthesis`.

---

## The config file

Provider configuration lives in your project's **`.wiki/config.yml`**, under the `llm:` block. `wiki init` writes a working default (all tasks → local Ollama). You can edit it by hand, or use the web UI's **Settings** page (`wiki serve` → Settings), which builds valid profiles for you from a dropdown of presets.

Shape:

```yaml
llm:
  # Fallback provider for any task not named in task_providers.
  default_provider: ollama

  # Which provider handles each pipeline task.
  task_providers:
    ingest_extract: ollama
    ingest_extract_retry: ollama
    ingest_draft: ollama
    query_intent: ollama
    query_chitchat: ollama
    query_synthesis: ollama
    lint_deep: ollama

  # Named, self-contained provider profiles.
  providers:
    ollama:
      type: ollama-native
      model: qwen3.6:35b
      host: http://localhost:11434
      thinking: true
      # sampling defaults (Ollama vocabulary) …
      temperature: 0.6
      num_ctx: 131072
      num_predict: 8192
      # sparse per-task overrides
      tasks:
        ingest_extract: { num_predict: -1 }
        query_chitchat: { temperature: 1.0 }
```

Legacy configs (pre-multi-provider, with a flat `llm.provider` + `llm.model`) are upgraded automatically on load into a single `ollama` profile — no manual migration needed.

---

## Provider profiles

A profile is a self-contained description of one backend. Keys:

| Key | Meaning |
|---|---|
| `type` | Client type: `ollama-native`, `litellm`, `claude-code`, or `codex` |
| `model` | Model identifier. For `litellm`, a full LiteLLM model string incl. vendor prefix (e.g. `openai/gpt-4o`, `anthropic/claude-opus-4-8`) |
| `host` | Endpoint for `ollama-native` (default `http://localhost:11434`) |
| `api_base` | Endpoint for `litellm` OpenAI-compatible/local servers (e.g. `http://localhost:8000/v1`) |
| `api_key_env` | *(optional)* Name of the env var / keyring handle to read this profile's API key from. Lets several profiles share one key and survive key rotation |
| `thinking` | Enable the model's reasoning/thinking mode where supported |
| `timeout` | Per-request timeout (seconds) |
| `tasks` | Sparse per-task overrides, merged over the profile's base sampling settings |
| *(anything else)* | Passed through as a **sampling parameter** in that provider's own vocabulary (see [Sampling parameters](#sampling-parameters)) |

Everything not in the reserved set (`type`, `model`, `host`, `api_base`, `api_key_env`, `isolated`, `command`, `thinking`, `tasks`, `timeout`) is treated as a sampling option and forwarded to the client.

---

## Provider types & model strings

When you add a provider in the **Settings** UI you pick a *preset*, which fills in the right `type` and applies the correct LiteLLM model-string prefix. Editing `config.yml` by hand, you supply these yourself:

| Preset | `type` | Model string | Endpoint field | API key |
|---|---|---|---|---|
| Ollama (native) | `ollama-native` | `qwen3.6:35b` | `host` | not needed |
| Ollama (via LiteLLM) | `litellm` | `ollama/qwen3.6:35b` | `api_base` | not needed |
| OpenAI-compatible (vLLM, LocalAI, …) | `litellm` | `hosted_vllm/<model>` | `api_base` | usually not needed |
| LM Studio | `litellm` | `lm_studio/<model>` | `api_base` | not needed |
| OpenAI | `litellm` | `openai/gpt-4o` | — (fixed URL) | **required** |
| Anthropic | `litellm` | `anthropic/claude-opus-4-8` | — | **required** |
| Google Gemini | `litellm` | `gemini/<model>` | — | **required** |
| Claude Code (no API key) | `claude-code` | e.g. `claude-opus-4-8` | — | not needed (CLI login) |
| Codex (no API key) | `codex` | the model your `codex` CLI accepts via `-m` | — | not needed (CLI login) |
| Custom (full LiteLLM string) | `litellm` | you supply the prefix | `api_base` (optional) | depends on vendor |

Any [LiteLLM-supported vendor](https://docs.litellm.ai/docs/providers) works via the **Custom** preset — write the full model string (e.g. `openrouter/anthropic/claude-3.5-sonnet`) and, if it needs a key, point `api_key_env` at the right handle.

---

## API keys

Keys are **per-vendor**, not per-profile — LiteLLM reads them from the environment by model-string prefix, so two profiles pointing at the same vendor share one key. Keys are **never** stored in `config.yml`.

**Resolution order on read:**

1. Process environment — including a gitignored **`.env`** file at the project root (loaded via python-dotenv).
2. **OS keyring** — Windows Credential Manager / macOS Keychain / Linux Secret Service. On startup, any keyring-stored key whose env var is unset is copied into the environment so LiteLLM finds it.

**Vendors with a canonical key handle:**

| Vendor slug | Env var |
|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `gemini` | `GEMINI_API_KEY` |
| `mistral` | `MISTRAL_API_KEY` |
| `cohere` | `COHERE_API_KEY` |
| `groq` | `GROQ_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `xai` | `XAI_API_KEY` |

Local backends (`ollama`, `vllm`, `lm_studio`, `hosted_vllm`) and the agent-CLI types (`claude-code`, `codex`) need no key.

**Setting a key** (stored in the OS keyring, hidden prompt):

```bash
wiki providers set-key anthropic
wiki providers set-key openai
```

Or set it in the environment / a `.env` file instead (best on headless servers without a keyring backend):

```dotenv
# .env  (gitignored)
ANTHROPIC_API_KEY=sk-ant-…
OPENAI_API_KEY=sk-…
```

For a vendor without a canonical handle, or to share one key across profiles, set `api_key_env` on the profile and store the key under that name:

```bash
wiki providers set-key OPENROUTER_API_KEY
```

---

## Per-task routing

The point of profiles is mixing them. Set `default_provider` to your everyday backend and override only the tasks you want elsewhere:

```yaml
llm:
  default_provider: ollama          # local for everything by default
  task_providers:
    query_synthesis: anthropic      # …but answer queries with Claude
    lint_deep: anthropic            # …and run deep-lint reasoning on Claude
  providers:
    ollama: { type: ollama-native, model: qwen3.6:35b, host: http://localhost:11434 }
    anthropic: { type: litellm, model: anthropic/claude-opus-4-8, temperature: 1.0, max_tokens: 8192 }
```

To force a single provider for one query regardless of routing:

```bash
wiki query "compare X and Y" --provider anthropic
```

---

## Managing providers

| Command | Purpose |
|---|---|
| `wiki providers list` | Show configured profiles, per-task routing, and API-key status |
| `wiki providers set-key <vendor\|ENV_VAR>` | Store an API key in the OS keyring (hidden prompt) |
| `wiki providers rm-key <vendor\|ENV_VAR>` | Remove a stored API key |
| `wiki providers test <profile>` | Check that a profile is reachable / authenticated |
| `wiki query "…" --provider <profile>` | Use a specific profile for one query |

**Adding or editing a profile** is done either in the web UI **Settings** page or by editing `.wiki/config.yml` directly — there's no `providers add` CLI command. After any change, verify with:

```bash
wiki providers list
wiki providers test <profile>
```

---

## Worked examples

### 1. Answer queries with a cloud model, keep ingest local

Ingesting is token-heavy; querying is where a stronger model pays off.

```bash
wiki providers set-key openai
```

```yaml
llm:
  default_provider: ollama
  task_providers:
    query_synthesis: openai
  providers:
    ollama: { type: ollama-native, model: qwen3.6:35b, host: http://localhost:11434, thinking: true }
    openai: { type: litellm, model: openai/gpt-4o, temperature: 1.0, max_tokens: 8192 }
```

```bash
wiki providers test openai
wiki query "what's the main argument about X?"      # query_synthesis now runs on GPT-4o
```

### 2. Point everything at a local vLLM server

```yaml
llm:
  default_provider: vllm
  task_providers: {}     # empty → everything falls back to default_provider
  providers:
    vllm:
      type: litellm
      model: hosted_vllm/Qwen/Qwen2.5-32B-Instruct
      api_base: http://localhost:8000/v1
      temperature: 0.6
      max_tokens: 8192
```

No API key needed. `wiki providers test vllm` confirms reachability.

### 3. Use LM Studio

```yaml
llm:
  default_provider: lmstudio
  providers:
    lmstudio:
      type: litellm
      model: lm_studio/qwen2.5-14b-instruct
      api_base: http://localhost:1234/v1
      temperature: 0.6
      max_tokens: 8192
```

### 4. Use Claude Code with no API key

If you already have the `claude` CLI logged in:

```yaml
llm:
  default_provider: claude-code
  providers:
    claude-code:
      type: claude-code
      model: claude-opus-4-8
```

---

## Sampling parameters

Sampling knobs are passed through in **each provider's own vocabulary** — LLM-Wiki does not translate between them:

- **Ollama** (`ollama-native`): `temperature`, `top_k`, `top_p`, `min_p`, `num_ctx`, `num_predict` (`-1` = unlimited), …
- **OpenAI / Anthropic / most LiteLLM vendors**: `temperature`, `top_p`, `max_tokens`, …

Set base values at the profile's top level; override per task under `tasks.<task>`. For example, the default Ollama profile lets `extraction` run unbounded (`num_predict: -1`) while keeping other tasks capped, and raises `chitchat` temperature to `1.0`. Use the parameter names your chosen backend expects (e.g. `max_tokens` for cloud vendors, not `num_predict`).

---

## Troubleshooting

**`wiki providers list` shows a key as `missing` / `unset`.** The vendor needs a key that isn't resolvable. Run `wiki providers set-key <vendor>`, or set the env var / add it to `.env`. Re-check with `wiki providers list`.

**`wiki providers test <profile>` fails with a connection error.** For local backends, make sure the server is running and `host`/`api_base` matches (Ollama `:11434`, vLLM `:8000/v1`, LM Studio `:1234/v1`). For cloud, check the key and model string.

**"No OS keyring backend is available."** You're on a headless box without Keychain/Credential Manager/Secret Service. Set the key in the environment or a gitignored `.env` file instead — resolution falls back to those automatically.

**A cloud model rejects a parameter.** You likely left an Ollama-only knob (e.g. `num_ctx`, `num_predict`, `min_p`) on a `litellm` cloud profile. Remove it or replace with the vendor's equivalent (`max_tokens`).

**Model string errors from LiteLLM.** Double-check the vendor prefix (`openai/`, `anthropic/`, `gemini/`, `ollama/`, `hosted_vllm/`, `lm_studio/`). See the [LiteLLM providers list](https://docs.litellm.ai/docs/providers).
