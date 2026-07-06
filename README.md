# LLM-Wiki

> A local, LLM-maintained personal knowledge base, focused on understanding research papers. Drop documents in, watch an LLM compile them into a living, interlinked Obsidian wiki you can search and query.

Feel free to fork and don't forget to give it a Star ⭐️ for better reach!

-------------------------

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/LLM-Qwen3.6--35B-purple.svg)](https://ollama.com/library/qwen3)
[![Local-first](https://img.shields.io/badge/runs-100%25_local-green.svg)](#)

Built on the pattern Andrej Karpathy described in his [LLM Wiki gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f): instead of retrieving from raw documents at query time (classic RAG), an LLM incrementally **compiles** your sources into a structured, cross-linked markdown wiki that sits between you and the raw documents. The wiki is a **persistent, compounding artifact** — the cross-references are already there, the contradictions have already been flagged, the synthesis already reflects everything you've read.

You never write the wiki yourself. The LLM does all the grunt work: summarizing, cross-referencing, filing, bookkeeping. You bring the sources and ask the questions.

Runs **100% locally by default** — Ollama + Qwen3, no API keys, no cloud, no data leaving your machine. The LLM layer is pluggable: when you want to, route any part of the pipeline to another local backend (vLLM, LM Studio) or a cloud provider (OpenAI, Anthropic, Gemini, …). Fully local, fully cloud, or a mix — your call, per task. See [PROVIDERS.md](./PROVIDERS.md).

## What it does

```bash
# Drop files in (PDFs, markdown, HTML, DOCX, text)
wiki add ~/Documents/papers --recursive

# Watch Qwen3 read them and build an interlinked wiki
wiki ingest

# Ask questions — it searches the compiled wiki and cites its sources
wiki query "what's the main argument about X?"

# Health-check the knowledge base
wiki lint --fix

# Browse the whole thing in Obsidian (graph view, backlinks, everything)
open wiki/
```

Every ingest produces a cluster of `sources/`, `entities/`, and `concepts/` pages with YAML frontmatter and `[[wikilinks]]` between them. Every query pulls the top-ranked pages via hybrid BM25 + vector + LLM-rerank search, then synthesizes a cited answer. Every lint run catches broken links, orphan pages, malformed frontmatter, and (optionally, using the LLM) contradictions between pages.

## Features

### Core capabilities
- **Incremental ingest** — drop a file, run `wiki ingest`, get 8–15 cross-linked wiki pages
- **Structured extraction** — Qwen3 identifies entities (people, organizations, places), concepts (models, techniques, architectures, ideas), and key takeaways per source, always framed around the problem being solved and its limitations
- **Smart merging** — re-ingesting related sources updates existing entity/concept pages instead of overwriting them, preserving provenance
- **Pluggable LLM providers** — fully local on Ollama by default; also routes to vLLM, LM Studio, OpenAI, Anthropic, Gemini (via [LiteLLM](https://litellm.ai)), or the Claude Code / Codex CLIs — configurable **per pipeline task** (see [PROVIDERS.md](./PROVIDERS.md))
- **Hybrid search** — BM25 full-text + vector embeddings + LLM reranking (all local, via [QMD](https://github.com/tobi/qmd))
- **3-way query scope** — `Wiki` (thematic answers from LLM-compiled pages), `Raw` (exact lookups in original documents), or `Hybrid` (both)
- **Intent classification** — casual messages ("hi", "thanks") skip retrieval and get a quick reply, saving ~30 seconds per chitchat turn
- **Cited synthesis** — queries return markdown answers with `[[wikilinks]]` pointing to the pages that support each claim
- **Write-back** — save good answers as new `synthesis/` pages with `--save-as`, so your explorations compound in the knowledge base
- **Wiki linting** — automated health checks for broken links, orphans, malformed frontmatter, noise in sources, and (with `--deep`) LLM-powered contradiction detection between pages
- **Auto-fix** — most stylistic issues resolve with one command
- **Auto-reindex** — search index refreshes automatically after ingest and lint; new pages are queryable immediately

### Web UI
A full web interface at `http://127.0.0.1:8000` after `wiki serve`:
- **Dashboard** — project stats and recent activity
- **Sources** — list, inspect, delete, or re-ingest sources with one click
- **Ingest** — drag-and-drop upload, live progress log, persistent jobs that **survive tab close and server restart**
- **Jobs** — history of all ingest runs with live progress bars and error details
- **Query** — chat-style interface with streaming synthesis, scope toggle, save-as-synthesis button
- **Lint** — interactive lint report with one-click auto-fix
- **Graph** — D3 force-directed visualization of the full wiki, color-coded by page type
- **Settings** — configure LLM providers (add profiles from presets, set per-task routing, manage API keys)

### Supported input formats
`.pdf` · `.md` · `.html` · `.docx` · `.txt`

### Obsidian integration
The `wiki/` folder is a ready-made Obsidian vault with:
- Color-coded graph view (sources, entities, concepts, synthesis each get their own color)
- YAML frontmatter compatible with the Dataview plugin
- All cross-references as native `[[wikilinks]]` so backlinks, outgoing-links, and graph traversal all work

## Architecture

Three layers, per Karpathy:

```
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   raw/        │ → │   LLM Agent   │ → │   wiki/       │
│ Your docs     │   │ (Qwen3.6-35B) │   │ Markdown,     │
│ (immutable)   │   │               │   │ auto-linked   │
└───────────────┘   └───────────────┘   └───────────────┘
                          │                     │
                          ▼                     ▼
                    ┌───────────────┐   ┌───────────────┐
                    │ schema/       │   │   Obsidian    │
                    │ AGENTS.md     │   │  graph view   │
                    │ (the rules)   │   │  + editing    │
                    └───────────────┘   └───────────────┘
```

- **`raw/`** — your source documents. Immutable. The agent reads but never modifies.
- **`wiki/`** — LLM-maintained markdown. One folder per page type (`sources/`, `entities/`, `concepts/`, `synthesis/`) plus auto-generated `index.md` and `log.md`. Open this in Obsidian.
- **`schema/AGENTS.md`** — the conventions file. Tells the LLM how to format pages, when to merge vs create, how to cite, how to handle contradictions. Edit as your preferences evolve.
- **`.wiki/`** — internal state: SQLite ingest history, QMD search index, config, and per-source diagnostic logs (`.wiki/logs/`). Git-ignored.

### The ingest pipeline

Each source goes through three LLM passes:

1. **Extraction** (thinking mode on) — Qwen3 reads the source and returns structured JSON: a summary that leads with the problem being solved, key takeaways (including what's novel and what's limited), named entities (people, organizations, places only), concepts (models, techniques, architectures, and other ideas), tags. LLM calls retry automatically once after a 5s backoff on failure, and a per-source diagnostic log is written to `.wiki/logs/` for troubleshooting.
2. **Page drafting** (streaming, thinking mode off) — one call per entity/concept. Draft a new page from scratch, or *merge* new information into an existing page (preserving prior content, updating dates, appending to `sources:` frontmatter).
3. **Source summary** — write the `sources/<slug>.md` page listing every wiki page touched by this source for provenance.

After the three passes: `index.md` is rebuilt, `log.md` is appended, and QMD's search index is updated automatically.

### The query pipeline

1. **Hybrid search** via QMD — BM25 full-text + vector similarity + LLM reranker, all local
2. **Top-K page hydration** — load full content of the top 5–8 hits
3. **Synthesis** — Qwen3 writes a cited markdown answer using `[[wikilinks]]` to reference the pages
4. **(Optional) save-back** — `--save-as` files the answer as a new `synthesis/` page

### Search indexing & chunking (inside QMD)

Search is delegated to [QMD](https://github.com/tobi/qmd), which only understands markdown/plain text. Two collections are registered per wiki:

| Collection | Indexes | Contents |
|---|---|---|
| `llm-wiki-pages` | `wiki/**/*.md` | LLM-compiled pages (entities, concepts, sources, synthesis) |
| `llm-wiki-raw` | `raw-text/**/*.md` | Parser-extracted text mirrors of every tracked source |

The raw collection deliberately points at `raw-text/` mirrors, **never** at the original binaries in `raw/` — QMD has no PDF/DOCX parser and would otherwise index raw file bytes as text. Mirrors are written by `wiki add` (and backfilled by `wiki reindex --full`) with slug-safe filenames so QMD's reported paths map back to files on disk.

Indexing happens in two phases after each ingest:

1. **`qmd update`** — rebuilds the BM25 full-text index synchronously (seconds; keyword search works immediately)
2. **`qmd embed`** — generates vector embeddings (EmbeddingGemma-300M) in a background thread; failures are logged to `.wiki/logs/embed-fail-*.log`

**Chunking** is QMD's job, done at embed time — this project always hands over whole files:

- Documents ≤ ~3,600 chars become a **single chunk** (most wiki pages)
- Longer documents are cut at a target of **900 tokens (~3,600 chars) with 15% overlap** (135 tokens / ~540 chars)
- Cuts snap to natural break points within a ~200-token window of the target — headings score highest (h1 > h2 > …), with a distance penalty — and never land inside code fences
- These constants are **hard-coded in QMD** (no config/env override), and its embed fingerprint includes them, so changing them would invalidate and re-embed every chunk

QMD's state is fully **per-wiki**: both `XDG_CACHE_HOME` and `XDG_CONFIG_HOME` are pointed at `.wiki/qmd-cache/`, which holds the SQLite index, the collection registry (`index.yml`), and the downloaded GGUF models (embedder, reranker, query-expansion). Overriding only the cache would leave the collection registry global — deleting the index would then silently resurrect stale collection definitions.

## Providers & models

The LLM layer is **pluggable**. Every pipeline stage (extraction, drafting, synthesis, intent classification, chitchat, contradiction checks) is a named *task* that you route to a *provider profile*. Out of the box every task points at a local Ollama profile running `qwen3.6:35b` — nothing leaves your machine. Change that whenever you like:

- **Other local backends** — vLLM, LM Studio, or any OpenAI-compatible server
- **Cloud vendors** — OpenAI, Anthropic, Gemini, and other [LiteLLM](https://litellm.ai)-supported providers
- **Local agent CLIs** — Claude Code or Codex, using their own login (no API key)

Routing is **per task**, so you can, say, keep token-heavy ingest local and answer queries with a cloud model:

```yaml
# .wiki/config.yml
llm:
  default_provider: ollama          # local for everything by default
  task_providers:
    query_synthesis: anthropic      # …but answer queries with Claude
  providers:
    ollama:    { type: ollama-native, model: qwen3.6:35b, host: http://localhost:11434 }
    anthropic: { type: litellm, model: anthropic/claude-opus-4-8, max_tokens: 8192 }
```

Manage it from the CLI (`wiki providers list` / `set-key` / `test`, `wiki query --provider`) or the web UI **Settings** page. **Full guide: [PROVIDERS.md](./PROVIDERS.md).**

## Stack

| Layer | Component | Why |
|---|---|---|
| LLM (default) | [Ollama](https://ollama.com) + [Qwen3.6-35B](https://ollama.com/library/qwen3) Q4_K_M | Strong reasoning, 40K context, thinking mode — runs fully local |
| Provider routing | [LiteLLM](https://litellm.ai) abstraction | Swap in vLLM, LM Studio, OpenAI, Anthropic, Gemini, Claude Code / Codex — per task |
| Search | [QMD](https://github.com/tobi/qmd) (BM25 + vector + rerank) | All local, SQLite-backed, handles the heavy lifting |
| Embeddings | EmbeddingGemma-300M (via QMD) | Small footprint, high quality |
| Reranker | Qwen3-Reranker-0.6B (via QMD) | Fast cross-encoder rerank |
| CLI | [Typer](https://typer.tiangolo.com) + [Rich](https://rich.readthedocs.io) | Great UX, colored output, progress bars |
| Parsers | pypdf, python-docx, beautifulsoup4, lxml | Cover the main document formats |
| Vault | [Obsidian](https://obsidian.md) | Best-in-class graph view and backlink UX — you don't have to build it |

Local by default: with the Ollama provider, no cloud services, no API keys, and no data leaves your machine. Cloud providers are strictly opt-in.

## Requirements

- **Python 3.11+**
- **Node.js 18+** (for QMD)
- **QMD** (`npm install -g @tobilu/qmd`) — the local search backend
- **An LLM provider** — for the default local setup, **Ollama** with the `qwen3.6:35b` model pulled (large download, tens of GB). You can point tasks at a smaller local model or a cloud provider instead; see [PROVIDERS.md](./PROVIDERS.md).
- **Homebrew SQLite** on macOS (`brew install sqlite`)
- **~30GB free disk space** for models and embeddings (much less if you use a cloud provider for inference and only run QMD's embed/rerank models locally)
- **~24GB RAM** recommended for the default local LLM (32GB+ for comfort); far less if inference runs in the cloud
- **Obsidian** (optional but strongly recommended for browsing)

Developed and tested on macOS (Apple Silicon, M3 Pro 18GB). Also runs on Windows and should work on Linux.

## Installation

```bash
# Clone
git clone https://github.com/YOUR-USERNAME/llm-wiki.git
cd llm-wiki

# Create a virtual environment (uv is faster than pip, either works)
uv venv
source .venv/bin/activate
uv pip install -e .

# Pull the LLM (one-time, large download)
ollama pull qwen3.6:35b

# Install QMD (the search backend)
npm install -g @tobilu/qmd

# Verify
wiki version
wiki --help
```

## Quick start

```bash
# 1. Create a wiki in a folder of your choosing
mkdir my-wiki && cd my-wiki
wiki init

# 2. Drop some source documents in raw/, or use:
wiki add ~/Documents/papers --recursive

# 3. Run ingest (interactive by default — shows you entities/concepts
#    before filing, with a y/n prompt per source)
wiki ingest

# First query triggers QMD to download its embedding + reranker models
# (~2GB, one-time). Subsequent queries are fast.

# 4. Ask questions
wiki query "what are the main themes across these documents?"

# 5. Save a good answer as a synthesis page
wiki query "compare X vs Y" --save-as x-vs-y-comparison

# 6. Health-check and auto-fix
wiki lint --fix

# 7. Browse the vault in Obsidian
open wiki/   # then "Open folder as vault"
```

## Commands

| Command | Purpose |
|---|---|
| `wiki init [path]` | Scaffold a new wiki project |
| `wiki add <file-or-folder> [-r]` | Copy sources into `raw/` and register for ingest |
| `wiki sources list` | List all tracked sources with status |
| `wiki sources show <id>` | Show metadata + text preview for one source |
| `wiki sources rm <id>` | Remove a source from tracking |
| `wiki ingest [source_id]` | Run the 3-pass LLM ingest pipeline |
| `wiki reingest <source_id>` | Reset a source to `pending` so the next `wiki ingest` reprocesses it |
| `wiki query "<question>" [--scope wiki\|raw\|hybrid] [--mode hybrid\|lex\|vec] [--types <t1,t2>] [--provider <name>] [--save-as <slug>]` | Search + synthesize a cited answer |
| `wiki reindex [--full]` | Force rebuild of the QMD search index (`--full`: delete index, backfill raw-text mirrors, re-embed everything) |
| `wiki lint [--deep] [--fix]` | Health-check the wiki |
| `wiki providers list` | Show configured LLM providers, per-task routing, and API-key status |
| `wiki providers set-key <vendor>` | Store a cloud provider's API key in the OS keyring |
| `wiki providers test <provider>` | Check that a provider profile is reachable / authenticated |
| `wiki status` | Show project stats, paths, config, backend health |
| `wiki serve [--port N]` | Launch the web UI at `http://127.0.0.1:8000` |

Run `wiki <command> --help` for full options on any command. See [USAGE.md](./USAGE.md) for a full walkthrough and [PROVIDERS.md](./PROVIDERS.md) for configuring LLM providers.

## Example output

A real ingest against `notes.txt` (28 words about Qwen3):

```
Source #1  raw/notes.txt
  parsing…
  extracting entities and concepts (thinking mode)…

Title: Quick Notes on Qwen
Slug:  quick-notes-on-qwen

Summary:
  Qwen is a family of large language models developed by Alibaba Cloud.
  The latest version, Qwen3, introduces a thinking mode designed to enhance
  performance on complex reasoning tasks.

Entities (1):
  + alibaba-cloud (organization)  Alibaba Cloud

Concepts (3):
  + qwen                                  Qwen
  + qwen3                                 Qwen3
  + thinking-mode-for-complex-reasoning   Thinking Mode for Complex Reasoning

File these? Will create/update ~5 wiki pages. [Y/n]: Y

created entity alibaba-cloud
created concept qwen
created concept qwen3
created concept thinking-mode-for-complex-reasoning
created source  quick-notes-on-qwen

✓ Ingested Quick Notes on Qwen — 5 created, 0 updated
```

That's **5 cross-linked pages from a 28-word input**, each with YAML frontmatter, `[[wikilinks]]` between them, and provenance back to the source. Open Obsidian's graph view and you'll see the cluster light up.

A real query against 11 ingested pages:

```
> wiki query "how does multi-head attention differ from self-attention?"

  searching wiki (BM25 + vector + rerank)…
  found 8 relevant page(s):
    1. 0.93 concepts/multi-head-attention.md      Multi-Head Attention
    2. 0.55 concepts/self-attention-mechanism.md  Self-Attention Mechanism
    3. 0.40 entities/attention-is-all-you-need.md Attention is All You Need
    ...

  synthesizing answer…

Multi-head attention and self-attention are related but distinct mechanisms:

1. **Scope and Parallelism**
   - Self-attention is a single mechanism where each position in the input
     computes attention weights based on all other positions
     [[concepts/self-attention-mechanism]].
   - Multi-head attention extends this by using multiple parallel attention
     heads, allowing the model to focus on diverse patterns simultaneously
     [[concepts/multi-head-attention]].

2. **Information Capture**
   - Self-attention focuses on a single representation.
   - Multi-head aggregates information from multiple heads, each capturing
     different aspects (syntactic vs semantic, etc.)
     [[concepts/multi-head-attention]].

[... etc.]
```

Every claim is cited. Every citation points to a page that actually exists.

## Lint example

```
> wiki lint

╭─────────── Lint Report ────────────╮
│ Health score: 57/100               │
│ Pages checked: 12                  │
│                                    │
│   2 errors · 21 warnings · 0 infos │
╰────────────────────────────────────╯

──── Errors (2) ────

  synthesis/transformers-and-llms.md
    ✗ broken_wikilink: Broken wikilink: [[entities/introduction-to-transformers]]
      → Either create the page or remove the link.

──── Warnings (21) ────

  entities/qwen.md
    ! malformed_wikilink: 'sources/quick-notes-on-qwen.md' should be
      'sources/quick-notes-on-qwen'
      ✓ auto-fixable

  [... 20 more warnings ...]

> wiki lint --fix
✓ auto-fixed: 11

> wiki lint
╭─────────── Lint Report ───────────╮
│ Health score: 100/100             │
│   0 errors · 0 warnings · 0 infos │
╰───────────────────────────────────╯
✓ No issues found. Your wiki is in good shape!
```

## Project status

**Current version: v1.1.2** — production-ready for personal use.

| Stage | Scope | Status |
|---|---|---|
| 1 | Scaffolding, CLI, Obsidian vault config | ✅ Done |
| 2 | Parsers (PDF, MD, HTML, DOCX, TXT), dedupe, `wiki add` | ✅ Done |
| 3 | LLM ingest pipeline (3 passes, streaming, merge-path) | ✅ Done |
| 4 | QMD search + `wiki query` with citation + save-back | ✅ Done |
| 5 | Lint checks + auto-fix + deep contradiction detection | ✅ Done |
| 6 | FastAPI + HTMX web UI (7 pages: Dashboard, Sources, Ingest, Jobs, Query, Lint, Graph) | ✅ Done |
| 7 (v0.7.0) | Source CRUD, intent classification, 3-way scope toggle | ✅ Done |
| 8 (v0.8.0) | Persistent ingest jobs (survive tab close, server restart) | ✅ Done |
| 9 (v0.9.0) | Focus on research problem and limitations. Proper extraction JSON and document formatting. Auto-reindex after ingest and lint (BM25 updates immediately, vector embeddings build in the background) | ✅ Done |
| 9.1 (v0.9.1) | `wiki reingest`, stabilize with retry-with-backoff, per-source diagnostic logs | ✅ Done |
| 10 (v1.0.0) | Major overhaul of indexing; uses raw text from parser instead of raw file. Also exposed LLM settings to `config.yml` | ✅ Done |
| 11 (v1.1.1) | Support multiple providers, local and cloud. Configurable providers per task. | ✅ Done |

### Possible future work
- Hugging Face Spaces deployment (smaller model, API-compatible)
- Cloud API, if preferred over local model
- Better graph and community detection support
- Dashboard showing live active-job count
- Static HTML export for sharing the wiki
- Multi-user / team features
- Mobile-friendly web UI
- Fine-tuned query expansion model
- Confidence scoring per extracted claim
- OCR support for scanned PDFs
- EPUB support

## Credits

- **[Andrej Karpathy](https://karpathy.ai/)** — for the LLM-Wiki pattern described in [this gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). This project is a direct implementation of the idea.
- **[QMD](https://github.com/tobi/qmd)** by Tobi Lütke — the hybrid search backend that does all the heavy lifting for query-time retrieval.
- **[Qwen3](https://qwenlm.github.io/blog/qwen3/)** by Alibaba Cloud — the local LLM doing the reading, writing, and synthesis.
- **[Ollama](https://ollama.com)** — the runtime that makes local LLM inference painless on Apple Silicon.
- **[Obsidian](https://obsidian.md)** — saved me from writing my own graph view.

## License

MIT — see [LICENSE](LICENSE).

---

*"The tedious part of maintaining a knowledge base is not the reading or the thinking — it's the bookkeeping. Humans abandon wikis because the maintenance burden grows faster than the value. LLMs don't get bored, don't forget to update a cross-reference, and can touch 15 files in one pass."*
— Karpathy
