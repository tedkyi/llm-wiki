"""Prompt templates for the LLM ingest pipeline.

Three passes:
    1. extract — read source, return structured JSON with entities/concepts
    2. draft_page — generate a single entity/concept/source page
    3. merge_page — update an existing page with new information

Each template returns a list of ChatMessage ready for the Ollama client.
"""

from __future__ import annotations

from .llm import ChatMessage


# ---------------------------------------------------------------------------
# System prompt — shared across all passes
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the LLM agent that maintains an LLM-Wiki knowledge base.

You follow these conventions strictly:

1. Wiki pages use YAML frontmatter delimited by triple dashes, exactly like this:

   ---
   title: "Page Title"
   type: source | entity | concept | synthesis
   tags: [tag1, tag2]
   created: YYYY-MM-DD
   updated: YYYY-MM-DD
   sources: ["sources/source-slug"]
   confidence: high | medium | low
   ---

   Never wrap frontmatter in a ```yaml code block. The --- delimiters are the only correct format.
   Never append .md to entries in the sources list — use bare paths like sources/source-slug, not sources/source-slug.md.

2. Always use [[wikilinks]] for cross-references between wiki pages.
   Never use plain markdown links for internal pages.
   Every wikilink must begin with one of these prefixes:
     [[entities/slug]], [[concepts/slug]], [[sources/slug]], [[synthesis/slug]]
   Never write a bare [[slug]] without a prefix.
   Never wrap [[wikilinks]] in backticks — they must appear as plain text, not inline code.
   Only create [[wikilinks]] for items you are explicitly given in a related list or
   instructed to link. Never invent wikilinks for entities or concepts not in that list,
   even if you recognise them from your training data.

3. Slugs are kebab-case lowercase ASCII (e.g. andrej-karpathy, not Karpathy.md).
   Use the canonical name. Acronyms stay together (rag, llm, not r-a-g).

4. Be factual. Never invent citations or claims not in the source.
   If unsure, mark confidence as 'medium' or 'low'.

5. Page bodies should be concise but substantive: 150-400 words is typical
   for entity pages, 200-500 words for concept pages.

6. Preserve existing content when updating a page. Add new info in new
   sections or under an '## Updates' heading. Never silently overwrite.
"""


# ---------------------------------------------------------------------------
# Pass 1 — Extraction
# ---------------------------------------------------------------------------

EXTRACTION_INSTRUCTIONS = """Read the source document below and extract a structured summary.

Return ONLY a valid JSON object matching this exact schema:

{
  "title": "A clear, specific title for this source (max 80 chars)",
  "source_slug": "kebab-case-slug-for-this-source",
  "summary": "A 2-3 sentence paragraph summarizing the source, always starting with what is the problem being solved",
  "key_takeaways": [
    "Bullet 1 — a substantive takeaway (1-2 sentences)",
    "Bullet 2",
    "Bullet 3"
  ],
  "entities": [
    {
      "name": "Canonical name as it would appear in a wiki",
      "slug": "kebab-case-slug",
      "type": "person | organization | model | product | place",
      "description": "1-2 sentences describing this entity based on the source"
    }
  ],
  "concepts": [
    {
      "name": "Canonical name",
      "slug": "kebab-case-slug",
      "type": "concept",
      "description": "1-2 sentences describing this concept based on the source"
    }
  ],
  "tags": ["tag1", "tag2", "tag3"]
}

Rules:
- Extract 3-8 key takeaways, each substantive. Include what is novel compared to prior work. 
  Also include any limitations, caveats, or open questions.
- Extract 2-10 entities (people, organizations, models, products, places mentioned).
  Entities must be proper nouns with a specific, named identity (e.g. GPT-4, OpenAI, Yann LeCun).
  CRITICAL: Do NOT put algorithms, architectures, techniques, or building blocks in entities — those are concepts.
  The test: could this be mistaken for a generic type or class of thing? If yes, it is a concept, not an entity.
  FORBIDDEN as entities (these are always concepts, never entities, no matter what):
    Transformer, Attention, Self-Attention, Multi-Head Attention, Backpropagation,
    Wake-Sleep Algorithm, Variational Autoencoder, Diffusion Model, Neural Network,
    Convolutional Neural Network, Recurrent Neural Network, Language Model,
    Residual Network, Feed-Forward Network, and any other general architecture or algorithm.
  Specific named model instances ARE entities only when they have a unique product identity
  (e.g. GPT-4, DALL-E 3, Gemini 1.5 Pro) — generic architecture names like "Transformer"
  or "BERT" used as an architectural class are not.
- Extract 2-10 concepts (techniques, ideas, algorithms, architectures, and topics discussed).
  Concepts are generic or named ideas, not specific named instances of a product or organization.
- Slugs must be kebab-case ASCII. For people use last name if unambiguous
  (karpathy, not andrej-karpathy). For concepts use the shortest canonical
  form (rag, not retrieval-augmented-generation — but use the full form in 'name').
- Tags should be 3-5 broad topic labels for the whole source.
- Do NOT extract trivial mentions — only things substantive enough to deserve
  their own wiki page.
- Return ONLY the JSON object. No preamble, no explanation, no markdown fences.
"""


def build_extraction_messages(source_title: str, source_text: str) -> list[ChatMessage]:
    """Pass 1 — extract structured information from a source document."""
    user_content = (
        f"{EXTRACTION_INSTRUCTIONS}\n\n"
        f"---SOURCE TITLE---\n{source_title}\n\n"
        f"---SOURCE TEXT---\n{source_text}\n"
    )
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
    ]


def build_extraction_retry_messages(
    source_title: str, source_text: str, bad_response: str
) -> list[ChatMessage]:
    """Retry prompt after a JSON parse failure."""
    user_content = (
        f"Your previous response was not valid JSON. Return ONLY a valid JSON "
        f"object matching the schema — no markdown fences, no preamble.\n\n"
        f"{EXTRACTION_INSTRUCTIONS}\n\n"
        f"---SOURCE TITLE---\n{source_title}\n\n"
        f"---SOURCE TEXT---\n{source_text}\n"
    )
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
        ChatMessage(role="assistant", content=bad_response[:2000]),
        ChatMessage(
            role="user",
            content="That was not valid JSON. Try again. Return ONLY the JSON object.",
        ),
    ]


# ---------------------------------------------------------------------------
# Pass 2 — Draft a new entity or concept page
# ---------------------------------------------------------------------------

NEW_ENTITY_PAGE_TEMPLATE = """Draft a wiki entity page for '{name}'.

This entity was extracted from the source: '{source_title}' (sources/{source_slug})

The source describes it as:
{description}

Here are some relevant excerpts from the source:
{excerpts}

Related entities and concepts also mentioned in this source (use prefixed [[wikilinks]] to connect to them):
{related}

Write a complete markdown page with:
1. YAML frontmatter in the --- delimited format described in your instructions (title, type: entity, tags, created: {today}, updated: {today}, sources: ["sources/{source_slug}"], confidence)
2. An H1 heading matching the title
3. A 2-3 paragraph body (150-300 words) describing this entity
4. Use [[entities/slug]] or [[concepts/slug]] wikilinks ONLY for items explicitly listed
   in the Related section above. Do not add wikilinks for any other entity or concept,
   even if mentioned in the excerpts or known from your training data.
5. End with a '## Sources' section listing [[sources/{source_slug}]]

Do not invent facts. Only use information from the excerpts. Return ONLY the markdown content — no preamble, no code fences.
"""


NEW_CONCEPT_PAGE_TEMPLATE = """Draft a wiki concept page for '{name}'.

This concept was extracted from the source: '{source_title}' (sources/{source_slug})

The source describes it as:
{description}

Here are some relevant excerpts from the source:
{excerpts}

Related entities and concepts also mentioned in this source (use prefixed [[wikilinks]] to connect to them):
{related}

Write a complete markdown page with:
1. YAML frontmatter in the --- delimited format described in your instructions (title, type: concept, tags, created: {today}, updated: {today}, sources: ["sources/{source_slug}"], confidence)
2. An H1 heading matching the title
3. A 2-4 paragraph body (200-400 words) explaining this concept:
   - What it is
   - Why it matters
   - How it relates to other concepts/entities (use [[entities/slug]] or [[concepts/slug]] wikilinks
     ONLY for items explicitly listed in the Related section above)
4. End with a '## Sources' section listing [[sources/{source_slug}]]

Do not invent facts. Only use information from the excerpts. Return ONLY the markdown content — no preamble, no code fences.
"""


def build_draft_page_messages(
    kind: str,
    name: str,
    source_title: str,
    source_slug: str,
    description: str,
    excerpts: str,
    related: list[str],
    today: str,
) -> list[ChatMessage]:
    """Pass 2 — draft a single new entity or concept page."""
    template = NEW_ENTITY_PAGE_TEMPLATE if kind == "entity" else NEW_CONCEPT_PAGE_TEMPLATE
    related_str = "\n".join(f"  - [[{r}]]" for r in related) if related else "  (none)"
    user_content = template.format(
        name=name,
        source_title=source_title,
        source_slug=source_slug,
        description=description,
        excerpts=excerpts,
        related=related_str,
        today=today,
    )
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
    ]


# ---------------------------------------------------------------------------
# Pass 2b — Merge new info into an existing page
# ---------------------------------------------------------------------------

MERGE_PAGE_TEMPLATE = """Update the following existing wiki page with new information from a new source.

---EXISTING PAGE---
{existing_content}
---END EXISTING PAGE---

---NEW SOURCE---
Title: {source_title}
Source slug: {source_slug}

The source describes '{name}' as:
{description}

Relevant excerpts from the new source:
{excerpts}
---END NEW SOURCE---

Update the page by:
1. Preserving ALL existing content — do not delete or rewrite existing paragraphs.
2. Adding new information as either:
   - A new paragraph in the appropriate section, OR
   - A new section if the information is substantively new, OR
   - An '## Updates' section if the new info contradicts something in the existing page.
3. Updating the 'updated:' date in frontmatter to {today}.
4. Adding "sources/{source_slug}" to the 'sources:' list in frontmatter (keep existing entries).
5. Adding [[sources/{source_slug}]] to the '## Sources' section at the bottom.
6. Keeping any existing [[wikilinks]] intact. Do NOT add new [[wikilinks]] for entities or
   concepts beyond those already present in the existing page — adding a wikilink implies
   that page exists in the wiki, and you cannot verify that.

Return ONLY the complete updated markdown page — no preamble, no code fences.
"""


def build_merge_page_messages(
    name: str,
    existing_content: str,
    source_title: str,
    source_slug: str,
    description: str,
    excerpts: str,
    today: str,
) -> list[ChatMessage]:
    """Pass 2b — merge new information into an existing page."""
    user_content = MERGE_PAGE_TEMPLATE.format(
        name=name,
        existing_content=existing_content,
        source_title=source_title,
        source_slug=source_slug,
        description=description,
        excerpts=excerpts,
        today=today,
    )
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
    ]


# ---------------------------------------------------------------------------
# Pass 3 — Source summary page
# ---------------------------------------------------------------------------

SOURCE_PAGE_TEMPLATE = """Draft a source summary page for the ingested document.

Source details:
- Title: {source_title}
- Slug: {source_slug}
- File path: {file_path}
- File type: {file_type}
- Ingested: {today}

Summary: {summary}

Key takeaways:
{key_takeaways}

Tags: {tags}

Entity pages created/updated from this source:
{entity_links}

Concept pages created/updated from this source:
{concept_links}

Write a complete markdown page with:
1. YAML frontmatter in the --- delimited format described in your instructions: title, type: source, tags, created: {today}, updated: {today}, file_path, file_type
2. An H1 heading matching the title
3. A 'Summary' section with the summary paragraph
4. A 'Key Takeaways' section with the takeaways as bullets
5. A 'Related Pages' section with two subsections (Entities, Concepts), each listing [[wikilinks]] to the pages above
6. No made-up facts — only use what's provided above

Return ONLY the markdown content — no preamble, no code fences.
"""


def build_source_page_messages(
    source_title: str,
    source_slug: str,
    file_path: str,
    file_type: str,
    summary: str,
    key_takeaways: list[str],
    tags: list[str],
    entity_slugs: list[str],
    concept_slugs: list[str],
    today: str,
) -> list[ChatMessage]:
    """Pass 3 — draft the sources/<slug>.md summary page."""
    takeaways_str = "\n".join(f"- {t}" for t in key_takeaways)
    entity_links = (
        "\n".join(f"- [[entities/{s}]]" for s in entity_slugs)
        if entity_slugs
        else "  (none)"
    )
    concept_links = (
        "\n".join(f"- [[concepts/{s}]]" for s in concept_slugs)
        if concept_slugs
        else "  (none)"
    )
    user_content = SOURCE_PAGE_TEMPLATE.format(
        source_title=source_title,
        source_slug=source_slug,
        file_path=file_path,
        file_type=file_type,
        summary=summary,
        key_takeaways=takeaways_str,
        tags=", ".join(tags),
        entity_links=entity_links,
        concept_links=concept_links,
        today=today,
    )
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
    ]


# ---------------------------------------------------------------------------
# Stage 5 — Contradiction detection (used by `wiki lint --deep`)
# ---------------------------------------------------------------------------

CONTRADICTION_DETECTION_PROMPT = """You are reviewing two wiki pages for potential contradictions.

Page A: {path_a}
---
{content_a}
---

Page B: {path_b}
---
{content_b}
---

Compare the factual claims made in these two pages. If you find a clear
contradiction between them, describe it concisely in 1-3 sentences, naming
the specific conflicting claims.

Only flag REAL contradictions — direct factual disagreements, not stylistic
differences or different levels of detail. If a claim in one page simply
elaborates on a claim in the other, that's NOT a contradiction.

If there is no contradiction, respond with exactly the word: NONE

Otherwise, respond with a brief description of the contradiction. Do not
include preamble like "I found" — just state the conflict directly.
"""
