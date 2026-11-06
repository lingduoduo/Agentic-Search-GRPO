# Hierarchical Agent Memory + Lifecycle Hooks — Design

Date: 2026-11-05
Status: approved in conversation; calls marked *(builder's call)* or *(Claude's call)*

## Intent

The search agents should remember each signed-in user across chats: their
facts, preferences, constraints and decisions. That memory should organize
itself, be found quickly when it matters, capture itself without the user
asking, and stay tidy as it grows.

This upgrades the long-term memory that already exists in
`src/internal/memory/`. It does not replace it. Today memory is a flat list.
Search is either lexical or e5, never both, and re-embeds every memory on
every call. Capture is off by default and runs only when a long session gets
compressed. Consolidation is exact-dedup plus "newest wins per tag". Only
the CHAT route sees memory.

The user asked for all four pieces:

1. **Index of indexes**: memories organized into a navigable hierarchy.
2. **Memory grep**: fast keyword + semantic retrieval.
3. **Hooks**: automatic capture and retrieval driven by turn lifecycle events.
4. **Consolidation**: deduplication, summarization, restructuring.

## Goals

- Every agent route the web backend runs (CHAT/AgenticRAG, classic RAG,
  SEARCH, TOOL) gets the user's relevant memory in its prompt.
- Recall stays fast and adds no model download or LLM call to the request
  path. The e5 encoder is used only when it is already loaded
  (`app.state.memory_encoder`).
- Capture happens per turn, not only on compression, once the existing
  opt-in flag is set.
- Memory size stays bounded in practice, because consolidation merges and
  supersedes instead of only appending.

## Non-goals

- Frontend changes (that is the separate Agentic-Search-UI repo).
- Changes to the MCP memory server beyond what it inherits from shared
  service functions.
- A user-defined or LLM-invented category taxonomy (see Categories).
- Postgres support. The store is SQLite and stays that way.
- Memory for anonymous callers on the agent path. They keep getting none.
- Removing the dead memory code the recon found (`src/internal/db/memory.py`,
  `secondary_llm_flows/memory_update.py`). Mention it, don't delete it.

## Anti-goals (a failure even if it "works")

- A memory failure that fails or slows a user request. Recall and capture
  are best-effort, as `resolve_capabilities` is today.
- Memories leaking across users, or an anonymous transcript being filed under
  a signed-in user. The existing ownership rules (`_readable`, the
  `user_id is None` skip in `schedule_compression`) must hold on every new path.
- Secrets stored as memories. Every write still goes through the store's
  `deterministic_capture` redaction.
- Capture that floods memory with trivia, such as one memory per turn.
- Tests that need torch, a real model, or a live LLM. The repo's guards
  (`test_no_live_llm_in_tests.py`, `test_no_blocking_calls_in_async.py`) must
  stay green, and the unit-test install has no torch.

## Constraints

- SQLite via `AgenticSearchStore`, a single connection behind an RLock.
  Schema changes follow the store's existing idempotent pattern (add in both
  `_init_schema` and `_migrate_schema`, using `ALTER TABLE ... ADD COLUMN`
  guarded by a column check), or a real `SCHEMA_VERSION` bump. *(builder's call)*
- The LLM client (`OpenAICompatibleLLM`) is synchronous. Every call from
  async code goes through `asyncio.to_thread`.
- The LLM may be absent (`llm is None`). Every LLM-backed step has a
  deterministic fallback, or is skipped and logged.
- The encoder may be absent. Everything works lexically without it.

## Design

### 1. Data model and the index of indexes

The hierarchy has three levels:

```
root index (one per user)        "You know 47 things about this user: ..."
 └─ category index (one each)    "preferences (12): prefers peer-reviewed ..."
     └─ memories                 the existing user_memories rows
```

**Categories** are a fixed set *(Claude's call: a fixed set is testable,
keeps prompts stable, and lets recall treat some categories specially)*:

| category      | holds                                                  |
|---------------|--------------------------------------------------------|
| `constraints` | allergies, hard limits, must/never rules               |
| `preferences` | tastes, source/format preferences, style               |
| `profile`     | facts about the user (role, location, expertise)       |
| `projects`    | ongoing work, goals, things they're researching        |
| `decisions`   | choices the user made, with their reason when known    |
| `general`     | anything else, and the default when unclassified       |

Schema changes:

- `user_memories` gains:
  - `category TEXT NOT NULL DEFAULT 'general'`
  - `embedding BLOB` (nullable; float32 bytes)
  - `embedding_model TEXT` (nullable; the encoder identity it was computed with)
  - Existing rows migrate to `general`. Existing `metadata.tags` stay as they are.
- New table `memory_index_nodes`, with primary key `(user_id, node)`:
  - `user_id`
  - `node`: a category name, or `'__root__'`
  - `summary TEXT`
  - `memory_count INTEGER`
  - `updated_at`
  - `dirty INTEGER`
- Store accessors cover adding/updating with a category, listing by
  category, reading and writing index nodes, writing cached embeddings, and
  marking nodes dirty. Any memory write marks that memory's category node and
  the root node dirty.

**Classification at write time** *(Claude's call)*:
- Every write path that doesn't supply a category runs a deterministic
  keyword classifier in the service layer. That covers `/api/memory/save`
  without `category`, the curation tools' `add_memory`, MCP `save_memory`
  (via the shared service function), and compression-time `curate_span`.
- Examples of what it catches:
  - allergy/allergic/intolerant/vegan/never/must not/can't eat → `constraints`
  - prefer/like/hate/favorite → `preferences`
  - decided/chose/going with → `decisions`
  - working on/researching/project/goal → `projects`
  - I am/I'm a/I live/my job → `profile`
- Anything else is `general`.
- The migration backfills existing rows with the same classifier, so an
  existing "allergic to peanuts" lands in `constraints` and the always-on
  guarantee covers it.
- LLM extraction and consolidation may assign a better category later. The
  classifier is the floor.
- `tests/unit/memory/test_injection.py:132` currently asserts that an
  off-query allergy can be dropped above the cap. Rewrite it to assert the
  opposite.

Embedding cache rule: a cached vector is valid only if `embedding_model`
matches the current encoder's identity and the text hasn't changed since. An
update clears the embedding. Missing or stale vectors are computed lazily,
in one batch, the first time a search needs them, and then written back.

### 2. Memory grep (`src/internal/memory/search.py`, new)

`search_memories(store, user_id, query, *, max_results, encoder=None, category=None)`
keeps its current call shape (plus `category`) and return type
(`list[tuple[UserMemoryRecord, float]]`), so existing callers and tests keep
working.

- **Lexical leg:** a small in-process BM25 (k1≈1.2, b≈0.75) over the user's
  active memories, tokenized with the repo's existing tokenizer
  (`document_index/text.py` or the service's `_tokenize`, *builder's call*)
  and stopword-filtered. Pure Python, no new dependency. This replaces token
  overlap.
- **Dense leg** (only when an encoder is given): cosine against cached
  embeddings, using the e5 `query:`/`passage:` prefixes as today.
- **Fusion:** `src.internal.retrieval.fusion.rrf_rank` over the two ranked
  lists. The returned score is the fused RRF score. Lexical-only returns the
  BM25 score.
- `category` restricts the candidate set before scoring.
- A memory scoring zero on both legs is never returned.

`/api/memory/search` uses `request.app.state.memory_encoder` instead of
calling `maybe_build_encoder()` per request. That fixes the per-request model
load. It also accepts an optional `category`.

### 3. Hierarchical recall (the preamble)

`memory_preamble(...)` keeps its signature and its "empty string when there's
nothing" contract, and still opens with the instructional header that drives
proactive use. Above the existing cap it now builds:

1. **Root index line(s):** the root summary, or when no summary exists yet,
   a deterministic line of category counts.
2. **Always-on:** every `constraints` memory, capped by `max_items`. An
   allergy must never be dropped because it's irrelevant to the query.
3. **Relevant memories:** grep hits for the current query across all
   categories, grouped under their category heading, each with a one-line
   category summary when one exists.
4. **Recency fill:** the newest memories fill any slots left under the cap.

At or below the cap, every memory is listed as today, grouped by category.
Total length is bounded *(builder's call: a char budget of a few thousand)*.
Exact formatting is the builder's call. It must be deterministic for a given
store state so it's testable.

### 4. Lifecycle hooks (`src/internal/hooks/lifecycle.py`, new)

A small in-process registry, separate from the external HTTP webhooks in
`hooks/executor.py`. It does not replace them.

- **Events:** `before_turn`, `after_turn`, `after_tool_call`.
- **Context object:** `TurnContext` carries `user_id`, `session_id`, `query`,
  `route` (once known), `store`, `llm`, `encoder`, and for `after_turn` the
  final answer. `ToolCallContext` carries the tool name, arguments, a result
  summary, `user_id` and `session_id`.
- **`before_turn` handlers** return text fragments, which are joined into the
  turn's memory/context block. Handlers run in registration order.
- **`after_turn` handlers** are fire-and-forget background work, scheduled
  like `schedule_compression` (task set, no drain at shutdown).
- **Isolation:** a handler that raises or times out is logged and skipped. It
  never fails the turn. Handlers that touch the sync store or LLM do so via
  `asyncio.to_thread`.
- **Registry ownership:** one registry per web app, built in `create_web_app`
  and stored on `app.state`. Memory registers its handlers there. Tests build
  their own.
- **Emission points:**
  - `before_turn` fires in `_run_agent_impl` after `_ensure_session` and after
    the QUERY_PROCESSING webhook, so `session_id` is set and recall sees the
    rewritten query *(Claude's call)*. `resolve_capabilities` keeps doing
    identity and ACL resolution where it is now, which keeps the webhook
    getting `user_id`. It no longer builds the memory preamble. `route` is
    `None` in `before_turn` because routing happens later; it is set for
    `after_turn`.
  - `after_turn` fires in the existing `finally` block next to
    `schedule_compression`.
  - `after_tool_call` fires from the agent loops' shared tool dispatch
    (`AgentLoopBase`), carried in the same way `on_turn` is passed through.
    *(builder's call on the exact plumbing)*

Memory's registrations:

- **Recall** is the `before_turn` handler that produces the preamble. Today's
  direct `memory_preamble` call in `resolve_capabilities` moves behind this
  handler. Whether `RequestCapabilities.memory_preamble` stays (filled from
  the hook) or goes is the builder's call, as long as existing callers and
  tests stay meaningful.
- **Capture** is the `after_turn` handler (section 5).
- **Consolidation trigger** is the `after_turn` handler (section 6).
- `after_tool_call`: memory registers nothing in v1. The event exists because
  the user asked for tool-call hooks, and so other features can use it. It
  needs one test showing it fires with the right context.

### 5. Automatic capture

Gated by the existing `AGENTIC_SEARCH_MEMORY_AUTO_CURATE` flag, which stays
**off by default** *(Claude's call: per-turn capture costs one LLM call per
qualifying turn and writes user data silently, so an operator should opt in)*.
It also requires a signed-in user, a session that user owns, and a configured
LLM.

Per turn, in the background:

1. **Cheap gate:** skip the LLM entirely unless the user's message carries a
   durable-fact signal. Examples are first-person statements ("I", "my",
   "I'm"), preference and constraint verbs (prefer, like, hate, allergic,
   always, never, don't), or decision language (decided, going with, chose).
   Most search questions don't qualify, and that is the main cost optimization.
2. **Extract:** one `llm.complete` call over the last user message plus the
   assistant answer. It returns JSON `{"memories": [{"text", "category"}]}`
   with zero or more items, using the same "no secrets, one contextual
   sentence" rules as the curation prompt. An unknown category maps to
   `general`. Unparseable output means nothing is captured.
3. **Reconcile each item** against existing memories via memory grep within
   the same category:
   - near-duplicate (*builder's call on the threshold*) → update the existing
     memory to the newer wording
   - otherwise → add
   - capture never deletes; contradictions are consolidation's job
4. Record a `memory_trajectories` row (`source: "auto_turn"`) as curation does.

When per-turn capture is on, the compression-time `curate_span` path is
switched off **only in `_run_agent_impl`**, where per-turn capture runs, so
the same turns aren't captured twice. `/send-chat-message` and
`/send-tool-message` (`chat_backend.py`, `tool_backend.py`) emit no hooks
and keep compression-time `curate_span` exactly as today, so they lose
nothing *(Claude's call: hooks for those endpoints are out of scope)*.
Compression itself (summaries) is unchanged.

### 6. Consolidation

`consolidate_memories(store, user_id, *, llm=None, encoder=None, resolve_conflicts=True)`.
It keeps the existing report keys (`initial`, `duplicates_removed`,
`conflicts_resolved`, `final`) and adds `merged`, `reclassified` and
`summaries_rebuilt`.

Steps, in order:

1. **Exact dedup**, as today.
2. **Near-duplicate merge:** cluster memories within a category whose cosine
   (encoder) or BM25-normalized/Jaccard similarity (no encoder) passes a
   threshold. With an LLM, merge each cluster into one memory (update the
   newest, soft-delete the rest). Without one, keep the newest.
3. **Contradictions:** with an LLM, for each category (batched, bounded
   size), ask which memories are superseded by newer ones and soft-delete
   those. Without one, apply today's newest-per-`tags[0]` rule. Superseded
   rows get `metadata.superseded_by`.
4. **Reclassify:** with an LLM, assign a category to `general` memories that
   fit one. Without one, a keyword heuristic, or skip. *(builder's call)*
**Constraints are protected** *(Claude's call)*:
- Automatic consolidation never soft-deletes or supersedes a `constraints`
  memory. Steps 2 and 3 skip that category.
- It also never reclassifies one out of `constraints`.
- Exact duplicates (step 1) of a constraint may still collapse to one row,
  since nothing is lost.
- Only an explicit user action removes a constraint (`delete_memory`, or the
  curation loop acting on what the user said).

5. **Summaries:** rebuild dirty category summaries and the root summary. With
   an LLM, one short summary each. Without one, a deterministic summary:
   count, top keywords, and the newest few memories truncated.

Triggers:

- manual: `POST /api/memory/consolidate`, which gains the LLM and encoder
- automatic: an `after_turn` handler, when the user's memory count has grown
  by ≥ N since the last consolidation (env
  `AGENTIC_SEARCH_MEMORY_CONSOLIDATE_EVERY`, default 10), run in the background
- index summaries may also be rebuilt on their own when dirty nodes exist and
  no full consolidation is due *(builder's call)*

**Concurrency:** a per-user in-process guard so capture and consolidation for
the same user don't interleave. Skip-if-busy is fine, since the next turn
retries.

### 7. Wiring memory into SEARCH and TOOL routes

- **SEARCH agent:** `_run_search_agent` already appends the working-memory
  summary to the loop's instruction (`app.py:811-830`). The memory block
  rides the same way, and the instruction is built whenever there's a summary
  *or* memory. The direct-search and escalation paths that synthesize an
  answer with the LLM pass `user_memory` through the same prompt argument
  CHAT uses, where one exists.
- **TOOL agent:** the memory block goes into the tool agent's system
  instruction. A read-only, user-scoped `search_memory` tool (wrapping memory
  grep, with optional `category`) is added to that request's tool set when
  the user is signed in. It's bound per request the way the ACL'd corpus
  search is (`with_search_tool=True`), never registered globally. When it is
  bound, the remote MCP `search_memories` tool (`mcp_client.DEFAULT_USER_SCOPED`)
  is left out of that request's tool set so the model doesn't see two
  near-identical tools *(Claude's call)*.
- Remove the "intentionally scoped to CHAT" comment at `app.py:1471` and
  update `docs/chat-engine.md` to match.

### 8. API surface

- `GET /api/memory/index`: the caller's hierarchy (root summary, each
  category's summary, count, and memories).
- `POST /api/memory/save`: optional `category`.
- `GET /api/memory/list`: each item includes `category`; optional
  `?category=` filter.
- `POST /api/memory/search`: optional `category`; uses the app encoder.
- `POST /api/memory/consolidate`: unchanged request, richer report.

Auth semantics (`_uid`, `require_auth`, shared `default_user` bucket) are
unchanged. Document the new endpoints in `docs/api-reference.md`.

## Testing expectations

Unit tests in `tests/unit/memory/` and `tests/unit/db/`, using the existing
patterns: `AgenticSearchStore(":memory:")`, fake encoders, and scripted fake
LLMs. Cover at least:

- store migration from a pre-change database
- BM25 ranking
- RRF fusion with and without an encoder
- embedding cache reuse and invalidation on update
- category-filtered search
- hierarchical preamble: constraints always present, determinism
- hook isolation (a raising handler doesn't fail the turn)
- the after_tool_call event
- capture gate skips the LLM
- capture add vs update
- no capture for anonymous or unowned sessions
- consolidation with and without an LLM
- the auto-consolidation trigger threshold
- SEARCH and TOOL routes receiving the memory block
- the `search_memory` tool being user-scoped

## Left to the builder

Exact thresholds, char budgets, BM25 parameters, prompt wording, preamble
formatting, migration mechanics, hook plumbing into the agent loops, and
module boundaries inside `src/internal/memory/`, all within the constraints
above.
