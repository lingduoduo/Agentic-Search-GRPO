# Relevance-aware memory recall above the cap

## Origin

Memory-augmented generation (#460) injects a signed-in user's stored memories
into the answer prompt unconditionally, capped at the 20 most recent
(`memory_preamble`, `MEMORY_INJECTION_MAX`). That design was deliberate:
proactive recall needs durable facts *always present*, since a question about
Thai food would never retrieve "allergic to peanuts" by similarity.

It has one blind spot. Once a user has more than 20 memories, the cap keeps
the newest and silently drops the rest, so an old but relevant fact (a
preference stated months ago) never reaches the model. `search_memories`
already ranks memories by relevance (e5 cosine with an encoder, token overlap
without) but is reachable only from the CLI, the `/api/memory` router, and
the MCP tool.

This PR, the third of three borrowing from the reference `AgentMemory` class
(its `recall(query, top_k)`), makes the preamble relevance-aware above the
cap. PRs 1 and 2 (#578, #579) are on main.

## Scope

- Above the cap, the preamble is half relevance hits for the current query,
  half most-recent fill. At or below the cap it is unchanged.
- The current query reaches the preamble through `resolve_capabilities`.
- The web app builds the optional e5 encoder once at startup and hands it to
  the preamble, instead of the per-call build the CLI and MCP paths do.
- No new flag. Nothing changes for a user with 20 or fewer memories, half the
  slots above the cap stay the most recent, and the semantic leg is already
  opt-in through `AGENTIC_SEARCH_MEMORY_SEMANTIC`.

## Non-goals

- Injecting memory on the Chat or Tools surfaces. Tools resolves capabilities
  but does not inject the preamble today; that stays as is, and its
  `resolve_capabilities` call is unchanged (no query, recency rule).
- Changing `search_memories`, the manual search endpoint, or the MCP tool.
- Changing the cap (`MEMORY_INJECTION_MAX = 20`) or the instructional header.
- Reading `curated_through`. It remains write-only bookkeeping.

## Design

### Selection rule

`src/internal/memory/service.py`:

```python
MEMORY_RELEVANT_SLOTS = MEMORY_INJECTION_MAX // 2   # 10

def memory_preamble(store, user_id, *, max_items=MEMORY_INJECTION_MAX,
                    query: str | None = None, encoder: Encoder | None = None) -> str
```

1. `memories = store.get_user_memories(user_id)` (chronological texts). Empty
   returns `""`, as today.
2. If `len(memories) <= max_items` or `query` is empty/whitespace: the last
   `max_items`, as today. This path calls nothing new, so every existing
   caller and test double (which expose only `get_user_memories`) is
   unchanged.
3. Otherwise `_select_relevant(store, user_id, query, memories, max_items,
   encoder)`:
   - The lexical leg (no encoder) scores on `_content_query(query)`, which
     drops function words, so the query is never matched on tokens like "to"
     or "for"; the e5 leg scores on the original query.
   - `hits = search_memories(store, user_id, search_query, max_results=min(MEMORY_RELEVANT_SLOTS, max_items // 2), encoder=encoder)`,
     a list of `(record, score)`; take `record.memory_text` in rank order.
   - Fill: walk `memories` newest-first, adding texts not already selected,
     until `max_items` are chosen.
   - Return the selected texts **in chronological order** (filter `memories`
     by membership), so the block is stable and independent of score ties.
   - If `search_memories` raises, log at warning and return the recency
     result. Selection never fails a request; the existing guard in
     `resolve_capabilities` still wraps the whole preamble.
4. Format as today.

Ties, duplicates, and the lexical fallback: `search_memories` returns fewer
than the slot count when few memories overlap the query; the fill then takes
more recent ones, so the block is always `max_items` long above the cap.
Two memories with identical text produce one bullet, at the first
occurrence; the store never deduplicates rows, so the preamble must.

### Trade-off

Below the cap nothing changes. Above it, the 10 most recent memories stay
always present; positions 11–20 of the old recency rule become displaceable
by relevance hits. For a user with 30 memories, 10 memories that were in
every prompt no longer are. The lexical leg matches on content tokens only
(function words dropped); with no content overlap all 20 slots fall back to
recency. The e5 leg ranks by embedding similarity and takes the top 10
without a floor: e5 similarities have a high baseline, so a floor would be
arbitrary, and the top 10 of a user's memories are the most related ones by
construction.

### Threading the query and the encoder

`src/internal/access/capabilities.py`:

```python
def resolve_capabilities(user, store, *, query: str | None = None,
                         encoder: Encoder | None = None) -> RequestCapabilities
```

forwards both to `memory_preamble`. Anonymous callers still get `ANONYMOUS`
before the store is touched.

Assist (`app.py` `_run_agent_impl`): `resolve_capabilities(auth_user, db,
query=query, encoder=getattr(http_request.app.state, "memory_encoder", None))`.
`query` is already the stripped request query at that point. The
`getattr` default keeps every test that builds the app without running the
lifespan working.

Tools (`tool_backend.py`): unchanged.

### The encoder, built once

The lifespan in `create_web_app` sets `_app.state.memory_encoder =
maybe_build_encoder()` next to the other `_app.state` assignments.
`maybe_build_encoder` already returns `None` unless
`AGENTIC_SEARCH_MEMORY_SEMANTIC` is set and already logs and returns `None`
when the model cannot load, so the default stack pays nothing and uses the
lexical fallback. Building it once at startup, not per request, is the whole
reason it lives on `app.state` rather than being called inside the preamble.

`docs/configuration.md`: the `AGENTIC_SEARCH_MEMORY_SEMANTIC` row (or the
memory section, wherever that variable is documented) gains one sentence: it
also selects the semantic leg for the per-request memory preamble above the
20-memory cap, with the model loaded once at web-app startup.

### Failure

- `search_memories` raising (encoder error, store error): logged, recency
  result used.
- Encoder load failure at startup: `maybe_build_encoder` already degrades
  to `None`; lexical fallback for the process lifetime.
- No path can fail a request; `resolve_capabilities` already catches any
  exception from the preamble and degrades to `""`.

## Testing

`tests/unit/memory/test_injection.py`:

- Below the cap with a query: block identical to the no-query block.
- Above the cap, lexical: 25 memories where memory 0 is "User is allergic to
  peanuts" and the rest are "memory number i"; `query="thai food peanuts"`
  includes the peanut memory and still includes the newest memory; exactly
  `MEMORY_INJECTION_MAX` bullets; no duplicates; bullets appear in
  chronological order (peanut bullet first).
- Above the cap with no query (or whitespace): unchanged recency block
  (`memory number 0` absent).
- Above the cap with a fake encoder (returns unit vectors so one old memory
  scores highest): that memory is included; the fake records that it was
  called once with the passage list and once with the query.
- `search_memories` raising: recency block, no exception.
- `MEMORY_RELEVANT_SLOTS == 10`.

`tests/unit/access/test_capabilities.py`: `resolve_capabilities(user, store,
query="q", encoder=enc)` forwards both (a store double whose
`get_user_memory_records`/`get_user_memories` return >20 items and a
recording fake encoder); existing tests unchanged.

`tests/unit/servers/web/test_memory_injection.py`: the Assist call passes the
request query and the `app.state` encoder (monkeypatch
`web_app.resolve_capabilities` with a recorder; set `app.state.memory_encoder`
to a sentinel after `create_web_app`); and the lifespan sets
`app.state.memory_encoder` from `maybe_build_encoder` exactly once (patch
`web_app.maybe_build_encoder` to return a sentinel and count calls, run the
app under `with TestClient(app):`).

Every new test gets a mutation check.
