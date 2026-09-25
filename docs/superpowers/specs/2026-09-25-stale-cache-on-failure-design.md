# Serve stale cached results when a provider fails: design

## Problem

No path serves a cached result after a live call times out or fails (fallback
investigation, 2026-09-25):

- Every cache is read-through and fresh-only.
- `TTLCache.get` (`src/internal/cache/ttl_cache.py`) deletes an expired entry
  when it reads it.
- Web search (`search_tool`, `src/internal/tools/search.py`) and retrieval
  rows (`SearchClient.retrieve`, `src/context/retrieval/client.py`) consult
  the serving cache *before* the live call, and never after it fails.

So when SerpAPI times out, a user who got an answer five minutes ago now gets
nothing, even though that answer is still sitting in memory. The tool
agent's `web_search` cascade does not even use the cache: its SerpAPI leg
calls `serpapi_search` directly.

## Decision (approved by the user)

### 1. `TTLCache` grace window

`TTLCache(ttl_seconds, *, max_entries=1024, stale_seconds=0.0, clock=...)`:

- **`get(key)`** returns only **fresh** entries, as today. An entry that is
  expired but still inside the grace window is **kept** (a miss, not
  deleted). An entry beyond `ttl + stale_seconds` is deleted on read, as
  today.
- **`get_stale(key)`** returns the value if the entry is anywhere within
  `ttl + stale_seconds`, fresh or stale. Otherwise it returns None. It counts
  toward a new `stale_hits` stat.
- **LRU bound.** `max_entries` still bounds memory, and stale entries count
  toward it and are evicted by LRU as today.
- **`stale_seconds=0`** reproduces today's behavior exactly. Negative values
  raise.

### 2. Configuration

- `configure_serving_cache(ttl_seconds, *, max_entries=1024,
  stale_seconds=0.0)`.
- The web app lifespan passes
  `ServiceSettings.search_cache_stale_seconds`, read from the new env var
  `AGENTIC_SEARCH_SEARCH_CACHE_STALE_SECONDS`, **default 3600**, where 0 means
  off. It is loaded where `AGENTIC_SEARCH_SEARCH_CACHE_TTL` is, validated the
  same way, and documented in `docs/configuration.md`.

### 3. Web search falls back to stale

In `search_tool`, after the live provider call:

- **When it applies.** The result counts as a failure when the list is
  non-empty and every page has an error, is `timed_out`, or `is_blank`. That
  includes an open circuit breaker's error page. On a failure, when
  `cache.get_stale(cache_key)` has an entry, return a deep copy of it with
  each page's `metadata["stale"] = True`, and log at INFO.
- **What is untouched.** A genuinely empty list `[]` is a successful empty
  search and stays empty (user decision). Fresh behavior, the caching guard,
  and the "never cache a failure" rule are unchanged.
- **Mixed lists.** A list with some real pages and some error pages is
  returned as-is. The fallback only happens when nothing usable came back.

### 4. The cascade uses the cache

`make_web_cascade_search`'s default SerpAPI leg becomes `search_tool(query,
provider="serpapi", page=..., page_size=..., timeout_seconds=...)`, so it gets
the same key, fresh hits, and the stale fallback.

- The `serpapi_fn` injection seam stays for tests. Its default is a small
  wrapper around `search_tool`.
- The circuit breaker is still consulted inside `serpapi_search`.

**Breaker accounting (ruling made during review).** The cascade's browser
leg records a circuit-breaker **failure** when every page it got back is an
error page or a stale answer. A stale answer means the live call failed.
Recording it as a success would reset the breaker, or close it from
half-open, while the browser server is down. The stale pages are still
returned.

### 5. Retrieval rows fall back to stale

In `SearchClient.retrieve`, when `_post_json` raises:

- If **every** missing query has a `cache.get_stale(key)` entry, use those
  rows. Each row dict gets a copy of its `metadata` with `"stale": True`, via
  a deep copy, so the cached row is never mutated. Log at INFO.
- Otherwise re-raise, exactly as today. There are no partial answers from a
  mix of stale and missing rows.
- The stage-metrics accounting for the failed call is unchanged.

### 6. Labels reach the caller

- `SearchPage.metadata["stale"]` and row `metadata["stale"]` flow into
  documents through the existing metadata plumbing
  (`_documents_from_search_pages` and `SearchResult.from_api_item`). No UI
  change.
- ACL metadata is preserved, because only the `stale` key is added.

## Out of scope

- The rerank cache (a rerank failure already degrades to the fused order).
- The Redis caches.
- Public-data tools.
- Sharing across processes.
- Background refresh (stale-while-revalidate).

## Testing

- **`TTLCache`** (fake clock):
  - fresh `get`;
  - an expired entry inside the grace window: `get` is None but the entry is
    retained, and `get_stale` returns it;
  - beyond the grace window: both None and the entry deleted;
  - `stale_seconds=0` is identical to today;
  - LRU bound;
  - `stale_hits` stat;
  - negative `stale_seconds` raises.
- **`search_tool`:**
  - fresh result, then expiry, then a provider timeout (blank), error page,
    or circuit-open page: returns the stale pages labelled `stale`;
  - an empty `[]` does not fall back;
  - no stale entry means today's failure result;
  - a stale hit is a copy, so mutation cannot poison the cache.
- **Cascade:**
  - the SerpAPI leg goes through `search_tool`: a second identical call is a
    cache hit, with no provider call;
  - a SerpAPI failure with a stale entry means no browser call is needed.
- **`SearchClient.retrieve`:**
  - `_post_json` raises with stale rows for all queries: stale rows returned
    and labelled, with the cached row unmodified;
  - one query missing a stale row: raises;
  - the ACL metadata survives.
- **Config:** env var parsed, default 3600, 0 means off, a bad value is
  rejected; the lifespan passes it through.
- **Mutation checks:**
  - Make `get` delete in-grace entries and watch the stale test go red.
  - Remove the `search_tool` fallback and watch its test go red.
  - Remove the retrieval fallback and watch its test go red.
