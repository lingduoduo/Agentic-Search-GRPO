# Serving-path ACL filtering and caching

**Date:** 2026-09-10
**Status:** Implemented on `feat/serving-acl-and-cache`, pending PR review

## Problem

"Online serving" here means the request-path processes: the web backend
(`src/internal/servers/web/app.py`), the retrieval servers
(`src/internal/servers/retrieval/`), and the rerank server. Two things are
uneven across them.

**ACL filtering.** The web path serialises `access_acl` and re-enforces it on
every result list (`_enforce_access`, the search agent's inline check, the tool
agent's per-request tool). `demo.py` and `hybrid.py` honour `access_acl`. Two
serving parts do not:

- `src/internal/retrieval/backends/local.py::_apply_filters` — the backend
  behind the full `RetrievalService` (`server.py`) — treats every filter key as
  a metadata equality test. An `{"access_acl": [...]}` request compares the
  list against `metadata["access_acl"]`, a key no document has, so a filtered
  request returns nothing and an unfiltered one is never ACL-checked. The
  `ResultCache` in front of it is keyed by the same dict and caches that.
- `src/internal/search/stages.py::SearchClientRetrievalStage` forwards `filters`
  to the retrieval server and stores them on the `CandidateSet`, but never
  checks what came back. That is the unpaired-serialisation shape
  `docs/request-routing.md#access-control` says the web layer must not rely on.

**Caching.** Nothing on the web path is cached. Every request re-posts to the
retrieval server (including the hybrid server's dense encode), re-calls the
web providers (SerpAPI ~12 s, browser ~48 s), and re-scores through the
cross-encoder. The three caches that exist (`ResultCache`, `CachedReranker`,
`CachedQueryTransformPipeline`) are Redis-only, sit inside `RetrievalService`,
and none is on the `app.py → demo/hybrid` path the local stack runs.

## Goals

- Every serving part that receives `access_acl` applies the same rule as
  `SearchFilters.matches`: intersect with the document's declared ACL; a
  document with no declared ACL is public.
- Repeated identical searches, web lookups and rerank calls within a short
  window are served from process memory, with no Redis required.
- A cache hit can never widen what a caller may read: the ACL is part of the
  cache key, and the existing call-site enforcement still runs on hits.

## Non-goals

- LLM response caching (answers stream as verified claims; not idempotent).
- Caching page fetches (`fetch_pages_concurrently`) — a follow-up if needed.
- An in-memory backend for `RetrievalService`'s Redis `ResultCache`. Once the
  client caches, the server-side cache is redundant for the web path.
- Filters for `single_turn.py`'s retrieval — it is CLI-only, not served.
- Filtering inside `rerank.py` — it scores only what a filtered caller hands it.

## Design

### ACL

`src/internal/retrieval/acl.py` (torch-free, no web imports) holds one
function:

```python
def acl_allows(metadata: dict | None, filters: dict | None) -> bool
```

No filters, no `access_acl` in the filters, or no declared ACL → allowed.
Otherwise allowed iff the declared ACL intersects `access_acl`. The declared
ACL is read from `metadata["acl"]` and `metadata["tags"]["acl"]` (str or list),
exactly as `SearchFilters._metadata_acl_values` does. This is
`demo.py::_allowed_by_acl` with the `document["metadata"]` lookup moved to the
caller plus the `tags.acl` source it lacked; `demo.py` keeps `_allowed_by_acl`
as a one-line wrapper so `hybrid.py` and the tests are unchanged.

`backends/local.py::_apply_filters` pops `access_acl` and applies
`acl_allows`; the remaining keys keep the metadata-equality semantics. The
local backend flattens unknown corpus keys into `RetrievalResult.metadata`, so
a corpus document with `"metadata": {"acl": [...]}` (the shape
`metadata_with_acl` produces) arrives as `metadata["metadata"]["acl"]`; the
backend reads the nested dict when it carries an ACL, so both that shape and a
top-level `"acl"` key work.

`SearchClientRetrievalStage.retrieve` filters the returned candidates with
`acl_allows(result.metadata, filters)` before building the `CandidateSet`.

### Cache primitive

`src/internal/cache/ttl_cache.py::TTLCache(ttl_seconds, max_entries=1024,
clock=time.monotonic)`: a lock-guarded `OrderedDict` with per-entry expiry.
`get` evicts an expired entry and returns `None`; `set` evicts the oldest
entry when full; `clear()`; `stats()` returns hits/misses. Keys are any
hashable; callers build tuples.

`src/internal/cache/serving.py` owns the process-level instance:

```python
def configure_serving_cache(ttl_seconds: float, *, max_entries: int = 1024) -> None
def reset_serving_cache() -> None
def serving_cache() -> TTLCache | None      # None ⇒ caching off
```

Caching is off until configured. The web app's lifespan calls
`configure_serving_cache` on startup when the TTL is positive and
`reset_serving_cache` on shutdown, so a `TestClient` that never enters the
lifespan never turns it on, and one that does turns it off again.

### Insertion points and keys

| Site | Key | Cached value |
|---|---|---|
| `SearchClient.retrieve` (`src/context/retrieval/client.py`) | `("retrieve", url, query, topk, json(filters, sort_keys))` per query | raw API items for that query |
| `search_tool` web providers (`src/internal/tools/search.py`) | `("web", provider, query, page, page_size)` | `list[SearchPage]` |
| `RerankHTTPRankingStage.rank` (`src/internal/search/stages.py`) | `("rerank", url, query, top_k-or-None, tuple(contents))` | the server's ranked list |

`SearchClient.retrieve` looks each query up, posts only the misses in one
request, and merges; it always returns one row per query, logging a warning
when the server answered with a different number of rows. Cached values are
deep-copied on every hit and store, so a caller mutating a result (including a
nested metadata value) never poisons the cache. Values that are empty or contain an error page are not cached: an empty
web result usually means a transient provider failure, and empty retrieval is
cheap to recompute.

Why the retrieval key carries the serialised filters: two callers with
different ACLs must never share an entry. Every existing call site still runs
its own enforcement on the returned list, so a hit is checked exactly as a
miss is.

### Configuration

`ServiceSettings.search_cache_ttl_seconds: int = 300`, loaded from
`AGENTIC_SEARCH_SEARCH_CACHE_TTL`; `0` disables. `SearchExperienceSettings`
carries it as `search_cache_ttl` and the lifespan reads it. One knob, one
env var; `max_entries` stays at its default.

## Error handling

A cache lookup or store never raises into the request: the primitive is pure
Python over a dict. Provider errors are not cached, so a transient failure is
retried on the next request rather than pinned for the TTL.

## Testing

- `acl_allows`: no filters, no declared ACL, str ACL, intersect / disjoint.
- Local backend: filtered request with `access_acl` keeps public and matching
  docs and drops disjoint ones, for both the nested and top-level ACL shapes;
  other keys still equality-match.
- `SearchClientRetrievalStage`: a result the filters reject never reaches the
  `CandidateSet` even though the fake client returned it.
- `TTLCache`: hit, miss, expiry via an injected clock, LRU eviction at
  `max_entries`, `stats`.
- `SearchClient.retrieve`: second identical call makes no HTTP request; a
  different `filters` payload does; a batch with one cached and one new query
  posts only the new one and returns rows in order; empty rows are not cached.
- `search_tool`: a web provider is called once for two identical lookups;
  error pages are not cached; the `retrieval` provider is untouched here.
- `RerankHTTPRankingStage`: one HTTP call for two identical rank calls;
  different candidate contents miss.
- Settings: env → `ServiceSettings` → `SearchExperienceSettings`; lifespan
  configures and resets the cache.

Each new test is mutation-checked by removing the feature it pins.

## Rejected alternatives

- **Cache inside the retrieval servers.** Would not cover web providers or the
  reranker, and the hybrid server's encode is only one of the costs. The
  client-side choke points cover every caller of the `retrieval` provider.
- **Reuse `src/internal/cache/InMemoryCache`.** It has no TTL (`expire` is a
  no-op) and carries the lock/list API of the Redis backend it mirrors.
- **Default the cache off.** The point is the local stack, which runs with no
  extra setup; the env var is the off switch.
