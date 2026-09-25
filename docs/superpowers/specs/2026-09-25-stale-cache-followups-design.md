# Stale-cache follow-ups: design

Follow-ups deferred in the #655 review (stale results served on failure).
Approved by the user.

## 1. A retrieval 4xx is never answered from stale rows

**Problem.**
- `SearchClient.retrieve` (`src/context/retrieval/client.py`) serves stale
  rows for *any* `Exception` raised by `_post_json`.
- `_post_json` re-raises a 4xx immediately, as an
  `aiohttp.ClientResponseError`. Examples: a filter shape the server now
  rejects, or newly required auth.
- So a 400, 401 or 403 is hidden behind stale rows for up to an hour, and the
  only signal is an INFO log.

**Decision.** Do not fall back when the exception, or its `__cause__`, is an
`aiohttp.ClientResponseError` with `status < 500` and `status != 429`.

- A 429 or a 5xx still falls back.
- So do transport errors and the exhausted-retries `RuntimeError`, whose
  cause is the last attempt's error.

The client and request errors re-raise exactly as they do without a cache.

## 2. Stale serves are visible on `/metrics`

**Problem.** A stale answer is not a failure anywhere. Nothing consumes
`TTLCache.stats()["stale_hits"]`, and the #651 tool metrics count
stale-answered calls as `success`, so an outage the cache hides looks healthy.

**Decision.**
- **The counter.** A new counter in
  `src/internal/observability/prometheus.py`:
  `agentic_search_stale_cache_serves_total{source}`. `source` is `web` (the
  `search_tool` fallback) or `retrieval` (the `SearchClient` fallback), a
  fixed vocabulary validated like the existing helpers.
- **The helper.** `observe_stale_serve(source: str) -> None` is incremented
  once per fallback that actually serves. For retrieval that is once per
  `retrieve` call, not once per row.
- **The docs.** `docs/observability-metrics.md` gets a section with the
  metric, and a PromQL example of the stale-serve rate by source.
- **No torch.** Importing the prometheus module from `src/context/...` must
  keep the torch-free CI job working. The module only imports
  `prometheus_client`.

## 3. The stale-row copy is tested, and the ACL window is documented

- **The test.** Changing `rows_by_index[index] = copy.deepcopy(row)` to `=
  row` currently passes the suite. Add a test that mutates a returned stale
  result's nested metadata (the `acl` list), re-serves stale, and asserts the
  second answer is unchanged.
- **The docs.** `docs/retrieval.md` (the serving-cache section) gets one
  sentence: during a retrieval outage, a document made private can still be
  served from a stale row for up to TTL + grace, 65 minutes by default. Set
  `AGENTIC_SEARCH_SEARCH_CACHE_STALE_SECONDS=0` where that is unacceptable.

## Testing

- **4xx.** A 400 and a 403 re-raise even when a stale row exists. A 429, a
  503 and an exhausted-retries `RuntimeError` whose cause is a 500 are all
  served stale.
- **Counter.**
  - The web fallback increments `{source="web"}` by 1.
  - The retrieval fallback over 3 queries increments `{source="retrieval"}`
    by 1.
  - No fallback means no increment.
  - An unknown source raises.
- **Copy.** Covered by the mutation-sensitive test above.
- **Mutation checks.**
  - Drop the 4xx guard and watch the 4xx tests go red.
  - Remove each `observe_stale_serve` call and watch its test go red.
  - Replace the deep copy with an alias and watch the copy test go red.
