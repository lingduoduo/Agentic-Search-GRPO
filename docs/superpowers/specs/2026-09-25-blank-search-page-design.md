# A blank search page is an empty result, never a cached source: design

## Problem

`str(asyncio.TimeoutError())` is empty. When SerpAPI (or Google / Serper)
times out, the provider returns `[SearchPage(error="", timed_out=True)]`, a
page with no title, summary, URL or error text. The user decided on
2026-09-25 that a timeout reading as a *successful empty search* is fine. Two
paths turn that blank page into something worse than empty:

1. **It is cached.** `search_tool` caches when `not any(p.error for p in
   pages)` (`src/internal/tools/search.py` ~:575). `""` is falsy, so the
   blank page is pinned for the serving-cache TTL (300 s by default).
2. **It becomes a fake source.** On `/api/agent`'s hybrid path,
   `_documents_from_search_pages` (`src/internal/servers/web/app.py`
   ~:3060) turns it into a document titled `"Result 1"` with the content
   `"No summary available."` and `metadata.error = False`. `_has_usable` then
   counts it as a real result, so **the browser fallback is skipped**, and the
   user sees a junk source card. The cache hit repeats this for the whole TTL.

## Decision (approved by the user)

- **The predicate.** `SearchPage.is_blank` is a property that is true when
  the page has no title, no summary, no URL and no error text. This is
  exactly the shape of an empty-message provider timeout, and never the shape
  of a real result or an explicit error.
- **The cache.** `search_tool` does not cache a list that contains any page
  that has an error, is `timed_out`, or `is_blank`. Real results are still
  cached, and explicit errors were already skipped.
- **The documents.** `_documents_from_search_pages` drops blank pages, and
  numbers the remaining documents contiguously (`D1..Dn`) with no gaps. A
  SerpAPI timeout on the hybrid path therefore yields no documents:
  - `_has_usable` is False, so the browser fallback runs when it is
    configured.
  - With no browser, `_finalize_hybrid` reports `status="empty"`, a
    successful empty search, which matches the user's decision.

**Unchanged:**
- tool-loop recovery and the `web_search` cascade, which already ignores
  blank pages because it only collects pages where `p.error` is set;
- the error text itself;
- the `timed_out` flag;
- the metrics from #651.

## Out of scope

- Stale-on-error caching.
- Making timeout pages carry error text. The user declined that, because it
  changes recovery.
- `fetch_pages_concurrently` dropping `timed_out`. It does not matter here,
  because blankness survives that rebuild: a page with no URL is not fetched.

## Testing

- **Cache** (`tests/unit/test_search_tools_cache.py`):
  - a timed-out blank page is not cached: two identical lookups make two
    provider calls;
  - a blank page without the flag is not cached either;
  - real results are still cached (an existing test).
- **Hybrid** (`tests/unit/servers/web/test_hybrid_web_fallback.py`):
  - a SerpAPI blank page with a browser configured leads to the browser
    results, and no `"Result 1"` document;
  - with no browser configured it gives `status == "empty"` and no
    documents;
  - a real SerpAPI result still skips the browser (an existing test).
- **`_documents_from_search_pages`:** a mix of blank and real pages keeps
  only the real ones, numbered `D1..Dn`.
- **Mutation checks:**
  - Remove the blank filter in `_documents_from_search_pages` and watch the
    hybrid tests go red.
  - Remove `is_blank` from the cache guard and watch the cache test go red.
