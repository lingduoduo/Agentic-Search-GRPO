# Blank Search Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A blank search page (an empty-message provider timeout) is never
cached and never becomes a source document, so the hybrid path treats it as an
empty result and falls back to the browser.

**Architecture:** Add a `SearchPage.is_blank` property. Use it in
`search_tool`'s cache guard and in `_documents_from_search_pages`. No change to
error text, recovery, or the cascade.

**Tech Stack:** Python 3.10+, pytest, pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-09-25-blank-search-page-design.md`

## Global Constraints

- The error text is unchanged. A timeout stays a *successful empty search*
  (user decision, 2026-09-25).
- Tool-loop recovery and the `web_search` cascade are unchanged.
- Document ids stay contiguous (`D1..Dn`) after blank pages are dropped.

## Review Focus

- A page with only a URL (no title or summary) is not blank, and must still
  become a document (Task 2 test).
- An explicit error page is still not cached and still becomes a "Search
  error" document on the hybrid path. Existing tests cover this.
- Local retrieval results in the same hybrid request are unaffected. The
  existing hybrid tests run local retrieval in parallel.

---

### Task 1: `SearchPage.is_blank` and the cache guard

**Files:**
- Modify: `src/internal/tools/search.py` (`SearchPage`; `search_tool` cache
  guard ~:575)
- Test: `tests/unit/test_search_tools_cache.py`

**Interfaces:**
- Produces: `SearchPage.is_blank -> bool`, a property.

- [ ] **Step 1: Failing tests.** Extend the `serp_calls` fake so that
  `query == "timeout"` returns `[SearchPage(error="", timed_out=True)]` and
  `query == "blank"` returns `[SearchPage()]`. Add these tests:

```python
@pytest.mark.parametrize("query", ["timeout", "blank"])
def test_blank_page_is_not_cached(serp_calls, cache, query):
    _search(query)
    _search(query)
    assert [c[0] for c in serp_calls] == [query, query]


def test_is_blank_only_for_an_empty_page():
    assert SearchPage().is_blank
    assert SearchPage(timed_out=True).is_blank
    assert not SearchPage(url="https://x").is_blank
    assert not SearchPage(error="boom").is_blank
    assert not SearchPage(title="t").is_blank
```

- [ ] **Step 2: Run.** `.venv/bin/python -m pytest
  tests/unit/test_search_tools_cache.py -q`. Expected: FAIL, because
  `is_blank` is missing and the blank page is cached, so only one call is
  made.
- [ ] **Step 3: Implement.**

```python
    @property
    def is_blank(self) -> bool:
        """No title, summary, URL or error: an empty-message provider timeout,
        never a real result or an explicit error."""
        return not (self.title or self.summary or self.url or self.error)
```

  In `search_tool`:

```python
    if cache is not None and pages and not any(
        p.error or p.timed_out or p.is_blank for p in pages
    ):
```

- [ ] **Step 4: Run again.** Expected: PASS. Then commit.

### Task 2: Drop blank pages from search documents

**Files:**
- Modify: `src/internal/servers/web/app.py` (`_documents_from_search_pages`)
- Test: `tests/unit/servers/web/test_hybrid_web_fallback.py`

- [ ] **Step 1: Failing tests.**

```python
@pytest.mark.asyncio
async def test_auto_serpapi_timeout_page_falls_back_to_browser(monkeypatch):
    browser_doc = ContextDocument(
        id="D1", title="GRPO explained", content="Group Relative ...",
        url="https://example.com/grpo", score=0.0,
        metadata={"error": False, "source_provider": "browser"},
    )
    result = await _run_auto(
        monkeypatch,
        serpapi_pages=[SearchPage(error="", timed_out=True)],
        browser_docs=[browser_doc],
        browser_url="http://browser",
    )
    assert [d.title for d in result.documents] == ["GRPO explained"]


@pytest.mark.asyncio
async def test_auto_serpapi_timeout_page_without_browser_is_empty(monkeypatch):
    result = await _run_auto(
        monkeypatch,
        serpapi_pages=[SearchPage(error="", timed_out=True)],
        browser_docs=None,
        browser_url=None,
    )
    assert result.status == "empty"
    assert result.documents == []


def test_blank_pages_are_dropped_and_ids_stay_contiguous():
    docs = _documents_from_search_pages(
        [SearchPage(), SearchPage(url="https://a"), SearchPage(timed_out=True),
         SearchPage(title="b", url="https://b")],
        source_provider="serpapi", query="q",
    )
    assert [d.id for d in docs] == ["D1", "D2"]
    assert [d.url for d in docs] == ["https://a", "https://b"]
```

- [ ] **Step 2: Run.** `.venv/bin/python -m pytest
  tests/unit/servers/web/test_hybrid_web_fallback.py -q`. Expected: FAIL.
  The browser fake raises "must not be called", because the blank doc counts
  as usable, and the dropped-page test fails on its ids.
- [ ] **Step 3: Implement.** In `_documents_from_search_pages`, iterate over
  `[p for p in pages if not p.is_blank]`, so `start_index + offset` stays
  contiguous.
- [ ] **Step 4: Run again.** Expected: PASS. Then run the hybrid, cache and
  search suites and commit.

### Task 3: Verify and ship

- [ ] **Mutation checks.**
  - Remove the blank filter and expect the hybrid tests to go red.
  - Remove `p.is_blank` from the cache guard and expect the cache test to
    go red.
  - Restore both, then delete `__pycache__`.
- [ ] **Full verification.** Run `.venv/bin/python -m pytest tests/unit/ -q`,
  then `ruff check . && ruff format --check .`, then `git diff --check`.
- [ ] **Ship.** Get a fresh review, push, and open the PR.
