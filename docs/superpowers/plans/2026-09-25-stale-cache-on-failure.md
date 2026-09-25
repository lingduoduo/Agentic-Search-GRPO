# Stale Cache On Failure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a web provider or the retrieval server fails, serve the last good cached result (labelled `stale`) instead of nothing, for up to a configurable grace window past the TTL.

**Architecture:** `TTLCache` keeps expired entries for `stale_seconds` more and exposes them only through a new `get_stale`. `search_tool` (web providers) and `SearchClient.retrieve` (retrieval rows) consult `get_stale` only after the live call failed. The tool agent's web cascade routes its SerpAPI leg through `search_tool` so it shares the key, the fresh hits and the fallback.

**Tech Stack:** Python 3.12, pytest (asyncio_mode=auto), FastAPI lifespan.

**Spec:** `docs/superpowers/specs/2026-09-25-stale-cache-on-failure-design.md`

## Global Constraints

- `TTLCache(ttl_seconds, *, max_entries=1024, stale_seconds=0.0, clock=...)`; `stale_seconds=0` reproduces today's behavior exactly; negative raises.
- `get` returns only fresh entries; an in-grace entry is kept (a miss, not deleted); beyond `ttl + stale_seconds` it is deleted on read.
- New env var `AGENTIC_SEARCH_SEARCH_CACHE_STALE_SECONDS`, **default 3600**, `0` means off; documented in `docs/configuration.md`.
- Binding user decision: a genuinely empty list `[]` — and a SerpAPI empty-message timeout with **no** stale entry — stays what it is today (a successful empty search / the blank page). Do not give timeout pages error text; do not change tool-loop recovery.
- Never cache a failure (the existing caching guard is unchanged). Fresh behavior unchanged.
- Mixed lists (some real pages, some error pages) are returned as-is.
- Retrieval: all-or-nothing — if any missing query has no stale row, re-raise exactly as today. Stage-metrics accounting for the failed call is unchanged.
- Stale results are deep copies; the cached value is never mutated. Only the `stale` key is added to metadata (ACL survives).
- Log stale serving at INFO.
- Out of scope: rerank cache, Redis caches, public-data tools, cross-process sharing, stale-while-revalidate.

## Review Focus

1. A `CancelledError` (or other `BaseException`) during the retrieval POST must re-raise, never serve stale — a cancelled request is not a provider failure.
2. An in-grace entry read by `get` must count as a miss and must not be deleted or treated as a hit (else stale data leaks into the fresh path).
3. After a stale fallback, the next successful live call must replace the entry and make it fresh again (the fallback must not pin the stale value).
4. A retrieval batch where some queries are fresh-cached and only the missing ones have stale rows must still fall back, labelling only the stale ones.
5. A stale entry past `ttl + stale_seconds` must not be served (the failure is returned as today).

Each is pinned by a test in the owning task below (1, 4 → Task 5; 2 → Task 1; 3, 5 → Task 3).

## Deviations from the spec (recorded)

- **Row labelling (spec §5).** The spec says each row dict gets a copy of its `metadata` with `"stale": True`. Retrieval items come in three shapes (`{"document": {...}}`, a bare document dict, or a non-dict document with no metadata at all), so the label is set on each built `SearchResult.metadata` instead, after `from_api_item`. The rows are deep-copied out of the cache first, and `from_api_item` builds a fresh metadata dict, so the cached row is never mutated. Same observable result: every stale `SearchResult` carries `metadata["stale"] is True` next to its ACL.
- **Negative env value.** "Validated the same way" as the TTL means `get_env_int` (non-integer rejected). A negative stale value is also rejected at load time with `ValueError`, because otherwise `TTLCache` would raise inside the web lifespan at startup.
- **`stats()` shape.** Adding `stale_hits` changes the dict `stats()` returns; the existing `test_stats_count_hits_and_misses` assertion gains the key.

## File structure

- Modify `src/internal/cache/ttl_cache.py` — grace window, `get_stale`, `stale_hits`.
- Modify `src/internal/cache/serving.py` — `configure_serving_cache(..., stale_seconds=0.0)`.
- Modify `src/internal/configs/app_configs.py` — `ServiceSettings.search_cache_stale_seconds`.
- Modify `src/internal/servers/web/app.py` — `SearchExperienceSettings.search_cache_stale`, lifespan passes it.
- Modify `src/internal/tools/search.py` — `search_tool` fallback; cascade default SerpAPI leg.
- Modify `src/context/retrieval/client.py` — `SearchClient.retrieve` fallback.
- Modify `docs/configuration.md`, `docs/retrieval.md` — env var + behavior.
- Tests: `tests/unit/cache/test_ttl_cache.py`, `tests/unit/cache/test_serving_cache.py`, `tests/unit/servers/web/test_serving_cache_lifespan.py`, `tests/unit/test_search_tools_cache.py`, `tests/unit/test_web_cascade_search.py`, `tests/unit/test_search_client_cache.py`.

---

### Task 1: `TTLCache` grace window + `configure_serving_cache(stale_seconds=)`

**Files:**
- Modify: `src/internal/cache/ttl_cache.py`, `src/internal/cache/serving.py`
- Test: `tests/unit/cache/test_ttl_cache.py`, `tests/unit/cache/test_serving_cache.py`

**Interfaces:**
- Produces: `TTLCache(..., stale_seconds: float = 0.0, ...)`, `TTLCache.get_stale(key) -> Any | None`, `stats()` gains `"stale_hits"`; `configure_serving_cache(ttl_seconds, *, max_entries=1024, stale_seconds=0.0)`.

- [x] **Step 1: Write the failing tests** (append to `tests/unit/cache/test_ttl_cache.py`; add `import pytest`; update the existing stats assertion to include `"stale_hits": 0`)

```python
def test_in_grace_entry_is_a_miss_but_kept_and_served_stale():
    clock = _Clock()
    cache = TTLCache(ttl_seconds=10, stale_seconds=100, clock=clock)
    cache.set("k", "v")
    clock.now += 50
    assert cache.get("k") is None
    assert cache.stats()["size"] == 1
    assert cache.stats()["hits"] == 0
    assert cache.get_stale("k") == "v"
    assert cache.get("k") is None  # still not fresh


def test_get_stale_returns_fresh_entries_too():
    cache = TTLCache(ttl_seconds=10, stale_seconds=100)
    cache.set("k", "v")
    assert cache.get_stale("k") == "v"
    assert cache.get_stale("missing") is None


def test_beyond_grace_both_miss_and_entry_is_deleted():
    clock = _Clock()
    cache = TTLCache(ttl_seconds=10, stale_seconds=100, clock=clock)
    cache.set("k", "v")
    cache.set("j", "w")
    clock.now += 110
    assert cache.get("k") is None
    assert cache.get_stale("j") is None
    assert cache.get_stale("k") is None
    assert cache.stats()["size"] == 0


def test_zero_stale_seconds_is_todays_behavior():
    clock = _Clock()
    cache = TTLCache(ttl_seconds=10, clock=clock)
    cache.set("k", "v")
    clock.now += 10
    assert cache.get_stale("k") is None
    cache.set("j", "w")
    clock.now += 10
    assert cache.get("j") is None
    assert cache.stats()["size"] == 0


def test_stale_entries_count_toward_the_lru_bound():
    clock = _Clock()
    cache = TTLCache(ttl_seconds=10, max_entries=2, stale_seconds=100, clock=clock)
    cache.set("a", 1)
    clock.now += 20  # a is stale
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get_stale("a") is None
    assert cache.stats()["size"] == 2


def test_stale_hits_are_counted():
    clock = _Clock()
    cache = TTLCache(ttl_seconds=10, stale_seconds=100, clock=clock)
    cache.set("k", "v")
    clock.now += 20
    cache.get_stale("k")
    cache.get_stale("k")
    cache.get_stale("missing")
    assert cache.stats()["stale_hits"] == 2


def test_negative_stale_seconds_raises():
    with pytest.raises(ValueError):
        TTLCache(ttl_seconds=10, stale_seconds=-1)
```

In `tests/unit/cache/test_serving_cache.py`:

```python
def test_stale_seconds_reach_the_cache():
    serving.configure_serving_cache(30, stale_seconds=600)
    assert serving.serving_cache()._stale == 600.0


def test_stale_window_defaults_to_off():
    serving.configure_serving_cache(30)
    assert serving.serving_cache()._stale == 0.0
```

- [x] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/unit/cache -q -p no:cacheprovider` → FAIL (`unexpected keyword argument 'stale_seconds'`).

- [x] **Step 3: Implement**

```python
    def __init__(self, ttl_seconds, *, max_entries=1024, stale_seconds: float = 0.0, clock=time.monotonic):
        ...
        if stale_seconds < 0:
            raise ValueError("stale_seconds must not be negative")
        self._stale = float(stale_seconds)
        self._stale_hits = 0

    def get(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            expires_at, value = entry
            now = self._clock()
            if now >= expires_at:
                # Past the grace window it is gone; inside it, keep it for get_stale.
                if now >= expires_at + self._stale:
                    del self._entries[key]
                self._misses += 1
                return None
            ...

    def get_stale(self, key):
        """The entry if it is fresh or inside the grace window; for callers whose live call just failed."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if self._clock() >= expires_at + self._stale:
                del self._entries[key]
                return None
            self._stale_hits += 1
            return value
```

`stats()` adds `"stale_hits": self._stale_hits`. `configure_serving_cache` passes `stale_seconds` through.

- [x] **Step 4: Run tests** → PASS.
- [x] **Step 5: Commit** — `git add src/internal/cache tests/unit/cache && git commit`.

### Task 2: Configuration, lifespan, docs

**Files:**
- Modify: `src/internal/configs/app_configs.py`, `src/internal/servers/web/app.py`, `docs/configuration.md`, `docs/retrieval.md`
- Test: `tests/unit/servers/web/test_serving_cache_lifespan.py`

**Interfaces:**
- Consumes: `configure_serving_cache(ttl, stale_seconds=...)`, `TTLCache._stale` (test only).
- Produces: `ServiceSettings.search_cache_stale_seconds: int = 3600`; `SearchExperienceSettings.search_cache_stale: int = 3600`.

- [x] **Step 1: Failing tests** (append)

```python
def test_stale_window_defaults_to_an_hour():
    settings = load_app_settings({})
    assert settings.services.search_cache_stale_seconds == 3600
    assert SearchExperienceSettings.from_app_settings(settings).search_cache_stale == 3600


def test_stale_window_env_is_parsed_and_zero_is_off():
    settings = load_app_settings({"AGENTIC_SEARCH_SEARCH_CACHE_STALE_SECONDS": "0"})
    assert settings.services.search_cache_stale_seconds == 0
    assert SearchExperienceSettings.from_app_settings(settings).search_cache_stale == 0


@pytest.mark.parametrize("value", ["soon", "-5"])
def test_bad_stale_window_is_rejected(value):
    with pytest.raises(ValueError, match="AGENTIC_SEARCH_SEARCH_CACHE_STALE_SECONDS"):
        load_app_settings({"AGENTIC_SEARCH_SEARCH_CACHE_STALE_SECONDS": value})


def test_lifespan_passes_the_stale_window(tmp_path):
    app = create_web_app(
        SearchExperienceSettings(
            db_path=tmp_path / "c.sqlite3", search_cache_ttl=42, search_cache_stale=900
        )
    )
    with TestClient(app):
        assert serving.serving_cache()._stale == 900.0
```

- [x] **Step 2: Run** → FAIL (`AttributeError: search_cache_stale_seconds`).
- [x] **Step 3: Implement** — `ServiceSettings.search_cache_stale_seconds: int = 3600`; in `load_app_settings` read it with `get_env_int(source, "AGENTIC_SEARCH_SEARCH_CACHE_STALE_SECONDS", 3600)` into a local, raise `ValueError("AGENTIC_SEARCH_SEARCH_CACHE_STALE_SECONDS must not be negative.")` when `< 0`, pass it in. `SearchExperienceSettings.search_cache_stale: int = 3600` mapped in `from_app_settings`; lifespan: `configure_serving_cache(settings.search_cache_ttl, stale_seconds=settings.search_cache_stale)`. Docs row in `docs/configuration.md` after the TTL row:

`| AGENTIC_SEARCH_SEARCH_CACHE_STALE_SECONDS | Seconds past the TTL an expired serving-cache entry is kept to answer when the live call fails (a web provider's error, timeout or open circuit; a retrieval-server error); such results carry metadata stale: true. Defaults to 3600, 0 disables the fallback. See [Serving cache](retrieval.md#serving-cache) |`

and a paragraph under `docs/retrieval.md` "Serving cache".
- [x] **Step 4: Run** lifespan tests + `tests/unit/test_documented_env_vars.py` → PASS.
- [x] **Step 5: Commit.**

### Task 3: `search_tool` falls back to stale pages

**Files:**
- Modify: `src/internal/tools/search.py` (`search_tool`, import `replace` from `dataclasses`)
- Test: `tests/unit/test_search_tools_cache.py`

**Interfaces:**
- Consumes: `TTLCache.get_stale`.
- Produces: on failure `search_tool` returns deep copies of the cached pages with `metadata["stale"] = True`.

- [x] **Step 1: Failing tests** (append; imports `copy`, `TTLCache`, `SERPAPI_CIRCUIT_OPEN_ERROR`)

```python
class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return _Clock()


@pytest.fixture
def stale_cache(monkeypatch, clock):
    cache = TTLCache(60, stale_seconds=3600, clock=clock)
    monkeypatch.setattr(serving, "_cache", cache)
    return cache


@pytest.fixture
def serp(monkeypatch):
    state = {
        "reply": [SearchPage(title="t", summary="s", url="https://x", metadata={"acl": ["public"]})],
        "calls": 0,
    }

    async def _fake(query, *, page, page_size, timeout_seconds):
        state["calls"] += 1
        return copy.deepcopy(state["reply"])

    monkeypatch.setattr("src.internal.tools.search.serpapi_search", _fake)
    return state


FAILURES = {
    "timeout": [SearchPage(timed_out=True)],
    "blank": [SearchPage()],
    "error": [SearchPage(error="rate limited")],
    "circuit_open": [SearchPage(error=SERPAPI_CIRCUIT_OPEN_ERROR)],
}


@pytest.mark.parametrize("failure", FAILURES)
def test_failure_after_expiry_serves_stale_pages(serp, stale_cache, clock, failure):
    _search("q")
    clock.now += 61
    serp["reply"] = FAILURES[failure]
    pages = _search("q")
    assert serp["calls"] == 2
    assert [p.url for p in pages] == ["https://x"]
    assert pages[0].metadata == {"acl": ["public"], "stale": True}


def test_empty_result_never_falls_back(serp, stale_cache, clock):
    _search("q")
    clock.now += 61
    serp["reply"] = []
    assert _search("q") == []


@pytest.mark.parametrize("failure", FAILURES)
def test_no_stale_entry_returns_the_failure_unchanged(serp, stale_cache, failure):
    serp["reply"] = FAILURES[failure]
    assert _search("q") == FAILURES[failure]


def test_beyond_the_grace_window_returns_the_failure(serp, stale_cache, clock):
    _search("q")
    clock.now += 60 + 3600 + 1
    serp["reply"] = FAILURES["timeout"]
    assert _search("q") == FAILURES["timeout"]


def test_mixed_list_is_returned_as_is(serp, stale_cache, clock):
    _search("q")
    clock.now += 61
    mixed = [SearchPage(title="new", url="https://new"), SearchPage(error="partial")]
    serp["reply"] = mixed
    assert _search("q") == mixed


def test_stale_hit_is_a_copy(serp, stale_cache, clock):
    _search("q")
    clock.now += 61
    serp["reply"] = FAILURES["error"]
    _search("q")[0].metadata["acl"].append("user:mallory")
    again = _search("q")
    assert again[0].metadata == {"acl": ["public"], "stale": True}
    cached = stale_cache.get_stale(("web", "serpapi", "q", 1, 5))
    assert cached[0].metadata == {"acl": ["public"]}


def test_recovery_after_a_stale_fallback_is_fresh_again(serp, stale_cache, clock):
    _search("q")
    clock.now += 61
    serp["reply"] = FAILURES["error"]
    _search("q")
    serp["reply"] = [SearchPage(title="t2", url="https://y")]
    assert [p.url for p in _search("q")] == ["https://y"]
    fresh = _search("q")
    assert serp["calls"] == 3
    assert fresh[0].metadata == {}
```

- [x] **Step 2: Run** → the fallback tests FAIL (failure pages returned).
- [x] **Step 3: Implement** — after the provider call, before the caching guard:

```python
    # A failed live lookup (every page an error, timeout or blank; an open
    # circuit is an error page) is answered from a stale cached copy when one
    # is inside the grace window. An empty list is a successful empty search.
    if cache is not None and pages and all(
        p.error or p.timed_out or p.is_blank for p in pages
    ):
        stale = cache.get_stale(cache_key)
        if stale is not None:
            logger.info(
                "search_tool: %s failed for %r; serving stale cached pages",
                provider,
                query,
            )
            return [
                replace(p, metadata={**p.metadata, "stale": True})
                for p in copy.deepcopy(stale)
            ]
```

- [x] **Step 4: Run** `tests/unit/test_search_tools_cache.py` → PASS.
- [x] **Step 5: Commit.**

### Task 4: The cascade's SerpAPI leg goes through `search_tool`

**Files:**
- Modify: `src/internal/tools/search.py` (`make_web_cascade_search`)
- Test: `tests/unit/test_web_cascade_search.py`

**Interfaces:**
- Consumes: `search_tool(query, provider="serpapi", page=, page_size=, timeout_seconds=)` and its stale fallback (Task 3).
- Produces: `make_web_cascade_search(serpapi_fn=None, ...)` — `None` means the default wrapper around `search_tool`; injected `serpapi_fn` unchanged.

- [x] **Step 1: Failing tests** (append; imports `copy`, `pytest`, `serving`, `TTLCache`)

```python
@pytest.fixture
def serp_calls(monkeypatch):
    calls: list = []
    reply = {"pages": [SearchPage(title="t", summary="s", url="https://x")]}

    async def _fake(query, *, page, page_size, timeout_seconds):
        calls.append(query)
        return copy.deepcopy(reply["pages"])

    monkeypatch.setattr("src.internal.tools.search.serpapi_search", _fake)
    return calls, reply


def test_default_serpapi_leg_is_served_from_the_serving_cache(serp_calls):
    calls, _ = serp_calls
    serving.configure_serving_cache(60)
    try:
        fn = make_web_cascade_search(browser_search_url=None)
        asyncio.run(fn("q"))
        pages = asyncio.run(fn("q"))
    finally:
        serving.reset_serving_cache()
    assert calls == ["q"]
    assert [p.url for p in pages] == ["https://x"]


def test_serpapi_failure_with_stale_entry_needs_no_browser(monkeypatch, serp_calls):
    calls, reply = serp_calls
    now = [100.0]
    monkeypatch.setattr(serving, "_cache", TTLCache(60, stale_seconds=3600, clock=lambda: now[0]))

    async def browser(query, **kw):
        raise AssertionError("browser must not run when a stale SerpAPI result exists")

    fn = make_web_cascade_search(browser_search_url="http://browser/retrieve", browser_fn=browser)
    asyncio.run(fn("q"))
    now[0] += 61
    reply["pages"] = [SearchPage(timed_out=True)]
    pages = asyncio.run(fn("q"))
    assert calls == ["q", "q"]
    assert pages[0].url == "https://x"
    assert pages[0].metadata["stale"] is True
```

- [x] **Step 2: Run** → FAIL (`calls == ["q", "q"]` in the first; the default leg bypasses the cache).
- [x] **Step 3: Implement**

```python
async def _serpapi_via_search_tool(query, *, page, page_size, timeout_seconds):
    """The cascade's default SerpAPI leg: through search_tool, so it shares the
    serving cache's key, fresh hits and stale fallback (the circuit breaker is
    still consulted inside serpapi_search)."""
    return await search_tool(query, provider="serpapi", page=page, page_size=page_size, timeout_seconds=timeout_seconds)


def make_web_cascade_search(*, browser_search_url=None, serpapi_fn=_serpapi_via_search_tool, browser_fn=search_tool):
```

- [x] **Step 4: Run** cascade + domain + timeout-identity + circuit-breaker-site tests → PASS.
- [x] **Step 5: Commit.**

### Task 5: `SearchClient.retrieve` falls back to stale rows

**Files:**
- Modify: `src/context/retrieval/client.py`
- Test: `tests/unit/test_search_client_cache.py`

**Interfaces:**
- Consumes: `TTLCache.get_stale`.
- Produces: on a failed POST with stale rows for every missing query, `retrieve` returns them with each `SearchResult.metadata["stale"] = True`.

- [x] **Step 1: Failing tests** (append; import `TTLCache`)

```python
class _FailingSession(_FakeSession):
    def __init__(self, posts, exc):
        super().__init__(posts)
        self._exc = exc

    def post(self, url, json):
        self._posts.append(json)
        raise self._exc


@pytest.fixture
def stale_cache(monkeypatch):
    now = [100.0]
    cache = TTLCache(60, stale_seconds=3600, clock=lambda: now[0])
    monkeypatch.setattr(serving, "_cache", cache)
    return cache, now


def _fail_with(monkeypatch, exc, posts):
    monkeypatch.setattr(
        "src.context.retrieval.client.aiohttp.ClientSession",
        lambda *, timeout: _FailingSession(posts, exc),
    )
    monkeypatch.setattr("src.context.retrieval.client._client_policy", lambda: _NoBackoff())


class _NoBackoff:
    backoff_base_seconds = 0.0
    timeout_seconds = 1.0
    max_retries = 1


def test_failed_post_serves_stale_rows_labelled(monkeypatch, posts, stale_cache):
    cache, now = stale_cache
    _run(_client().retrieve(["a", "b"]))
    now[0] += 61
    _fail_with(monkeypatch, ConnectionError("down"), [])
    rows = _run(_client().retrieve(["a", "b"]))
    assert [row[0].title for row in rows] == ["a", "b"]
    assert rows[0][0].metadata == {"acl": ["public"], "stale": True}
    key = ("retrieve", "http://localhost:8001/retrieve", "a", 5, "")  # default topk
    assert "stale" not in cache.get_stale(key)[0]["document"]["metadata"]


def test_one_query_without_a_stale_row_raises(monkeypatch, posts, stale_cache):
    _, now = stale_cache
    _run(_client().retrieve(["a"]))
    now[0] += 61
    _fail_with(monkeypatch, ConnectionError("down"), [])
    with pytest.raises(RuntimeError):
        _run(_client().retrieve(["a", "b"]))


def test_fresh_hits_plus_stale_misses_label_only_the_stale(monkeypatch, posts, stale_cache):
    _, now = stale_cache
    _run(_client().retrieve(["a"]))
    now[0] += 61
    _run(_client().retrieve(["b"]))  # b is fresh, a is stale
    _fail_with(monkeypatch, ConnectionError("down"), [])
    rows = _run(_client().retrieve(["a", "b"]))
    assert rows[0][0].metadata.get("stale") is True
    assert "stale" not in rows[1][0].metadata


def test_cancellation_never_serves_stale(monkeypatch, posts, stale_cache):
    _, now = stale_cache
    _run(_client().retrieve(["a"]))
    now[0] += 61
    _fail_with(monkeypatch, asyncio.CancelledError(), [])
    with pytest.raises(asyncio.CancelledError):
        _run(_client().retrieve(["a"]))
```

- [x] **Step 2: Run** → FAIL (RuntimeError raised instead of stale rows).
- [x] **Step 3: Implement** — inside the existing `except BaseException` after `note_retrieval(...)`:

```python
            except BaseException as exc:
                note_retrieval(...)  # unchanged
                # A failed server answers from stale cached rows, but only when
                # every missing query has one: no mix of stale and missing.
                # Cancellation is not a failure and always propagates.
                stale = (
                    [cache.get_stale(keys[i]) for i in missing]
                    if cache is not None and isinstance(exc, Exception)
                    else []
                )
                if not stale or any(row is None for row in stale):
                    raise
                logger.info("SearchClient.retrieve: %s failed; serving %d stale rows", self.config.url, len(missing))
                for index, row in zip(missing, stale):
                    rows_by_index[index] = copy.deepcopy(row)
                results = [[SearchResult.from_api_item(item) for item in rows_by_index.get(i, [])] for i in range(len(queries))]
                for index in missing:
                    for result in results[index]:
                        result.metadata["stale"] = True
                return results
```

- [x] **Step 4: Run** `tests/unit/test_search_client_cache.py` and the stage-metrics tests → PASS.
- [x] **Step 5: Commit.**

### Task 6: Mutation checks and full verification

- [x] Make `get` delete in-grace entries (`if now >= expires_at + self._stale:` → unconditional delete) → `test_in_grace_entry_is_a_miss_but_kept_and_served_stale` red. Restore, `find . -name __pycache__ -path '*src*' -exec rm -rf {} +`.
- [x] Remove the `search_tool` fallback block → `test_failure_after_expiry_serves_stale_pages` red. Restore, clear pycache.
- [x] Remove the retrieval fallback (`raise` unconditionally) → `test_failed_post_serves_stale_rows_labelled` red. Restore, clear pycache.
- [x] `git diff` shows no residue; `.venv/bin/python -m pytest tests/unit/ -q -p no:cacheprovider` → 0 failures; `ruff check . && ruff format --check .`; `git diff --check origin/main...HEAD`.
