# Stale-cache follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A retrieval 4xx is never hidden behind stale rows, every stale serve is counted on `/metrics`, and the stale-row copy is pinned by a test.

**Architecture:** One new Prometheus counter + validated helper in `src/internal/observability/prometheus.py`, called once from each existing stale fallback (`search_tool` for web, `SearchClient.retrieve` for retrieval). `SearchClient.retrieve` gains a small `_is_client_error` predicate that suppresses the stale fallback for a 4xx other than 429. No new modules.

**Tech Stack:** Python 3.12, aiohttp 3.9.3, prometheus_client, pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-stale-cache-followups-design.md`

## Global Constraints

- No fallback when the exception, or its `__cause__`, is an `aiohttp.ClientResponseError` with `status < 500` and `status != 429`. A 429, a 5xx, transport errors and the exhausted-retries `RuntimeError` still fall back.
- Metric name: `agentic_search_stale_cache_serves_total{source}`; `source` is exactly `web` or `retrieval`; an unknown source raises `ValueError`.
- Helper: `observe_stale_serve(source: str) -> None`, incremented once per fallback that actually serves (once per `retrieve` call, not per row).
- `prometheus.py` stays `prometheus_client`-only; importing it from `src/context/...` must not pull in torch.
- Binding user decision: a SerpAPI empty-message timeout with no stale entry stays a successful empty search. Tool-loop recovery is not changed.
- ACL doc sentence: a document made private can still be served from a stale row for up to TTL + grace, 65 minutes by default; `AGENTIC_SEARCH_SEARCH_CACHE_STALE_SECONDS=0` disables.

## Review Focus

- A 4xx raised with a stale row present must re-raise the *original* `ClientResponseError` (same type, same status), not a `RuntimeError` — pinned in Task 2 (`exc_info.value.status`).
- A failed retrieval that raises (a missing stale row, or a 4xx) must not bump the counter — pinned in Task 3.
- A batch of 3 stale-served queries counts 1, not 3 — pinned in Task 3.
- A web failure with no stale entry (including the blank empty-message timeout) counts nothing and still returns the failure unchanged — pinned in Task 3.
- `asyncio.CancelledError` still never serves stale and never counts — existing `test_cancellation_never_serves_stale`, re-run in Task 3.

## Spec deviations (recorded)

- `docs/retrieval.md` also gets a clause that a 4xx other than 429 never falls back, and a pointer to the new metric: the serving-cache section describes the fallback's trigger, and it would otherwise be wrong.
- Checked: `_post_json` re-raises every `status < 500` directly (including 429), so a 4xx never reaches the `RuntimeError` cause today. The `__cause__` check is kept as the spec requires; it is cheap and robust to a future retry change.

---

### Task 1: The stale-serve counter

**Files:**
- Modify: `src/internal/observability/prometheus.py` (append after `observe_tool_attempt`)
- Modify: `docs/observability-metrics.md`
- Test: `tests/unit/observability/test_prometheus.py`

**Interfaces:**
- Produces: `observe_stale_serve(source: str) -> None` in `src.internal.observability.prometheus`; sample name `agentic_search_stale_cache_serves_total`, label `source`.

- [x] **Step 1: Write the failing tests** (append to `tests/unit/observability/test_prometheus.py`, add `observe_stale_serve` and `import pytest`)

```python
def _stale(source: str) -> float:
    return _value("agentic_search_stale_cache_serves_total", {"source": source})


@pytest.mark.parametrize("source", ["web", "retrieval"])
def test_observe_stale_serve_counts_by_source(source):
    before = {s: _stale(s) for s in ("web", "retrieval")}
    observe_stale_serve(source)
    for s in ("web", "retrieval"):
        assert _stale(s) == before[s] + (1 if s == source else 0)


def test_observe_stale_serve_rejects_an_unknown_source():
    with pytest.raises(ValueError):
        observe_stale_serve("rerank")
    assert _stale("rerank") == 0.0
```

- [x] **Step 2: Run** `.venv/bin/python -m pytest tests/unit/observability/test_prometheus.py -q -p no:cacheprovider` — expect ImportError on `observe_stale_serve`.

- [x] **Step 3: Implement**

```python
_STALE_SERVES = Counter(
    "agentic_search_stale_cache_serves_total",
    "Failed live lookups answered from a stale serving-cache entry, by source.",
    ("source",),
    registry=REGISTRY,
)


def observe_stale_serve(source: str) -> None:
    """Count one fallback that served stale rows (once per call, not per row)."""
    if source not in {"web", "retrieval"}:
        raise ValueError("Unknown stale-serve source")
    _STALE_SERVES.labels(source).inc()
```

- [x] **Step 4: Docs.** In `docs/observability-metrics.md`: add the table row; add a "Stale cache serves" semantics paragraph (web = `search_tool` fallback, retrieval = `SearchClient.retrieve` fallback, once per serving call, a stale-answered tool call still counts as tool `success`); add the query

```promql
sum by (source) (rate(agentic_search_stale_cache_serves_total[5m]))
```

- [x] **Step 5: Run tests** — pass. **Commit** `Metrics: a stale-cache serve counter by source`.

### Task 2: A retrieval 4xx is never answered from stale rows

**Files:**
- Modify: `src/context/retrieval/client.py` (new `_is_client_error`; the `stale = (...)` condition in `retrieve`)
- Modify: `docs/retrieval.md` (serving-cache section)
- Test: `tests/unit/test_search_client_cache.py`

**Interfaces:**
- Produces: module-private `_is_client_error(exc: BaseException) -> bool`.

- [x] **Step 1: Write the failing tests** (append; add `import aiohttp` at top)

```python
def _http_error(status: int) -> aiohttp.ClientResponseError:
    return aiohttp.ClientResponseError(None, (), status=status)


@pytest.mark.parametrize("status", [400, 403])
def test_client_error_reraises_despite_a_stale_row(
    monkeypatch, posts, stale_cache, status
):
    _, now = stale_cache
    _run(_client().retrieve(["a"]))
    now[0] += 61
    _fail_with(monkeypatch, _http_error(status))
    with pytest.raises(aiohttp.ClientResponseError) as exc_info:
        _run(_client().retrieve(["a"]))
    assert exc_info.value.status == status


@pytest.mark.parametrize("status", [429, 503, 500])
def test_rate_limit_and_server_errors_serve_stale(
    monkeypatch, posts, stale_cache, status
):
    # 429 re-raises directly from _post_json; 5xx exhaust the single retry and
    # arrive as RuntimeError whose __cause__ is the ClientResponseError.
    _, now = stale_cache
    _run(_client().retrieve(["a"]))
    now[0] += 61
    _fail_with(monkeypatch, _http_error(status))
    rows = _run(_client().retrieve(["a"]))
    assert rows[0][0].metadata == {"acl": ["public"], "stale": True}


def test_is_client_error_reads_the_cause():
    from src.context.retrieval.client import _is_client_error

    wrapped = RuntimeError("retries exhausted")
    wrapped.__cause__ = _http_error(404)
    assert _is_client_error(wrapped)
    wrapped.__cause__ = _http_error(500)
    assert not _is_client_error(wrapped)
    assert not _is_client_error(_http_error(429))
```

- [x] **Step 2: Run** — the 400/403 tests and `test_is_client_error_reads_the_cause` fail (stale served / ImportError); 429/503/500 pass already.

- [x] **Step 3: Implement**

```python
def _is_client_error(exc: BaseException) -> bool:
    """A 4xx other than 429, raised directly or as the retry error's cause.

    Such a request is wrong (a rejected filter, missing auth), so it must
    surface rather than be hidden behind stale rows.
    """
    return any(
        isinstance(err, aiohttp.ClientResponseError)
        and err.status < 500
        and err.status != 429
        for err in (exc, exc.__cause__)
    )
```

and in `retrieve`:

```python
                stale = (
                    [cache.get_stale(keys[i]) for i in missing]
                    if cache is not None
                    and isinstance(exc, Exception)
                    and not _is_client_error(exc)
                    else []
                )
```

and update the comment above it to mention the 4xx rule.

- [x] **Step 4: Docs.** `docs/retrieval.md`: a 4xx other than 429 is raised, never answered from stale rows.

- [x] **Step 5: Run tests** — pass. **Commit** `Retrieval: a 4xx is raised, never answered from stale rows`.

### Task 3: Count each stale serve

**Files:**
- Modify: `src/context/retrieval/client.py` (import; call after the stale `logger.info`)
- Modify: `src/internal/tools/search.py` (import; call after the stale `logger.info`)
- Test: `tests/unit/test_search_client_cache.py`, `tests/unit/test_search_tools_cache.py`

**Interfaces:**
- Consumes: `observe_stale_serve` from Task 1; `_is_client_error` behaviour from Task 2.

- [x] **Step 1: Write the failing tests**

In `tests/unit/test_search_client_cache.py`:

```python
from src.internal.observability.prometheus import REGISTRY


def _stale_serves() -> float:
    return (
        REGISTRY.get_sample_value(
            "agentic_search_stale_cache_serves_total", {"source": "retrieval"}
        )
        or 0.0
    )


def test_stale_serve_counts_once_per_call(monkeypatch, posts, stale_cache):
    _, now = stale_cache
    _run(_client().retrieve(["a", "b", "c"]))
    now[0] += 61
    _fail_with(monkeypatch, ConnectionError("down"))
    before = _stale_serves()
    _run(_client().retrieve(["a", "b", "c"]))
    assert _stale_serves() == before + 1


def test_no_stale_serve_counts_nothing(monkeypatch, posts, stale_cache):
    _, now = stale_cache
    _run(_client().retrieve(["a"]))
    now[0] += 61
    before = _stale_serves()
    _fail_with(monkeypatch, ConnectionError("down"))
    with pytest.raises(RuntimeError):
        _run(_client().retrieve(["a", "b"]))  # b has no stale row
    _fail_with(monkeypatch, _http_error(400))
    with pytest.raises(aiohttp.ClientResponseError):
        _run(_client().retrieve(["a"]))
    assert _stale_serves() == before
```

In `tests/unit/test_search_tools_cache.py`:

```python
from src.internal.observability.prometheus import REGISTRY


def _stale_serves() -> float:
    return (
        REGISTRY.get_sample_value(
            "agentic_search_stale_cache_serves_total", {"source": "web"}
        )
        or 0.0
    )


def test_stale_fallback_counts_one_web_serve(serp, stale_cache, clock):
    _search("q")
    clock.now += 61
    serp["reply"] = FAILURES["error"]
    before = _stale_serves()
    _search("q")
    assert _stale_serves() == before + 1


@pytest.mark.parametrize("failure", FAILURES)
def test_failure_without_stale_entry_counts_nothing(serp, stale_cache, failure):
    before = _stale_serves()
    serp["reply"] = FAILURES[failure]
    assert _search("q") == FAILURES[failure]
    assert _stale_serves() == before
```

- [x] **Step 2: Run** — the two "counts once/one" tests fail; the "counts nothing" tests pass.

- [x] **Step 3: Implement.** `client.py`: `from src.internal.observability.prometheus import observe_stale_serve`; call `observe_stale_serve("retrieval")` right after the stale `logger.info`. `search.py`: `from ..observability.prometheus import observe_stale_serve`; call `observe_stale_serve("web")` right after its stale `logger.info`.

- [x] **Step 4: Run** both files + `tests/unit/observability/`; run the torch-blocked import probe (`sys.meta_path` finder raising for `torch`, then import `src.context.retrieval.client` and `src.internal.tools.search`) — expect `ok False`.

- [x] **Step 5: Commit** `Stale serves are counted on /metrics (web and retrieval)`.

### Task 4: Pin the stale-row copy; document the ACL window

**Files:**
- Test: `tests/unit/test_search_client_cache.py`
- Modify: `docs/retrieval.md`

- [x] **Step 1: Write the test** (passes today; its red is proven by the mutation check in Step 2)

```python
def test_mutating_a_stale_result_cannot_poison_the_next_stale_serve(
    monkeypatch, posts, stale_cache
):
    _, now = stale_cache
    _run(_client().retrieve(["a"]))
    now[0] += 61
    _fail_with(monkeypatch, ConnectionError("down"))
    first = _run(_client().retrieve(["a"]))
    first[0][0].metadata["acl"].append("user:mallory")
    second = _run(_client().retrieve(["a"]))
    assert second[0][0].metadata == {"acl": ["public"], "stale": True}
```

- [x] **Step 2: Mutation.** Change `rows_by_index[index] = copy.deepcopy(row)` (stale loop) to `= row`; the test must go red; restore; delete `__pycache__` under `src/context/retrieval`.

- [x] **Step 3: Docs.** `docs/retrieval.md` serving-cache section: "During a retrieval outage, a document made private can still be served from a stale row for up to TTL + grace, 65 minutes by default; set `AGENTIC_SEARCH_SEARCH_CACHE_STALE_SECONDS=0` where that is unacceptable."

- [x] **Step 4: Commit** `Tests: a mutated stale result cannot poison the next stale serve; document the ACL window`.

### Final verification

- Mutation checks (restore + purge `__pycache__` after each): drop the `_is_client_error` guard → 400/403 tests red; remove each `observe_stale_serve` call → its counter test red; alias the stale deepcopy → copy test red.
- `.venv/bin/python -m pytest tests/unit/ -q -p no:cacheprovider` — 0 failures.
- `ruff check . && ruff format --check .`; `git diff --check origin/main...HEAD`.
