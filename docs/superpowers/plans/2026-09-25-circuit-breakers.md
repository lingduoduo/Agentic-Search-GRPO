# Circuit Breakers for Serving Dependencies — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SerpAPI, the browser search leg, the HTTP reranker and the remote LLM each get a per-process circuit breaker, so that once a dependency is down later requests skip straight to the existing degradation path instead of paying the full timeout.

**Architecture:** A stdlib-only `src/internal/resilience/circuit_breaker.py` holds a thread-safe closed/open/half-open state machine and a lazy process-wide registry keyed by name. Policy (`failure_threshold`, `open_seconds`) is a new `[circuit_breaker]` table in `timeouts.toml`. Each call site calls `before_call()` and then records success or failure itself; `/api/admin/metrics` reports every breaker's snapshot.

**Tech Stack:** Python ≥3.10 stdlib (`threading`, `dataclasses`, `time.monotonic`), aiohttp 3.9.3, httpx, FastAPI, pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-circuit-breakers-design.md`

## Global Constraints

- `src/internal/resilience/circuit_breaker.py` is torch-free, stdlib only (plus `src.internal.configs.timeouts`).
- Policy lives in `timeouts.toml` only: `[circuit_breaker] failure_threshold = 5`, `open_seconds = 30.0`. No new timeout/retry literals in code (the drift guard `tests/unit/test_timeout_policy_drift.py` must stay green).
- Breakers read `get_timeout_policies()` when they are created, never at import.
- Breaker names exactly: `serpapi`, `browser_search`, `rerank`, `remote_llm`.
- Failure = transport/connect error, timeout, HTTP 5xx, HTTP 429. Success = any response the dependency produced, including 4xx and empty results. Configuration errors (missing SerpAPI key) touch no breaker. `asyncio.CancelledError` records nothing.
- Degraded messages, verbatim:
  - SerpAPI: `"SerpAPI is temporarily skipped after repeated failures (circuit open)."`
  - Browser: `"Browser search is temporarily skipped after repeated failures (circuit open)."`
  - Remote LLM: `f"Inference server at {base_url} is temporarily skipped after repeated failures (circuit open)."` raised as `RuntimeError`.
  - Rerank: raises `CircuitOpenError`; `DefaultRankingStage` reports `rerank_status = "circuit_open"`, `degraded = True`.
- `CircuitOpenError` subclasses `RuntimeError`.
- Python 3.10 floor: no `datetime.UTC`, no 3.11-only syntax.
- Branch `feat/circuit-breakers`; never commit to `main`; commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Spec deviations (adapted to the real code)

1. **A half-open probe that never reports does not wedge the breaker.** The spec says `CancelledError` records nothing, and each site records nothing for exceptions that say nothing about the dependency (e.g. a streaming `on_token` callback raising). Under the spec's literal state machine the probe would then be "out" forever and the breaker would fail fast permanently. Fix: while half-open with a probe out, a new probe is admitted once `open_seconds` have passed since the last probe was admitted. Pinned by a test in Task 2.
2. **Remote LLM transport errors beyond connect/timeout.** Today `generate` converts only `ClientConnectorError`/`asyncio.TimeoutError` to `RuntimeError`; other `aiohttp.ClientError`s (e.g. `ServerDisconnectedError`) propagate unchanged. They now also count as failures (spec: "a transport or connect error"), but their propagation is unchanged.
3. **Autouse reset fixture lives in `tests/conftest.py`**, beside the existing `_fresh_timeout_policies`, not only in the new test dir: existing SerpAPI/cascade/rerank tests use the same process-wide registry and would otherwise leak state into each other.

## Review Focus

1. **A cancelled half-open probe** (client disconnects mid-request) — the breaker must admit a new probe after `open_seconds`, not fail fast forever — test in Task 2 (`test_unreported_probe_is_replaced_after_open_seconds`).
2. **A streaming client that disconnects** (`on_token` raises) — the LLM server did nothing wrong; this must not count toward opening `remote_llm` — test in Task 6 (`test_stream_callback_error_is_not_a_failure`).
3. **SerpAPI rate-limits (HTTP 429)** — counts as a failure and opens the breaker, while a 4xx such as 401 does not — test in Task 3.
4. **Many worker threads hit a half-open breaker at once** — exactly one is admitted — test in Task 2 (`test_half_open_admits_one_probe_across_threads`).
5. **A 4xx from the remote LLM** — still re-raised as the same `aiohttp.ClientResponseError` (not converted), recorded as success — test in Task 6.

---

## File Structure

- Create `src/internal/resilience/__init__.py` — empty package marker.
- Create `src/internal/resilience/circuit_breaker.py` — state machine, registry, `is_failure_status`.
- Modify `src/internal/configs/timeouts.py` — `CircuitBreakerPolicy`, `TimeoutPolicies.circuit_breaker`.
- Modify `src/internal/configs/timeouts.toml` — `[circuit_breaker]` table.
- Modify `docs/configuration/timeouts.md` — document the table.
- Modify `src/internal/tools/search.py` — `serpapi_search`, `make_web_cascade_search` browser leg.
- Modify `src/internal/search/stages.py` — `RerankHTTPRankingStage.rank`.
- Modify `src/internal/search/ranking.py` — `DefaultRankingStage.rank` `except CircuitOpenError` arm.
- Modify `src/model/serving.py` — `OpenAIServerManager.generate` / `generate_stream`.
- Modify `src/internal/servers/web/metrics_router.py` — `"circuits"`.
- Modify `tests/conftest.py` — autouse `reset_breakers()`.
- Modify `tests/unit/test_timeout_policies.py` — `TODAY` gains `circuit_breaker`.
- Create `tests/unit/resilience/__init__.py`, `tests/unit/resilience/test_circuit_breaker.py`, `tests/unit/resilience/test_circuit_breaker_sites.py`.
- Modify `tests/unit/servers/web/test_metrics_router.py`.

All commands run from `/Users/linghuang/Git/Agentic-Search/.worktrees/circuit-breakers` as `PYTHONPATH=. python -m pytest ... -p no:cacheprovider`.

---

### Task 1: Circuit-breaker policy in `timeouts.toml`

**Files:**
- Modify: `src/internal/configs/timeouts.py` (dataclasses near `SSEPolicy`, `TimeoutPolicies`)
- Modify: `src/internal/configs/timeouts.toml` (append)
- Modify: `docs/configuration/timeouts.md` (append schema section)
- Test: `tests/unit/test_timeout_policies.py`

**Interfaces:**
- Produces: `CircuitBreakerPolicy(failure_threshold: int, open_seconds: float)`; `get_timeout_policies().circuit_breaker`.

- [ ] **Step 1: Write the failing test** — in `tests/unit/test_timeout_policies.py`, add to `TODAY` after `"sse"`:

```python
    "circuit_breaker": {"failure_threshold": 5, "open_seconds": 30.0},
```

and add to the `test_invalid_values_error_naming_the_key` parametrize list:

```python
        (
            {"circuit_breaker": {"failure_threshold": 0}},
            "circuit_breaker.failure_threshold",
        ),
        ({"circuit_breaker": {"open_seconds": 0}}, "circuit_breaker.open_seconds"),
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/unit/test_timeout_policies.py -q -p no:cacheprovider`
Expected: FAIL — `test_bundled_defaults_are_todays_values` (no `circuit_breaker` key) and the two new invalid cases (`unknown timeout policy key 'circuit_breaker'`, which does not match the dotted path).

- [ ] **Step 3: Implement** — in `timeouts.py`, after `SSEPolicy`:

```python
@dataclass(frozen=True)
class CircuitBreakerPolicy:
    failure_threshold: int
    open_seconds: float
```

and add `circuit_breaker: CircuitBreakerPolicy` as the last field of `TimeoutPolicies`. Append to `timeouts.toml`:

```toml

[circuit_breaker]
failure_threshold = 5              # consecutive failures that open a breaker
open_seconds = 30.0                # how long an open breaker fails fast before one probe
```

Append to `docs/configuration/timeouts.md`:

```markdown

### `[circuit_breaker]`

One policy for every serving-dependency breaker (`serpapi`, `browser_search`,
`rerank`, `remote_llm`; see `src/internal/resilience/circuit_breaker.py`).
A breaker opens after `failure_threshold` consecutive failures — a transport
error, a timeout, HTTP 5xx or 429 — and then fails fast for `open_seconds`
before letting one probe call through. Breaker state is per process and is
reported under `circuits` by `GET /api/admin/metrics`.

| key | default | bounds |
|---|---|---|
| `failure_threshold` | 5 | consecutive failures that open a breaker, `>= 1` |
| `open_seconds` | 30.0 | how long an open breaker fails fast before one probe |
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/unit/test_timeout_policies.py tests/unit/test_timeout_policy_drift.py tests/unit/test_timeout_policy_sites.py -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/internal/configs/timeouts.py src/internal/configs/timeouts.toml docs/configuration/timeouts.md tests/unit/test_timeout_policies.py
git commit -m "Circuit-breaker policy table in timeouts.toml"
```

---

### Task 2: The circuit-breaker state machine and registry

**Files:**
- Create: `src/internal/resilience/__init__.py`, `src/internal/resilience/circuit_breaker.py`
- Modify: `tests/conftest.py` (autouse fixture)
- Test: `tests/unit/resilience/__init__.py`, `tests/unit/resilience/test_circuit_breaker.py`

**Interfaces:**
- Consumes: `get_timeout_policies().circuit_breaker` (Task 1).
- Produces:
  - `class CircuitOpenError(RuntimeError)` with `.name: str`, `.retry_in_seconds: float`; `__init__(self, name: str, retry_in_seconds: float)`.
  - `@dataclass(frozen=True) BreakerSnapshot(name: str, state: str, consecutive_failures: int, opened_at: float | None, retry_in_seconds: float)`.
  - `class CircuitBreaker(name, *, failure_threshold: int, open_seconds: float, clock=time.monotonic)` with `before_call()`, `record_success()`, `record_failure()`, `snapshot() -> BreakerSnapshot`.
  - `is_failure_status(status: int) -> bool` (≥500 or 429).
  - `get_breaker(name: str) -> CircuitBreaker`, `breaker_snapshots() -> list[BreakerSnapshot]` (sorted by name), `reset_breakers() -> None`.

- [ ] **Step 1: Write the failing tests** — `tests/unit/resilience/__init__.py` empty; `tests/unit/resilience/test_circuit_breaker.py`:

```python
"""The circuit-breaker state machine, driven by a fake clock."""

from __future__ import annotations

import threading

import pytest

from src.internal.configs.timeouts import TIMEOUTS_PATH_ENV, reset_timeout_policies
from src.internal.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    breaker_snapshots,
    get_breaker,
    is_failure_status,
    reset_breakers,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _breaker(clock, threshold=3, open_seconds=30.0):
    return CircuitBreaker(
        "dep", failure_threshold=threshold, open_seconds=open_seconds, clock=clock
    )


def _open(breaker, threshold=3):
    for _ in range(threshold):
        breaker.before_call()
        breaker.record_failure()


def test_opens_after_exactly_threshold_consecutive_failures():
    b = _breaker(_Clock())
    for _ in range(2):
        b.before_call()
        b.record_failure()
    assert b.snapshot().state == "closed"
    assert b.snapshot().consecutive_failures == 2
    b.before_call()
    b.record_failure()
    assert b.snapshot().state == "open"
    with pytest.raises(CircuitOpenError):
        b.before_call()


def test_success_resets_the_consecutive_count():
    b = _breaker(_Clock())
    for _ in range(2):
        b.record_failure()
    b.record_success()
    for _ in range(2):
        b.record_failure()
    assert b.snapshot().state == "closed"
    assert b.snapshot().consecutive_failures == 2


def test_open_raises_and_retry_in_counts_down():
    clock = _Clock()
    b = _breaker(clock)
    _open(b)
    snap = b.snapshot()
    assert snap.opened_at == 1000.0
    assert snap.retry_in_seconds == 30.0
    clock.now += 10
    with pytest.raises(CircuitOpenError) as info:
        b.before_call()
    assert info.value.name == "dep"
    assert info.value.retry_in_seconds == pytest.approx(20.0)
    assert b.snapshot().retry_in_seconds == pytest.approx(20.0)
    assert isinstance(info.value, RuntimeError)


def test_after_open_seconds_exactly_one_probe_is_admitted():
    clock = _Clock()
    b = _breaker(clock)
    _open(b)
    clock.now += 30
    b.before_call()  # the probe
    assert b.snapshot().state == "half_open"
    assert b.snapshot().retry_in_seconds == 0.0
    with pytest.raises(CircuitOpenError):
        b.before_call()  # a concurrent second call


def test_probe_success_closes():
    clock = _Clock()
    b = _breaker(clock)
    _open(b)
    clock.now += 30
    b.before_call()
    b.record_success()
    snap = b.snapshot()
    assert (snap.state, snap.consecutive_failures, snap.opened_at) == (
        "closed",
        0,
        None,
    )
    b.before_call()  # passes


def test_probe_failure_reopens_with_a_fresh_timer():
    clock = _Clock()
    b = _breaker(clock)
    _open(b)
    clock.now += 30
    b.before_call()
    b.record_failure()
    snap = b.snapshot()
    assert snap.state == "open"
    assert snap.opened_at == 1030.0
    clock.now += 29
    with pytest.raises(CircuitOpenError):
        b.before_call()
    clock.now += 1
    b.before_call()  # next probe


def test_unreported_probe_is_replaced_after_open_seconds():
    # A cancelled probe records nothing; the breaker must not fail fast forever.
    clock = _Clock()
    b = _breaker(clock)
    _open(b)
    clock.now += 30
    b.before_call()  # probe, never reports
    clock.now += 29
    with pytest.raises(CircuitOpenError):
        b.before_call()
    clock.now += 1
    b.before_call()  # a replacement probe


def test_half_open_admits_one_probe_across_threads():
    clock = _Clock()
    b = _breaker(clock)
    _open(b)
    clock.now += 30
    admitted = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        try:
            b.before_call()
            admitted.append(1)
        except CircuitOpenError:
            pass

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(admitted) == 1


@pytest.mark.parametrize(
    "status, failure", [(500, True), (503, True), (429, True), (404, False), (401, False), (200, False)]
)
def test_is_failure_status(status, failure):
    assert is_failure_status(status) is failure


def test_registry_is_lazy_shared_and_uses_bundled_policy():
    assert breaker_snapshots() == []
    b = get_breaker("serpapi")
    assert get_breaker("serpapi") is b
    for _ in range(4):
        b.record_failure()
    assert b.snapshot().state == "closed"  # bundled threshold is 5
    b.record_failure()
    assert b.snapshot().state == "open"
    get_breaker("alpha")
    assert [s.name for s in breaker_snapshots()] == ["alpha", "serpapi"]
    reset_breakers()
    assert breaker_snapshots() == []


def test_override_file_changes_the_policy(tmp_path, monkeypatch):
    f = tmp_path / "t.toml"
    f.write_text("[circuit_breaker]\nfailure_threshold = 1\nopen_seconds = 2\n")
    monkeypatch.setenv(TIMEOUTS_PATH_ENV, str(f))
    reset_timeout_policies()
    b = get_breaker("rerank")
    b.record_failure()
    snap = b.snapshot()
    assert snap.state == "open"
    assert snap.retry_in_seconds == pytest.approx(2.0, abs=0.5)


def test_transitions_log(caplog):
    clock = _Clock()
    b = _breaker(clock, threshold=1)
    with caplog.at_level("INFO", logger="src.internal.resilience.circuit_breaker"):
        b.record_failure()
        clock.now += 30
        b.before_call()
        b.record_failure()
        clock.now += 30
        b.before_call()
        b.record_success()
    levels = [(r.levelname, r.getMessage()) for r in caplog.records]
    assert [lvl for lvl, _ in levels] == ["WARNING", "WARNING", "INFO"]
    assert "opened" in levels[0][1] and "re-opened" in levels[1][1]
    assert "closed" in levels[2][1]
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/unit/resilience/test_circuit_breaker.py -q -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.internal.resilience'`.

- [ ] **Step 3: Implement** — `src/internal/resilience/__init__.py`:

```python
"""Resilience primitives for serving dependencies."""
```

`src/internal/resilience/circuit_breaker.py`:

```python
"""Per-process circuit breakers for serving dependencies.

A breaker remembers that a dependency is down so later requests skip straight
to the caller's existing degradation path instead of paying the full timeout.
Calls are never wrapped: a site calls ``before_call()`` and then
``record_success()`` or ``record_failure()``, and decides for itself what
counts as a failure. Policy comes from ``[circuit_breaker]`` in timeouts.toml,
read when a breaker is first created.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from src.internal.configs.timeouts import get_timeout_policies

logger = logging.getLogger(__name__)

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised by ``before_call`` while the dependency is being skipped."""

    def __init__(self, name: str, retry_in_seconds: float) -> None:
        super().__init__(
            f"circuit {name!r} is open; retry in {retry_in_seconds:.1f}s"
        )
        self.name = name
        self.retry_in_seconds = retry_in_seconds


@dataclass(frozen=True)
class BreakerSnapshot:
    name: str
    state: str
    consecutive_failures: int
    opened_at: float | None
    retry_in_seconds: float


def is_failure_status(status: int) -> bool:
    """HTTP statuses that mean the dependency is unhealthy: 5xx and 429."""
    return status >= 500 or status == 429


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int,
        open_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self._threshold = failure_threshold
        self._open_seconds = open_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CLOSED
        self._failures = 0
        self._opened_at: float | None = None
        # When the current half-open probe was admitted. A probe that never
        # reports (cancelled) is replaced after open_seconds.
        self._probe_at: float | None = None

    def before_call(self) -> None:
        with self._lock:
            if self._state == CLOSED:
                return
            now = self._clock()
            since = self._opened_at if self._state == OPEN else self._probe_at
            wait = since + self._open_seconds - now
            if wait > 0:
                raise CircuitOpenError(
                    self.name, wait if self._state == OPEN else 0.0
                )
            self._state = HALF_OPEN
            self._probe_at = now

    def record_success(self) -> None:
        with self._lock:
            if self._state != CLOSED:
                logger.info("Circuit %r closed: the dependency answered", self.name)
            self._state = CLOSED
            self._failures = 0
            self._opened_at = None
            self._probe_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            reopen = self._state == HALF_OPEN
            if reopen or (self._state == CLOSED and self._failures >= self._threshold):
                self._state = OPEN
                self._opened_at = self._clock()
                self._probe_at = None
                logger.warning(
                    "Circuit %r %s after %d consecutive failures; "
                    "skipping it for %.1fs",
                    self.name,
                    "re-opened" if reopen else "opened",
                    self._failures,
                    self._open_seconds,
                )

    def snapshot(self) -> BreakerSnapshot:
        with self._lock:
            retry_in = 0.0
            if self._state == OPEN:
                retry_in = max(
                    0.0, self._opened_at + self._open_seconds - self._clock()
                )
            return BreakerSnapshot(
                name=self.name,
                state=self._state,
                consecutive_failures=self._failures,
                opened_at=self._opened_at,
                retry_in_seconds=retry_in,
            )


_registry: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_breaker(name: str) -> CircuitBreaker:
    """The process-wide breaker for ``name``, created from policy on first use."""
    with _registry_lock:
        breaker = _registry.get(name)
        if breaker is None:
            policy = get_timeout_policies().circuit_breaker
            breaker = CircuitBreaker(
                name,
                failure_threshold=policy.failure_threshold,
                open_seconds=policy.open_seconds,
            )
            _registry[name] = breaker
        return breaker


def breaker_snapshots() -> list[BreakerSnapshot]:
    with _registry_lock:
        breakers = sorted(_registry.values(), key=lambda b: b.name)
    return [b.snapshot() for b in breakers]


def reset_breakers() -> None:
    """Forget every breaker. Tests only."""
    with _registry_lock:
        _registry.clear()
```

Note on the half-open `CircuitOpenError`: its `retry_in_seconds` is `0.0` (snapshot contract: "0 unless open"); the caller only needs to know the call is skipped.

Add to `tests/conftest.py`, after `_fresh_timeout_policies`:

```python
@pytest.fixture(autouse=True)
def _fresh_circuit_breakers():
    """Breakers are process-wide; one test's failures must not open another's."""
    from src.internal.resilience.circuit_breaker import reset_breakers

    reset_breakers()
    yield
    reset_breakers()
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/unit/resilience/test_circuit_breaker.py -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/internal/resilience tests/unit/resilience tests/conftest.py
git commit -m "Circuit-breaker state machine and process-wide registry"
```

---

### Task 3: `serpapi` breaker in `serpapi_search`

**Files:**
- Modify: `src/internal/tools/search.py` (`serpapi_search`, imports)
- Test: `tests/unit/resilience/test_circuit_breaker_sites.py` (create)

**Interfaces:**
- Consumes: `get_breaker`, `CircuitOpenError`, `is_failure_status` (Task 2).
- Produces: module constant `SERPAPI_CIRCUIT_OPEN_ERROR` in `search.py`.

- [ ] **Step 1: Write the failing tests** — create `tests/unit/resilience/test_circuit_breaker_sites.py`:

```python
"""Each serving dependency skips its call once its breaker is open."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager

import aiohttp
import pytest
from yarl import URL

from src.internal.configs.timeouts import load_timeout_policies, use_timeout_policies
from src.internal.resilience.circuit_breaker import get_breaker


@contextmanager
def threshold(n: int):
    overrides = {"circuit_breaker": {"failure_threshold": n}}
    with use_timeout_policies(load_timeout_policies({}, overrides=overrides)):
        yield


def _aiohttp_status_error(status: int) -> aiohttp.ClientResponseError:
    url = URL("http://dep.test/x")
    info = aiohttp.RequestInfo(url=url, method="GET", headers={}, real_url=url)
    return aiohttp.ClientResponseError(info, (), status=status, message="x")


# --- serpapi -----------------------------------------------------------------


def _serp(monkeypatch, outcome):
    """Fake _get_json: raise ``outcome`` if it is an exception, else return it."""
    from src.internal.tools import search

    calls = []

    async def fake_get_json(url, **kw):
        calls.append(url)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(search, "_get_json", fake_get_json)
    monkeypatch.setenv("SERPAPI_API_KEY", "k")
    return search, calls


def test_serpapi_open_breaker_skips_the_call(monkeypatch):
    search, calls = _serp(monkeypatch, asyncio.TimeoutError())
    with threshold(2):
        for _ in range(2):
            asyncio.run(search.serpapi_search("q"))
        assert len(calls) == 2
        pages = asyncio.run(search.serpapi_search("q"))
    assert len(calls) == 2  # no outbound call
    assert [p.error for p in pages] == [
        "SerpAPI is temporarily skipped after repeated failures (circuit open)."
    ]


@pytest.mark.parametrize("status", [429, 503])
def test_serpapi_429_and_5xx_count(monkeypatch, status):
    search, _ = _serp(monkeypatch, _aiohttp_status_error(status))
    with threshold(2):
        for _ in range(2):
            asyncio.run(search.serpapi_search("q"))
    assert get_breaker("serpapi").snapshot().state == "open"


def test_serpapi_4xx_does_not_count(monkeypatch):
    search, calls = _serp(monkeypatch, _aiohttp_status_error(401))
    with threshold(2):
        for _ in range(3):
            asyncio.run(search.serpapi_search("q"))
    assert len(calls) == 3
    snap = get_breaker("serpapi").snapshot()
    assert (snap.state, snap.consecutive_failures) == ("closed", 0)


def test_serpapi_success_resets(monkeypatch):
    search, _ = _serp(monkeypatch, {"organic_results": []})
    b = get_breaker("serpapi")
    b.record_failure()
    assert asyncio.run(search.serpapi_search("q")) == []
    assert b.snapshot().consecutive_failures == 0


def test_serpapi_missing_key_never_touches_the_breaker(monkeypatch):
    from src.internal.resilience.circuit_breaker import breaker_snapshots
    from src.internal.tools import search

    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.delenv("SERP_API_KEY", raising=False)
    pages = asyncio.run(search.serpapi_search("q"))
    assert "required" in pages[0].error
    assert breaker_snapshots() == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/unit/resilience/test_circuit_breaker_sites.py -q -p no:cacheprovider`
Expected: FAIL — `test_serpapi_open_breaker_skips_the_call` (3 calls, generic error), `test_serpapi_429_and_5xx_count` (state `closed`), `test_serpapi_success_resets` (count 1). The 4xx and missing-key tests pass already (nothing records yet); they are the regression half.

- [ ] **Step 3: Implement** — in `search.py` add the import after `from ..configs.timeouts import get_timeout_policies`:

```python
from ..resilience.circuit_breaker import (
    CircuitOpenError,
    get_breaker,
    is_failure_status,
)
```

module constants near the other module-level constants (after the imports / endpoints):

```python
SERPAPI_CIRCUIT_OPEN_ERROR = (
    "SerpAPI is temporarily skipped after repeated failures (circuit open)."
)
```

and in `serpapi_search`, replace the `try: data = await _get_json(...) except Exception` block with:

```python
    breaker = get_breaker("serpapi")
    try:
        breaker.before_call()
    except CircuitOpenError:
        return [SearchPage(error=SERPAPI_CIRCUIT_OPEN_ERROR)]
    try:
        data = await _get_json(
            SERPAPI_SEARCH_ENDPOINT,
            params={
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "num": page_size,
                "start": (page - 1) * page_size,
            },
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        # A 4xx is an answer from a healthy SerpAPI (bad key, bad query).
        if isinstance(exc, aiohttp.ClientResponseError) and not is_failure_status(
            exc.status
        ):
            breaker.record_success()
        else:
            breaker.record_failure()
        return [SearchPage(error=_redact_secret_params(str(exc)))]
    breaker.record_success()
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/unit/resilience/ tests/unit/test_search_tools.py tests/unit/test_web_cascade_search.py -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/internal/tools/search.py tests/unit/resilience/test_circuit_breaker_sites.py
git commit -m "serpapi breaker: skip SerpAPI after repeated failures"
```

---

### Task 4: `browser_search` breaker in the web cascade

**Files:**
- Modify: `src/internal/tools/search.py` (`make_web_cascade_search`)
- Test: `tests/unit/resilience/test_circuit_breaker_sites.py` (append)

**Interfaces:**
- Consumes: Task 2 API; Task 3 imports in `search.py`.
- Produces: module constant `BROWSER_CIRCUIT_OPEN_ERROR` in `search.py`.

- [ ] **Step 1: Write the failing tests** — append:

```python
# --- browser_search ----------------------------------------------------------


def _cascade(browser_outcome):
    from src.internal.tools.search import SearchPage, make_web_cascade_search

    calls = []

    async def fake_serp(query, **kw):
        return [SearchPage(error="serp down")]

    async def fake_browser(query, **kw):
        calls.append(query)
        if isinstance(browser_outcome, BaseException):
            raise browser_outcome
        return browser_outcome

    fn = make_web_cascade_search(
        browser_search_url="http://browser/retrieve",
        serpapi_fn=fake_serp,
        browser_fn=fake_browser,
    )
    return fn, calls


def test_browser_open_breaker_skips_the_call():
    fn, calls = _cascade(RuntimeError("connection refused"))
    with threshold(2):
        for _ in range(2):
            asyncio.run(fn("q"))
        pages = asyncio.run(fn("q"))
    assert len(calls) == 2
    errors = [p.error for p in pages]
    assert "serp down" in errors
    assert (
        "Browser search is temporarily skipped after repeated failures (circuit open)."
        in errors
    )


def test_browser_all_error_pages_count_as_failure():
    from src.internal.tools.search import SearchPage

    fn, _ = _cascade([SearchPage(error="503")])
    with threshold(2):
        for _ in range(2):
            asyncio.run(fn("q"))
    assert get_breaker("browser_search").snapshot().state == "open"


def test_browser_empty_result_is_a_success():
    fn, calls = _cascade([])
    get_breaker("browser_search").record_failure()
    with threshold(2):
        for _ in range(3):
            asyncio.run(fn("q"))
    assert len(calls) == 3
    snap = get_breaker("browser_search").snapshot()
    assert (snap.state, snap.consecutive_failures) == ("closed", 0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/unit/resilience/test_circuit_breaker_sites.py -q -p no:cacheprovider -k browser`
Expected: FAIL — skip test (3 calls), all-error test (`closed`), empty test (`consecutive_failures == 1`).

- [ ] **Step 3: Implement** — constant beside `SERPAPI_CIRCUIT_OPEN_ERROR`:

```python
BROWSER_CIRCUIT_OPEN_ERROR = (
    "Browser search is temporarily skipped after repeated failures (circuit open)."
)
```

In `_cascade`, replace the `if browser_search_url:` body (the `try/except` around `browser_fn`) with:

```python
        if browser_search_url:
            breaker = get_breaker("browser_search")
            try:
                breaker.before_call()
            except CircuitOpenError:
                failures.append(SearchPage(error=BROWSER_CIRCUIT_OPEN_ERROR))
            else:
                try:
                    browser_pages = await browser_fn(
                        query,
                        provider="retrieval",
                        search_url=browser_search_url,
                        page=page,
                        page_size=page_size,
                    )
                except Exception as exc:  # noqa: BLE001
                    breaker.record_failure()
                    logger.warning("browser cascade leg failed for %r: %s", query, exc)
                    failures.append(SearchPage(error=f"Browser search failed: {exc}"))
                else:
                    # search_tool reports HTTP failures as error pages, not raises.
                    if browser_pages and all(p.error for p in browser_pages):
                        breaker.record_failure()
                    else:
                        breaker.record_success()
                    if _pages_are_usable(browser_pages):
                        return browser_pages
                    failures.extend(p for p in browser_pages if p.error)
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/unit/resilience/ tests/unit/test_web_cascade_search.py tests/unit/test_search_tools.py -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/internal/tools/search.py tests/unit/resilience/test_circuit_breaker_sites.py
git commit -m "browser_search breaker: skip the browser cascade leg after repeated failures"
```

---

### Task 5: `rerank` breaker and the `circuit_open` ranking status

**Files:**
- Modify: `src/internal/search/stages.py` (`RerankHTTPRankingStage.rank`)
- Modify: `src/internal/search/ranking.py` (`DefaultRankingStage.rank`)
- Test: `tests/unit/resilience/test_circuit_breaker_sites.py` (append)

**Interfaces:**
- Consumes: Task 2 API.
- Produces: `RerankHTTPRankingStage.rank` raises `CircuitOpenError` when open; `DefaultRankingStage` metadata `rerank_status == "circuit_open"`.

- [ ] **Step 1: Write the failing tests** — append:

```python
# --- rerank ------------------------------------------------------------------


def _rerank_stage(monkeypatch, outcome):
    """Patch httpx.AsyncClient: post raises ``outcome`` or returns it."""
    import httpx

    from src.internal.search import stages

    calls = []

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, *, json, timeout):
            calls.append(url)
            if isinstance(outcome, BaseException):
                raise outcome
            if isinstance(outcome, int):
                return httpx.Response(outcome, request=httpx.Request("POST", url))
            return httpx.Response(
                200, json=outcome, request=httpx.Request("POST", url)
            )

    monkeypatch.setattr(stages.httpx, "AsyncClient", _Client)
    return stages.RerankHTTPRankingStage("http://r"), calls


def _candidates():
    from src.context.search import SearchResult
    from src.internal.search.models import CandidateSet

    return CandidateSet(
        query="q",
        candidates=[SearchResult(contents="one", title="One", score=0.2)],
        provider="retrieval",
    )


def test_rerank_open_breaker_raises_without_calling(monkeypatch):
    import httpx

    from src.internal.resilience.circuit_breaker import CircuitOpenError

    stage, calls = _rerank_stage(monkeypatch, httpx.ConnectError("down"))
    with threshold(2):
        for _ in range(2):
            with pytest.raises(httpx.ConnectError):
                asyncio.run(stage.rank("q", _candidates(), 1))
        with pytest.raises(CircuitOpenError):
            asyncio.run(stage.rank("q", _candidates(), 1))
    assert len(calls) == 2


def test_rerank_5xx_counts_and_4xx_does_not(monkeypatch):
    import httpx

    stage, _ = _rerank_stage(monkeypatch, 422)
    with threshold(2):
        for _ in range(3):
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(stage.rank("q", _candidates(), 1))
        assert get_breaker("rerank").snapshot().state == "closed"
        stage, _ = _rerank_stage(monkeypatch, 503)
        for _ in range(2):
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(stage.rank("q", _candidates(), 1))
    assert get_breaker("rerank").snapshot().state == "open"


def test_rerank_cache_hit_never_touches_the_breaker(monkeypatch):
    from src.internal.resilience.circuit_breaker import breaker_snapshots
    from src.internal.search import stages

    class _Cache:
        def get(self, key):
            return [{"document": {"_idx": "0"}, "score": 0.9}]

        def set(self, key, value):
            raise AssertionError("a hit is not re-cached")

    monkeypatch.setattr(stages, "serving_cache", lambda: _Cache())
    stage, calls = _rerank_stage(monkeypatch, RuntimeError("must not be called"))
    result = asyncio.run(stage.rank("q", _candidates(), 1))
    assert [d.title for d in result.evidence] == ["One"]
    assert calls == []
    assert breaker_snapshots() == []


def test_default_ranking_reports_circuit_open():
    from src.internal.resilience.circuit_breaker import CircuitOpenError
    from src.internal.search.ranking import DefaultRankingStage

    class _OpenReranker:
        async def rank(self, query, candidates, top_k):
            raise CircuitOpenError("rerank", 12.0)

    result = asyncio.run(
        DefaultRankingStage(_OpenReranker()).rank("q", _candidates(), 1)
    )
    assert result.metadata["rerank_status"] == "circuit_open"
    assert result.metadata["degraded"] is True
    assert [d.title for d in result.evidence] == ["One"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/unit/resilience/test_circuit_breaker_sites.py -q -p no:cacheprovider -k "rerank or ranking"`
Expected: FAIL — open-breaker test (third call raises `ConnectError`, 3 calls), 5xx test (`closed` after 503s), `rerank_status == "error"` instead of `circuit_open`. The cache-hit test passes already (regression half).

- [ ] **Step 3: Implement** — `stages.py` import:

```python
from src.internal.resilience.circuit_breaker import get_breaker, is_failure_status
```

replace the `if ranked is None:` block's HTTP section with:

```python
        if ranked is None:
            breaker = get_breaker("rerank")
            breaker.before_call()  # raises CircuitOpenError while skipped
            try:
                async with httpx.AsyncClient() as client:
                    body = {
                        "queries": [query],
                        "documents": [payloads],
                        "return_scores": True,
                    }
                    if self._send_top_k:
                        body["rerank_topk"] = top_k
                    response = await client.post(
                        self._url, json=body, timeout=self._timeout
                    )
                    response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if is_failure_status(exc.response.status_code):
                    breaker.record_failure()
                else:
                    breaker.record_success()
                raise
            except httpx.HTTPError:
                breaker.record_failure()
                raise
            breaker.record_success()
            ranked = response.json()["result"][0]
            if cache is not None and ranked:
                cache.set(cache_key, ranked)
```

`ranking.py` import:

```python
from src.internal.resilience.circuit_breaker import CircuitOpenError
```

and in `DefaultRankingStage.rank`, insert before `except httpx.TimeoutException`:

```python
            except CircuitOpenError as exc:
                logger.warning("Rerank skipped, using original order: %s", exc)
                rerank_status = "circuit_open"
                degraded = True
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/unit/resilience/ tests/unit/search/ tests/unit/test_timeout_policy_sites.py -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/internal/search/stages.py src/internal/search/ranking.py tests/unit/resilience/test_circuit_breaker_sites.py
git commit -m "rerank breaker: skip the HTTP reranker and report circuit_open"
```

---

### Task 6: `remote_llm` breaker in `OpenAIServerManager`

**Files:**
- Modify: `src/model/serving.py` (`OpenAIServerManager.generate`, `generate_stream`, imports)
- Test: `tests/unit/resilience/test_circuit_breaker_sites.py` (append)

**Interfaces:**
- Consumes: Task 2 API.
- Produces: `generate`/`generate_stream` raise `RuntimeError(f"Inference server at {base_url} is temporarily skipped after repeated failures (circuit open).")` when open.

`serving.py` imports torch lazily inside methods only, and the existing `test_remote_server_manager_timeout_follows_policy` imports it without `importorskip`, so these tests need none.

- [ ] **Step 1: Write the failing tests** — append:

```python
# --- remote_llm --------------------------------------------------------------


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "prompt"

    def encode(self, text):
        return [1] * len(text)


class _Resp:
    def __init__(self, *, status=200, lines=()):
        self.status = status
        self.content = self._iter(lines)

    @staticmethod
    async def _iter(lines):
        for line in lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise _aiohttp_status_error(self.status)

    async def json(self):
        return {"choices": [{"text": "ok"}], "usage": {}}


def _llm(monkeypatch, outcome):
    """A remote manager whose session.post raises ``outcome`` or returns it."""
    from src.model.serving import OpenAIServerManager

    calls = []

    class _Session:
        closed = False

        def post(self, url, json):
            calls.append(url)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    m = OpenAIServerManager(tokenizer=_Tokenizer(), base_url="http://llm", model="m")
    monkeypatch.setattr(m, "_get_session", lambda: _Session())
    return m, calls


async def _noop(text):
    return None


def _call(m, stream):
    if stream:
        return asyncio.run(m.generate_stream("r", [1], {}, _noop))
    return asyncio.run(m.generate("r", [1], {}))


@pytest.mark.parametrize("stream", [False, True])
def test_remote_llm_open_breaker_raises_without_calling(monkeypatch, stream):
    m, calls = _llm(monkeypatch, asyncio.TimeoutError())
    with threshold(2):
        for _ in range(2):
            with pytest.raises(RuntimeError, match="Cannot connect"):
                _call(m, stream)
        with pytest.raises(RuntimeError) as info:
            _call(m, stream)
    assert len(calls) == 2
    assert str(info.value) == (
        "Inference server at http://llm is temporarily skipped after "
        "repeated failures (circuit open)."
    )


@pytest.mark.parametrize("stream", [False, True])
def test_remote_llm_5xx_and_429_count(monkeypatch, stream):
    for status in (503, 429):
        from src.internal.resilience.circuit_breaker import reset_breakers

        reset_breakers()
        m, _ = _llm(monkeypatch, _Resp(status=status))
        with threshold(2):
            for _ in range(2):
                with pytest.raises(aiohttp.ClientResponseError):
                    _call(m, stream)
        assert get_breaker("remote_llm").snapshot().state == "open"


@pytest.mark.parametrize("stream", [False, True])
def test_remote_llm_4xx_is_reraised_and_not_a_failure(monkeypatch, stream):
    get_breaker("remote_llm").record_failure()
    m, calls = _llm(monkeypatch, _Resp(status=400))
    with threshold(2):
        for _ in range(3):
            with pytest.raises(aiohttp.ClientResponseError) as info:
                _call(m, stream)
            assert info.value.status == 400
    assert len(calls) == 3
    snap = get_breaker("remote_llm").snapshot()
    assert (snap.state, snap.consecutive_failures) == ("closed", 0)


def test_remote_llm_other_transport_error_counts(monkeypatch):
    m, _ = _llm(monkeypatch, aiohttp.ServerDisconnectedError())
    with threshold(2):
        for _ in range(2):
            with pytest.raises(aiohttp.ServerDisconnectedError):
                _call(m, False)
    assert get_breaker("remote_llm").snapshot().state == "open"


def test_remote_llm_success_resets(monkeypatch):
    get_breaker("remote_llm").record_failure()
    m, _ = _llm(monkeypatch, _Resp())
    assert _call(m, False) == [1, 1]
    stream_m, _ = _llm(
        monkeypatch, _Resp(lines=[b'data: {"choices": [{"text": "hi"}]}', b"data: [DONE]"])
    )
    get_breaker("remote_llm").record_failure()
    assert _call(stream_m, True) == [1, 1]
    assert get_breaker("remote_llm").snapshot().consecutive_failures == 0


def test_stream_callback_error_is_not_a_failure(monkeypatch):
    # A client that disconnects mid-stream says nothing about the LLM server.
    m, _ = _llm(monkeypatch, _Resp(lines=[b'data: {"choices": [{"text": "hi"}]}']))

    async def on_token(text):
        raise ConnectionResetError("client went away")

    with threshold(1):
        with pytest.raises(ConnectionResetError):
            asyncio.run(m.generate_stream("r", [1], {}, on_token))
    assert get_breaker("remote_llm").snapshot().consecutive_failures == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/unit/resilience/test_circuit_breaker_sites.py -q -p no:cacheprovider -k remote_llm`
Expected: FAIL — open-breaker (3 calls, "Cannot connect"), 5xx/429 and transport-error (`closed`), 4xx and success tests (`consecutive_failures == 1`).

- [ ] **Step 3: Implement** — `serving.py` import after `from src.internal.configs.timeouts import get_timeout_policies`:

```python
from src.internal.resilience.circuit_breaker import (
    CircuitOpenError,
    get_breaker,
    is_failure_status,
)
```

Add two methods to `OpenAIServerManager` (after `aclose`):

```python
    def _admit(self):
        """This server's breaker, or RuntimeError while it is being skipped."""
        breaker = get_breaker("remote_llm")
        try:
            breaker.before_call()
        except CircuitOpenError:
            raise RuntimeError(
                f"Inference server at {self.base_url} is temporarily skipped "
                "after repeated failures (circuit open)."
            ) from None
        return breaker

    def _connect_error(self) -> RuntimeError:
        return RuntimeError(
            f"Cannot connect to inference server at {self.base_url}. "
            f"Start one first, e.g.: mlx_lm.server --model {self.model} --port 8080"
        )
```

In `generate`, replace from `started = time.perf_counter()` through the end of the `except` with:

```python
        breaker = self._admit()
        started = time.perf_counter()
        try:
            session = self._get_session()
            async with session.post(
                f"{self.base_url}/v1/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except aiohttp.ClientResponseError as exc:
            _record_status(breaker, exc.status)
            raise
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError):
            breaker.record_failure()
            raise self._connect_error()
        except aiohttp.ClientError:
            breaker.record_failure()
            raise
        breaker.record_success()
```

In `generate_stream`, do the same around the streaming block: `breaker = self._admit()` before `started`, same three `except` arms (the connect arm raising `self._connect_error()`), and `breaker.record_success()` after the `try`. Any other exception (e.g. `on_token` raising) records nothing.

Module-level helper (above `class OpenAIServerManager`):

```python
def _record_status(breaker, status: int) -> None:
    """A 5xx or 429 is an unhealthy server; any other status is an answer."""
    if is_failure_status(status):
        breaker.record_failure()
    else:
        breaker.record_success()
```

(Plan note: behaviour for `ClientConnectorError` / `asyncio.TimeoutError` is unchanged — same message, same bare `raise RuntimeError` — only a breaker record is added.)

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/unit/resilience/ tests/unit/test_timeout_policy_sites.py -q -p no:cacheprovider` and `PYTHONPATH=. python -m pytest tests/unit -q -p no:cacheprovider -k "serving or remote or OpenAIServer"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/serving.py tests/unit/resilience/test_circuit_breaker_sites.py
git commit -m "remote_llm breaker: fail fast on a remote inference server that keeps failing"
```

---

### Task 7: Breaker state on `/api/admin/metrics`

**Files:**
- Modify: `src/internal/servers/web/metrics_router.py`
- Test: `tests/unit/servers/web/test_metrics_router.py`

**Interfaces:**
- Consumes: `breaker_snapshots()` (Task 2).
- Produces: response key `"circuits": list[dict]` (asdict of `BreakerSnapshot`).

- [ ] **Step 1: Write the failing test** — append:

```python
def test_reports_circuit_breakers_once_used():
    from src.internal.resilience.circuit_breaker import get_breaker

    client = _client(AgenticSearchStore(":memory:"), admin=True)
    assert client.get("/api/admin/metrics").json()["circuits"] == []
    get_breaker("serpapi").record_failure()

    body = client.get("/api/admin/metrics").json()

    assert body["circuits"] == [
        {
            "name": "serpapi",
            "state": "closed",
            "consecutive_failures": 1,
            "opened_at": None,
            "retry_in_seconds": 0.0,
        }
    ]
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/unit/servers/web/test_metrics_router.py -q -p no:cacheprovider`
Expected: FAIL — `KeyError: 'circuits'`.

- [ ] **Step 3: Implement** — imports:

```python
from dataclasses import asdict
...
from src.internal.resilience.circuit_breaker import breaker_snapshots
```

and in the response dict add:

```python
            "circuits": [asdict(s) for s in breaker_snapshots()],
```

Update the docstring line of `metrics` to `"""Route latency, per-stage latency, feedback rates by target, breakers."""`.

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/unit/servers/web/test_metrics_router.py -q -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/internal/servers/web/metrics_router.py tests/unit/servers/web/test_metrics_router.py
git commit -m "Report circuit-breaker state on /api/admin/metrics"
```

---

### Task 8: Mutation checks, torch-free check, full verification

- [ ] **Step 1: Mutation checks** (revert each after observing red; then `find src tests -name __pycache__ -type d -exec rm -rf {} +`):
  1. Delete the `breaker.before_call()` block in `serpapi_search` → `test_serpapi_open_breaker_skips_the_call` must fail.
  2. Delete `breaker.before_call()` in `RerankHTTPRankingStage.rank` → `test_rerank_open_breaker_raises_without_calling` must fail.
  3. In `CircuitBreaker.record_failure`, change `self._failures >= self._threshold` to `False` → `test_opens_after_exactly_threshold_consecutive_failures` must fail.
  4. Delete `breaker = self._admit()` gating in `generate` (make `_admit` return the breaker without calling `before_call`) → remote_llm open test must fail.
- [ ] **Step 2: Torch-free** — run `tests/unit/resilience` with a `sys.meta_path` finder that raises `ImportError` for `torch`; all must pass.
- [ ] **Step 3: Full suite** — `PYTHONPATH=. python -m pytest -q -p no:cacheprovider`; `ruff check . && ruff format --check .`. Both clean; fix and commit any formatting.
