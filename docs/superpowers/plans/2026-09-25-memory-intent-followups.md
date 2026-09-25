# Memory and Intent Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `InMemoryCache` honours expiry, deleting a chat session deletes its working-memory state, and the two intent cold-load caches load once under concurrency.

**Architecture:** `InMemoryCache` gains a per-key monotonic deadline (injectable clock) checked lazily on `get`/`exists`/`ttl`. `working.forget_session` deletes `session_memory:{id}` and never raises; the chat session-delete route calls it after a successful delete. `_INTENT_INDEXES` and `_MODEL_CACHE` each get a module-level `threading.Lock` with double-checked locking.

**Tech Stack:** Python 3.12, pytest, FastAPI TestClient, `threading`.

**Spec:** `docs/superpowers/specs/2026-09-25-memory-intent-followups-design.md`

## Global Constraints

- `InMemoryCache(clock=time.monotonic)`; `set(..., ex=None)` means no expiry; expiry recorded as `clock() + ex`.
- `get`, `exists`, `ttl` treat an expired key as missing and delete it; `ttl` keeps `TTL_NO_EXPIRY` (-1) and `TTL_KEY_NOT_FOUND` (-2).
- `expire(key, seconds)` sets the expiry only when the key exists; `delete` clears value and expiry.
- List and lock helpers unchanged. Semantics match Redis for these methods (a `set` without `ex` clears an existing TTL, as Redis `SET` does).
- `forget_session(cache, session_id) -> None` in `src/internal/memory/working.py`; deletes the state key; never raises; failure logged at WARNING.
- The session-delete route calls it after `store.delete_chat_session` succeeds, using `get_cache_backend()`.
- One module-level `threading.Lock` per intent cache; fast path without the lock on a hit; failure-caching semantics unchanged.
- Concurrency tests are bounded: every `Event.wait` and `Thread.join` has a timeout.

## Deviations / findings recorded against the spec

- **Other routes that delete chat sessions: none.** Grep of `delete_chat_session`, `DELETE FROM chat_sessions` and route decorators found exactly one caller, `DELETE /chat/delete-chat-session/{session_id}` in `chat_backend.py`. `/api/sessions` has no delete route. User deletion (`store.delete_user`, used only by the integration-mode `/manage/admin/reset-test-data`) does not delete sessions: `chat_sessions.user_id` is `ON DELETE SET NULL`, so the sessions (and their state) remain valid. SCIM `DELETE /Users/{id}` only deactivates the user. So only the one route is wired.
- **`ttl` rounding.** "Remaining whole seconds" is implemented as `math.ceil(remaining)` so a live key never reports 0 (0 would read as "about to vanish" while `get` still returns it).

## Review Focus

1. A `set` without `ex` over a key that has a TTL must drop the TTL (Redis semantics), else a rewritten key silently expires on its old deadline. → Task 1 test `test_set_without_ex_clears_an_existing_expiry`.
2. A key read exactly at its deadline is expired (Redis: expired at `now >= deadline`). → Task 1 test `test_key_is_expired_exactly_at_its_deadline`.
3. `forget_session` for a session that never had state must not raise or log. → Task 2 test `test_forget_session_without_state_is_a_quiet_no_op`.
4. A refused delete (another user's session) must leave the state intact — forgetting before the ownership check would let a caller wipe someone else's summary. → Task 2 route test.
5. Two threads cold-loading an encoder that fails: the loader still runs once, the first caller sees the original error, the waiter sees the cached-failure `RuntimeError`. → Task 3 test `test_concurrent_cold_load_failure_is_decided_once`.

---

### Task 1: `InMemoryCache` honours expiry

**Files:**
- Modify: `src/internal/cache/interface.py` (`InMemoryCache`)
- Test: `tests/unit/cache/test_in_memory_cache.py` (create)

**Interfaces:**
- Produces: `InMemoryCache(clock: Callable[[], float] = time.monotonic)`; existing method signatures unchanged.

- [ ] **Step 1: Write the failing tests**

```python
"""InMemoryCache expiry: the TTL semantics Redis gives, in-process."""

from __future__ import annotations

import pytest

from src.internal.cache.interface import (
    TTL_KEY_NOT_FOUND,
    TTL_NO_EXPIRY,
    InMemoryCache,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def cache(clock: FakeClock) -> InMemoryCache:
    return InMemoryCache(clock=clock)


def test_get_returns_the_value_before_expiry_and_none_after(cache, clock):
    cache.set("k", "v", ex=10)
    clock.now += 9
    assert cache.get("k") == b"v"
    clock.now += 1
    assert cache.get("k") is None


def test_key_is_expired_exactly_at_its_deadline(cache, clock):
    cache.set("k", "v", ex=10)
    clock.now += 10
    assert cache.exists("k") is False


def test_exists_and_ttl_after_expiry(cache, clock):
    cache.set("k", "v", ex=5)
    assert cache.exists("k") is True
    assert cache.ttl("k") == 5
    clock.now += 6
    assert cache.exists("k") is False
    assert cache.ttl("k") == TTL_KEY_NOT_FOUND


def test_ttl_reports_remaining_whole_seconds_rounded_up(cache, clock):
    cache.set("k", "v", ex=100)
    clock.now += 0.5
    assert cache.ttl("k") == 100


def test_an_expired_key_is_deleted_on_read(cache, clock):
    cache.set("k", "v", ex=1)
    clock.now += 2
    assert cache.get("k") is None
    assert "k" not in cache._store
    assert "k" not in cache._expires


def test_expire_on_an_existing_key_sets_its_expiry(cache, clock):
    cache.set("k", "v")
    cache.expire("k", 3)
    assert cache.ttl("k") == 3
    clock.now += 3
    assert cache.get("k") is None


def test_expire_on_a_missing_key_does_nothing(cache, clock):
    cache.expire("ghost", 3)
    assert cache.ttl("ghost") == TTL_KEY_NOT_FOUND
    cache.set("ghost", "v")
    assert cache.ttl("ghost") == TTL_NO_EXPIRY


def test_ex_none_never_expires(cache, clock):
    cache.set("k", "v", ex=None)
    clock.now += 10**9
    assert cache.get("k") == b"v"
    assert cache.ttl("k") == TTL_NO_EXPIRY


def test_set_without_ex_clears_an_existing_expiry(cache, clock):
    cache.set("k", "v1", ex=5)
    cache.set("k", "v2")
    clock.now += 10
    assert cache.get("k") == b"v2"
    assert cache.ttl("k") == TTL_NO_EXPIRY


def test_delete_clears_the_expiry(cache, clock):
    cache.set("k", "v", ex=5)
    cache.delete("k")
    cache.set("k", "v")
    clock.now += 10
    assert cache.get("k") == b"v"
    assert cache.ttl("k") == TTL_NO_EXPIRY


def test_default_clock_is_monotonic():
    cache = InMemoryCache()
    cache.set("k", "v", ex=60)
    assert cache.get("k") == b"v"
    assert 0 < cache.ttl("k") <= 60
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/cache/test_in_memory_cache.py -q -p no:cacheprovider`
Expected: FAIL (`InMemoryCache() got an unexpected keyword argument 'clock'`).

- [ ] **Step 3: Implement**

In `src/internal/cache/interface.py` add `import math`, `import time`, `from collections.abc import Callable`, then:

```python
class InMemoryCache(CacheBackend):
    """Thread-unsafe in-memory cache for local / test mode.

    Keys may carry an expiry (``set(ex=...)`` / ``expire``), checked lazily on
    read against *clock* (monotonic seconds), with Redis semantics.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._store: dict[str, Any] = {}
        self._expires: dict[str, float] = {}
        self._lists: dict[str, list[bytes]] = {}
        self._clock = clock

    def _live(self, key: str) -> bool:
        """Is *key* present and unexpired? Deletes it if it has expired."""
        deadline = self._expires.get(key)
        if deadline is not None and self._clock() >= deadline:
            self.delete(key)
        return key in self._store

    def get(self, key: str) -> bytes | None:
        if not self._live(key):
            return None
        v = self._store[key]
        if v is None:
            return None
        return v if isinstance(v, bytes) else str(v).encode()

    def set(self, key, value, ex=None) -> None:
        self._store[key] = value
        if ex is None:
            self._expires.pop(key, None)
        else:
            self._expires[key] = self._clock() + ex

    def delete(self, key: str) -> None:
        self._store.pop(key, None)
        self._expires.pop(key, None)

    def exists(self, key: str) -> bool:
        return self._live(key)

    def expire(self, key: str, seconds: int) -> None:
        if self._live(key):
            self._expires[key] = self._clock() + seconds

    def ttl(self, key: str) -> int:
        if not self._live(key):
            return TTL_KEY_NOT_FOUND
        deadline = self._expires.get(key)
        if deadline is None:
            return TTL_NO_EXPIRY
        return math.ceil(deadline - self._clock())
```

(keep the existing `set` type annotations; lock/rpush/blpop unchanged.)

- [ ] **Step 4: Run to verify pass** — same command plus `tests/unit/cache/ tests/unit/memory/test_working_memory.py tests/unit/test_tool_backend.py`. Expected: PASS.

- [ ] **Step 5: Commit** — `git add src/internal/cache/interface.py tests/unit/cache/test_in_memory_cache.py && git commit -m "InMemoryCache honours ex, expire and ttl"`

---

### Task 2: Deleting a chat session forgets its working memory

**Files:**
- Modify: `src/internal/memory/working.py` (add `forget_session` after `save_state`)
- Modify: `src/internal/servers/query_and_chat/chat_backend.py` (import + call in `delete_chat_session_by_id`)
- Test: `tests/unit/memory/test_working_memory.py`, `tests/unit/test_chat_backend.py`

**Interfaces:**
- Produces: `forget_session(cache: CacheBackend, session_id: str) -> None`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/memory/test_working_memory.py` (add `import logging`, import `forget_session`):

```python
def test_forget_session_removes_the_state(cache):
    save_state(cache, "sid", SessionMemoryState(summary="s"))
    forget_session(cache, "sid")
    assert cache.get("session_memory:sid") is None
    assert load_state(cache, "sid") == SessionMemoryState()


def test_forget_session_without_state_is_a_quiet_no_op(cache, caplog):
    with caplog.at_level(logging.WARNING, logger="src.internal.memory.working"):
        forget_session(cache, "never-stored")
    assert caplog.records == []


def test_forget_session_swallows_and_logs_a_cache_failure(caplog):
    class BrokenCache(InMemoryCache):
        def delete(self, key):
            raise ConnectionError("cache down")

    with caplog.at_level(logging.WARNING, logger="src.internal.memory.working"):
        forget_session(BrokenCache(), "sid")
    assert any(
        r.levelno == logging.WARNING and "sid" in r.getMessage()
        for r in caplog.records
    )
```

In `tests/unit/test_chat_backend.py` (add imports `from src.internal.cache.interface import InMemoryCache`, `from src.internal.memory.working import SessionMemoryState, load_state, save_state`, `from src.internal.servers.query_and_chat import chat_backend`):

```python
def test_delete_session_forgets_its_working_memory(
    client: TestClient, store: AgenticSearchStore, monkeypatch: pytest.MonkeyPatch
):
    cache = InMemoryCache()
    monkeypatch.setattr(chat_backend, "get_cache_backend", lambda: cache)
    session = store.create_chat_session(user_id=_USER_ID)
    save_state(cache, session.id, SessionMemoryState(summary="secret"))

    resp = client.delete(f"/chat/delete-chat-session/{session.id}")

    assert resp.status_code == 200
    assert load_state(cache, session.id) == SessionMemoryState()


def test_refused_delete_keeps_the_owners_working_memory(
    client: TestClient, store: AgenticSearchStore, monkeypatch: pytest.MonkeyPatch
):
    cache = InMemoryCache()
    monkeypatch.setattr(chat_backend, "get_cache_backend", lambda: cache)
    store.upsert_user(UserRecord(id="u-other", email="other@example.com"))
    session = store.create_chat_session(user_id="u-other")
    save_state(cache, session.id, SessionMemoryState(summary="secret"))

    resp = client.delete(f"/chat/delete-chat-session/{session.id}")

    assert resp.status_code == 404
    assert load_state(cache, session.id).summary == "secret"
```

- [ ] **Step 2: Run to verify fail** — `.venv/bin/python -m pytest tests/unit/memory/test_working_memory.py tests/unit/test_chat_backend.py -q -p no:cacheprovider`. Expected: ImportError on `forget_session`.

- [ ] **Step 3: Implement**

`working.py`, after `save_state`:

```python
def forget_session(cache: CacheBackend, session_id: str) -> None:
    """Drop the session's state, so a deleted conversation's summary goes with
    it. Never raises: the session is already gone, and a cache problem must
    not turn a successful delete into an error (the key still has its TTL)."""
    try:
        cache.delete(_STATE_KEY.format(session_id=session_id))
    except Exception as exc:  # noqa: BLE001 - best effort; TTL is the backstop
        logger.warning("session memory delete failed for %s: %s", session_id, exc)
```

`chat_backend.py`: add `forget_session` to the `working` import; in the route:

```python
        found = store.delete_chat_session(session_id)
        if not found:
            raise HTTPException(status_code=404, detail="Chat session not found")
        forget_session(get_cache_backend(), session_id)
```

- [ ] **Step 4: Run to verify pass** — same command plus `tests/unit/servers/web/test_chat_session_ownership.py`. Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -m "Deleting a chat session deletes its working-memory state"`

---

### Task 3: Intent index and encoder load once under concurrency

**Files:**
- Modify: `src/internal/servers/web/intent/similarity.py` (`_INTENT_INDEXES_LOCK`, `load_intent_index`)
- Modify: `src/model/pre_training/intents/model.py` (`_MODEL_LOCK`, `_model`)
- Test: `tests/unit/test_intent_cache_concurrency.py` (create)

**Interfaces:** none new beyond the two module-level locks.

- [ ] **Step 1: Write the failing tests**

```python
"""The intent cold-load caches: two concurrent cold requests load once.

Since #657 ``recognize_intent`` runs in worker threads, so the index and the
encoder caches can be hit cold from two threads at once. The loader here
blocks until released; without the lock a second thread enters it too.
Every wait is bounded, so a regression fails rather than hangs.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest

from src.internal.configs import AppSettings
from src.internal.servers.web.intent import similarity
from src.model.pre_training.intents import model as model_mod

_WAIT = 5.0
# How long to give a second thread to (wrongly) enter the loader.
_SECOND_ENTRY_GRACE = 0.5


class _BlockingLoader:
    """Counts calls; the first blocks until ``release`` is set."""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.calls = 0
        self._lock = threading.Lock()
        self.entered = threading.Event()
        self.second_entered = threading.Event()
        self.release = threading.Event()
        self._result = result
        self._error = error

    def __call__(self, *args, **kwargs):
        with self._lock:
            self.calls += 1
            if self.calls == 1:
                self.entered.set()
            else:
                self.second_entered.set()
        assert self.release.wait(_WAIT), "loader never released"
        if self._error is not None:
            raise self._error
        return self._result if self._result is not None else object()


def _race(target, loader):
    """Run *target* in two threads with the loader held; return outcomes."""
    outcomes: list[object] = [None, None]

    def run(i):
        try:
            outcomes[i] = target()
        except Exception as exc:  # noqa: BLE001 - recorded for the assertion
            outcomes[i] = exc

    first = threading.Thread(target=run, args=(0,), daemon=True)
    second = threading.Thread(target=run, args=(1,), daemon=True)
    first.start()
    assert loader.entered.wait(_WAIT), "first thread never reached the loader"
    second.start()
    loader.second_entered.wait(_SECOND_ENTRY_GRACE)
    loader.release.set()
    first.join(_WAIT)
    second.join(_WAIT)
    assert not first.is_alive() and not second.is_alive()
    return outcomes


class _FakeIndex:
    encoder = model_mod.DEFAULT_ENCODER

    def low_support_modules(self):
        return []


def test_concurrent_cold_index_load_loads_once(tmp_path, monkeypatch):
    monkeypatch.setattr(similarity, "_INTENT_INDEXES", {})
    index = _FakeIndex()
    loader = _BlockingLoader(result=index)
    monkeypatch.setattr(
        model_mod, "IntentIndex", types.SimpleNamespace(load=loader)
    )
    settings = AppSettings(intent_index_path=tmp_path)

    outcomes = _race(lambda: similarity.load_intent_index(settings), loader)

    assert loader.calls == 1
    assert outcomes == [index, index]


def _install_fake_sentence_transformers(monkeypatch, loader):
    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = loader
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    monkeypatch.setattr(model_mod, "_MODEL_CACHE", {})


def test_concurrent_cold_encoder_load_loads_once(monkeypatch):
    encoder = object()
    loader = _BlockingLoader(result=encoder)
    _install_fake_sentence_transformers(monkeypatch, loader)

    outcomes = _race(lambda: model_mod._model("some/encoder"), loader)

    assert loader.calls == 1
    assert outcomes == [encoder, encoder]


def test_concurrent_cold_load_failure_is_decided_once(monkeypatch):
    boom = OSError("download failed")
    loader = _BlockingLoader(error=boom)
    _install_fake_sentence_transformers(monkeypatch, loader)

    outcomes = _race(lambda: model_mod._model("some/encoder"), loader)

    assert loader.calls == 1
    assert outcomes[0] is boom
    assert isinstance(outcomes[1], RuntimeError)
    assert outcomes[1].__cause__ is boom
    with pytest.raises(RuntimeError, match="not retrying"):
        model_mod._model("some/encoder")
```

- [ ] **Step 2: Run to verify fail** — `.venv/bin/python -m pytest tests/unit/test_intent_cache_concurrency.py -q -p no:cacheprovider`. Expected: all three FAIL with `loader.calls == 2`.

- [ ] **Step 3: Implement**

`similarity.py` (add `import threading`):

```python
_INTENT_INDEXES: dict[Path, object | None] = {}
# recognize_intent runs in worker threads: without this two cold requests each
# load the index, and a transient failure in either is cached for good.
_INTENT_INDEXES_LOCK = threading.Lock()
```

In `load_intent_index`, keep the fast-path check, then wrap the load:

```python
    if directory in _INTENT_INDEXES:
        return _INTENT_INDEXES[directory]
    with _INTENT_INDEXES_LOCK:
        if directory in _INTENT_INDEXES:
            return _INTENT_INDEXES[directory]
        try:
            ...existing load, unchanged, indented one level...
        except Exception:
            ...unchanged...
        else:
            ...unchanged...
    return _INTENT_INDEXES[directory]
```

`model.py` (add `import threading`):

```python
_MODEL_CACHE: dict[str, object] = {}
# One load per model even when worker threads arrive cold together.
_MODEL_LOCK = threading.Lock()


def _model(model_name: str):
    """Load and cache the encoder. Loading costs seconds; encoding costs ms."""
    cached = _MODEL_CACHE.get(model_name)
    if cached is None:
        with _MODEL_LOCK:
            cached = _MODEL_CACHE.get(model_name)
            if cached is None:
                from sentence_transformers import SentenceTransformer

                try:
                    cached = SentenceTransformer(model_name, device="cpu")
                except Exception as exc:
                    _MODEL_CACHE[model_name] = exc
                    raise
                _MODEL_CACHE[model_name] = cached
    if isinstance(cached, Exception):
        raise RuntimeError(
            f"intent encoder {model_name!r} failed to load previously; not retrying"
        ) from cached
    return cached
```

- [ ] **Step 4: Run to verify pass** — new file plus `tests/unit/test_intent_encoder.py tests/unit/test_intent_routing.py tests/unit/test_ml_intent.py tests/unit/test_intent_evaluation.py tests/unit/test_run_agentic_search.py`. Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -m "Intent index and encoder load once under concurrent cold requests"`

---

### Task 4: Mutation checks and final verification

- [ ] Mutation 1: in `InMemoryCache.set` drop the `_expires` write (ignore `ex`) → expiry tests red. Restore, `find src tests -name __pycache__ -type d -prune -exec rm -rf {} +`.
- [ ] Mutation 2: remove the `forget_session(...)` call from the route → `test_delete_session_forgets_its_working_memory` red. Restore, clear `__pycache__`.
- [ ] Mutation 3: remove the lock in `load_intent_index` (replace `with _INTENT_INDEXES_LOCK:` by `if True:`) → index test red. Restore, clear.
- [ ] Mutation 4: same for `_MODEL_LOCK` → both encoder tests red. Restore, clear.
- [ ] `git diff` shows no source change after restoring.
- [ ] `.venv/bin/python -m pytest tests/unit/ -q -p no:cacheprovider` (0 failures), `ruff check . && ruff format --check .`, `git diff --check origin/main...HEAD`.
