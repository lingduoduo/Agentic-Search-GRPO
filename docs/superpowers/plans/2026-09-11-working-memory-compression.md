# Working-Memory Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a session's history overflows the tail cap, summarize the dropped turns in the background and hand the next turn `[summary] + tail` instead of `tail`, across the Assist, Chat and Tool surfaces, behind a default-off flag.

**Architecture:** A new `src/internal/memory/working.py` owns working memory: a `SessionMemoryState` (summary + cursors) stored as JSON in the existing `CacheBackend`, a `load_working_memory` loader that replaces the three per-surface tail-slices, and a locked `compress_session` task that calls the configured `llm.complete`. Surfaces call `schedule_compression` after loading history; nothing in the loops changes.

**Tech Stack:** Python 3.11, FastAPI, SQLite via `AgenticSearchStore`, `src.internal.cache.interface.CacheBackend` (`InMemoryCache` default, Redis via `CACHE_BACKEND=redis`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-11-working-memory-compression-design.md`

## Global Constraints

- Flag name is exactly `AGENTIC_SEARCH_MEMORY_COMPRESSION`; default off; read via the `_flag(...)` idiom in `SearchExperienceSettings.from_app_settings`.
- Cache key is exactly `session_memory:<session_id>`; lock name `session_memory:<session_id>:compress`.
- `SUMMARY_PREFIX` is exactly `"Earlier in this conversation: "`; the summary is a `ChatMessage(role="system", ...)`.
- Summarize with `llm.complete(messages, max_tokens=400, temperature=0.0)` via `asyncio.to_thread`. `complete` returns `LLMResponse | str`.
- With the flag off, or `llm is None`, request behavior is byte-for-byte unchanged.
- No new dependency. No change inside `src/agents/`.
- Never `git stash pop` in this checkout; never commit to `main`. Work on `feat/working-memory-compression`.
- Run `ruff check . --fix && ruff format .` before each commit (pre-commit runs ruff-format and aborts the commit if it reformats; re-add and re-commit).
- Every new test gets a mutation check: temporarily remove the behavior it guards, confirm the test goes red, restore it (`git diff` must be clean of the mutation before committing — a stale `.pyc` can fake a red suite, so diff before you "fix" anything).
- Chat and Tool routers must not import `src.internal.servers.web.app` at module level (import cycle); they receive plain `llm` and `memory_compression: bool` arguments.
- Every surface test that turns the flag on must stub `schedule_compression` in the module under test (a no-op, or a recorder when the test asserts on it). `create_web_app` builds a real LLM client from `OPENAI_API_KEY` when one is in the developer's environment, and an unstubbed scheduler would then make a live summarization call from a unit test.

## Spec amendments (apply in Task 1, same commit)

Three details discovered while planning; edit the spec file so it matches the code that lands:

1. **Process-local in-flight guard.** `_InMemoryCacheLock.acquire` always returns `True`, so the cache lock alone gives no mutual exclusion on the default backend. `compress_session` also keeps a module-level `_inflight: set[str]` of session ids being summarized in this process. The cache lock still covers cross-process (Redis) deployments.
2. **Search mode keeps a leading system message.** Instead of a `summary=` parameter, `_build_search_agent_messages` preserves a leading `role == "system"` message ahead of the capped tail. No new argument threads through the four dispatchers that take `history`.
3. **Router arguments.** `create_chat_router(store, *, llm=None, memory_compression=False)` and `create_tool_router(..., llm=None, memory_compression=False)` take a bool, not the settings object (passing `SearchExperienceSettings` would import `app.py` into the routers). Chat and Tool schedule the task right after persisting the user message: their answer comes from the local model, so the remote `llm` summarizing concurrently contends with nothing, and one call site beats two. Assist schedules in its `finally`, after the reply, because there the same `llm` produces the answer.
4. `compress_session` does not take `store`; `pending` already carries the records.

---

### Task 1: State record and `load_working_memory`

**Files:**
- Create: `src/internal/memory/working.py`
- Create: `tests/unit/memory/test_working_memory.py`
- Modify: `docs/superpowers/specs/2026-09-11-working-memory-compression-design.md` (amendments above)

**Interfaces:**
- Consumes: `AgenticSearchStore.list_chat_messages(session_id) -> list[ChatMessageRecord]` (`src/internal/db/store.py:736`), `ChatMessageRecord.id/.role/.content` (`src/internal/db/models.py:87`), `ChatMessage(role, content)` (`src/context/models.py:19`), `CacheBackend.get/set` (`src/internal/cache/interface.py`).
- Produces:
  - `MAX_HISTORY_MESSAGES: int = 40`
  - `SUMMARY_PREFIX: str`
  - `SessionMemoryState(summary: str = "", summarized_through: str | None = None, curated_through: str | None = None)` (frozen dataclass)
  - `load_state(cache: CacheBackend, session_id: str) -> SessionMemoryState`
  - `save_state(cache: CacheBackend, session_id: str, state: SessionMemoryState) -> None`
  - `WorkingMemory(messages: list[ChatMessage], summary: str, pending: list[ChatMessageRecord])` (frozen dataclass)
  - `load_working_memory(store, session_id: str, *, keep_last: int = MAX_HISTORY_MESSAGES, cache: CacheBackend | None = None) -> WorkingMemory`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/memory/test_working_memory.py
"""Working memory: session tail + compressed summary of what fell off it."""

from __future__ import annotations

import json

import pytest

from src.internal.cache.interface import InMemoryCache
from src.internal.db import AgenticSearchStore
from src.internal.memory.working import (
    SUMMARY_PREFIX,
    SessionMemoryState,
    load_state,
    load_working_memory,
    save_state,
)


@pytest.fixture()
def store() -> AgenticSearchStore:
    return AgenticSearchStore(":memory:")


@pytest.fixture()
def cache() -> InMemoryCache:
    return InMemoryCache()


def _seed(store: AgenticSearchStore, n: int) -> tuple[str, list]:
    session = store.create_chat_session(title="s")
    records = [
        store.add_chat_message(
            session.id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"m{i}",
        )
        for i in range(n)
    ]
    return session.id, records


# --- state -------------------------------------------------------------------


def test_load_state_missing_key_is_default(cache):
    assert load_state(cache, "none") == SessionMemoryState()


def test_save_then_load_roundtrips(cache):
    state = SessionMemoryState(summary="s", summarized_through="m_3")
    save_state(cache, "sid", state)
    assert load_state(cache, "sid") == state
    assert cache.get("session_memory:sid") is not None


def test_load_state_malformed_json_is_default(cache):
    cache.set("session_memory:sid", b"{not json")
    assert load_state(cache, "sid") == SessionMemoryState()


def test_load_state_cache_error_is_default():
    class Broken(InMemoryCache):
        def get(self, key):
            raise RuntimeError("redis down")

    assert load_state(Broken(), "sid") == SessionMemoryState()


# --- load_working_memory ------------------------------------------------------


def test_below_cap_is_full_history_no_pending(store, cache):
    sid, records = _seed(store, 5)
    wm = load_working_memory(store, sid, keep_last=10, cache=cache)
    assert [m.content for m in wm.messages] == [r.content for r in records]
    assert wm.pending == []
    assert wm.summary == ""


def test_above_cap_no_state_is_tail_and_pending_prefix(store, cache):
    sid, records = _seed(store, 12)
    wm = load_working_memory(store, sid, keep_last=10, cache=cache)
    assert [m.content for m in wm.messages] == [r.content for r in records[2:]]
    assert [r.id for r in wm.pending] == [r.id for r in records[:2]]
    assert wm.summary == ""


def test_summary_covering_dropped_prefix_is_prepended(store, cache):
    sid, records = _seed(store, 12)
    save_state(
        cache, sid, SessionMemoryState(summary="they said hi", summarized_through=records[1].id)
    )
    wm = load_working_memory(store, sid, keep_last=10, cache=cache)
    assert wm.messages[0].role == "system"
    assert wm.messages[0].content == SUMMARY_PREFIX + "they said hi"
    assert [m.content for m in wm.messages[1:]] == [r.content for r in records[2:]]
    assert wm.pending == []
    assert wm.summary == "they said hi"


def test_summary_covering_part_of_prefix_leaves_rest_pending(store, cache):
    sid, records = _seed(store, 14)
    save_state(
        cache, sid, SessionMemoryState(summary="old", summarized_through=records[1].id)
    )
    wm = load_working_memory(store, sid, keep_last=10, cache=cache)
    assert wm.messages[0].content == SUMMARY_PREFIX + "old"
    assert [r.id for r in wm.pending] == [r.id for r in records[2:4]]


def test_unknown_cursor_ignores_summary_and_marks_all_pending(store, cache):
    sid, records = _seed(store, 12)
    save_state(
        cache, sid, SessionMemoryState(summary="stale", summarized_through="msg_not_here")
    )
    wm = load_working_memory(store, sid, keep_last=10, cache=cache)
    assert wm.messages[0].role == "user"
    assert wm.summary == ""
    assert [r.id for r in wm.pending] == [r.id for r in records[:2]]


def test_cursor_inside_tail_ignores_summary(store, cache):
    # keep_last grew since the summary was written: the covered turns are
    # verbatim in the tail, so the summary must not duplicate them.
    sid, records = _seed(store, 12)
    save_state(
        cache, sid, SessionMemoryState(summary="dup", summarized_through=records[5].id)
    )
    wm = load_working_memory(store, sid, keep_last=10, cache=cache)
    assert wm.summary == ""
    assert wm.messages[0].role == "user"


def test_cache_none_is_plain_tail_and_never_pending(store):
    sid, records = _seed(store, 12)
    wm = load_working_memory(store, sid, keep_last=10, cache=None)
    assert [m.content for m in wm.messages] == [r.content for r in records[2:]]
    assert wm.pending == []
    assert wm.summary == ""


def test_default_keep_last_is_forty(store, cache):
    sid, records = _seed(store, 45)
    wm = load_working_memory(store, sid, cache=cache)
    assert len(wm.messages) == 40
    assert len(wm.pending) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/memory/test_working_memory.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'src.internal.memory.working'`

- [ ] **Step 3: Write the implementation**

```python
# src/internal/memory/working.py
"""Working memory: the session tail plus a compressed summary of what fell off it.

Every conversational surface used to keep the last N messages and forget the
rest. This module keeps the same tail and, when turns fall outside it, hands
them to a background summarizer whose output is prepended to the next turn as
one system message. Per-session state (the summary and the id of the last
message it covers) lives in the process's ``CacheBackend`` -- in-memory by
default, Redis when ``CACHE_BACKEND=redis`` -- so no new dependency is added.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

from src.context.models import ChatMessage
from src.internal.cache.interface import CacheBackend
from src.internal.db.models import ChatMessageRecord

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 40
SUMMARY_PREFIX = "Earlier in this conversation: "

_STATE_KEY = "session_memory:{session_id}"


@dataclass(frozen=True)
class SessionMemoryState:
    """What the cache remembers about one session's working memory."""

    summary: str = ""
    # Id of the last message the summary covers. None until the first compress.
    summarized_through: str | None = None
    # Reserved for auto-curation (next PR); never written here.
    curated_through: str | None = None


def load_state(cache: CacheBackend, session_id: str) -> SessionMemoryState:
    """Read the session's state; the default on a missing key, bad JSON, or a
    failing cache. A cache problem must never fail the request that reads it."""
    key = _STATE_KEY.format(session_id=session_id)
    try:
        raw = cache.get(key)
    except Exception as exc:  # noqa: BLE001 - degrade to no summary
        logger.warning("session memory read failed for %s: %s", session_id, exc)
        return SessionMemoryState()
    if raw is None:
        return SessionMemoryState()
    try:
        data = json.loads(raw)
        return SessionMemoryState(
            summary=str(data.get("summary", "")),
            summarized_through=data.get("summarized_through"),
            curated_through=data.get("curated_through"),
        )
    except (ValueError, AttributeError):
        return SessionMemoryState()


def save_state(cache: CacheBackend, session_id: str, state: SessionMemoryState) -> None:
    cache.set(_STATE_KEY.format(session_id=session_id), json.dumps(asdict(state)))


@dataclass(frozen=True)
class WorkingMemory:
    """What a surface hands to its loop, plus what still needs summarizing."""

    messages: list[ChatMessage]
    summary: str
    pending: list[ChatMessageRecord]


def load_working_memory(
    store,
    session_id: str,
    *,
    keep_last: int = MAX_HISTORY_MESSAGES,
    cache: CacheBackend | None = None,
) -> WorkingMemory:
    """Return the last ``keep_last`` messages, prefixed by the stored summary
    when one covers the dropped prefix.

    ``cache=None`` (the flag off) skips state entirely: the result is exactly
    the tail-slice every surface used before, and ``pending`` is empty so no
    compression is ever scheduled.
    """
    records = store.list_chat_messages(session_id)
    tail = records[-keep_last:]
    dropped = records[:-keep_last]
    messages = [ChatMessage(role=r.role, content=r.content) for r in tail]
    if cache is None or not dropped:
        return WorkingMemory(messages=messages, summary="", pending=[])

    state = load_state(cache, session_id)
    dropped_ids = [r.id for r in dropped]
    if state.summary and state.summarized_through in dropped_ids:
        cut = dropped_ids.index(state.summarized_through) + 1
        summary_message = ChatMessage(role="system", content=SUMMARY_PREFIX + state.summary)
        return WorkingMemory(
            messages=[summary_message, *messages],
            summary=state.summary,
            pending=dropped[cut:],
        )
    # No summary, or its cursor is not in this session's dropped prefix (state
    # lost, or the cap grew so the covered turns are back in the tail): start over.
    return WorkingMemory(messages=messages, summary="", pending=list(dropped))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/memory/test_working_memory.py -v`
Expected: 13 passed

- [ ] **Step 5: Mutation check**

In `load_working_memory`, change `cut = dropped_ids.index(...) + 1` to `cut = 0`. Run the file. Expected: `test_summary_covering_dropped_prefix_is_prepended` and `test_summary_covering_part_of_prefix_leaves_rest_pending` FAIL. Restore the line. Then change `if cache is None or not dropped:` to `if not dropped:`. Expected: `test_cache_none_is_plain_tail_and_never_pending` FAILS. Restore. Confirm `git diff src/internal/memory/working.py` shows only the intended file content.

- [ ] **Step 6: Apply the spec amendments**

Edit `docs/superpowers/specs/2026-09-11-working-memory-compression-design.md`:

1. In **Compression**, change the signature to `compress_session(session_id, llm, *, pending, cache=None) -> bool` and add after the lock bullet: "The task also keeps a module-level `_inflight` set of session ids being summarized in this process. `_InMemoryCacheLock` always acquires, so on the default backend this set is the real guard; the cache lock covers cross-process deployments."
2. In **Call sites**, replace the Search-mode row's "After" cell with: "`_build_search_agent_messages` keeps a leading `role == \"system\"` message (the summary) ahead of the tail it caps to 6. No new argument."
3. Replace the router-construction paragraph with: "`create_chat_router(db)` becomes `create_chat_router(db, *, llm=None, memory_compression=False)` and `create_tool_router(...)` gains the same two keyword arguments. A bool rather than the settings object, because the settings class lives in `app.py`, which imports both routers. Chat and Tool schedule the task right after persisting the user message: their answer comes from the local model, so the remote `llm` contends with nothing. Assist schedules in its `finally`, after the reply, because there the same `llm` produces the answer."
4. In **Flag**, replace "the Chat router gains it (see Call sites)" with "the Chat and Tool routers receive `memory_compression` as a bool (see Call sites)".

- [ ] **Step 7: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/memory/working.py tests/unit/memory/test_working_memory.py docs/superpowers/specs/2026-09-11-working-memory-compression-design.md
git commit -m "feat(memory): session working-memory state and loader

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `compress_session` and `schedule_compression`

**Files:**
- Modify: `src/internal/memory/working.py`
- Modify: `tests/unit/memory/test_working_memory.py`

**Interfaces:**
- Consumes: Task 1's `load_state`, `save_state`, `SessionMemoryState`, `WorkingMemory`; `CacheBackend.lock(name, timeout) -> CacheLock` with `acquire(blocking=False) -> bool` and `release()`; `get_cache_backend()` from `src.internal.cache.interface`.
- Produces:
  - `async compress_session(session_id: str, llm, *, pending: list[ChatMessageRecord], cache: CacheBackend | None = None) -> bool`
  - `schedule_compression(wm: WorkingMemory, *, session_id: str, llm, enabled: bool, cache: CacheBackend | None = None) -> asyncio.Task | None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/memory/test_working_memory.py`:

```python
import asyncio
import time

from src.internal.memory.working import (
    WorkingMemory,
    compress_session,
    schedule_compression,
)


class FakeLLM:
    def __init__(self, text="SUMMARY", *, delay=0.0, fail=False):
        self.text, self.delay, self.fail = text, delay, fail
        self.prompts: list[list[dict]] = []

    def complete(self, messages, **kwargs):
        self.prompts.append(messages)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("llm down")
        return self.text


# --- compress_session ---------------------------------------------------------


def test_compress_advances_cursor_and_stores_text(store, cache):
    sid, records = _seed(store, 12)
    pending = records[:2]
    ok = asyncio.run(compress_session(sid, FakeLLM("they said hi"), pending=pending, cache=cache))
    assert ok is True
    state = load_state(cache, sid)
    assert state.summary == "they said hi"
    assert state.summarized_through == records[1].id
    assert state.curated_through is None


def test_compress_prompt_carries_prior_summary_and_every_turn(store, cache):
    sid, records = _seed(store, 12)
    save_state(cache, sid, SessionMemoryState(summary="PRIOR", summarized_through=records[0].id))
    llm = FakeLLM()
    asyncio.run(compress_session(sid, llm, pending=records[1:3], cache=cache))
    user_prompt = llm.prompts[0][-1]["content"]
    assert "PRIOR" in user_prompt
    assert "USER: m2" in user_prompt
    assert "ASSISTANT: m1" in user_prompt


def test_compress_preserves_curated_through(store, cache):
    sid, records = _seed(store, 12)
    save_state(cache, sid, SessionMemoryState(curated_through="keep-me"))
    asyncio.run(compress_session(sid, FakeLLM(), pending=records[:2], cache=cache))
    assert load_state(cache, sid).curated_through == "keep-me"


def test_compress_llm_failure_leaves_state_and_allows_retry(store, cache):
    sid, records = _seed(store, 12)
    ok = asyncio.run(compress_session(sid, FakeLLM(fail=True), pending=records[:2], cache=cache))
    assert ok is False
    assert load_state(cache, sid) == SessionMemoryState()
    ok = asyncio.run(compress_session(sid, FakeLLM("later"), pending=records[:2], cache=cache))
    assert ok is True
    assert load_state(cache, sid).summary == "later"


def test_compress_empty_text_leaves_state(store, cache):
    sid, records = _seed(store, 12)
    ok = asyncio.run(compress_session(sid, FakeLLM("   "), pending=records[:2], cache=cache))
    assert ok is False
    assert load_state(cache, sid) == SessionMemoryState()


def test_compress_skips_when_cursor_already_at_last_pending(store, cache):
    sid, records = _seed(store, 12)
    save_state(cache, sid, SessionMemoryState(summary="done", summarized_through=records[1].id))
    llm = FakeLLM()
    ok = asyncio.run(compress_session(sid, llm, pending=records[:2], cache=cache))
    assert ok is False
    assert llm.prompts == []


def test_compress_no_pending_or_no_llm_is_noop(store, cache):
    sid, records = _seed(store, 12)
    assert asyncio.run(compress_session(sid, FakeLLM(), pending=[], cache=cache)) is False
    assert asyncio.run(compress_session(sid, None, pending=records[:2], cache=cache)) is False


def test_concurrent_compress_calls_llm_once(store, cache):
    sid, records = _seed(store, 12)
    llm = FakeLLM(delay=0.05)

    async def both():
        return await asyncio.gather(
            compress_session(sid, llm, pending=records[:2], cache=cache),
            compress_session(sid, llm, pending=records[:2], cache=cache),
        )

    results = asyncio.run(both())
    assert sorted(results) == [False, True]
    assert len(llm.prompts) == 1


# --- schedule_compression -----------------------------------------------------


def _wm(pending):
    return WorkingMemory(messages=[], summary="", pending=pending)


def test_schedule_returns_none_when_disabled_or_no_llm_or_nothing_pending(store, cache):
    sid, records = _seed(store, 12)

    async def run():
        assert schedule_compression(_wm(records[:2]), session_id=sid, llm=FakeLLM(), enabled=False, cache=cache) is None
        assert schedule_compression(_wm(records[:2]), session_id=sid, llm=None, enabled=True, cache=cache) is None
        assert schedule_compression(_wm([]), session_id=sid, llm=FakeLLM(), enabled=True, cache=cache) is None

    asyncio.run(run())
    assert load_state(cache, sid) == SessionMemoryState()


def test_schedule_runs_compress_in_background(store, cache):
    sid, records = _seed(store, 12)

    async def run():
        task = schedule_compression(_wm(records[:2]), session_id=sid, llm=FakeLLM("bg"), enabled=True, cache=cache)
        assert task is not None
        await task

    asyncio.run(run())
    assert load_state(cache, sid).summary == "bg"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/memory/test_working_memory.py -v`
Expected: FAIL at import with `ImportError: cannot import name 'compress_session'`

- [ ] **Step 3: Write the implementation**

Append to `src/internal/memory/working.py` (add `import asyncio` to the imports and `from src.internal.cache.interface import CacheBackend, get_cache_backend`):

```python
_LOCK_KEY = "session_memory:{session_id}:compress"
_SUMMARY_MAX_TOKENS = 400
_SUMMARY_SYSTEM = (
    "You compress a conversation. Rewrite the prior summary and the new turns "
    "into one concise summary that keeps facts, decisions, user preferences, "
    "and open questions. Output the summary only."
)

# Sessions being summarized in this process. `_InMemoryCacheLock` always
# acquires, so on the default backend this set is the real mutual exclusion;
# the cache lock below covers a Redis deployment with several processes.
_inflight: set[str] = set()
# Strong references to scheduled tasks, so the event loop cannot drop them.
_tasks: set[asyncio.Task] = set()


def _summary_prompt(prior: str, pending: list[ChatMessageRecord]) -> list[dict]:
    turns = "\n".join(f"{r.role.upper()}: {r.content}" for r in pending)
    return [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {
            "role": "user",
            "content": f"Prior summary:\n{prior or '(none)'}\n\nNew turns:\n{turns}",
        },
    ]


def _complete(llm, prompt: list[dict]) -> str:
    raw = llm.complete(prompt, max_tokens=_SUMMARY_MAX_TOKENS, temperature=0.0)
    text = raw if isinstance(raw, str) else getattr(raw, "text", "")
    return (text or "").strip()


async def compress_session(
    session_id: str,
    llm,
    *,
    pending: list[ChatMessageRecord],
    cache: CacheBackend | None = None,
) -> bool:
    """Summarize ``pending`` into the session's stored summary.

    Returns True when the state advanced. False means nothing to do, another
    task owns this span, or the summarizer failed -- in which case the state
    is untouched and the next turn retries the same span.
    """
    if not pending or llm is None or session_id in _inflight:
        return False
    cache = cache if cache is not None else get_cache_backend()
    lock = cache.lock(_LOCK_KEY.format(session_id=session_id), timeout=120)
    if not lock.acquire(blocking=False):
        return False
    _inflight.add(session_id)
    try:
        state = load_state(cache, session_id)
        last_id = pending[-1].id
        if state.summarized_through == last_id:
            return False
        text = await asyncio.to_thread(_complete, llm, _summary_prompt(state.summary, pending))
        if not text:
            return False
        save_state(
            cache,
            session_id,
            SessionMemoryState(
                summary=text,
                summarized_through=last_id,
                curated_through=state.curated_through,
            ),
        )
        return True
    except Exception as exc:  # noqa: BLE001 - compression is a delivery detail
        logger.warning("session memory compression failed for %s: %s", session_id, exc)
        return False
    finally:
        _inflight.discard(session_id)
        lock.release()


def schedule_compression(
    wm: WorkingMemory,
    *,
    session_id: str,
    llm,
    enabled: bool,
    cache: CacheBackend | None = None,
) -> asyncio.Task | None:
    """Fire-and-forget ``compress_session`` when there is something to compress.

    Must be called from a running event loop. Returns the task (tests await
    it) or None when nothing was scheduled.
    """
    if not enabled or llm is None or not wm.pending:
        return None
    task = asyncio.create_task(
        compress_session(session_id, llm, pending=wm.pending, cache=cache)
    )
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/memory/test_working_memory.py -v`
Expected: 23 passed

- [ ] **Step 5: Mutation check**

Remove `session_id in _inflight` from the first guard and the `_inflight.add(session_id)` line. Run. Expected: `test_concurrent_compress_calls_llm_once` FAILS (two prompts). Restore. Change `if state.summarized_through == last_id: return False` to `pass`. Expected: `test_compress_skips_when_cursor_already_at_last_pending` FAILS. Restore. `git diff` shows only the intended additions.

- [ ] **Step 6: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/memory/working.py tests/unit/memory/test_working_memory.py
git commit -m "feat(memory): background session compression with in-flight guard

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Assist wiring, flag, and search-mode summary preservation

**Files:**
- Modify: `src/internal/servers/web/app.py` (settings dataclass ~140-185; history load ~1554-1559; `finally` ~1888; `MAX_HISTORY_MESSAGES` and `_build_search_agent_messages` ~2123-2148)
- Modify: `docs/configuration.md` (env table, after the `AGENTIC_SEARCH_MEMORY_REQUIRE_AUTH` row at line 92)
- Modify: `tests/unit/test_search_agent_history.py`
- Modify: `tests/unit/servers/web/test_web_experience_app.py`

**Interfaces:**
- Consumes: Task 1 `load_working_memory`, `MAX_HISTORY_MESSAGES`, `SUMMARY_PREFIX`, `save_state`, `SessionMemoryState`; Task 2 `schedule_compression`; `get_cache_backend()`.
- Produces: `SearchExperienceSettings.memory_compression: bool = False`; `src.internal.servers.web.app.MAX_HISTORY_MESSAGES` still importable (re-export); `_build_search_agent_messages(query, history)` keeps a leading system message.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_search_agent_history.py`:

```python
def test_leading_system_summary_survives_the_cap():
    history = [ChatMessage(role="system", content="Earlier in this conversation: x")] + [
        ChatMessage(role=("user" if i % 2 == 0 else "assistant"), content=f"m{i}")
        for i in range(20)
    ]
    msgs = _build_search_agent_messages("q", history)
    assert msgs[0] == {"role": "system", "content": "Earlier in this conversation: x"}
    assert len(msgs) == SEARCH_AGENT_HISTORY_MESSAGES + 2
    assert [m["content"] for m in msgs[1:-1]] == [
        f"m{i}" for i in range(20 - SEARCH_AGENT_HISTORY_MESSAGES, 20)
    ]
    assert msgs[-1] == {"role": "user", "content": "q"}
```

Append to `tests/unit/servers/web/test_web_experience_app.py`:

```python
def test_memory_compression_flag_defaults_off_and_reads_env(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_MEMORY_COMPRESSION", raising=False)
    assert SearchExperienceSettings.from_app_settings().memory_compression is False
    monkeypatch.setenv("AGENTIC_SEARCH_MEMORY_COMPRESSION", "true")
    assert SearchExperienceSettings.from_app_settings().memory_compression is True


def _seed_long_session(store, n=45):
    session = store.create_chat_session(title="long")
    records = [
        store.add_chat_message(
            session.id, role="user" if i % 2 == 0 else "assistant", content=f"msg {i}"
        )
        for i in range(n)
    ]
    return session.id, records


def test_run_agent_prepends_stored_summary_when_flag_on(monkeypatch, tmp_path):
    from src.internal.cache.interface import InMemoryCache
    from src.internal.memory.working import SUMMARY_PREFIX, SessionMemoryState, save_state

    cache = InMemoryCache()
    monkeypatch.setattr("src.internal.cache.interface._default_cache", cache)
    # Never let a developer's OPENAI_API_KEY turn this into a live call.
    monkeypatch.setattr(
        "src.internal.servers.web.app.schedule_compression", lambda wm, **kw: None
    )
    captured: list = []

    async def fake_answer(question, *, llm=None, chat_history=None, **kw):
        captured.append(list(chat_history or []))
        return _answer_result(question)

    monkeypatch.setattr("src.internal.servers.web.app.answer_with_retrieval", fake_answer)

    store = AgenticSearchStore(tmp_path / "state.sqlite3")
    session_id, records = _seed_long_session(store)
    save_state(cache, session_id, SessionMemoryState(summary="S", summarized_through=records[4].id))

    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3", memory_compression=True),
        store=store,
    )
    TestClient(app).post(
        "/api/agent", json={"query": "follow up", "mode": "chat_once", "session_id": session_id}
    )

    assert len(captured) == 1
    assert captured[0][0].role == "system"
    assert captured[0][0].content == SUMMARY_PREFIX + "S"
    assert len(captured[0]) == 41


def test_run_agent_flag_off_ignores_stored_summary(monkeypatch, tmp_path):
    from src.internal.cache.interface import InMemoryCache
    from src.internal.memory.working import SessionMemoryState, save_state

    cache = InMemoryCache()
    monkeypatch.setattr("src.internal.cache.interface._default_cache", cache)
    captured: list = []

    async def fake_answer(question, *, llm=None, chat_history=None, **kw):
        captured.append(list(chat_history or []))
        return _answer_result(question)

    monkeypatch.setattr("src.internal.servers.web.app.answer_with_retrieval", fake_answer)

    store = AgenticSearchStore(tmp_path / "state.sqlite3")
    session_id, records = _seed_long_session(store)
    save_state(cache, session_id, SessionMemoryState(summary="S", summarized_through=records[4].id))

    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"), store=store)
    TestClient(app).post(
        "/api/agent", json={"query": "follow up", "mode": "chat_once", "session_id": session_id}
    )

    assert [m.role for m in captured[0]][:1] == ["user"]
    assert len(captured[0]) == 40


def test_run_agent_schedules_compression_after_reply(monkeypatch, tmp_path):
    from src.internal.cache.interface import InMemoryCache
    from src.internal.memory import working

    cache = InMemoryCache()
    monkeypatch.setattr("src.internal.cache.interface._default_cache", cache)
    scheduled: list = []

    def fake_schedule(wm, **kw):
        scheduled.append((len(wm.pending), kw["enabled"], kw["llm"]))
        return None

    monkeypatch.setattr("src.internal.servers.web.app.schedule_compression", fake_schedule)

    async def fake_answer(question, **kw):
        return _answer_result(question)

    monkeypatch.setattr("src.internal.servers.web.app.answer_with_retrieval", fake_answer)

    store = AgenticSearchStore(tmp_path / "state.sqlite3")
    session_id, _ = _seed_long_session(store)
    sentinel_llm = object()
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3", memory_compression=True),
        store=store,
        llm=sentinel_llm,
    )
    TestClient(app).post(
        "/api/agent", json={"query": "follow up", "mode": "chat_once", "session_id": session_id}
    )

    assert scheduled == [(5, True, sentinel_llm)]
```

Note on `llm=sentinel_llm`: `create_web_app` accepts any object as `llm`; the `chat_once` path hands it to `answer_with_retrieval`, which is patched, so a bare `object()` is safe here.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_search_agent_history.py tests/unit/servers/web/test_web_experience_app.py -v -k "summary or compression or flag_defaults"`
Expected: `test_leading_system_summary_survives_the_cap` FAILS (summary sliced off); the three web tests FAIL with `TypeError: ... unexpected keyword argument 'memory_compression'` / `AttributeError: ... has no attribute 'schedule_compression'`.

- [ ] **Step 3: Add the flag to `SearchExperienceSettings`**

In `src/internal/servers/web/app.py`, after the `memory_require_auth: bool = False` field:

```python
    # Summarize turns that fall off the history tail into a system message the
    # next turn sees, using the configured LLM client. Off by default; with no
    # LLM client the flag is inert.
    memory_compression: bool = False
```

and in `from_app_settings`, after the `memory_require_auth=` line:

```python
            memory_compression=_flag("AGENTIC_SEARCH_MEMORY_COMPRESSION"),
```

- [ ] **Step 4: Move `MAX_HISTORY_MESSAGES` and preserve a leading system message**

Add to the imports near the other `src.internal.*` imports in `app.py`:

```python
from src.internal.cache.interface import get_cache_backend
from src.internal.memory.working import (
    MAX_HISTORY_MESSAGES,
    load_working_memory,
    schedule_compression,
)
```

Delete the line `MAX_HISTORY_MESSAGES = 40` (keep the `_trim_history` function; its default now refers to the import). Replace `_build_search_agent_messages`:

```python
def _build_search_agent_messages(query: str, history: list) -> list[dict[str, str]]:
    """Build the SearchAgentLoop message buffer: capped prior turns + the query.

    A leading ``system`` message is the working-memory summary of turns that
    already fell off the tail; it is kept ahead of the cap. The rest of the
    history is capped to the last ``SEARCH_AGENT_HISTORY_MESSAGES`` messages
    and mapped to ``{"role", "content"}`` dicts; the current user query is
    appended last. ``SearchAgentLoop._with_system_prompt`` prepends the system
    prompt.
    """
    leading = history[:1] if history and history[0].role == "system" else []
    capped = _trim_history(
        history[len(leading) :], max_messages=SEARCH_AGENT_HISTORY_MESSAGES
    )
    messages = [{"role": m.role, "content": m.content} for m in [*leading, *capped]]
    messages.append({"role": "user", "content": query})
    return messages
```

- [ ] **Step 5: Load working memory and schedule compression in `_run_agent_impl`**

Replace the history load (currently `history = _trim_history([...])` right before `db.add_chat_message(session_id, role="user", content=query)`) with:

```python
        working = load_working_memory(
            db,
            session_id,
            keep_last=MAX_HISTORY_MESSAGES,
            cache=get_cache_backend() if settings.memory_compression else None,
        )
        history = working.messages
        db.add_chat_message(session_id, role="user", content=query)
```

`history` was `list[ChatMessage]` before and still is. In the outer `finally:` of `_run_agent_impl` (the one that calls `STAGE_LATENCY.record(...)`), add as the first statement:

```python
            schedule_compression(
                working,
                session_id=session_id,
                llm=llm,
                enabled=settings.memory_compression,
            )
```

`llm` and `settings` are already closed over by `_run_agent_impl` (see `create_web_app`). Remove the now-unused `ChatMessage` import from `app.py` only if `ruff` reports it unused; other code in the file may still use it.

- [ ] **Step 6: Document the flag**

In `docs/configuration.md`, directly after the `AGENTIC_SEARCH_MEMORY_REQUIRE_AUTH` row:

```markdown
| `AGENTIC_SEARCH_MEMORY_COMPRESSION` | `false` | When session history exceeds the 40-message tail, summarize the dropped turns in the background with the configured LLM client and prepend the summary as a system message on the next turn. Inert without an LLM client. State lives in the cache backend (`CACHE_BACKEND`), in-memory by default |
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/unit/test_search_agent_history.py tests/unit/servers/web/test_web_experience_app.py tests/unit/memory/test_working_memory.py -v`
Expected: all pass, including the pre-existing `test_run_agent_trims_long_history` and `test_cap_default_is_six`.

- [ ] **Step 8: Mutation check**

In `_build_search_agent_messages`, set `leading = []`. Expected: `test_leading_system_summary_survives_the_cap` FAILS. Restore. In `_run_agent_impl`, change `cache=get_cache_backend() if settings.memory_compression else None` to `cache=None`. Expected: `test_run_agent_prepends_stored_summary_when_flag_on` FAILS. Restore. Remove the `schedule_compression(...)` call from the `finally`. Expected: `test_run_agent_schedules_compression_after_reply` FAILS. Restore. `git diff` clean of mutations.

- [ ] **Step 9: Lint, full suite, commit**

```bash
ruff check . --fix && ruff format .
pytest -q
git add src/internal/servers/web/app.py docs/configuration.md tests/unit/test_search_agent_history.py tests/unit/servers/web/test_web_experience_app.py
git commit -m "feat(web): working-memory compression on the Assist path behind AGENTIC_SEARCH_MEMORY_COMPRESSION

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Chat surface

**Files:**
- Modify: `src/internal/servers/query_and_chat/chat_backend.py` (`_MAX_HISTORY_MESSAGES` line 36; `create_chat_router` signature line 70; history load ~208-212)
- Modify: `tests/unit/test_chat_backend.py`

**Interfaces:**
- Consumes: Task 1 `load_working_memory`, `MAX_HISTORY_MESSAGES`; Task 2 `schedule_compression`; `get_cache_backend()`.
- Produces: `create_chat_router(store, *, llm=None, memory_compression: bool = False) -> APIRouter`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_chat_backend.py`:

```python
def _seed_long(store, n=45):
    session = store.create_chat_session(user_id=_USER_ID, title="long")
    records = [
        store.add_chat_message(
            session.id, role="user" if i % 2 == 0 else "assistant", content=f"m{i}"
        )
        for i in range(n)
    ]
    return session.id, records


def _client_with(store, monkeypatch, *, llm=None, memory_compression=False):
    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.chat_backend.resolve_active_user",
        lambda _request, _store: _USER,
    )
    app = FastAPI()
    app.include_router(
        create_chat_router(store, llm=llm, memory_compression=memory_compression)
    )
    app.state.search_agent_manager = object()
    app.state.search_agent_tokenizer = object()
    return TestClient(app)


def _capture_plain_chat(monkeypatch):
    captured: list = []

    async def fake_run(message, *, manager, tokenizer, history, on_turn=None, **kw):
        captured.append(list(history))
        return "ok"

    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.chat_backend._run_plain_chat", fake_run
    )
    return captured


def test_send_chat_flag_off_hands_runner_last_forty(store, monkeypatch):
    from src.internal.cache.interface import InMemoryCache
    from src.internal.memory.working import SessionMemoryState, save_state

    cache = InMemoryCache()
    monkeypatch.setattr("src.internal.cache.interface._default_cache", cache)
    session_id, records = _seed_long(store)
    save_state(cache, session_id, SessionMemoryState(summary="S", summarized_through=records[4].id))
    captured = _capture_plain_chat(monkeypatch)

    client = _client_with(store, monkeypatch)
    resp = client.post(
        "/chat/send-chat-message",
        json={"message": "next", "session_id": session_id, "stream": False},
    )
    assert resp.status_code == 200
    assert len(captured[0]) == 40
    assert captured[0][0].role == "user"


def test_send_chat_flag_on_prepends_stored_summary(store, monkeypatch):
    from src.internal.cache.interface import InMemoryCache
    from src.internal.memory.working import SUMMARY_PREFIX, SessionMemoryState, save_state

    cache = InMemoryCache()
    monkeypatch.setattr("src.internal.cache.interface._default_cache", cache)
    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.chat_backend.schedule_compression",
        lambda wm, **kw: None,
    )
    session_id, records = _seed_long(store)
    save_state(cache, session_id, SessionMemoryState(summary="S", summarized_through=records[4].id))
    captured = _capture_plain_chat(monkeypatch)

    client = _client_with(store, monkeypatch, llm=object(), memory_compression=True)
    client.post(
        "/chat/send-chat-message",
        json={"message": "next", "session_id": session_id, "stream": False},
    )
    assert len(captured[0]) == 41
    assert captured[0][0].role == "system"
    assert captured[0][0].content == SUMMARY_PREFIX + "S"


def test_send_chat_schedules_compression(store, monkeypatch):
    from src.internal.cache.interface import InMemoryCache

    monkeypatch.setattr("src.internal.cache.interface._default_cache", InMemoryCache())
    session_id, _ = _seed_long(store)
    _capture_plain_chat(monkeypatch)
    scheduled: list = []

    def fake_schedule(wm, **kw):
        scheduled.append((len(wm.pending), kw["enabled"], kw["llm"]))
        return None

    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.chat_backend.schedule_compression",
        fake_schedule,
    )
    sentinel = object()
    client = _client_with(store, monkeypatch, llm=sentinel, memory_compression=True)
    client.post(
        "/chat/send-chat-message",
        json={"message": "next", "session_id": session_id, "stream": False},
    )
    assert scheduled == [(5, True, sentinel)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_chat_backend.py -v -k "send_chat_flag or schedules_compression"`
Expected: FAIL with `TypeError: create_chat_router() got an unexpected keyword argument 'llm'`.

- [ ] **Step 3: Wire the chat router**

In `chat_backend.py`, replace the `_MAX_HISTORY_MESSAGES = 40` line with imports (put them with the other `src.internal.*` imports):

```python
from src.internal.cache.interface import get_cache_backend
from src.internal.memory.working import (
    MAX_HISTORY_MESSAGES,
    load_working_memory,
    schedule_compression,
)
```

Change the signature and docstring:

```python
def create_chat_router(
    store: AgenticSearchStore,
    *,
    llm=None,
    memory_compression: bool = False,
) -> APIRouter:
    """Return an APIRouter for chat session endpoints bound to *store*.

    ``llm`` and ``memory_compression`` drive working-memory compression: when
    both are set, turns that fall off the history tail are summarized in the
    background and the next turn sees the summary. Plain chat itself still
    runs on the local model.
    """
```

Replace the history load in `send_chat_message`:

```python
        working = load_working_memory(
            store,
            session_id,
            keep_last=MAX_HISTORY_MESSAGES,
            cache=get_cache_backend() if memory_compression else None,
        )
        history = working.messages
        store.add_chat_message(session_id, role="user", content=body.message)
        # The answer comes from the local model, so the remote llm summarizing
        # now contends with nothing; one site covers both the stream and
        # non-stream branches.
        schedule_compression(
            working, session_id=session_id, llm=llm, enabled=memory_compression
        )
```

Delete the now-unused `from src.context import ChatMessage` import if `ruff` flags it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_chat_backend.py -v`
Expected: all pass, including every pre-existing test (the signature change is keyword-only with defaults).

- [ ] **Step 5: Mutation check**

Change `cache=get_cache_backend() if memory_compression else None` to `cache=None`. Expected: `test_send_chat_flag_on_prepends_stored_summary` FAILS. Restore. Remove the `schedule_compression(...)` call. Expected: `test_send_chat_schedules_compression` FAILS. Restore. `git diff` clean.

- [ ] **Step 6: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/servers/query_and_chat/chat_backend.py tests/unit/test_chat_backend.py
git commit -m "feat(chat): working-memory compression on the Chat surface

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Tool surface and router plumbing

**Files:**
- Modify: `src/internal/servers/query_and_chat/tool_backend.py` (`_MAX_HISTORY_MESSAGES` line 33; `create_tool_router` signature line 36; `_history` ~66-71; history load in `send_tool_message` ~93-95)
- Modify: `src/internal/servers/web/app.py` (`_register_routers` signature ~365-373, chat/tool mounts ~380-385, call ~1420-1428)
- Modify: `tests/unit/test_tool_backend.py`
- Modify: `tests/unit/servers/web/test_web_experience_app.py`

**Interfaces:**
- Consumes: Task 1 `load_working_memory`, `MAX_HISTORY_MESSAGES`; Task 2 `schedule_compression`; `get_cache_backend()`; Task 4's `create_chat_router(store, *, llm, memory_compression)`.
- Produces: `create_tool_router(store, *, search_url=..., resolved, llm=None, memory_compression: bool = False)`; `_register_routers(..., memory_compression: bool = False)` passes `llm` and `memory_compression` to both routers.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_tool_backend.py`:

```python
def _make_app_with_memory(*, llm=None, memory_compression=False):
    store = AgenticSearchStore(":memory:")
    app = FastAPI()
    app.include_router(
        create_tool_router(
            store,
            search_url="http://x/retrieve",
            resolved=load_app_settings(),
            llm=llm,
            memory_compression=memory_compression,
        )
    )
    app.state.search_agent_manager = object()
    app.state.search_agent_tokenizer = object()
    app.state.tool_approval_broker = None
    app.state._store = store
    return app


def _seed_long(store, n=45):
    session = store.create_chat_session(title="long")
    records = [
        store.add_chat_message(
            session.id, role="user" if i % 2 == 0 else "assistant", content=f"m{i}"
        )
        for i in range(n)
    ]
    return session.id, records


def _capture_tool_agent(monkeypatch):
    from src.internal.servers.web import tool_agent_runner

    captured: list = []

    async def fake_run_tool_agent(query, *, history, **kw):
        captured.append(list(history))
        return ("ok", [], [], "tool", {"tool_calls": [], "num_turns": 1})

    monkeypatch.setattr(tool_agent_runner, "_run_tool_agent", fake_run_tool_agent)
    return captured


def test_send_tool_flag_off_hands_runner_last_forty(monkeypatch):
    from src.internal.cache.interface import InMemoryCache
    from src.internal.memory.working import SessionMemoryState, save_state

    cache = InMemoryCache()
    monkeypatch.setattr("src.internal.cache.interface._default_cache", cache)
    app = _make_app_with_memory()
    session_id, records = _seed_long(app.state._store)
    save_state(cache, session_id, SessionMemoryState(summary="S", summarized_through=records[4].id))
    captured = _capture_tool_agent(monkeypatch)

    resp = TestClient(app).post(
        "/tool/send-tool-message",
        json={"message": "next", "session_id": session_id, "stream": False},
    )
    assert resp.status_code == 200
    assert len(captured[0]) == 40
    assert captured[0][0].role == "user"


def test_send_tool_flag_on_prepends_stored_summary(monkeypatch):
    from src.internal.cache.interface import InMemoryCache
    from src.internal.memory.working import SUMMARY_PREFIX, SessionMemoryState, save_state

    cache = InMemoryCache()
    monkeypatch.setattr("src.internal.cache.interface._default_cache", cache)
    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.tool_backend.schedule_compression",
        lambda wm, **kw: None,
    )
    app = _make_app_with_memory(llm=object(), memory_compression=True)
    session_id, records = _seed_long(app.state._store)
    save_state(cache, session_id, SessionMemoryState(summary="S", summarized_through=records[4].id))
    captured = _capture_tool_agent(monkeypatch)

    TestClient(app).post(
        "/tool/send-tool-message",
        json={"message": "next", "session_id": session_id, "stream": False},
    )
    assert len(captured[0]) == 41
    assert captured[0][0].role == "system"
    assert captured[0][0].content == SUMMARY_PREFIX + "S"


def test_send_tool_schedules_compression(monkeypatch):
    from src.internal.cache.interface import InMemoryCache
    from src.internal.servers.query_and_chat import tool_backend

    monkeypatch.setattr("src.internal.cache.interface._default_cache", InMemoryCache())
    sentinel = object()
    app = _make_app_with_memory(llm=sentinel, memory_compression=True)
    session_id, _ = _seed_long(app.state._store)
    _capture_tool_agent(monkeypatch)
    scheduled: list = []

    def fake_schedule(wm, **kw):
        scheduled.append((len(wm.pending), kw["enabled"], kw["llm"]))
        return None

    monkeypatch.setattr(tool_backend, "schedule_compression", fake_schedule)
    TestClient(app).post(
        "/tool/send-tool-message",
        json={"message": "next", "session_id": session_id, "stream": False},
    )
    assert scheduled == [(5, True, sentinel)]
```

Append to `tests/unit/servers/web/test_web_experience_app.py`:

```python
def test_register_routers_passes_memory_settings_to_chat_and_tool(monkeypatch, tmp_path):
    seen: dict = {}

    def fake_chat_router(store, *, llm=None, memory_compression=False):
        seen["chat"] = (llm, memory_compression)
        from fastapi import APIRouter

        return APIRouter()

    def fake_tool_router(store, *, search_url, resolved, llm=None, memory_compression=False):
        seen["tool"] = (llm, memory_compression)
        from fastapi import APIRouter

        return APIRouter()

    monkeypatch.setattr("src.internal.servers.web.app.create_chat_router", fake_chat_router)
    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.tool_backend.create_tool_router", fake_tool_router
    )
    sentinel = object()
    create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "s.sqlite3", memory_compression=True),
        llm=sentinel,
    )
    assert seen == {"chat": (sentinel, True), "tool": (sentinel, True)}
```

The tool router is imported function-locally inside `_register_routers` (`from src.internal.servers.query_and_chat.tool_backend import create_tool_router`), which is why the patch target is the `tool_backend` module, while the chat router is a module-level import in `app.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_tool_backend.py tests/unit/servers/web/test_web_experience_app.py -v -k "send_tool_flag or send_tool_schedules or register_routers_passes"`
Expected: tool tests FAIL with `TypeError: create_tool_router() got an unexpected keyword argument 'llm'`; the app test FAILS with `assert {...} == {...}` (both entries `(None, False)`).

- [ ] **Step 3: Wire the tool router**

In `tool_backend.py`, replace `_MAX_HISTORY_MESSAGES = 40` with the imports (alongside the other `src.internal.*` imports):

```python
from src.internal.cache.interface import get_cache_backend
from src.internal.memory.working import (
    MAX_HISTORY_MESSAGES,
    load_working_memory,
    schedule_compression,
)
```

Change the signature:

```python
def create_tool_router(
    store: AgenticSearchStore,
    *,
    search_url: str = "http://localhost:8000/retrieve",
    resolved,
    llm=None,
    memory_compression: bool = False,
) -> APIRouter:
```

Delete the `_history` helper. In `send_tool_message`, replace `history = _history(session_id)` and the following `store.add_chat_message(...)` with:

```python
        working = load_working_memory(
            store,
            session_id,
            keep_last=MAX_HISTORY_MESSAGES,
            cache=get_cache_backend() if memory_compression else None,
        )
        history = working.messages
        store.add_chat_message(session_id, role="user", content=body.message)
        # The answer comes from the local model, so the remote llm summarizing
        # now contends with nothing; one site covers both branches.
        schedule_compression(
            working, session_id=session_id, llm=llm, enabled=memory_compression
        )
```

Delete the `from src.context import ChatMessage` import if `ruff` flags it unused.

- [ ] **Step 4: Thread the settings through `_register_routers`**

In `app.py`, add `memory_compression: bool = False,` to the `_register_routers` signature after `memory_require_auth`, and change the two mounts:

```python
    app.include_router(
        create_chat_router(db, llm=llm, memory_compression=memory_compression)
    )
    ...
    app.include_router(
        create_tool_router(
            db,
            search_url=search_url,
            resolved=settings,
            llm=llm,
            memory_compression=memory_compression,
        )
    )
```

and at the call in `create_web_app`, add `memory_compression=settings.memory_compression,` after `memory_require_auth=settings.memory_require_auth,`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_tool_backend.py tests/unit/servers/web/test_web_experience_app.py tests/unit/test_chat_backend.py -v`
Expected: all pass.

- [ ] **Step 6: Mutation check**

In `tool_backend.py` set `cache=None`. Expected: `test_send_tool_flag_on_prepends_stored_summary` FAILS. Restore. In `_register_routers`, drop `memory_compression=memory_compression` from the chat mount. Expected: `test_register_routers_passes_memory_settings_to_chat_and_tool` FAILS. Restore. `git diff` clean.

- [ ] **Step 7: Lint, full suite, import guard, commit**

```bash
ruff check . --fix && ruff format .
pytest -q
python -c "import src.internal.servers.web.app, src.internal.servers.query_and_chat.chat_backend, src.internal.servers.query_and_chat.tool_backend, src.internal.memory.working"
git add src/internal/servers/query_and_chat/tool_backend.py src/internal/servers/web/app.py tests/unit/test_tool_backend.py tests/unit/servers/web/test_web_experience_app.py
git commit -m "feat(tools): working-memory compression on the Tool surface; thread the flag through router registration

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Done criteria

- `pytest -q` green; `ruff check .` clean.
- `grep -rn "_MAX_HISTORY_MESSAGES" src/` returns nothing; `grep -rn "MAX_HISTORY_MESSAGES = " src/` returns only `src/internal/memory/working.py`.
- With `AGENTIC_SEARCH_MEMORY_COMPRESSION` unset, `git stash`-free manual check: start the web backend, send 45 messages to one session on `/chat/send-chat-message`, confirm the 46th turn's history is 40 messages (add a temporary log line, then remove it).
- Push `feat/working-memory-compression` and open a PR titled "feat(memory): working-memory compression behind AGENTIC_SEARCH_MEMORY_COMPRESSION" whose body links the spec and this plan.
