# Memory Auto-Curation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the working-memory compression task summarizes a span of dropped turns, also curate that span into the user's long-term memories and advance the `curated_through` cursor, behind a default-off flag.

**Architecture:** `curate_from_conversation` accepts the conversation text directly; a new `curate_span` helper formats a list of message records the way whole sessions are formatted. `compress_session` takes an optional `curate` coroutine and runs it after the summary is saved. `schedule_compression` builds that coroutine from `store`, `user_id`, and `llm` when `auto_curate` is on and the session has an owner. The three surfaces pass their resolved user id.

**Tech Stack:** Python 3.11, FastAPI, SQLite via `AgenticSearchStore`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-11-memory-auto-curation-design.md`

## Global Constraints

- Flag name exactly `AGENTIC_SEARCH_MEMORY_AUTO_CURATE`; default off; read via `_flag(...)` in `SearchExperienceSettings.from_app_settings`; effective only when `memory_compression` is also on (enforced by `schedule_compression`'s existing early return, not by a second check).
- `user_id is None` (anonymous session) means no curation, regardless of `AGENTIC_SEARCH_MEMORY_REQUIRE_AUTH`.
- Curation runs only after the summary's `save_state` succeeded, over the post-trim `pending`; on `False` or an exception it logs at warning, leaves `curated_through` unchanged, does not retry, and `compress_session` still returns `True`.
- `curate_span` formats `f"{r.role.upper()}: {r.content}"` lines joined by `\n`, keeps the last `MEMORY_GATHER_CHAR_BUDGET` characters, returns `True` only when `curate_from_conversation` reports `status == "ok"`, and returns `False` for an empty span without calling the LLM.
- No new dependency. No change inside `src/agents/`. No change to the curation prompt, `consolidate_memories`, or profiles.
- Never commit to `main`; branch `feat/memory-auto-curation`; never bare `git stash`/`git stash pop`.
- `ruff check . --fix && ruff format .` before each commit; the pre-commit hook aborts if ruff-format reformats; re-add and re-commit.
- Every new test gets a mutation check (remove the behavior, watch it go red, restore, `git diff` clean before committing).
- Every surface test that turns `memory_compression` on keeps its `schedule_compression` stub (a developer's `OPENAI_API_KEY` would otherwise make `create_web_app` build a real client and the scheduler make a live call).

---

### Task 1: `curate_from_conversation(conversation=...)` and `curate_span`

**Files:**
- Modify: `src/internal/memory/service.py` (`curate_from_conversation` at ~line 248; add `curate_span` after it)
- Modify: `tests/unit/memory/test_curation.py`

**Interfaces:**
- Consumes: `_gather_sources(store, user_id, session_id) -> str`, `MEMORY_GATHER_CHAR_BUDGET = 12000`, `ChatMessageRecord` (`src/internal/db/models.py:87`), the file's existing `_FakeLLM(turns)`, `_tool_chunk(index, call_id, name, arguments)`, `_text_chunk(text)` helpers.
- Produces:
  - `curate_from_conversation(store, user_id, llm, session_id=None, max_turns=MAX_CURATION_TURNS, *, conversation: str | None = None) -> dict`
  - `async curate_span(store, user_id: str, llm, session_id: str, records: list[ChatMessageRecord]) -> bool`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/memory/test_curation.py`:

```python
class _NoSessionsStore(AgenticSearchStore):
    """Raises if curation tries to read sessions: the caller supplied the text."""

    def list_sessions_for_user(self, *a, **k):  # pragma: no cover - guard
        raise AssertionError("curation read sessions despite conversation=")

    def get_chat_session(self, *a, **k):  # pragma: no cover - guard
        raise AssertionError("curation read a session despite conversation=")


def test_curate_from_conversation_uses_supplied_text_and_skips_the_store():
    store = _NoSessionsStore(":memory:")
    store.upsert_user(UserRecord(id="u1"))
    prompts: list = []

    class _RecordingLLM(_FakeLLM):
        def stream(self, prompt, **kw):
            prompts.append(prompt)
            return super().stream(prompt, **kw)

    llm = _RecordingLLM(turns=[[_text_chunk("STOP")]])
    result = asyncio.run(
        service.curate_from_conversation(
            store, "u1", llm, session_id="sess-x", conversation="USER: I like tea."
        )
    )
    assert result["status"] == "ok"
    assert "USER: I like tea." in prompts[0][1]["content"]
    trajectories = store.list_memory_trajectories("u1")
    assert trajectories[-1].session_id == "sess-x"


def test_curate_span_formats_records_and_reports_ok():
    store = AgenticSearchStore(":memory:")
    store.upsert_user(UserRecord(id="u1"))
    session = store.create_chat_session(user_id="u1")
    r1 = store.add_chat_message(session.id, role="user", content="I moved to Lyon.")
    r2 = store.add_chat_message(session.id, role="assistant", content="Nice.")
    prompts: list = []

    class _RecordingLLM(_FakeLLM):
        def stream(self, prompt, **kw):
            prompts.append(prompt)
            return super().stream(prompt, **kw)

    llm = _RecordingLLM(
        turns=[
            [_tool_chunk(0, "c1", "add_memory", '{"content": "User lives in Lyon"}')],
            [_text_chunk("STOP")],
        ]
    )
    ok = asyncio.run(service.curate_span(store, "u1", llm, session.id, [r1, r2]))
    assert ok is True
    assert "USER: I moved to Lyon.\nASSISTANT: Nice." in prompts[0][1]["content"]
    assert [r.memory_text for r in store.get_user_memory_records("u1")] == [
        "User lives in Lyon"
    ]


def test_curate_span_empty_records_is_false_without_llm():
    store = AgenticSearchStore(":memory:")
    store.upsert_user(UserRecord(id="u1"))

    class _Boom:
        def stream(self, *a, **k):
            raise AssertionError("LLM must not be called for an empty span")

    assert asyncio.run(service.curate_span(store, "u1", _Boom(), "s", [])) is False
```

`list_memory_trajectories` is defined at `src/internal/db/store.py:2163`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/memory/test_curation.py -v -k "supplied_text or curate_span"`
Expected: FAIL with `TypeError: curate_from_conversation() got an unexpected keyword argument 'conversation'` and `AttributeError: module ... has no attribute 'curate_span'`.

- [ ] **Step 3: Implement**

In `src/internal/memory/service.py`, change the signature and first line of `curate_from_conversation`:

```python
async def curate_from_conversation(
    store,
    user_id: str,
    llm,
    session_id: str | None = None,
    max_turns: int = MAX_CURATION_TURNS,
    *,
    conversation: str | None = None,
) -> dict[str, Any]:
    # A caller that already holds the text (the compression task, which
    # curates exactly the span it summarized) hands it over; everyone else
    # reads the user's sessions as before.
    sources = (
        conversation
        if conversation is not None
        else _gather_sources(store, user_id, session_id)
    )
```

Add after `curate_from_conversation` (add `from src.internal.db.models import ChatMessageRecord` to the existing models import):

```python
def _format_span(records: list[ChatMessageRecord]) -> str:
    text = "\n".join(f"{r.role.upper()}: {r.content}" for r in records)
    return text[-MEMORY_GATHER_CHAR_BUDGET:]


async def curate_span(
    store, user_id: str, llm, session_id: str, records: list[ChatMessageRecord]
) -> bool:
    """Curate one span of a session into *user_id*'s memories.

    The compression task calls this over the turns it just summarized. True
    means curation ran; False means there was nothing to curate or it did
    not complete. Formats the span exactly as ``_gather_sources`` formats
    whole sessions so the curation prompt sees the same shape either way.
    """
    if not records:
        return False
    result = await curate_from_conversation(
        store, user_id, llm, session_id=session_id, conversation=_format_span(records)
    )
    return result.get("status") == "ok"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/memory/test_curation.py -v`
Expected: all pass, including every pre-existing test.

- [ ] **Step 5: Mutation check**

Change `sources = (conversation if conversation is not None else ...)` to always call `_gather_sources`. Expected: `test_curate_from_conversation_uses_supplied_text_and_skips_the_store` FAILS (AssertionError from the store guard). Restore. In `curate_span`, remove the `if not records: return False`. Expected: `test_curate_span_empty_records_is_false_without_llm` FAILS. Restore. `git diff` clean.

- [ ] **Step 6: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/memory/service.py tests/unit/memory/test_curation.py
git commit -m "feat(memory): curate a supplied span instead of whole sessions

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Curation inside `compress_session` and `schedule_compression`

**Files:**
- Modify: `src/internal/memory/working.py`
- Modify: `tests/unit/memory/test_working_memory.py`

**Interfaces:**
- Consumes: Task 1's `curate_span`; the existing `compress_session`, `schedule_compression`, `save_state`, `load_state`, `SessionMemoryState`; the test file's `store`, `cache` fixtures, `_seed(store, n)`, `FakeLLM`, `_wm(pending)`, and the autouse fixture clearing `_inflight`/`_tasks`.
- Produces:
  - `CurateFn = Callable[[list[ChatMessageRecord]], Awaitable[bool]]`
  - `compress_session(session_id, llm, *, pending, cache=None, curate: CurateFn | None = None) -> bool`
  - `schedule_compression(wm, *, session_id, llm, enabled, cache=None, store=None, user_id: str | None = None, auto_curate: bool = False) -> asyncio.Task | None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/memory/test_working_memory.py`:

```python
def _curator(result=True, *, raise_exc=False):
    calls: list = []

    async def curate(records):
        calls.append(list(records))
        if raise_exc:
            raise RuntimeError("curation down")
        return result

    curate.calls = calls
    return curate


def test_compress_curates_the_summarized_span_and_advances_cursor(store, cache):
    sid, records = _seed(store, 12)
    curate = _curator(True)
    ok = asyncio.run(
        compress_session(sid, FakeLLM("S"), pending=records[:2], cache=cache, curate=curate)
    )
    assert ok is True
    state = load_state(cache, sid)
    assert state.summary == "S"
    assert state.summarized_through == records[1].id
    assert state.curated_through == records[1].id
    assert [r.id for r in curate.calls[0]] == [records[0].id, records[1].id]


def test_compress_curates_only_the_post_trim_span(store, cache):
    sid, records = _seed(store, 12)
    save_state(cache, sid, SessionMemoryState(summary="old", summarized_through=records[1].id))
    curate = _curator(True)
    asyncio.run(
        compress_session(sid, FakeLLM("S"), pending=records[:4], cache=cache, curate=curate)
    )
    assert [r.id for r in curate.calls[0]] == [records[2].id, records[3].id]
    assert load_state(cache, sid).curated_through == records[3].id


def test_compress_curate_false_keeps_summary_and_cursor(store, cache):
    sid, records = _seed(store, 12)
    save_state(cache, sid, SessionMemoryState(curated_through="keep-me"))
    ok = asyncio.run(
        compress_session(sid, FakeLLM("S"), pending=records[:2], cache=cache, curate=_curator(False))
    )
    assert ok is True
    state = load_state(cache, sid)
    assert state.summary == "S"
    assert state.summarized_through == records[1].id
    assert state.curated_through == "keep-me"


def test_compress_curate_raising_keeps_summary_and_releases(store, cache):
    sid, records = _seed(store, 12)
    ok = asyncio.run(
        compress_session(
            sid, FakeLLM("S"), pending=records[:2], cache=cache, curate=_curator(raise_exc=True)
        )
    )
    assert ok is True
    state = load_state(cache, sid)
    assert state.summary == "S"
    assert state.curated_through is None
    # Lock and in-flight entry were released: a later span compresses fine.
    ok2 = asyncio.run(
        compress_session(sid, FakeLLM("S2"), pending=records[:4], cache=cache, curate=_curator(True))
    )
    assert ok2 is True
    assert load_state(cache, sid).curated_through == records[3].id


def test_compress_without_curate_preserves_curated_through(store, cache):
    sid, records = _seed(store, 12)
    save_state(cache, sid, SessionMemoryState(curated_through="keep-me"))
    asyncio.run(compress_session(sid, FakeLLM("S"), pending=records[:2], cache=cache))
    assert load_state(cache, sid).curated_through == "keep-me"


def test_compress_does_not_curate_when_summary_fails(store, cache):
    sid, records = _seed(store, 12)
    curate = _curator(True)
    asyncio.run(
        compress_session(sid, FakeLLM(fail=True), pending=records[:2], cache=cache, curate=curate)
    )
    assert curate.calls == []
    assert load_state(cache, sid) == SessionMemoryState()


def test_schedule_curates_only_with_user_store_and_flag(store, cache, monkeypatch):
    sid, records = _seed(store, 12)
    seen: list = []

    async def fake_curate_span(store_, user_id, llm, session_id, records_):
        seen.append((user_id, session_id, [r.id for r in records_]))
        return True

    monkeypatch.setattr("src.internal.memory.working.curate_span", fake_curate_span)

    async def run(**kw):
        task = schedule_compression(
            _wm(records[:2]), session_id=sid, llm=FakeLLM("S"), enabled=True, cache=cache, **kw
        )
        assert task is not None
        await task

    asyncio.run(run(store=store, user_id="u1", auto_curate=True))
    assert seen == [("u1", sid, [records[0].id, records[1].id])]
    assert load_state(cache, sid).curated_through == records[1].id

    seen.clear()
    save_state(cache, sid, SessionMemoryState())
    asyncio.run(run(store=store, user_id=None, auto_curate=True))
    assert seen == []
    assert load_state(cache, sid).curated_through is None

    save_state(cache, sid, SessionMemoryState())
    asyncio.run(run(store=store, user_id="u1", auto_curate=False))
    assert seen == []

    save_state(cache, sid, SessionMemoryState())
    asyncio.run(run(store=None, user_id="u1", auto_curate=True))
    assert seen == []


def test_schedule_disabled_never_curates_even_with_auto_curate(store, cache):
    sid, records = _seed(store, 12)

    async def run():
        return schedule_compression(
            _wm(records[:2]), session_id=sid, llm=FakeLLM(), enabled=False, cache=cache,
            store=store, user_id="u1", auto_curate=True,
        )

    assert asyncio.run(run()) is None
    assert load_state(cache, sid) == SessionMemoryState()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/memory/test_working_memory.py -v -k "curat"`
Expected: FAIL with `TypeError: compress_session() got an unexpected keyword argument 'curate'` and the same for `schedule_compression`'s `store`.

- [ ] **Step 3: Implement**

In `src/internal/memory/working.py`, add imports `import functools`, `from collections.abc import Awaitable, Callable`, and `from src.internal.memory.service import curate_span`. Add near the other module-level names:

```python
# Curates one span of records into a user's long-term memories. Built by
# ``schedule_compression`` from the store, user and llm; ``compress_session``
# only ever sees the callable.
CurateFn = Callable[[list[ChatMessageRecord]], Awaitable[bool]]
```

Change `compress_session`'s signature to add `curate: CurateFn | None = None` after `cache`, and replace the block from `save_state(` through `return True` with:

```python
        save_state(
            cache,
            session_id,
            SessionMemoryState(
                summary=text,
                summarized_through=last_id,
                curated_through=state.curated_through,
            ),
        )
        if curate is not None:
            await _curate_after_summary(cache, session_id, curate, pending, text, last_id)
        return True
```

Add the helper before `compress_session`:

```python
async def _curate_after_summary(
    cache: CacheBackend,
    session_id: str,
    curate: CurateFn,
    pending: list[ChatMessageRecord],
    summary: str,
    last_id: str,
) -> None:
    """Best-effort: curate the span the summary just covered, then move the
    curated cursor. A failure is logged and the span is not retried -- the
    summary is the record, long-term memory is a bonus, and a six-turn
    tool-calling loop is not worth re-running on every transient error."""
    try:
        curated = await curate(pending)
    except Exception as exc:  # noqa: BLE001 - curation is best-effort
        logger.warning("memory auto-curation failed for %s: %s", session_id, exc)
        return
    if not curated:
        return
    save_state(
        cache,
        session_id,
        SessionMemoryState(
            summary=summary, summarized_through=last_id, curated_through=last_id
        ),
    )
```

Update the docstring of `compress_session` to add: "When ``curate`` is given it runs after the summary is saved, over the same span; see ``_curate_after_summary``."

Change `schedule_compression`:

```python
def schedule_compression(
    wm: WorkingMemory,
    *,
    session_id: str,
    llm,
    enabled: bool,
    cache: CacheBackend | None = None,
    store=None,
    user_id: str | None = None,
    auto_curate: bool = False,
) -> asyncio.Task | None:
    """...existing docstring...

    With ``auto_curate`` on, a ``store`` and a ``user_id``, the same task also
    curates the summarized span into that user's memories. An anonymous
    session (``user_id`` is None) is never curated: auto-curation is silent,
    and silently pooling anonymous transcripts into a shared bucket is the
    leak ``AGENTIC_SEARCH_MEMORY_REQUIRE_AUTH`` exists to prevent.
    """
    if not enabled or llm is None or not wm.pending:
        return None
    curate: CurateFn | None = None
    if auto_curate and store is not None and user_id is not None:
        curate = functools.partial(curate_span, store, user_id, llm, session_id)
    task = asyncio.create_task(
        compress_session(session_id, llm, pending=wm.pending, cache=cache, curate=curate)
    )
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/memory/test_working_memory.py tests/unit/memory/test_curation.py -v`
Expected: all pass.

- [ ] **Step 5: Mutation check**

In `_curate_after_summary`, delete the final `save_state(...)` call. Expected: `test_compress_curates_the_summarized_span_and_advances_cursor` and `test_compress_curates_only_the_post_trim_span` FAIL. Restore. In `schedule_compression`, drop `and user_id is not None`. Expected: `test_schedule_curates_only_with_user_store_and_flag` FAILS on the anonymous case (the fake receives `None`). Restore. `git diff` clean.

- [ ] **Step 6: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/memory/working.py tests/unit/memory/test_working_memory.py
git commit -m "feat(memory): curate the summarized span and advance curated_through

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Flag, surfaces, router plumbing, docs

**Files:**
- Modify: `src/internal/servers/web/app.py` (settings dataclass; `_register_routers` signature, both mounts, and its call in `create_web_app`; the `schedule_compression` call at the end of `_run_agent_impl`'s `finally`)
- Modify: `src/internal/servers/query_and_chat/chat_backend.py` (`create_chat_router` signature; the `schedule_compression` call)
- Modify: `src/internal/servers/query_and_chat/tool_backend.py` (`create_tool_router` signature; the `schedule_compression` call)
- Modify: `docs/configuration.md` (row after `AGENTIC_SEARCH_MEMORY_COMPRESSION`)
- Modify: `tests/unit/servers/web/test_web_experience_app.py`, `tests/unit/test_chat_backend.py`, `tests/unit/test_tool_backend.py`

**Interfaces:**
- Consumes: Task 2's `schedule_compression(..., store=, user_id=, auto_curate=)`; in `_run_agent_impl` the locals `db` (closure), `user_id` (from `capabilities.user_id`), `settings`; in `chat_backend.send_chat_message` the local `user_id`; in `tool_backend.send_tool_message` `capabilities.user_id` and `store`.
- Produces: `SearchExperienceSettings.memory_auto_curate: bool = False`; `create_chat_router(store, *, llm=None, memory_compression=False, memory_auto_curate=False)`; `create_tool_router(..., llm=None, memory_compression=False, memory_auto_curate=False)`; `_register_routers(..., memory_auto_curate: bool = False)`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/servers/web/test_web_experience_app.py`:

- Extend `test_memory_compression_flag_defaults_off_and_reads_env` (or add a sibling `test_memory_auto_curate_flag_defaults_off_and_reads_env`) asserting `memory_auto_curate` is `False` with the env unset and `True` with `AGENTIC_SEARCH_MEMORY_AUTO_CURATE=true`.
- In `test_run_agent_schedules_compression_after_reply`, change the recorder to `scheduled.append((len(wm.pending), kw["enabled"], kw["llm"], kw["store"], kw["user_id"], kw["auto_curate"]))`, build the app with `SearchExperienceSettings(..., memory_compression=True, memory_auto_curate=True)`, and assert `scheduled == [(5, True, sentinel_llm, store, None, True)]` (an unauthenticated test request has `user_id None`; the point is the kwargs are threaded and the store is the app's store).
- In `test_register_routers_passes_memory_settings_to_chat_and_tool`, extend both fakes with `memory_auto_curate=False` in their signatures, record it, build with `memory_auto_curate=True`, and assert `seen == {"chat": (sentinel, True, True), "tool": (sentinel, True, True)}`.

In `tests/unit/test_chat_backend.py`, `test_send_chat_schedules_compression`: extend the recorder the same way; build with `_client_with(store, monkeypatch, llm=sentinel, memory_compression=True, memory_auto_curate=True)` (add the kwarg to `_client_with` and pass it through); assert `scheduled == [(5, True, sentinel, store, _USER_ID, True)]`.

In `tests/unit/test_tool_backend.py`, `test_send_tool_schedules_compression`: extend `_make_app_with_memory` with `memory_auto_curate=False` passed through; extend the recorder; build with `memory_auto_curate=True`; the tool tests run unauthenticated so assert `scheduled == [(5, True, sentinel, app.state._store, None, True)]`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/servers/web/test_web_experience_app.py tests/unit/test_chat_backend.py tests/unit/test_tool_backend.py -v -k "schedules_compression or register_routers or auto_curate"`
Expected: FAIL with `TypeError ... unexpected keyword argument 'memory_auto_curate'` and `KeyError: 'store'`.

- [ ] **Step 3: Implement**

`app.py`: after the `memory_compression: bool = False` field add

```python
    # On each compression event, also curate the summarized turns into the
    # signed-in user's long-term memories. Off by default; inert unless
    # memory_compression is on; anonymous sessions are never curated.
    memory_auto_curate: bool = False
```

and in `from_app_settings` after `memory_compression=...`: `memory_auto_curate=_flag("AGENTIC_SEARCH_MEMORY_AUTO_CURATE"),`. Add `memory_auto_curate: bool = False,` to `_register_routers` after `memory_compression`; pass `memory_auto_curate=memory_auto_curate` to both `create_chat_router(...)` and `create_tool_router(...)`; at the call in `create_web_app` add `memory_auto_curate=settings.memory_auto_curate,`. In `_run_agent_impl`'s `finally`, the call becomes:

```python
            schedule_compression(
                working,
                session_id=session_id,
                llm=llm,
                enabled=settings.memory_compression,
                store=db,
                user_id=user_id,
                auto_curate=settings.memory_auto_curate,
            )
```

`chat_backend.py`: `create_chat_router(store, *, llm=None, memory_compression: bool = False, memory_auto_curate: bool = False)`; docstring gains one sentence: "``memory_auto_curate`` additionally curates each summarized span into the signed-in user's memories; anonymous sessions are never curated." The call becomes:

```python
        schedule_compression(
            working,
            session_id=session_id,
            llm=llm,
            enabled=memory_compression,
            store=store,
            user_id=user_id,
            auto_curate=memory_auto_curate,
        )
```

`tool_backend.py`: `create_tool_router(store, *, search_url=..., resolved, llm=None, memory_compression: bool = False, memory_auto_curate: bool = False)`; the call becomes the same shape with `store=store, user_id=capabilities.user_id, auto_curate=memory_auto_curate`.

`docs/configuration.md`, directly after the `AGENTIC_SEARCH_MEMORY_COMPRESSION` row:

```markdown
| `AGENTIC_SEARCH_MEMORY_AUTO_CURATE` | `false` | On each compression event, also curate the summarized turns into the signed-in user's long-term memories (the same `curate` loop the CLI and `/api/memory/curate` run, scoped to that span). Does nothing unless `AGENTIC_SEARCH_MEMORY_COMPRESSION` is on. Anonymous sessions are never curated. Each overflow costs one curation loop (up to 6 tool-calling turns) against the configured remote LLM client |
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/servers/web/test_web_experience_app.py tests/unit/test_chat_backend.py tests/unit/test_tool_backend.py tests/unit/memory -v`
Expected: all pass.

- [ ] **Step 5: Mutation check**

Drop `auto_curate=memory_auto_curate` from the chat call. Expected: `test_send_chat_schedules_compression` FAILS. Restore. Drop `memory_auto_curate=memory_auto_curate` from the tool mount in `_register_routers`. Expected: `test_register_routers_passes_memory_settings_to_chat_and_tool` FAILS. Restore. `git diff` clean.

- [ ] **Step 6: Lint, full suite, import guard, commit**

```bash
ruff check . --fix && ruff format .
pytest -q
python -c "import src.internal.servers.web.app, src.internal.servers.query_and_chat.chat_backend, src.internal.servers.query_and_chat.tool_backend, src.internal.memory.working"
git add src/internal/servers/web/app.py src/internal/servers/query_and_chat/chat_backend.py src/internal/servers/query_and_chat/tool_backend.py docs/configuration.md tests/unit/servers/web/test_web_experience_app.py tests/unit/test_chat_backend.py tests/unit/test_tool_backend.py
git commit -m "feat(memory): AGENTIC_SEARCH_MEMORY_AUTO_CURATE wires curation into all three surfaces

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Done criteria

- `pytest -q` green; `ruff check .` clean.
- `grep -n "curated_through" src/internal/memory/working.py` shows the cursor written in `_curate_after_summary`.
- Push `feat/memory-auto-curation`; open a PR titled "feat(memory): auto-curate summarized turns into long-term memory behind AGENTIC_SEARCH_MEMORY_AUTO_CURATE" linking the spec and this plan.
