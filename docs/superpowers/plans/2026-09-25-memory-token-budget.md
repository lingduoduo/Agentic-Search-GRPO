# Working Memory Token Budget — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The session tail every surface hands its loop is cut by an estimated token budget (default 2500), summarization defaults on for `/api/agent`, the summarizer's input is bounded per call, and the dead token-ratio config is gone.

**Architecture:** `load_working_memory` gains `token_budget`: the message-count slice is taken first, then trimmed from the oldest end until its `ceil(len/4) + 4` estimate fits (newest record always kept). `compress_session` summarizes only the oldest ≤3000-token chunk of `pending` and advances the cursor to that chunk's last record. `SearchExperienceSettings` reads a tri-state `AGENTIC_SEARCH_MEMORY_COMPRESSION` into two fields plus `AGENTIC_SEARCH_MEMORY_HISTORY_TOKENS`, and every surface passes the budget.

**Tech Stack:** Python 3.10+ (CI 3.12), FastAPI, pytest (`asyncio.run` style in the memory tests).

**Spec:** `docs/superpowers/specs/2026-09-25-memory-token-budget-design.md`

## Global Constraints

- `estimate_tokens(text) = ceil(len(text) / 4)`, plus a per-message overhead of 4. Module-level, no new dependency.
- The newest record is always kept, even if it alone exceeds the budget. `keep_last` stays a cap.
- `token_budget=None` keeps today's behavior exactly (pure message count).
- `AGENTIC_SEARCH_MEMORY_HISTORY_TOKENS`, default **2500**; non-positive or non-integer raises `ValueError` at settings load.
- `AGENTIC_SEARCH_MEMORY_COMPRESSION` tri-state: unset → on for `/api/agent` only; truthy (`1/true/yes`) → every surface; falsy (`0/false/no`) → off everywhere.
- `SearchExperienceSettings.memory_compression` (`/api/agent`) and `memory_compression_direct` (`/chat`, `/tool`); the direct routers receive the latter.
- `_SUMMARY_INPUT_TOKENS = 3000`; always at least one record; an oversized single record is clipped with a marker in the prompt only; `last_id` = last included record; curation covers the same bounded span.
- `AGENTIC_SEARCH_MEMORY_AUTO_CURATE` unchanged (default off).
- Branch `feat/memory-token-budget`; never commit to `main`; commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Deviations from the spec (recorded)

1. **Unrecognized `AGENTIC_SEARCH_MEMORY_COMPRESSION` values** (e.g. `on`, `maybe`): the spec names only unset/truthy/falsy. This plan treats any non-empty value outside `1/true/yes` as off, matching the `_flag` helper every other flag in `SearchExperienceSettings` uses. Pinned by a test.
2. **Dataclass defaults follow the production default.** `SearchExperienceSettings()` (constructed directly, as most web tests do) now has `memory_compression=True`, `memory_compression_direct=False`, `memory_history_tokens=2500`, so direct construction and `from_app_settings()` with the env unset agree. Consequence: `test_run_agent_flag_off_ignores_stored_summary` now passes `memory_compression=False` explicitly (deliberate update).
3. **Router-level defaults stay opt-in.** `create_chat_router` / `create_tool_router` get `memory_history_tokens: int | None = None` (None = today's pure message count) so code that builds a router directly is unchanged; the web app always passes the settings value. `_register_routers`' `memory_compression` parameter is renamed `memory_compression_direct`, since that is the only thing it now carries.
4. **Dead config grep result:** `COMPRESSION_TRIGGER_RATIO` is read nowhere (only its definition in `chat_configs.py`, its entry in `default_config.py`'s `DEFAULT_CONFIG`, and `src/internal/chat/COMPRESSION.md`); no test or doc outside `docs/superpowers/` references it. Deleting is safe.

## Tests that pinned default-off (deliberately updated)

- `tests/unit/servers/web/test_web_experience_app.py::test_memory_compression_flag_defaults_off_and_reads_env` → replaced by the tri-state test.
- `tests/unit/servers/web/test_web_experience_app.py::test_run_agent_flag_off_ignores_stored_summary` → passes `memory_compression=False`.
- `tests/unit/servers/web/test_web_experience_app.py::test_register_routers_passes_memory_settings_to_chat_and_tool` → asserts the routers receive `memory_compression_direct` and `memory_history_tokens`.

## Review Focus

1. **The newest message is one huge paste** (larger than the whole budget) — it must still be sent, never an empty history — test in Task 1.
2. **An empty session under a budget** — returns no messages and no pending, no crash — test in Task 1.
3. **A stored summary cursor that now sits inside the budgeted tail** (the budget grew, or old turns were short) — the summary must be ignored so turns are not duplicated — test in Task 1.
4. **An operator writes `AGENTIC_SEARCH_MEMORY_COMPRESSION=on`** — behaves like every other flag in this class (off), and the doc says which values turn it on — test in Task 3.
5. **The summarizer fails on a bounded chunk** — nothing is written and the next call retries the same chunk (existing `test_compress_llm_failure_leaves_state_and_allows_retry` covers the write/retry path; the chunking adds no new failure branch).

---

### Task 1: Token-budgeted tail in `load_working_memory`

**Files:**
- Modify: `src/internal/memory/working.py` (imports, constants, new `estimate_tokens`/`_record_tokens`/`_fit_budget`, `load_working_memory`)
- Test: `tests/unit/memory/test_working_memory.py`

**Interfaces:**
- Produces: `estimate_tokens(text: str) -> int`; `DEFAULT_HISTORY_TOKENS = 2500`; `load_working_memory(store, session_id, *, keep_last=MAX_HISTORY_MESSAGES, token_budget: int | None = None, cache=None) -> WorkingMemory`; private `_record_tokens(record) -> int` (used by Task 2).

- [x] **Step 1: Write the failing tests** (append after `test_default_keep_last_is_forty`; add `estimate_tokens` to the import block)

```python
def _seed_sized(store: AgenticSearchStore, sizes: list[int]) -> tuple[str, list]:
    """Records whose content is exactly ``sizes[i]`` chars and unique per index."""
    session = store.create_chat_session(title="s")
    records = [
        store.add_chat_message(
            session.id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"{i}:".ljust(chars, "x"),
        )
        for i, chars in enumerate(sizes)
    ]
    return session.id, records


def test_estimate_tokens_is_ceil_chars_over_four():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


# 400 chars = 100 tokens + 4 overhead = 104 per record.
@pytest.mark.parametrize("budget,kept", [(312, 3), (311, 2)])
def test_token_budget_keeps_newest_records_that_fit(store, cache, budget, kept):
    sid, records = _seed_sized(store, [400] * 10)
    wm = load_working_memory(store, sid, token_budget=budget, cache=cache)
    assert [m.content for m in wm.messages] == [r.content for r in records[-kept:]]
    assert [r.id for r in wm.pending] == [r.id for r in records[:-kept]]


def test_newest_record_is_kept_even_when_over_budget(store, cache):
    sid, records = _seed_sized(store, [40, 40, 4000])
    wm = load_working_memory(store, sid, token_budget=100, cache=cache)
    assert [m.content for m in wm.messages] == [records[-1].content]
    assert [r.id for r in wm.pending] == [r.id for r in records[:2]]


def test_keep_last_still_caps_under_a_generous_budget(store, cache):
    sid, records = _seed(store, 12)
    wm = load_working_memory(store, sid, keep_last=10, token_budget=10_000, cache=cache)
    assert [m.content for m in wm.messages] == [r.content for r in records[2:]]
    assert [r.id for r in wm.pending] == [r.id for r in records[:2]]


@pytest.mark.parametrize("n,keep_last", [(0, 10), (5, 10), (12, 10), (45, 40)])
def test_no_budget_is_the_plain_message_count(store, cache, n, keep_last):
    sid, records = _seed_sized(store, [4000] * n)
    wm = load_working_memory(store, sid, keep_last=keep_last, cache=cache)
    assert [m.content for m in wm.messages] == [
        r.content for r in records[-keep_last:]
    ]
    assert [r.id for r in wm.pending] == [r.id for r in records[:-keep_last]]


def test_budget_applies_without_a_cache_and_never_pends(store):
    sid, records = _seed_sized(store, [400] * 10)
    wm = load_working_memory(store, sid, token_budget=312, cache=None)
    assert [m.content for m in wm.messages] == [r.content for r in records[-3:]]
    assert wm.pending == []


def test_empty_session_under_a_budget(store, cache):
    sid, _ = _seed_sized(store, [])
    wm = load_working_memory(store, sid, token_budget=100, cache=cache)
    assert wm == WorkingMemory(messages=[], summary="", pending=[])


def test_budget_dropped_prefix_with_cursor_gives_summary_and_pending(store, cache):
    sid, records = _seed_sized(store, [400] * 10)
    save_state(
        cache, sid, SessionMemoryState(summary="S", summarized_through=records[2].id)
    )
    wm = load_working_memory(store, sid, token_budget=312, cache=cache)
    assert wm.messages[0].content == SUMMARY_PREFIX + "S"
    assert [m.content for m in wm.messages[1:]] == [r.content for r in records[7:]]
    assert [r.id for r in wm.pending] == [r.id for r in records[3:7]]


def test_cursor_inside_the_budgeted_tail_ignores_summary(store, cache):
    sid, records = _seed_sized(store, [400] * 10)
    save_state(
        cache, sid, SessionMemoryState(summary="dup", summarized_through=records[8].id)
    )
    wm = load_working_memory(store, sid, token_budget=312, cache=cache)
    assert wm.summary == ""
    assert [m.content for m in wm.messages] == [r.content for r in records[7:]]
```

- [x] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/memory/test_working_memory.py -q -p no:cacheprovider`
Expected: collection ERROR — `ImportError: cannot import name 'estimate_tokens'`.

- [x] **Step 3: Implement** in `working.py`

Add `import math`. After `MAX_HISTORY_MESSAGES = 40`:

```python
# Default for AGENTIC_SEARCH_MEMORY_HISTORY_TOKENS: leaves room in the local
# loops' 4096-token prompt for the system prompt, the summary and the new turn.
DEFAULT_HISTORY_TOKENS = 2500
_MESSAGE_OVERHEAD_TOKENS = 4
```

Before `load_working_memory`:

```python
def estimate_tokens(text: str) -> int:
    """A rough token count, four characters per token; no tokenizer needed."""
    return math.ceil(len(text) / 4)


def _record_tokens(record: ChatMessageRecord) -> int:
    return estimate_tokens(record.content) + _MESSAGE_OVERHEAD_TOKENS


def _fit_budget(
    records: list[ChatMessageRecord], token_budget: int
) -> list[ChatMessageRecord]:
    """The newest ``records`` whose estimates fit ``token_budget``. The newest
    one is always kept, even alone over budget: a turn with no history beats
    a turn that silently loses the message it is answering."""
    used = 0
    kept = 0
    for record in reversed(records):
        used += _record_tokens(record)
        if kept and used > token_budget:
            break
        kept += 1
    return records[len(records) - kept :]
```

In `load_working_memory`, add the `token_budget: int | None = None` keyword (between `keep_last` and `cache`), and replace the two slice lines with:

```python
    tail = records[-keep_last:]
    if token_budget is not None:
        tail = _fit_budget(tail, token_budget)
    dropped = records[: len(records) - len(tail)]
```

Docstring: "Return the newest messages -- at most ``keep_last``, and with ``token_budget`` set, only as many as fit its estimate (the newest always) -- prefixed by the stored summary when one covers the dropped prefix." Keep the `cache=None` paragraph and add "``token_budget=None`` is the pure message count."

- [x] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/memory/test_working_memory.py -q -p no:cacheprovider`
Expected: all pass.

- [x] **Step 5: Commit** — `git add src/internal/memory/working.py tests/unit/memory/test_working_memory.py && git commit -m "Working memory: token-budgeted session tail"`

---

### Task 2: Bounded summarizer input

**Files:**
- Modify: `src/internal/memory/working.py` (`_summary_prompt`, `compress_session`, new `_SUMMARY_INPUT_TOKENS`, `_CLIP_MARKER`, `_bounded_span`, `_clip`)
- Test: `tests/unit/memory/test_working_memory.py`

**Interfaces:**
- Consumes: `_record_tokens`, `_seed_sized` (Task 1).
- Produces: `_SUMMARY_INPUT_TOKENS = 3000`, `_CLIP_MARKER: str`.

- [x] **Step 1: Write the failing tests** (append in the compress_session section)

```python
# 4000 chars = 1000 tokens + 4 overhead: two records fit the 3000-token input.
def test_compress_summarizes_one_bounded_chunk_of_a_backlog(store, cache):
    sid, records = _seed_sized(store, [4000] * 22)
    wm = load_working_memory(store, sid, keep_last=2, cache=cache)
    assert len(wm.pending) == 20
    llm = FakeLLM()
    assert asyncio.run(compress_session(sid, llm, pending=wm.pending, cache=cache))
    turns = llm.prompts[0][-1]["content"]
    assert records[0].content in turns
    assert records[1].content in turns
    assert records[2].content not in turns
    assert load_state(cache, sid).summarized_through == records[1].id


def test_repeated_compression_drains_the_backlog(store, cache):
    sid, records = _seed_sized(store, [4000] * 22)
    llm = FakeLLM()
    for _ in range(10):
        wm = load_working_memory(store, sid, keep_last=2, cache=cache)
        assert asyncio.run(compress_session(sid, llm, pending=wm.pending, cache=cache))
    assert load_working_memory(store, sid, keep_last=2, cache=cache).pending == []
    assert load_state(cache, sid).summarized_through == records[19].id
    assert len(llm.prompts) == 10


def test_oversized_single_record_is_clipped_in_the_prompt_only(store, cache):
    sid, records = _seed_sized(store, [20_000, 40, 40])
    wm = load_working_memory(store, sid, keep_last=2, cache=cache)
    llm = FakeLLM()
    assert asyncio.run(compress_session(sid, llm, pending=wm.pending, cache=cache))
    turns = llm.prompts[0][-1]["content"]
    assert working._CLIP_MARKER in turns
    assert records[0].content[: working._SUMMARY_INPUT_TOKENS * 4] in turns
    assert records[0].content not in turns
    assert store.list_chat_messages(sid)[0].content == records[0].content
    assert load_state(cache, sid).summarized_through == records[0].id


def test_curation_covers_the_bounded_span(store, cache):
    sid, records = _seed_sized(store, [4000] * 6)
    seen: list = []

    async def curate(span):
        seen.append([r.id for r in span])
        return True

    asyncio.run(
        compress_session(
            sid, FakeLLM(), pending=records[:4], cache=cache, curate=curate
        )
    )
    assert seen == [[records[0].id, records[1].id]]
    assert load_state(cache, sid).curated_through == records[1].id
```

- [x] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/memory/test_working_memory.py -q -p no:cacheprovider -k "bounded or drains or clipped"`
Expected: FAIL — `records[2].content` is in the prompt / cursor at `records[19]` after one call / `AttributeError: _CLIP_MARKER`.

- [x] **Step 3: Implement**

After `_SUMMARY_MAX_TOKENS = 400`:

```python
# Estimated tokens of turns one summarizer call may see. A backlog larger than
# this (the first compression of a long session) drains one chunk per turn.
_SUMMARY_INPUT_TOKENS = 3000
_CLIP_MARKER = " [...truncated]"
```

Before `_summary_prompt`:

```python
def _bounded_span(pending: list[ChatMessageRecord]) -> list[ChatMessageRecord]:
    """The oldest records of ``pending`` that fit ``_SUMMARY_INPUT_TOKENS``,
    always at least one so the cursor can advance."""
    used = 0
    span: list[ChatMessageRecord] = []
    for record in pending:
        used += _record_tokens(record)
        if span and used > _SUMMARY_INPUT_TOKENS:
            break
        span.append(record)
    return span


def _clip(text: str) -> str:
    """Cap one record at the input budget, for the prompt only."""
    limit = _SUMMARY_INPUT_TOKENS * 4
    return text if len(text) <= limit else text[:limit] + _CLIP_MARKER
```

In `_summary_prompt`: `f"{r.role.upper()}: {_clip(r.content)}"`.

In `compress_session`, immediately before `last_id = pending[-1].id`:

```python
        pending = _bounded_span(pending)
```

Docstring of `compress_session`: "Summarize the oldest chunk of ``pending`` (up to ``_SUMMARY_INPUT_TOKENS``) into the session's stored summary; the rest drains on later turns."

- [x] **Step 4: Run** the whole file — all pass.

- [x] **Step 5: Commit** — `git commit -m "Working memory: bound the summarizer's input to one chunk per call"`

---

### Task 3: Settings and surfaces

**Files:**
- Modify: `src/internal/servers/web/app.py` (imports, `SearchExperienceSettings`, `_register_routers`, `create_web_app` call, `/api/agent` `load_working_memory` call)
- Modify: `src/internal/servers/query_and_chat/chat_backend.py`, `src/internal/servers/query_and_chat/tool_backend.py`
- Modify: `docs/configuration.md`, module docstring of `src/internal/memory/working.py`
- Test: `tests/unit/servers/web/test_web_experience_app.py`, `tests/unit/test_chat_backend.py`, `tests/unit/test_tool_backend.py`

**Interfaces:**
- Consumes: `DEFAULT_HISTORY_TOKENS`, `load_working_memory(..., token_budget=...)` (Task 1).
- Produces: `SearchExperienceSettings.memory_compression: bool = True`, `.memory_compression_direct: bool = False`, `.memory_history_tokens: int = 2500`; `create_chat_router(..., memory_history_tokens: int | None = None)`; `create_tool_router(..., memory_history_tokens: int | None = None)`; `_register_routers(..., memory_compression_direct: bool = False, memory_history_tokens: int | None = None, ...)`.

- [x] **Step 1: Write the failing tests**

`test_web_experience_app.py` — replace `test_memory_compression_flag_defaults_off_and_reads_env` with:

```python
@pytest.mark.parametrize(
    "value,agent,direct",
    [
        (None, True, False),
        ("", True, False),
        ("1", True, True),
        ("true", True, True),
        ("YES", True, True),
        ("0", False, False),
        ("false", False, False),
        ("no", False, False),
        ("on", False, False),
    ],
)
def test_memory_compression_is_tri_state(monkeypatch, value, agent, direct):
    if value is None:
        monkeypatch.delenv("AGENTIC_SEARCH_MEMORY_COMPRESSION", raising=False)
    else:
        monkeypatch.setenv("AGENTIC_SEARCH_MEMORY_COMPRESSION", value)
    settings = SearchExperienceSettings.from_app_settings()
    assert (settings.memory_compression, settings.memory_compression_direct) == (
        agent,
        direct,
    )


def test_memory_defaults_match_between_dataclass_and_env(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_MEMORY_COMPRESSION", raising=False)
    monkeypatch.delenv("AGENTIC_SEARCH_MEMORY_HISTORY_TOKENS", raising=False)
    loaded = SearchExperienceSettings.from_app_settings()
    default = SearchExperienceSettings()
    for field in (
        "memory_compression",
        "memory_compression_direct",
        "memory_history_tokens",
    ):
        assert getattr(loaded, field) == getattr(default, field)
    assert default.memory_history_tokens == 2500


def test_memory_history_tokens_reads_env(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_MEMORY_HISTORY_TOKENS", "800")
    assert SearchExperienceSettings.from_app_settings().memory_history_tokens == 800


@pytest.mark.parametrize("value", ["0", "-5", "abc", "1.5"])
def test_memory_history_tokens_rejects_bad_values(monkeypatch, value):
    monkeypatch.setenv("AGENTIC_SEARCH_MEMORY_HISTORY_TOKENS", value)
    with pytest.raises(ValueError, match="AGENTIC_SEARCH_MEMORY_HISTORY_TOKENS"):
        SearchExperienceSettings.from_app_settings()
```

`test_run_agent_flag_off_ignores_stored_summary`: construct with `SearchExperienceSettings(db_path=tmp_path / "state.sqlite3", memory_compression=False)`.

Replace `test_register_routers_passes_memory_settings_to_chat_and_tool` with:

```python
@pytest.mark.parametrize("agent,direct", [(True, False), (False, True)])
def test_register_routers_passes_direct_memory_settings_to_chat_and_tool(
    monkeypatch, tmp_path, agent, direct
):
    seen: dict = {}

    def fake_chat_router(
        store,
        *,
        llm=None,
        memory_compression=False,
        memory_auto_curate=False,
        memory_history_tokens=None,
    ):
        seen["chat"] = (llm, memory_compression, memory_auto_curate, memory_history_tokens)
        from fastapi import APIRouter

        return APIRouter()

    def fake_tool_router(
        store,
        *,
        search_url,
        resolved,
        llm=None,
        memory_compression=False,
        memory_auto_curate=False,
        memory_history_tokens=None,
    ):
        seen["tool"] = (llm, memory_compression, memory_auto_curate, memory_history_tokens)
        from fastapi import APIRouter

        return APIRouter()

    monkeypatch.setattr(
        "src.internal.servers.web.app.create_chat_router", fake_chat_router
    )
    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.tool_backend.create_tool_router",
        fake_tool_router,
    )
    sentinel = object()
    create_web_app(
        SearchExperienceSettings(
            db_path=tmp_path / "s.sqlite3",
            memory_compression=agent,
            memory_compression_direct=direct,
            memory_auto_curate=True,
            memory_history_tokens=777,
        ),
        llm=sentinel,
    )
    assert seen == {
        "chat": (sentinel, direct, True, 777),
        "tool": (sentinel, direct, True, 777),
    }
```

New `/api/agent` default-settings test:

```python
class _SummaryLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        return "S"


async def _await(task):
    return await task


def test_run_agent_default_settings_compress_past_the_token_budget(
    monkeypatch, tmp_path
):
    from src.internal.cache.interface import InMemoryCache
    from src.internal.memory.working import load_state
    import src.internal.servers.web.app as app_module

    cache = InMemoryCache()
    monkeypatch.setattr("src.internal.cache.interface._default_cache", cache)
    captured: list = []

    async def fake_answer(question, *, llm=None, chat_history=None, **kw):
        captured.append(list(chat_history or []))
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )
    scheduled: list = []
    real_schedule = app_module.schedule_compression

    def spy(wm, **kw):
        task = real_schedule(wm, **kw)
        scheduled.append((len(wm.pending), task))
        return task

    monkeypatch.setattr("src.internal.servers.web.app.schedule_compression", spy)

    store = AgenticSearchStore(tmp_path / "state.sqlite3")
    session = store.create_chat_session(title="long")
    # 4000 chars ~ 1004 tokens: the 2500 default keeps 2, the other 8 pend.
    records = [
        store.add_chat_message(
            session.id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"{i}:".ljust(4000, "x"),
        )
        for i in range(10)
    ]
    llm = _SummaryLLM()
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"),
        store=store,
        llm=llm,
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/agent",
            json={"query": "follow up", "mode": "chat_once", "session_id": session.id},
        )
        assert response.status_code == 200
        assert [p for p, _ in scheduled] == [8]
        task = scheduled[0][1]
        assert task is not None
        assert client.portal.call(_await, task) is True

    assert len(captured[0]) == 2
    # First bounded chunk: the two oldest 1004-token records.
    assert load_state(cache, session.id).summarized_through == records[1].id
```

`test_chat_backend.py` — add `memory_history_tokens=None` to `_client_with` and pass it through to `create_chat_router`; then:

```python
class _SummaryLLM:
    def complete(self, messages, **kwargs):
        return "S"


def _seed_sized(store, n, chars=4000):
    session = store.create_chat_session(user_id=_USER_ID, title="long")
    for i in range(n):
        store.add_chat_message(
            session.id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"{i}:".ljust(chars, "x"),
        )
    return session.id


def test_send_chat_passes_the_token_budget(store, monkeypatch):
    session_id = _seed_sized(store, 10)
    captured = _capture_plain_chat(monkeypatch)
    client = _client_with(store, monkeypatch, memory_history_tokens=2500)
    client.post(
        "/chat/send-chat-message",
        json={"message": "next", "session_id": session_id, "stream": False},
    )
    assert len(captured[0]) == 2


@pytest.mark.parametrize("compression", [False, True])
def test_send_chat_compresses_only_when_direct_compression_is_on(
    store, monkeypatch, compression
):
    from src.internal.cache.interface import InMemoryCache
    from src.internal.servers.query_and_chat import chat_backend

    monkeypatch.setattr("src.internal.cache.interface._default_cache", InMemoryCache())
    session_id = _seed_sized(store, 10)
    _capture_plain_chat(monkeypatch)
    tasks: list = []
    real = chat_backend.schedule_compression

    def spy(wm, **kw):
        tasks.append(real(wm, **kw))
        return tasks[-1]

    monkeypatch.setattr(chat_backend, "schedule_compression", spy)
    with _client_with(
        store,
        monkeypatch,
        llm=_SummaryLLM(),
        memory_compression=compression,
        memory_history_tokens=2500,
    ) as client:
        client.post(
            "/chat/send-chat-message",
            json={"message": "next", "session_id": session_id, "stream": False},
        )
        assert (tasks[0] is not None) is compression
        if compression:
            assert client.portal.call(_await, tasks[0]) is True


async def _await(task):
    return await task
```

`test_tool_backend.py` — add `memory_history_tokens=None` to `_make_app_with_memory` and pass it through; then (same shape, `store = app.state._store`, `_capture_tool_agent`, `tool_backend.schedule_compression`, URL `/tool/send-tool-message`):

```python
class _SummaryLLM:
    def complete(self, messages, **kwargs):
        return "S"


async def _await(task):
    return await task


def _seed_sized(store, n, chars=4000):
    session = store.create_chat_session(title="long")
    for i in range(n):
        store.add_chat_message(
            session.id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"{i}:".ljust(chars, "x"),
        )
    return session.id


def test_send_tool_passes_the_token_budget(monkeypatch):
    app = _make_app_with_memory(memory_history_tokens=2500)
    session_id = _seed_sized(app.state._store, 10)
    captured = _capture_tool_agent(monkeypatch)
    TestClient(app).post(
        "/tool/send-tool-message",
        json={"message": "next", "session_id": session_id, "stream": False},
    )
    assert len(captured[0]) == 2


@pytest.mark.parametrize("compression", [False, True])
def test_send_tool_compresses_only_when_direct_compression_is_on(
    monkeypatch, compression
):
    from src.internal.cache.interface import InMemoryCache
    from src.internal.servers.query_and_chat import tool_backend

    monkeypatch.setattr("src.internal.cache.interface._default_cache", InMemoryCache())
    app = _make_app_with_memory(
        llm=_SummaryLLM(), memory_compression=compression, memory_history_tokens=2500
    )
    session_id = _seed_sized(app.state._store, 10)
    _capture_tool_agent(monkeypatch)
    tasks: list = []
    real = tool_backend.schedule_compression

    def spy(wm, **kw):
        tasks.append(real(wm, **kw))
        return tasks[-1]

    monkeypatch.setattr(tool_backend, "schedule_compression", spy)
    with TestClient(app) as client:
        client.post(
            "/tool/send-tool-message",
            json={"message": "next", "session_id": session_id, "stream": False},
        )
        assert (tasks[0] is not None) is compression
        if compression:
            assert client.portal.call(_await, tasks[0]) is True
```

(The `False` case is what `/chat` and `/tool` receive under default settings, proven by the `_register_routers` test; the `True` case is `=1`.)

- [x] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/servers/web/test_web_experience_app.py tests/unit/test_chat_backend.py tests/unit/test_tool_backend.py -q -p no:cacheprovider -k "memory or budget or compress"`
Expected: FAIL — `TypeError: unexpected keyword 'memory_compression_direct'` / `'memory_history_tokens'`, and the tri-state defaults.

- [x] **Step 3: Implement**

`app.py`: `from src.internal.configs import get_env_int`; import `DEFAULT_HISTORY_TOKENS` from `working`. Fields:

```python
    # Summarize turns that fall off the history tail into a system message the
    # next turn sees, using the configured LLM client; inert without one.
    # AGENTIC_SEARCH_MEMORY_COMPRESSION unset: on for /api/agent only, which
    # already sends history to that llm; truthy: every surface; else: off.
    memory_compression: bool = True
    # The same for /chat and /tool, which answer with the local model: on,
    # their older turns reach the remote llm, so it needs the explicit flag.
    memory_compression_direct: bool = False
    # Estimated-token budget of the history tail every surface sends.
    memory_history_tokens: int = DEFAULT_HISTORY_TOKENS
```

In `from_app_settings`, before `return cls(`:

```python
        compression_set = bool(
            os.environ.get("AGENTIC_SEARCH_MEMORY_COMPRESSION", "").strip()
        )
        history_tokens = get_env_int(
            os.environ, "AGENTIC_SEARCH_MEMORY_HISTORY_TOKENS", DEFAULT_HISTORY_TOKENS
        )
        if history_tokens <= 0:
            raise ValueError(
                "AGENTIC_SEARCH_MEMORY_HISTORY_TOKENS must be a positive integer."
            )
```

and in `cls(...)`:

```python
            memory_compression=(
                not compression_set or _flag("AGENTIC_SEARCH_MEMORY_COMPRESSION")
            ),
            memory_compression_direct=_flag("AGENTIC_SEARCH_MEMORY_COMPRESSION"),
            memory_history_tokens=history_tokens,
```

`_register_routers`: rename `memory_compression` → `memory_compression_direct`, add `memory_history_tokens: int | None = None`; pass `memory_compression=memory_compression_direct, memory_history_tokens=memory_history_tokens` to both routers. Caller in `create_web_app`: `memory_compression_direct=settings.memory_compression_direct, memory_history_tokens=settings.memory_history_tokens`.

`/api/agent`: `load_working_memory(db, session_id, keep_last=MAX_HISTORY_MESSAGES, token_budget=settings.memory_history_tokens, cache=...)`.

`chat_backend.create_chat_router` / `tool_backend.create_tool_router`: add `memory_history_tokens: int | None = None`, pass `token_budget=memory_history_tokens` to `load_working_memory`. Chat docstring: one sentence, "``memory_history_tokens`` caps the history tail by estimated tokens (None: the message count alone)."

`docs/configuration.md`: rewrite the `AGENTIC_SEARCH_MEMORY_COMPRESSION` row (default `unset`, tri-state, which values count as on, the /chat + /tool data-flow warning); add `AGENTIC_SEARCH_MEMORY_HISTORY_TOKENS` row (default `2500`, `ceil(chars/4)+4` estimate, newest message always kept, must be a positive integer); `AUTO_CURATE` row: "Does nothing unless compression is on for that surface".

`working.py` module docstring: the tail is capped by message count and an estimated-token budget; the dropped turns are summarized a bounded chunk at a time; on by default for `/api/agent`.

- [x] **Step 4: Run** the three files plus `tests/unit/memory/ tests/unit/test_documented_env_vars.py tests/unit/test_configs.py tests/unit/servers/web/test_loop_runners.py` — all pass.

- [x] **Step 5: Commit** — `git commit -m "Working memory: history token budget on every surface, compression on by default for /api/agent"`

---

### Task 4: Remove the dead token-ratio config

**Files:**
- Modify: `src/internal/configs/chat_configs.py:15-19`, `src/internal/configs/default_config.py:70`
- Delete: `src/internal/chat/COMPRESSION.md`

No behavior, so no new test: the check is that nothing reads what is deleted.

- [x] **Step 1:** `git grep -n "COMPRESSION_TRIGGER_RATIO\|COMPRESSION\.md" -- ':!docs/superpowers'` — expect only the three sites above.
- [x] **Step 2:** Delete the comment + `COMPRESSION_TRIGGER_RATIO` block in `chat_configs.py`, the `"COMPRESSION_TRIGGER_RATIO": 0.8,` line in `default_config.py`, and `git rm src/internal/chat/COMPRESSION.md`.
- [x] **Step 3:** Re-run the grep (expect nothing) and `.venv/bin/python -m pytest tests/unit/test_configs.py tests/unit/test_documented_env_vars.py -q -p no:cacheprovider`.
- [x] **Step 4: Commit** — `git commit -m "Remove the dead COMPRESSION_TRIGGER_RATIO config and COMPRESSION.md"`

---

### Task 5: Mutation checks and final verification

- [x] Mutation A: in `_fit_budget` delete `if kept and used > token_budget: break` → `test_token_budget_keeps_newest_records_that_fit` must go red. Restore, `find src tests -name __pycache__ -type d -prune -exec rm -rf {} +`, `git diff --stat` clean.
- [x] Mutation B: in `compress_session` delete `pending = _bounded_span(pending)` → `test_compress_summarizes_one_bounded_chunk_of_a_backlog` must go red. Restore, clear `__pycache__`, diff clean.
- [x] `.venv/bin/python -m pytest tests/unit/ -q -p no:cacheprovider` — 0 failures.
- [x] `ruff check . && ruff format --check .`; `git diff --check origin/main...HEAD`.
