"""Working memory: session tail + compressed summary of what fell off it."""

from __future__ import annotations

import asyncio
import logging
import time
import types

import pytest

from src.internal.cache.interface import InMemoryCache
from src.internal.db import AgenticSearchStore
from src.internal.memory import working
from src.internal.memory.working import (
    forget_session,
    SESSION_MEMORY_TTL_SECONDS,
    SUMMARY_PREFIX,
    SessionMemoryState,
    WorkingMemory,
    compress_session,
    estimate_tokens,
    load_state,
    load_working_memory,
    save_state,
    schedule_compression,
)


@pytest.fixture(autouse=True)
def _clear_working_memory_module_state():
    """`_inflight`/`_tasks` are module-level; clear them so tests don't leak
    session ids or task references into each other."""
    working._inflight.clear()
    working._tasks.clear()
    yield
    working._inflight.clear()
    working._tasks.clear()


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


def test_save_state_writes_the_session_memory_ttl(cache):
    class RecordingCache(InMemoryCache):
        def __init__(self):
            super().__init__()
            self.set_calls: list[tuple] = []

        def set(self, key, value, ex=None):
            self.set_calls.append((key, value, ex))
            super().set(key, value, ex=ex)

    recording = RecordingCache()
    save_state(recording, "sid", SessionMemoryState(summary="s"))
    assert recording.set_calls[-1][2] == SESSION_MEMORY_TTL_SECONDS


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
        cache,
        sid,
        SessionMemoryState(summary="they said hi", summarized_through=records[1].id),
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
        cache,
        sid,
        SessionMemoryState(summary="stale", summarized_through="msg_not_here"),
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
    assert [m.content for m in wm.messages] == [r.content for r in records[-keep_last:]]
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
    ok = asyncio.run(
        compress_session(sid, FakeLLM("they said hi"), pending=pending, cache=cache)
    )
    assert ok is True
    state = load_state(cache, sid)
    assert state.summary == "they said hi"
    assert state.summarized_through == records[1].id
    assert state.curated_through is None


def test_compress_prompt_carries_prior_summary_and_every_turn(store, cache):
    sid, records = _seed(store, 12)
    save_state(
        cache,
        sid,
        SessionMemoryState(summary="PRIOR", summarized_through=records[0].id),
    )
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
    ok = asyncio.run(
        compress_session(sid, FakeLLM(fail=True), pending=records[:2], cache=cache)
    )
    assert ok is False
    assert load_state(cache, sid) == SessionMemoryState()
    ok = asyncio.run(
        compress_session(sid, FakeLLM("later"), pending=records[:2], cache=cache)
    )
    assert ok is True
    assert load_state(cache, sid).summary == "later"


def test_compress_reads_text_off_an_llmresponse_shaped_object(store, cache):
    # llm.complete can return an LLMResponse (a .text attribute) instead of a
    # plain str; _complete must unwrap it either way.
    sid, records = _seed(store, 12)

    class LLMResponseLike:
        def complete(self, messages, **kwargs):
            return types.SimpleNamespace(text="from response")

    ok = asyncio.run(
        compress_session(sid, LLMResponseLike(), pending=records[:2], cache=cache)
    )
    assert ok is True
    assert load_state(cache, sid).summary == "from response"


def test_compress_empty_text_leaves_state(store, cache):
    sid, records = _seed(store, 12)
    ok = asyncio.run(
        compress_session(sid, FakeLLM("   "), pending=records[:2], cache=cache)
    )
    assert ok is False
    assert load_state(cache, sid) == SessionMemoryState()


def test_compress_skips_when_cursor_already_at_last_pending(store, cache):
    sid, records = _seed(store, 12)
    save_state(
        cache, sid, SessionMemoryState(summary="done", summarized_through=records[1].id)
    )
    llm = FakeLLM()
    ok = asyncio.run(compress_session(sid, llm, pending=records[:2], cache=cache))
    assert ok is False
    assert llm.prompts == []


def test_compress_trims_pending_already_covered_by_a_newer_cursor(store, cache):
    # Two overlapping compression tasks can run out of order; the stored
    # cursor may already be ahead of the start of `pending` by the time this
    # task re-reads state. Already-covered turns must be dropped, not
    # re-summarized, and the cursor must advance past the whole span given.
    sid, records = _seed(store, 12)
    save_state(
        cache, sid, SessionMemoryState(summary="old", summarized_through=records[1].id)
    )
    llm = FakeLLM("new summary")
    ok = asyncio.run(compress_session(sid, llm, pending=records[:4], cache=cache))
    assert ok is True
    user_prompt = llm.prompts[0][-1]["content"]
    assert "New turns" in user_prompt
    assert "m2" in user_prompt
    assert "m3" in user_prompt
    assert "m0" not in user_prompt
    assert "m1" not in user_prompt
    assert load_state(cache, sid).summarized_through == records[3].id


def test_compress_no_pending_or_no_llm_is_noop(store, cache):
    sid, records = _seed(store, 12)
    assert (
        asyncio.run(compress_session(sid, FakeLLM(), pending=[], cache=cache)) is False
    )
    assert (
        asyncio.run(compress_session(sid, None, pending=records[:2], cache=cache))
        is False
    )


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


def test_compress_lock_failure_leaves_state_and_clears_inflight(store, cache):
    sid, records = _seed(store, 12)

    class BrokenCache(InMemoryCache):
        def lock(self, name, timeout=None):
            raise RuntimeError("redis down")

    broken = BrokenCache()
    llm = FakeLLM()
    ok = asyncio.run(compress_session(sid, llm, pending=records[:2], cache=broken))
    assert ok is False
    assert load_state(broken, sid) == SessionMemoryState()
    assert llm.prompts == []

    # A second call on the same session, against a healthy cache, must still
    # succeed -- proving the failed lock attempt did not leave `_inflight` set.
    ok = asyncio.run(compress_session(sid, llm, pending=records[:2], cache=cache))
    assert ok is True


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


# --- schedule_compression -----------------------------------------------------


def _wm(pending):
    return WorkingMemory(messages=[], summary="", pending=pending)


def test_schedule_returns_none_when_disabled_or_no_llm_or_nothing_pending(store, cache):
    sid, records = _seed(store, 12)

    async def run():
        assert (
            schedule_compression(
                _wm(records[:2]),
                session_id=sid,
                llm=FakeLLM(),
                enabled=False,
                cache=cache,
            )
            is None
        )
        assert (
            schedule_compression(
                _wm(records[:2]), session_id=sid, llm=None, enabled=True, cache=cache
            )
            is None
        )
        assert (
            schedule_compression(
                _wm([]), session_id=sid, llm=FakeLLM(), enabled=True, cache=cache
            )
            is None
        )

    asyncio.run(run())
    assert load_state(cache, sid) == SessionMemoryState()


def test_schedule_runs_compress_in_background(store, cache):
    sid, records = _seed(store, 12)

    async def run():
        task = schedule_compression(
            _wm(records[:2]),
            session_id=sid,
            llm=FakeLLM("bg"),
            enabled=True,
            cache=cache,
        )
        assert task is not None
        await task

    asyncio.run(run())
    assert load_state(cache, sid).summary == "bg"


# --- curation ------------------------------------------------------------------


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
        compress_session(
            sid, FakeLLM("S"), pending=records[:2], cache=cache, curate=curate
        )
    )
    assert ok is True
    state = load_state(cache, sid)
    assert state.summary == "S"
    assert state.summarized_through == records[1].id
    assert state.curated_through == records[1].id
    assert [r.id for r in curate.calls[0]] == [records[0].id, records[1].id]


def test_compress_curates_only_the_post_trim_span(store, cache):
    sid, records = _seed(store, 12)
    save_state(
        cache, sid, SessionMemoryState(summary="old", summarized_through=records[1].id)
    )
    curate = _curator(True)
    asyncio.run(
        compress_session(
            sid, FakeLLM("S"), pending=records[:4], cache=cache, curate=curate
        )
    )
    assert [r.id for r in curate.calls[0]] == [records[2].id, records[3].id]
    assert load_state(cache, sid).curated_through == records[3].id


def test_curate_cursor_write_preserves_a_newer_summary(store, cache):
    # Curation can run for several LLM turns, longer than the 120s Redis lock
    # lease. If another process advanced the summary while this task's
    # curation was still running, the cursor write must merge onto that newer
    # state rather than reconstruct and overwrite it.
    sid, records = _seed(store, 12)

    async def curate(pending):
        save_state(
            cache,
            sid,
            SessionMemoryState(
                summary="newer", summarized_through="m_newer", curated_through=None
            ),
        )
        return True

    ok = asyncio.run(
        compress_session(
            sid, FakeLLM("S"), pending=records[:2], cache=cache, curate=curate
        )
    )
    assert ok is True
    state = load_state(cache, sid)
    assert state.summary == "newer"
    assert state.summarized_through == "m_newer"
    assert state.curated_through == records[1].id


def test_compress_curate_false_keeps_summary_and_cursor(store, cache):
    sid, records = _seed(store, 12)
    save_state(cache, sid, SessionMemoryState(curated_through="keep-me"))
    ok = asyncio.run(
        compress_session(
            sid, FakeLLM("S"), pending=records[:2], cache=cache, curate=_curator(False)
        )
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
            sid,
            FakeLLM("S"),
            pending=records[:2],
            cache=cache,
            curate=_curator(raise_exc=True),
        )
    )
    assert ok is True
    state = load_state(cache, sid)
    assert state.summary == "S"
    assert state.curated_through is None
    # Lock and in-flight entry were released: a later span compresses fine.
    ok2 = asyncio.run(
        compress_session(
            sid, FakeLLM("S2"), pending=records[:4], cache=cache, curate=_curator(True)
        )
    )
    assert ok2 is True
    assert load_state(cache, sid).curated_through == records[3].id


def test_compress_curate_cursor_write_failure_still_returns_true(store):
    sid, records = _seed(store, 12)

    class FlakySecondSetCache(InMemoryCache):
        def __init__(self):
            super().__init__()
            self.set_calls = 0

        def set(self, key, value, ex=None):
            self.set_calls += 1
            if self.set_calls == 2:
                raise RuntimeError("cache down")
            super().set(key, value, ex=ex)

    flaky = FlakySecondSetCache()
    ok = asyncio.run(
        compress_session(
            sid, FakeLLM("S"), pending=records[:2], cache=flaky, curate=_curator(True)
        )
    )
    assert ok is True
    state = load_state(flaky, sid)
    assert state.summary == "S"
    assert state.summarized_through == records[1].id
    assert state.curated_through is None


def test_compress_without_curate_preserves_curated_through(store, cache):
    sid, records = _seed(store, 12)
    save_state(cache, sid, SessionMemoryState(curated_through="keep-me"))
    asyncio.run(compress_session(sid, FakeLLM("S"), pending=records[:2], cache=cache))
    assert load_state(cache, sid).curated_through == "keep-me"


def test_compress_does_not_curate_when_summary_fails(store, cache):
    sid, records = _seed(store, 12)
    curate = _curator(True)
    asyncio.run(
        compress_session(
            sid, FakeLLM(fail=True), pending=records[:2], cache=cache, curate=curate
        )
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
            _wm(records[:2]),
            session_id=sid,
            llm=FakeLLM("S"),
            enabled=True,
            cache=cache,
            **kw,
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
            _wm(records[:2]),
            session_id=sid,
            llm=FakeLLM(),
            enabled=False,
            cache=cache,
            store=store,
            user_id="u1",
            auto_curate=True,
        )

    assert asyncio.run(run()) is None
    assert load_state(cache, sid) == SessionMemoryState()


def test_curate_cursor_write_skips_merge_when_reread_is_blank(store):
    # load_state turns a cache read failure into a blank state. If the cursor
    # merge trusted that blank, it would write summary="" over the summary
    # this same task saved a moment earlier. The merge must skip instead.
    class ReadsFailOnDemand(InMemoryCache):
        fail_reads = False

        def get(self, key):
            if self.fail_reads:
                raise RuntimeError("redis blip")
            return super().get(key)

    cache = ReadsFailOnDemand()
    sid, records = _seed(store, 12)

    async def curate(pending):
        cache.fail_reads = True  # the cursor re-read, not the summary write
        return True

    ok = asyncio.run(
        compress_session(
            sid, FakeLLM("S"), pending=records[:2], cache=cache, curate=curate
        )
    )
    assert ok is True
    cache.fail_reads = False
    state = load_state(cache, sid)
    assert state.summary == "S"
    assert state.summarized_through == records[1].id
    assert state.curated_through is None


class StateWritingLLM(FakeLLM):
    """Simulates another process writing the session state while this task's
    summarizer call is in flight (the 120s lock lease can expire mid-call)."""

    def __init__(self, text, *, cache, sid, write):
        super().__init__(text)
        self._cache, self._sid, self._write = cache, sid, write

    def complete(self, messages, **kwargs):
        save_state(self._cache, self._sid, self._write)
        return super().complete(messages, **kwargs)


def test_compress_skips_write_when_cursor_moved_during_llm_call(store, cache):
    sid, records = _seed(store, 12)
    newer = SessionMemoryState(summary="newer", summarized_through="m_newer")
    llm = StateWritingLLM("S", cache=cache, sid=sid, write=newer)
    curate = _curator(True)
    ok = asyncio.run(
        compress_session(sid, llm, pending=records[:2], cache=cache, curate=curate)
    )
    assert ok is False
    assert load_state(cache, sid) == newer
    assert curate.calls == []


def test_compress_write_preserves_curated_through_changed_during_llm_call(store, cache):
    sid, records = _seed(store, 12)
    moved = SessionMemoryState(curated_through="moved")  # cursor unchanged (None)
    llm = StateWritingLLM("S", cache=cache, sid=sid, write=moved)
    ok = asyncio.run(compress_session(sid, llm, pending=records[:2], cache=cache))
    assert ok is True
    state = load_state(cache, sid)
    assert state.summary == "S"
    assert state.summarized_through == records[1].id
    assert state.curated_through == "moved"


def test_compress_first_summary_writes_despite_failed_reread(store):
    class ReadsFailOnDemand(InMemoryCache):
        fail_reads = False

        def get(self, key):
            if self.fail_reads:
                raise RuntimeError("redis blip")
            return super().get(key)

    cache = ReadsFailOnDemand()
    sid, records = _seed(store, 12)

    class FailReadsLLM(FakeLLM):
        def complete(self, messages, **kwargs):
            cache.fail_reads = True  # the post-LLM re-read, not the pre-read
            return super().complete(messages, **kwargs)

    ok = asyncio.run(
        compress_session(sid, FailReadsLLM("S"), pending=records[:2], cache=cache)
    )
    assert ok is True
    cache.fail_reads = False
    state = load_state(cache, sid)
    assert state.summary == "S"
    assert state.summarized_through == records[1].id


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
        r.levelno == logging.WARNING and "sid" in r.getMessage() for r in caplog.records
    )
