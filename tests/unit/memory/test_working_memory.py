"""Working memory: session tail + compressed summary of what fell off it."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.internal.cache.interface import InMemoryCache
from src.internal.db import AgenticSearchStore
from src.internal.memory.working import (
    SUMMARY_PREFIX,
    SessionMemoryState,
    WorkingMemory,
    compress_session,
    load_state,
    load_working_memory,
    save_state,
    schedule_compression,
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
