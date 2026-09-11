"""Working memory: session tail + compressed summary of what fell off it."""

from __future__ import annotations


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
