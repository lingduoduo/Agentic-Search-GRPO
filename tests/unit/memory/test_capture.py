"""Per-turn automatic capture: gate, extract, reconcile, guard, handler."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.internal.db.models import UserRecord
from src.internal.db.store import AgenticSearchStore
from src.internal.hooks.lifecycle import LifecycleHooks, TurnContext
from src.internal.memory.capture import (
    capture_turn,
    extract_memories,
    has_durable_signal,
    is_near_duplicate,
)
from src.internal.memory.lifecycle import (
    capture_handler,
    register_memory_hooks,
    user_guard,
)


class _FakeLLM:
    config = SimpleNamespace(model_name="fake")

    def __init__(self, reply: str = "", raise_if_called: bool = False):
        self.reply = reply
        self.raise_if_called = raise_if_called
        self.calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        if self.raise_if_called:
            raise AssertionError("LLM must not be called")
        return self.reply


def _store(owner: str | None = "u1"):
    store = AgenticSearchStore(":memory:")
    for uid in {"u1", "u2"}:
        store.upsert_user(UserRecord(id=uid))
    sid = store.create_chat_session(user_id=owner).id
    return store, sid


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("I'm allergic to shellfish, find me restaurants", True),
        ("I prefer peer-reviewed sources", True),
        ("We decided to go with Postgres", True),
        ("What is FAISS?", False),
        ("compare dense and sparse retrieval", False),
    ],
)
def test_has_durable_signal(msg, expected):
    assert has_durable_signal(msg) is expected


def test_is_near_duplicate():
    assert is_near_duplicate(
        "User prefers peer reviewed sources", "User prefers peer-reviewed sources only"
    )
    assert not is_near_duplicate("User likes tea", "User works at Acme")
    assert is_near_duplicate("a b", "totally different words", cosine=0.95)
    assert not is_near_duplicate("User likes tea", "User works at Acme", cosine=0.5)


def test_extract_parses_leniently_and_normalizes_category():
    llm = _FakeLLM(
        'sure {"memories":[{"text":"User likes tea","category":"bogus"},'
        '{"text":5},{"category":"profile"}]} done'
    )
    assert extract_memories(llm, "I like tea", "ok") == [("User likes tea", "general")]


@pytest.mark.parametrize("reply", ["not json", '{"memories":[{"text":5}]}', "{"])
def test_extract_invalid_is_empty(reply):
    assert extract_memories(_FakeLLM(reply), "I like tea", "ok") == []


async def test_gate_skips_llm():
    store, sid = _store()
    llm = _FakeLLM(raise_if_called=True)
    assert await capture_turn(store, "u1", sid, llm, "What is FAISS?", "...") is None


async def test_capture_adds_with_category():
    store, sid = _store()
    llm = _FakeLLM(
        '{"memories":[{"text":"User is allergic to shellfish","category":"constraints"}]}'
    )
    await capture_turn(store, "u1", sid, llm, "I'm allergic to shellfish", "ok")
    assert [
        (r.memory_text, r.category) for r in store.get_user_memory_records("u1")
    ] == [("User is allergic to shellfish", "constraints")]
    assert store.list_memory_trajectories("u1")[0].trajectory["source"] == "auto_turn"


async def test_capture_updates_near_duplicate():
    store, sid = _store()
    store.add_user_memory(
        "u1", "User prefers peer reviewed sources", category="preferences"
    )
    llm = _FakeLLM(
        '{"memories":[{"text":"User prefers peer-reviewed sources only","category":"preferences"}]}'
    )
    out = await capture_turn(
        store, "u1", sid, llm, "I prefer peer-reviewed sources only", "ok"
    )
    recs = store.get_user_memory_records("u1")
    assert len(recs) == 1 and recs[0].memory_text.endswith("only")
    assert out["updated"] == 1 and out["added"] == 0


@pytest.mark.parametrize("reply", ["not json", '{"memories":[{"text":5}]}'])
async def test_capture_ignores_unparseable_llm_output(reply):
    store, sid = _store()
    await capture_turn(store, "u1", sid, _FakeLLM(reply), "I prefer tea", "ok")
    assert store.get_user_memory_records("u1") == []


async def test_no_capture_for_unowned_session():
    store, sid = _store(owner="u2")
    llm = _FakeLLM(raise_if_called=True)
    assert await capture_turn(store, "u1", sid, llm, "I prefer tea", "ok") is None


def test_user_guard_skips_when_busy():
    with user_guard("u1") as a:
        with user_guard("u1") as b:
            assert a is True and b is False
    with user_guard("u1") as c:
        assert c is True


def test_register_only_when_flag_on():
    on, off = LifecycleHooks(), LifecycleHooks()
    register_memory_hooks(on, SimpleNamespace(memory_auto_curate=True))
    register_memory_hooks(off, SimpleNamespace(memory_auto_curate=False))
    assert capture_handler in on._after_turn
    assert capture_handler not in off._after_turn


async def test_capture_handler_requires_user_llm_answer():
    store, sid = _store()
    llm = _FakeLLM(raise_if_called=True)
    for ctx in (
        TurnContext(None, sid, "I prefer tea", store, llm=llm, answer="ok"),
        TurnContext("u1", sid, "I prefer tea", store, llm=None, answer="ok"),
        TurnContext("u1", sid, "I prefer tea", store, llm=llm, answer=None),
    ):
        await capture_handler(ctx)
    assert llm.calls == 0


async def test_capture_handler_writes():
    store, sid = _store()
    llm = _FakeLLM(
        '{"memories":[{"text":"User prefers tea","category":"preferences"}]}'
    )
    await capture_handler(
        TurnContext("u1", sid, "I prefer tea", store, llm=llm, answer="ok")
    )
    assert [r.memory_text for r in store.get_user_memory_records("u1")] == [
        "User prefers tea"
    ]
