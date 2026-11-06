"""Consolidation, index summaries and the automatic trigger."""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace

import pytest

from src.internal.db.models import UserRecord
from src.internal.db.store import AgenticSearchStore
from src.internal.hooks.lifecycle import LifecycleHooks, TurnContext
from src.internal.memory import consolidation, lifecycle
from src.internal.memory.consolidation import (
    CONSOLIDATED_NODE,
    consolidate_memories,
    rebuild_index_summaries,
)
from src.internal.memory.lifecycle import (
    consolidation_handler,
    register_memory_hooks,
    user_guard,
)

_ID = re.compile(r"mem_[0-9a-f]+")


class _FakeLLM:
    """Scripted ``complete()``; dispatches on what the prompt asks for."""

    config = SimpleNamespace(model_name="fake")

    def __init__(
        self,
        *,
        supersede_all=False,
        merge_all=False,
        json_reply=None,
        merge_text="Merged memory",
        categories=None,
        summary=None,
        raw=None,
    ):
        self.supersede_all = supersede_all
        self.merge_all = merge_all
        self.json_reply = json_reply
        self.merge_text = merge_text
        self.categories = categories
        self.summary = summary
        self.raw = raw
        self.prompts: list[str] = []

    def complete(self, prompt, **kwargs):
        text = prompt if isinstance(prompt, str) else json.dumps(prompt)
        self.prompts.append(text)
        if self.raw is not None:
            return self.raw
        if '"superseded"' in text:
            if self.json_reply is not None:
                return json.dumps(self.json_reply)
            ids = _ID.findall(text) if self.supersede_all else []
            return json.dumps({"superseded": ids})
        if '"text"' in text:
            return json.dumps({"text": self.merge_text if self.merge_all else ""})
        if '"categories"' in text:
            return json.dumps({"categories": self.categories or {}})
        return self.summary or ""


@pytest.fixture
def store():
    s = AgenticSearchStore(":memory:")
    s.upsert_user(UserRecord(id="u1"))
    s.upsert_user(UserRecord(id="u2"))
    yield s
    s.close()


def _texts(store, user_id="u1", **kw):
    return sorted(r.memory_text for r in store.get_user_memory_records(user_id, **kw))


def test_near_duplicates_merge_without_llm_keeps_newest(store):
    store.add_user_memory("u1", "User prefers peer reviewed sources")
    store.add_user_memory("u1", "User prefers peer-reviewed sources")
    rep = consolidate_memories(store, "u1")
    assert rep["merged"] == 1
    assert _texts(store) == ["User prefers peer-reviewed sources"]


def test_near_duplicate_merge_with_llm_updates_newest(store):
    store.add_user_memory("u1", "User prefers peer reviewed sources")
    newest = store.add_user_memory("u1", "User prefers peer-reviewed sources")
    llm = _FakeLLM(merge_all=True, merge_text="User prefers peer-reviewed papers")
    rep = consolidate_memories(store, "u1", llm=llm)
    records = store.get_user_memory_records("u1")
    assert rep["merged"] == 1
    assert [(r.id, r.memory_text) for r in records] == [
        (newest.id, "User prefers peer-reviewed papers")
    ]


def test_constraints_never_superseded_or_merged_by_llm(store):
    store.add_user_memory("u1", "User is allergic to peanuts")
    store.add_user_memory("u1", "User is allergic to peanuts and tree nuts")
    llm = _FakeLLM(supersede_all=True, merge_all=True)
    consolidate_memories(store, "u1", llm=llm)
    assert len(store.get_user_memory_records("u1", category="constraints")) == 2
    decisions = [p for p in llm.prompts if '"superseded"' in p or '"text"' in p]
    assert not any("allergic" in p for p in decisions)


def test_constraints_skip_tag_rule_without_llm(store):
    store.add_user_memory("u1", "User is allergic to peanuts", {"tags": ["allergy"]})
    store.add_user_memory("u1", "User is allergic to shellfish", {"tags": ["allergy"]})
    rep = consolidate_memories(store, "u1")
    assert rep["conflicts_resolved"] == []
    assert len(store.get_user_memory_records("u1", category="constraints")) == 2


def test_exact_duplicate_constraint_collapses(store):
    store.add_user_memory("u1", "User is allergic to peanuts")
    store.add_user_memory("u1", "user is  allergic to peanuts")
    assert consolidate_memories(store, "u1")["duplicates_removed"] == 1
    assert len(store.get_user_memory_records("u1", category="constraints")) == 1


def test_llm_contradiction_supersedes_and_records_metadata(store):
    a = store.add_user_memory("u1", "Lives in Berlin", category="profile")
    b = store.add_user_memory("u1", "Moved to Munich", category="profile")
    llm = _FakeLLM(json_reply={"superseded": [a.id]})
    rep = consolidate_memories(store, "u1", llm=llm)
    assert [r.memory_text for r in store.get_user_memory_records("u1")] == [
        "Moved to Munich"
    ]
    assert rep["conflicts_resolved"]
    row = store._conn.execute(
        "SELECT metadata_json, is_active FROM user_memories WHERE id = ?", (a.id,)
    ).fetchone()
    assert row["is_active"] == 0
    assert json.loads(row["metadata_json"])["superseded_by"] == b.id


def test_llm_never_supersedes_the_newest_in_a_batch(store):
    store.add_user_memory("u1", "Lives in Berlin", category="profile")
    store.add_user_memory("u1", "Moved to Munich", category="profile")
    consolidate_memories(store, "u1", llm=_FakeLLM(supersede_all=True))
    assert _texts(store) == ["Moved to Munich"]


@pytest.mark.parametrize(
    "llm",
    [
        _FakeLLM(json_reply={"superseded": ["mem_nope"]}),
        _FakeLLM(raw="I think nothing is superseded {not json"),
        _FakeLLM(raw="[1, 2]"),
        _FakeLLM(raw=None, json_reply={"superseded": "mem_x"}),
    ],
)
def test_llm_consolidation_ignores_unknown_ids_and_junk(store, llm):
    store.add_user_memory("u1", "Lives in Berlin", category="profile")
    store.add_user_memory("u1", "Works as a chemist", category="profile")
    store.add_user_memory("u1", "Something miscellaneous happened")
    rep = consolidate_memories(store, "u1", llm=llm)
    assert len(store.get_user_memory_records("u1")) == 3
    assert rep["conflicts_resolved"] == [] and rep["reclassified"] == 0


def test_llm_cannot_touch_other_users_ids(store):
    other = store.add_user_memory("u2", "Lives in Paris", category="profile")
    store.add_user_memory("u1", "Lives in Berlin", category="profile")
    store.add_user_memory("u1", "Moved to Munich", category="profile")
    consolidate_memories(
        store, "u1", llm=_FakeLLM(json_reply={"superseded": [other.id]})
    )
    assert [r.id for r in store.get_user_memory_records("u2")] == [other.id]
    assert len(store.get_user_memory_records("u1")) == 2


def test_reclassify_general_with_llm_but_not_out_of_constraints(store):
    g = store.add_user_memory("u1", "Something about tea")
    c = store.add_user_memory("u1", "User is allergic to peanuts")
    assert g.category == "general" and c.category == "constraints"
    llm = _FakeLLM(categories={g.id: "preferences", c.id: "general"})
    rep = consolidate_memories(store, "u1", llm=llm)
    cats = {r.id: r.category for r in store.get_user_memory_records("u1")}
    assert rep["reclassified"] == 1
    assert cats == {g.id: "preferences", c.id: "constraints"}


def test_no_llm_tag_rule_still_applies_to_non_constraints(store):
    store.add_user_memory("u1", "lives in Beijing", metadata={"tags": ["home"]})
    store.add_user_memory("u1", "lives in Shanghai", metadata={"tags": ["home"]})
    rep = consolidate_memories(store, "u1")
    assert rep["conflicts_resolved"][0]["kept"] == "lives in Shanghai"
    assert _texts(store) == ["lives in Shanghai"]


def test_report_keys_and_consolidated_marker(store):
    store.add_user_memory("u1", "Prefers arXiv")
    rep = consolidate_memories(store, "u1")
    assert set(rep) == {
        "initial",
        "duplicates_removed",
        "conflicts_resolved",
        "merged",
        "reclassified",
        "summaries_rebuilt",
        "final",
    }
    assert rep["summaries_rebuilt"] >= 2 and rep["final"] == 1
    nodes = {n.node: n for n in store.get_memory_index_nodes("u1")}
    assert nodes[CONSOLIDATED_NODE].memory_count == 1
    assert nodes["__root__"].memory_count == 1


def test_rebuild_summaries_deterministic_and_clears_dirty(store):
    store.add_user_memory("u1", "Prefers arXiv")
    n = rebuild_index_summaries(store, "u1")
    nodes = {x.node: x for x in store.get_memory_index_nodes("u1")}
    assert n >= 2 and not nodes["preferences"].dirty
    assert nodes["__root__"].summary.startswith("preferences 1")
    assert nodes["preferences"].summary == (
        "1 memories; key terms: prefers, arxiv; latest: Prefers arXiv"
    )
    assert rebuild_index_summaries(store, "u1") == 0


def test_rebuild_summary_truncates_latest_and_orders_root(store):
    long_text = "User prefers " + "very " * 30 + "long answers"
    store.add_user_memory("u1", "User is allergic to peanuts")
    store.add_user_memory("u1", "User prefers short answers")
    store.add_user_memory("u1", long_text)
    rebuild_index_summaries(store, "u1")
    nodes = {x.node: x for x in store.get_memory_index_nodes("u1")}
    assert nodes["__root__"].summary == "constraints 1 · preferences 2"
    latest = nodes["preferences"].summary.split("; latest: ")[1].split(" | ")
    assert len(latest) == 2 and all(len(t) <= 80 for t in latest)
    assert latest[1] == "User prefers short answers"


def test_rebuild_summaries_uses_llm_and_falls_back(store):
    store.add_user_memory("u1", "Prefers arXiv")
    rebuild_index_summaries(store, "u1", llm=_FakeLLM(summary="Likes preprints."))
    nodes = {x.node: x for x in store.get_memory_index_nodes("u1")}
    assert nodes["preferences"].summary == "Likes preprints."
    store.add_user_memory("u1", "Prefers short answers")
    rebuild_index_summaries(store, "u1", llm=_FakeLLM(summary="   "))
    nodes = {x.node: x for x in store.get_memory_index_nodes("u1")}
    assert nodes["preferences"].summary.startswith("2 memories; key terms:")


def test_rebuild_summaries_clears_emptied_category(store):
    rec = store.add_user_memory("u1", "Prefers arXiv")
    rebuild_index_summaries(store, "u1")
    store.delete_user_memory("u1", rec.id)
    rebuild_index_summaries(store, "u1")
    nodes = {x.node: x for x in store.get_memory_index_nodes("u1")}
    assert nodes["preferences"].summary == "" and nodes["preferences"].memory_count == 0
    assert not any(x.dirty for x in nodes.values())


async def test_auto_trigger_threshold(store, monkeypatch):
    monkeypatch.setattr(consolidation, "CONSOLIDATE_EVERY", 3)
    calls = []
    real = consolidation.consolidate_memories

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(consolidation, "consolidate_memories", spy)
    ctx = TurnContext("u1", None, "q", store)
    store.add_user_memory("u1", "Prefers arXiv")
    store.add_user_memory("u1", "Works on retrieval project")
    await consolidation_handler(ctx)
    nodes = {x.node: x for x in store.get_memory_index_nodes("u1")}
    assert calls == [] and not nodes["preferences"].dirty
    assert CONSOLIDATED_NODE not in nodes

    store.add_user_memory("u1", "Decided to use Postgres")
    await consolidation_handler(ctx)
    assert len(calls) == 1
    await consolidation_handler(ctx)  # nothing grew since: no second run
    assert len(calls) == 1


async def test_consolidation_handler_anonymous_and_busy(store, monkeypatch):
    monkeypatch.setattr(consolidation, "CONSOLIDATE_EVERY", 1)
    monkeypatch.setattr(lifecycle, "CONSOLIDATION_WAIT_SECONDS", 0.0)
    store.add_user_memory("u1", "Prefers arXiv")
    await consolidation_handler(TurnContext(None, None, "q", store))
    with user_guard("u1") as free:
        assert free
        await consolidation_handler(TurnContext("u1", None, "q", store))
    nodes = {x.node for x in store.get_memory_index_nodes("u1")}
    assert CONSOLIDATED_NODE not in nodes


async def test_consolidation_waits_for_a_concurrent_capture(store, monkeypatch):
    """Capture holds the guard when after_turn starts consolidation; it must wait."""
    monkeypatch.setattr(consolidation, "CONSOLIDATE_EVERY", 1)
    monkeypatch.setattr(lifecycle, "_GUARD_POLL_SECONDS", 0.01)
    store.add_user_memory("u1", "Prefers arXiv")

    async def hold():
        with user_guard("u1"):
            await asyncio.sleep(0.05)

    holder = asyncio.create_task(hold())
    await asyncio.sleep(0)
    await consolidation_handler(TurnContext("u1", None, "q", store))
    await holder
    nodes = {x.node for x in store.get_memory_index_nodes("u1")}
    assert CONSOLIDATED_NODE in nodes


async def test_consolidation_handler_never_raises(store, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db gone")

    monkeypatch.setattr(consolidation, "run_due", boom)
    await consolidation_handler(TurnContext("u1", None, "q", store))


def test_register_consolidation_regardless_of_capture_flag():
    on, off = LifecycleHooks(), LifecycleHooks()
    register_memory_hooks(on, SimpleNamespace(memory_auto_curate=True))
    register_memory_hooks(off, SimpleNamespace(memory_auto_curate=False))
    assert consolidation_handler in on._after_turn
    assert consolidation_handler in off._after_turn
