import numpy as np

from src.context.models import SearchContextBundle
from src.context.prompts import build_answer_prompt, build_structured_answer_prompt
from src.internal.db.store import AgenticSearchStore
from src.internal.memory import service
from src.internal.memory.service import (
    MEMORY_INJECTION_MAX,
    MEMORY_RELEVANT_SLOTS,
    memory_preamble,
)


def test_memory_preamble_empty_user_returns_blank():
    store = AgenticSearchStore(":memory:")
    assert memory_preamble(store, "nobody") == ""
    store.close()


def test_memory_preamble_formats_instructional_block():
    store = AgenticSearchStore(":memory:")
    store.add_user_memory("u1", "User is allergic to peanuts")
    store.add_user_memory("u1", "User prefers window seats")
    pre = memory_preamble(store, "u1")
    assert pre.startswith("\n\nWhat you know about this user")
    assert "allergic to peanuts" in pre
    assert "- User prefers window seats" in pre
    store.close()


def test_memory_preamble_caps_to_most_recent():
    store = AgenticSearchStore(":memory:")
    for i in range(MEMORY_INJECTION_MAX + 5):
        store.add_user_memory("u1", f"memory number {i}")
    pre = memory_preamble(store, "u1", max_items=MEMORY_INJECTION_MAX)
    # exactly MEMORY_INJECTION_MAX bullet lines, and the most-recent ones kept
    assert pre.count("\n- ") == MEMORY_INJECTION_MAX
    assert f"memory number {MEMORY_INJECTION_MAX + 4}" in pre  # newest
    assert "memory number 0" not in pre  # oldest dropped
    store.close()


def test_prompt_builders_inject_memory_into_system():
    ctx = SearchContextBundle(query="q", documents=[])
    pre = "\n\nWhat you know about this user:\n- User is allergic to peanuts"

    answer = build_answer_prompt("Recommend Thai food", ctx, user_memory=pre)
    assert "allergic to peanuts" in answer.system

    structured = build_structured_answer_prompt(
        "Recommend Thai food", ctx, user_memory=pre
    )
    assert "allergic to peanuts" in structured.system


def test_prompt_builders_unchanged_without_memory():
    ctx = SearchContextBundle(query="q", documents=[])
    assert "allergic" not in build_answer_prompt("q", ctx).system
    assert "allergic" not in build_structured_answer_prompt("q", ctx).system


class _CapturingLLM:
    """Fake LLM that records the messages it was asked to complete."""

    def __init__(self):
        self.seen: list = []

    def complete(self, messages, **kwargs):
        self.seen = messages
        return "an answer"


def test_generate_answer_injects_memory_into_system_message():
    """End-to-end through the real pipeline: the memory reaches the LLM's system
    message, not just a helper string."""
    from src.context.models import AnswerGenerationRequest, GroundedGenerationConfig
    from src.context.pipeline import generate_answer

    ctx = SearchContextBundle(query="q", documents=[])
    pre = "\n\nUser memory:\n- User is allergic to peanuts"
    request = AnswerGenerationRequest(
        question="Recommend Thai food",
        context=ctx,
        user_memory=pre,
        grounded_generation=GroundedGenerationConfig(enabled=False),
    )
    llm = _CapturingLLM()
    generate_answer(request, llm=llm)

    system = next(m.content for m in llm.seen if m.role == "system")
    assert "allergic to peanuts" in system


def _seed_above_cap(store, n=MEMORY_INJECTION_MAX + 5):
    store.add_user_memory("u1", "User is allergic to peanuts")
    for i in range(1, n):
        store.add_user_memory("u1", f"memory number {i}")


def test_relevant_slots_is_half_the_cap():
    assert MEMORY_RELEVANT_SLOTS == MEMORY_INJECTION_MAX // 2


def test_memory_preamble_below_cap_ignores_query():
    store = AgenticSearchStore(":memory:")
    store.add_user_memory("u1", "User is allergic to peanuts")
    store.add_user_memory("u1", "User prefers window seats")
    assert memory_preamble(store, "u1", query="thai food") == memory_preamble(
        store, "u1"
    )
    store.close()


def test_memory_preamble_above_cap_keeps_relevant_old_memory_and_recent_fill():
    store = AgenticSearchStore(":memory:")
    _seed_above_cap(store)
    pre = memory_preamble(store, "u1", query="thai food with peanuts")
    bullets = [line[2:] for line in pre.split("\n") if line.startswith("- ")]
    assert len(bullets) == MEMORY_INJECTION_MAX
    assert len(set(bullets)) == MEMORY_INJECTION_MAX
    assert bullets[0] == "User is allergic to peanuts"  # oldest, kept, first
    assert bullets[-1] == f"memory number {MEMORY_INJECTION_MAX + 4}"  # newest fill
    assert "memory number 1" not in bullets  # old and irrelevant: dropped
    store.close()


def test_memory_preamble_above_cap_without_query_is_recency():
    store = AgenticSearchStore(":memory:")
    _seed_above_cap(store)
    for q in (None, "", "   "):
        pre = memory_preamble(store, "u1", query=q)
        assert "User is allergic to peanuts" not in pre
        assert pre.count("\n- ") == MEMORY_INJECTION_MAX
    store.close()


def test_memory_preamble_above_cap_uses_encoder_when_given():
    store = AgenticSearchStore(":memory:")
    _seed_above_cap(store)
    calls: list = []

    def encoder(texts):
        calls.append(list(texts))
        # One-hot: the peanut passage and the query share axis 0; everything else axis 1.
        return np.array(
            [
                [1.0, 0.0] if ("peanuts" in t or t.startswith("query:")) else [0.0, 1.0]
                for t in texts
            ]
        )

    pre = memory_preamble(store, "u1", query="anything", encoder=encoder)
    assert "User is allergic to peanuts" in pre
    assert len(calls) == 2
    assert all(t.startswith("passage: ") for t in calls[0])
    assert calls[1] == ["query: anything"]
    store.close()


def test_memory_preamble_search_failure_degrades_to_recency(monkeypatch):
    store = AgenticSearchStore(":memory:")
    _seed_above_cap(store)

    def boom(*a, **k):
        raise RuntimeError("encoder down")

    monkeypatch.setattr(service, "search_memories", boom)
    pre = memory_preamble(store, "u1", query="thai food with peanuts")
    assert pre.count("\n- ") == MEMORY_INJECTION_MAX
    assert "User is allergic to peanuts" not in pre
    store.close()
