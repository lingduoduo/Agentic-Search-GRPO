from src.internal.db.store import AgenticSearchStore
from src.internal.memory.service import (
    _MEMORY_PREAMBLE_HEADER,
    MEMORY_INJECTION_MAX,
    memory_preamble,
)


def _store_with(texts):
    store = AgenticSearchStore(":memory:")
    for t in texts:
        store.add_user_memory("u1", t)
    return store


def _seed_above_cap(store, n=MEMORY_INJECTION_MAX + 5):
    store.add_user_memory("u1", "User is allergic to peanuts")
    for i in range(1, n):
        store.add_user_memory("u1", f"memory number {i}")


def test_constraints_always_present_above_cap_without_query():
    store = AgenticSearchStore(":memory:")
    _seed_above_cap(store)
    for q in (None, "", "quantum computing"):
        assert "User is allergic to peanuts" in memory_preamble(store, "u1", query=q)


def test_grouped_by_category_with_root_line():
    store = _store_with(["User is allergic to peanuts", "Prefers arXiv"])
    pre = memory_preamble(store, "u1")
    assert "Memory index:" in pre
    assert pre.index("[constraints]") < pre.index("- User is allergic to peanuts")
    assert "[preferences]" in pre


def test_node_summary_shown_in_heading():
    store = _store_with(["Prefers arXiv"])
    store.upsert_memory_index_node(
        "u1", "preferences", summary="likes primary sources", memory_count=1
    )
    assert "[preferences]: likes primary sources" in memory_preamble(store, "u1")


def test_deterministic_and_bounded():
    store = _store_with([f"fact {i} " + "x" * 300 for i in range(40)])
    a = memory_preamble(store, "u1", query="fact")
    b = memory_preamble(store, "u1", query="fact")
    assert a == b and len(a) <= 4000 + len(_MEMORY_PREAMBLE_HEADER)


def test_constraints_survive_the_char_budget():
    store = _store_with(
        ["User is allergic to peanuts"] + [f"fact {i} " + "x" * 300 for i in range(40)]
    )
    pre = memory_preamble(store, "u1", query="fact")
    assert "User is allergic to peanuts" in pre


def test_empty_user_is_empty_string():
    assert memory_preamble(AgenticSearchStore(":memory:"), "nobody") == ""
