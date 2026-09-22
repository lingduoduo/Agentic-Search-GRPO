"""Unit tests for the canonical search fields on AgentState."""

from __future__ import annotations

from src.agents.core.state import (
    AgentState,
    Citation,
    Retriever,
    UserRequest,
)
from src.context.search import SearchResult


def _doc(doc_id: str, score: float = 1.0) -> SearchResult:
    return SearchResult(contents=f"text-{doc_id}", score=score, title=doc_id)


def _state(question: str = "q") -> AgentState:
    return AgentState(
        request_id="req-1",
        user_request=UserRequest(user_id="u1", channel="test", message=question),
        question=question,
    )


def test_initializes_with_question_and_empty_defaults() -> None:
    state = _state("what is faiss?")

    assert state.question == "what is faiss?"
    assert state.previous_queries == []
    assert state.retrieved_docs == []
    assert state.evidence_score == 0.0
    assert state.search_rounds == 0
    assert state.citations == []


def test_record_search_round_tracks_queries_docs_and_increments_once() -> None:
    state = _state()

    state.record_search_round(["faiss index", "vector index"], [_doc("a"), _doc("b")])

    assert state.previous_queries == ["faiss index", "vector index"]
    assert [d.title for d in state.retrieved_docs] == ["a", "b"]
    assert state.search_rounds == 1


def test_record_search_round_dedupes_queries_but_counts_each_round() -> None:
    state = _state()

    state.record_search_round(["dup", "dup"], [_doc("a")])
    state.record_search_round(["dup"], [_doc("b")])

    assert state.previous_queries == ["dup"]  # deduped, order preserved
    assert [d.title for d in state.retrieved_docs] == [
        "a",
        "b",
    ]  # both rounds' docs kept
    assert state.search_rounds == 2  # each retriever call counts


def test_record_search_round_preserves_distinct_query_order() -> None:
    state = _state()

    state.record_search_round(["first"], [_doc("a")])
    state.record_search_round(["second"], [_doc("b")])

    assert state.previous_queries == ["first", "second"]
    assert state.search_rounds == 2


def test_record_rerank_reorders_docs_without_incrementing_search_rounds() -> None:
    state = _state()
    state.record_search_round(["query"], [_doc("a", 0.1), _doc("b", 0.9)])

    reordered = list(reversed(state.retrieved_docs))
    state.record_rerank(reordered)

    assert [d.title for d in state.retrieved_docs] == ["b", "a"]
    assert state.search_rounds == 1  # rerank is not a retriever call


def test_set_evidence_clamps_to_unit_interval() -> None:
    state = _state()

    state.set_evidence(0.42)
    assert state.evidence_score == 0.42

    state.set_evidence(1.5)
    assert state.evidence_score == 1.0

    state.set_evidence(-0.3)
    assert state.evidence_score == 0.0


def test_set_citations_replaces_citations() -> None:
    state = _state()
    cites = [Citation(doc_id="a", marker="[1]", text="snippet")]

    state.set_citations(cites)

    assert state.citations == cites


def test_question_is_preserved_across_all_operations() -> None:
    state = _state("immutable?")

    state.record_search_round(["q"], [_doc("a")])
    state.record_rerank([_doc("a")])
    state.set_evidence(0.5)
    state.set_citations([Citation(doc_id="a", marker="[1]")])

    assert state.question == "immutable?"


def test_retriever_enum_has_web_and_vector_db() -> None:
    assert Retriever.WEB.value == "web"
    assert Retriever.VECTOR_DB.value == "vector_db"


def test_search_agent_state_is_not_exported() -> None:
    import src.agents.core.state as state_module

    assert not hasattr(state_module, "SearchAgentState")
    assert "SearchAgentState" not in state_module.__all__


def test_orchestration_vocabulary_stays_removed() -> None:
    """These types described a planning/routing layer no loop ever maintained.

    Each was reachable only through package re-exports, so nothing failed when
    they drifted from reality. Re-adding one is a claim that a loop writes it.
    """
    import src
    import src.agents.core.state as state_module

    removed = [
        "Plan",
        "PlanStep",
        "RetrievedDocument",
        "RouteDecision",
        "TaskNode",
        "TaskType",
        "ToolCall",
        "ToolResult",
        "ToolType",
    ]
    for name in removed:
        assert not hasattr(state_module, name), f"{name} came back to state.py"
        assert name not in state_module.__all__
        assert not hasattr(src, name), f"{name} is exported from src again"


def test_route_decision_cannot_resolve_to_the_unused_class() -> None:
    """`from src import RouteDecision` used to return the dead state.py class.

    The live one is src.internal.routing.route.RouteDecision. Two classes shared
    the name and the package-level export resolved to the one nothing wrote to.
    """
    import src
    from src.internal.routing.route import RouteDecision as LiveRouteDecision

    assert not hasattr(src, "RouteDecision")
    assert LiveRouteDecision.__module__ == "src.internal.routing.route"


def test_task_status_has_only_terminal_outcomes() -> None:
    """PENDING/RUNNING/RETRYING implied a scheduler and a retry path; neither exists."""
    from src.agents.core.state import TaskStatus

    assert {s.name for s in TaskStatus} == {"COMPLETED", "FAILED", "SKIPPED"}


def test_agent_state_has_no_unwritten_fields() -> None:
    """Every field must be one a loop or component actually writes."""
    import dataclasses

    from src.agents.core.state import AgentState

    assert {f.name for f in dataclasses.fields(AgentState)} == {
        "request_id",
        "user_request",
        "question",
        "previous_queries",
        "retrieved_docs",
        "evidence_score",
        "search_rounds",
        "citations",
    }
    for gone in ("record_trace", "add_tool_result"):
        assert not hasattr(AgentState, gone), f"{gone} was test-only; it stays removed"
