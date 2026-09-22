from src.agents.components.loop_controller import (
    AnswerVerb,
    LoopController,
    LoopSnapshot,
    StopReason,
)
from src.agents.search import SearchAgentLoopConfig


def _ctl(**over):
    return LoopController(SearchAgentLoopConfig(**over))


def _snap(**over):
    base = dict(
        rounds_used=1,
        num_subquestions=1,
        evidence_sufficient=True,
        prev_evidence_score=0.5,
        curr_evidence_score=0.9,
        consecutive_rejections=0,
    )
    base.update(over)
    return LoopSnapshot(**base)


def test_effective_limit_single_subquestion_unchanged():
    ctl = _ctl(
        max_search_limit=4, search_budget_per_subquestion=1, max_search_limit_cap=10
    )
    assert ctl.effective_search_limit(1) == 4
    assert ctl.effective_search_limit(0) == 4


def test_effective_limit_scales_then_caps():
    ctl = _ctl(
        max_search_limit=4, search_budget_per_subquestion=1, max_search_limit_cap=6
    )
    assert ctl.effective_search_limit(3) == 6  # 4 + (3-1)=6
    assert ctl.effective_search_limit(20) == 6  # capped


def test_continue_while_evidence_climbing():
    ctl = _ctl(max_search_limit=5, evidence_plateau_min_gain=0.05)
    d = ctl.should_continue_searching(
        _snap(prev_evidence_score=0.2, curr_evidence_score=0.9)
    )
    assert d.reason is StopReason.CONTINUE


def test_stop_budget_exhausted():
    ctl = _ctl(max_search_limit=2, max_search_limit_cap=10)
    d = ctl.should_continue_searching(_snap(rounds_used=2, num_subquestions=1))
    assert d.reason is StopReason.BUDGET_EXHAUSTED


def test_plateau_stops_only_when_sufficient():
    ctl = _ctl(
        max_search_limit=5,
        evidence_plateau_min_gain=0.05,
        plateau_requires_sufficient=True,
    )
    stalled = dict(
        prev_evidence_score=0.80, curr_evidence_score=0.82
    )  # gain 0.02 < 0.05
    assert (
        ctl.should_continue_searching(_snap(evidence_sufficient=True, **stalled)).reason
        is StopReason.PLATEAU
    )
    assert (
        ctl.should_continue_searching(
            _snap(evidence_sufficient=False, **stalled)
        ).reason
        is StopReason.CONTINUE
    )


def test_accept_when_sufficient():
    ctl = _ctl(require_sufficient_evidence_before_answer=True, max_answer_rejections=3)
    assert (
        ctl.final_answer_decision(_snap(evidence_sufficient=True)).verb
        is AnswerVerb.ACCEPT
    )


def test_reject_then_force_at_cap():
    ctl = _ctl(require_sufficient_evidence_before_answer=True, max_answer_rejections=3)
    assert (
        ctl.final_answer_decision(
            _snap(evidence_sufficient=False, consecutive_rejections=1)
        ).verb
        is AnswerVerb.REJECT
    )
    assert (
        ctl.final_answer_decision(
            _snap(evidence_sufficient=False, consecutive_rejections=3)
        ).verb
        is AnswerVerb.FORCE
    )


def test_accept_when_gate_disabled():
    ctl = _ctl(require_sufficient_evidence_before_answer=False)
    assert (
        ctl.final_answer_decision(_snap(evidence_sufficient=False)).verb
        is AnswerVerb.ACCEPT
    )


def test_none_gain_disables_plateau():
    """With evidence_plateau_min_gain=None, should_continue_searching returns CONTINUE
    even when evidence is sufficient and stalled (below budget)."""
    ctl = _ctl(max_search_limit=5, evidence_plateau_min_gain=None)
    stalled = dict(prev_evidence_score=0.80, curr_evidence_score=0.82)  # gain 0.02
    d = ctl.should_continue_searching(
        _snap(rounds_used=1, evidence_sufficient=True, **stalled)
    )
    assert d.reason is StopReason.CONTINUE


def test_effective_limit_falls_back_to_max_turns_when_limit_none():
    """With max_search_limit=None, effective_search_limit falls back to max_turns."""
    ctl = _ctl(max_search_limit=None, max_turns=3, max_search_limit_cap=10)
    assert ctl.effective_search_limit(1) == 3


def test_snapshot_carries_no_field_the_controller_cannot_read() -> None:
    """``model_emitted_answer`` was structurally redundant, not merely unread.

    It was set True immediately before every ``final_answer_decision`` call and
    False immediately before every ``should_continue_searching`` call. Each method
    is reachable from exactly one of those sites, so each only ever saw one
    possible value: the flag could not disambiguate anything, and a "Phase 2
    state machine" reading it would be reading which method it already is.

    A field on a snapshot is a claim that a decision depends on it.
    """
    import dataclasses

    fields = {f.name for f in dataclasses.fields(LoopSnapshot)}
    assert "model_emitted_answer" not in fields
    assert fields == {
        "rounds_used",
        "num_subquestions",
        "evidence_sufficient",
        "prev_evidence_score",
        "curr_evidence_score",
        "consecutive_rejections",
    }


def test_both_controller_decisions_still_work_without_it() -> None:
    ctl = _ctl(max_search_limit=5, max_answer_rejections=3)

    assert (
        ctl.should_continue_searching(
            _snap(prev_evidence_score=0.2, curr_evidence_score=0.9)
        ).reason
        is StopReason.CONTINUE
    )
    assert ctl.final_answer_decision(_snap(evidence_sufficient=True)).verb is (
        AnswerVerb.ACCEPT
    )
