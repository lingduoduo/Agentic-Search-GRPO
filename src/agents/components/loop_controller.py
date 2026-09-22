"""LoopController: the search loop's two control decisions, pure over a snapshot.

Owns no mutable state (only config). ``SearchAgentLoop.run`` builds a
``LoopSnapshot`` of the relevant loop state and consults the controller, keeping
all mutable state in the loop itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class LoopSnapshot:
    rounds_used: int
    num_subquestions: int
    evidence_sufficient: bool
    prev_evidence_score: float
    curr_evidence_score: float
    consecutive_rejections: int
    model_emitted_answer: bool  # reserved for Phase 2 state machine; not yet read


class StopReason(Enum):
    CONTINUE = "continue"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PLATEAU = "plateau"


class AnswerVerb(Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    FORCE = "force"


@dataclass(frozen=True)
class StopDecision:
    reason: StopReason


@dataclass(frozen=True)
class AnswerDecision:
    verb: AnswerVerb
    feedback: str = ""


class LoopController:
    """Stateless policy for the loop's keep-searching / how-to-answer decisions."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg

    def effective_search_limit(self, num_subquestions: int) -> int:
        cfg = self._cfg
        base = cfg.max_search_limit or cfg.max_turns
        bonus = cfg.search_budget_per_subquestion * max(0, num_subquestions - 1)
        return max(base, min(base + bonus, cfg.max_search_limit_cap))

    def should_continue_searching(self, s: LoopSnapshot) -> StopDecision:
        """Decide whether another search round is worth running.

        The budget check comes first on purpose, and ``SearchAgentLoop`` acts
        only on ``PLATEAU``. Together that means a round *at* the budget
        proceeds normally and gets its evidence injected, instead of
        early-stopping on a plateau it also satisfies — the last round's
        documents would otherwise never reach the model that must answer from
        them. ``BUDGET_EXHAUSTED`` therefore reads as "stop searching, the
        caller already knows": enforcement itself lives in
        ``Planner.partition_search_requests``, which refuses the queries.
        Removing this arm looks like dead-code cleanup and is not;
        ``test_the_final_round_within_budget_still_reaches_the_model`` pins it.
        """
        cfg = self._cfg
        if s.rounds_used >= self.effective_search_limit(s.num_subquestions):
            return StopDecision(StopReason.BUDGET_EXHAUSTED)
        if cfg.evidence_plateau_min_gain is not None:
            gain = s.curr_evidence_score - s.prev_evidence_score
            if gain < cfg.evidence_plateau_min_gain and (
                s.evidence_sufficient or not cfg.plateau_requires_sufficient
            ):
                return StopDecision(StopReason.PLATEAU)
        return StopDecision(StopReason.CONTINUE)

    _REJECT_FEEDBACK = (
        "Evidence is still insufficient for the question. Issue another search "
        "to gather more evidence before answering."
    )
    FORCE_FEEDBACK = (
        "You cannot gather more evidence (budget reached). Give your best answer "
        "now, grounded only in the evidence already collected. State explicitly "
        "what remains uncertain, and cite evidence labels."
    )

    def final_answer_decision(self, s: LoopSnapshot) -> AnswerDecision:
        cfg = self._cfg
        if s.evidence_sufficient or not cfg.require_sufficient_evidence_before_answer:
            return AnswerDecision(AnswerVerb.ACCEPT)
        if s.consecutive_rejections >= cfg.max_answer_rejections:
            return AnswerDecision(AnswerVerb.FORCE, feedback=self.FORCE_FEEDBACK)
        return AnswerDecision(AnswerVerb.REJECT, feedback=self._REJECT_FEEDBACK)
