"""Retry → degrade → escalate: the per-call recovery decision for tool failures.

Pure policy, no I/O: ``ToolAgentLoop`` owns the state and applies the decision.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.internal.tools import FailureCategory, ToolEffect, ToolFailure


class Action(Enum):
    RETRY = "retry"
    UNAVAILABLE = "unavailable"
    ESCALATE = "escalate"
    FEED_BACK = "feed_back"


@dataclass(frozen=True)
class Decision:
    action: Action
    delay: float = 0.0


@dataclass(frozen=True)
class RecoveryPolicy:
    max_retries: int = 2
    backoff: tuple[float, ...] = (0.5, 1.0)
    retry_after_cap: float = 4.0
    retry_budget: float = 10.0

    def decide(
        self,
        failure: ToolFailure,
        effect: ToolEffect,
        retries_so_far: int,
        budget_left: float,
        uniform: Callable[[float, float], float] = random.uniform,
    ) -> Decision:
        if failure.category in (
            FailureCategory.INVALID_INPUT,
            FailureCategory.NOT_FOUND,
        ):
            return Decision(Action.FEED_BACK)
        if effect is not ToolEffect.READ_ONLY:
            # Replaying might repeat an action; only the user may authorise it.
            return Decision(Action.ESCALATE)
        if (
            failure.category is FailureCategory.TRANSIENT
            # The provider already retried: retrying again would multiply attempts.
            and failure.provider_attempts <= 1
            and retries_so_far < self.max_retries
        ):
            step = self.backoff[min(retries_so_far, len(self.backoff) - 1)]
            delay = max(
                step * uniform(0.5, 1.5),
                min(failure.retry_after or 0.0, self.retry_after_cap),
            )
            if delay <= budget_left:
                return Decision(Action.RETRY, delay)
        return Decision(Action.UNAVAILABLE)


@dataclass
class RecoveryState:
    """Per-run recovery bookkeeping, shared by the calls of one run."""

    budget_left: float
    unavailable: set[str] = field(default_factory=set)
    retries: int = 0
    degraded: list[str] = field(default_factory=list)
    escalations: list[dict[str, Any]] = field(default_factory=list)
    escalation_count: int = 0
    outcome: str | None = None  # None | "cancelled" | "unresolved"
    stop_answer: str | None = None

    def mark_unavailable(self, tool_name: str) -> None:
        if tool_name not in self.unavailable:
            self.unavailable.add(tool_name)
            self.degraded.append(tool_name)

    def stop(self, outcome: str, answer: str) -> None:
        if self.stop_answer is None:  # the first stop wins
            self.outcome, self.stop_answer = outcome, answer

    def summary(self) -> dict[str, Any] | None:
        if not (self.retries or self.degraded or self.escalations):
            return None
        outcome = self.outcome or ("degraded" if self.degraded else "recovered")
        return {
            "outcome": outcome,
            "needs_user": outcome == "unresolved",
            "retries": self.retries,
            "degraded": list(self.degraded),
            "escalations": list(self.escalations),
        }
