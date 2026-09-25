"""Run-scoped decision accounting, independent of HTTP and output turn fields."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from functools import wraps
from typing import Awaitable, Callable, ParamSpec, TypeVar

from src.internal.observability.prometheus import observe_agent_run

_P = ParamSpec("_P")
_T = TypeVar("_T")
# Immutable values prevent a child task from changing its parent's count.
_ROUNDS: ContextVar[int | None] = ContextVar("agent_decision_rounds", default=None)


def record_decision_round() -> None:
    """Call immediately before an agent attempts a model generation."""
    rounds = _ROUNDS.get()
    if rounds is not None:
        _ROUNDS.set(rounds + 1)


def track_agent_run(
    agent: str,
) -> Callable[[Callable[_P, Awaitable[_T]]], Callable[_P, Awaitable[_T]]]:
    """Observe exactly once and restore the surrounding run, even on failure."""
    if agent not in {"search", "tool"}:
        raise ValueError("Unknown agent")

    def decorate(fn: Callable[_P, Awaitable[_T]]) -> Callable[_P, Awaitable[_T]]:
        @wraps(fn)
        async def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            token = _ROUNDS.set(0)
            outcome = "error"
            try:
                result = await fn(*args, **kwargs)
                outcome = "completed"
                return result
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            finally:
                rounds = _ROUNDS.get()
                _ROUNDS.reset(token)
                observe_agent_run(agent, outcome, rounds if rounds is not None else 0)

        return wrapped

    return decorate
