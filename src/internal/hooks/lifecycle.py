"""In-process lifecycle hooks around an agent turn.

Distinct from ``executor.py`` (external HTTP webhooks): handlers here are
async callables registered in the same process. A handler failure or timeout
never fails the turn; it is logged and that handler is skipped.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_SUMMARY_LIMIT = 500

# Strong references so fire-and-forget after_turn tasks are not garbage
# collected mid-flight; each task removes itself on completion.
_tasks: set[asyncio.Task] = set()


@dataclass
class TurnContext:
    user_id: str | None
    session_id: str | None
    query: str
    store: Any
    llm: Any = None
    encoder: Any = None
    route: str | None = None
    answer: str | None = None


@dataclass(frozen=True)
class ToolCallContext:
    tool_name: str
    arguments: dict
    result_summary: str
    user_id: str | None
    session_id: str | None

    def __post_init__(self) -> None:
        if len(self.result_summary) > _SUMMARY_LIMIT:
            object.__setattr__(
                self, "result_summary", self.result_summary[:_SUMMARY_LIMIT]
            )


BeforeTurnHandler = Callable[[TurnContext], Awaitable["str | None"]]
AfterTurnHandler = Callable[[TurnContext], Awaitable[None]]
AfterToolCallHandler = Callable[[ToolCallContext], Awaitable[None]]


def _name(handler: Callable) -> str:
    return getattr(handler, "__qualname__", repr(handler))


class LifecycleHooks:
    def __init__(self, *, timeout_seconds: float = 2.0) -> None:
        self._timeout = timeout_seconds
        self._before_turn: list[BeforeTurnHandler] = []
        self._after_turn: list[AfterTurnHandler] = []
        self._after_tool_call: list[AfterToolCallHandler] = []

    def on_before_turn(self, h: BeforeTurnHandler) -> None:
        self._before_turn.append(h)

    def on_after_turn(self, h: AfterTurnHandler) -> None:
        self._after_turn.append(h)

    def on_after_tool_call(self, h: AfterToolCallHandler) -> None:
        self._after_tool_call.append(h)

    async def _guarded(self, handler: Callable, ctx: Any, *, bounded: bool) -> Any:
        """Run ``handler(ctx)``; return None on failure or timeout."""
        try:
            coro = handler(ctx)
            if bounded:
                return await asyncio.wait_for(coro, self._timeout)
            return await coro
        except asyncio.TimeoutError:
            logger.warning("lifecycle hook %s timed out", _name(handler))
        except Exception:  # noqa: BLE001 - a hook never fails the turn
            logger.warning("lifecycle hook %s failed", _name(handler), exc_info=True)
        return None

    async def before_turn(self, ctx: TurnContext) -> str:
        fragments: list[str] = []
        for handler in self._before_turn:
            fragment = await self._guarded(handler, ctx, bounded=True)
            if fragment:
                fragments.append(fragment)
        return "\n\n".join(fragments)

    def after_turn(self, ctx: TurnContext) -> list[asyncio.Task]:
        """Schedule each handler as a background task (needs a running loop)."""
        tasks: list[asyncio.Task] = []
        for handler in self._after_turn:
            task = asyncio.create_task(self._guarded(handler, ctx, bounded=False))
            _tasks.add(task)
            task.add_done_callback(_tasks.discard)
            tasks.append(task)
        return tasks

    async def after_tool_call(self, ctx: ToolCallContext) -> None:
        for handler in self._after_tool_call:
            await self._guarded(handler, ctx, bounded=True)
