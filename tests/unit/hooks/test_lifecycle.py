"""Tests for the in-process lifecycle hook registry."""

from __future__ import annotations

import asyncio
import logging

import pytest

from src.internal.hooks.lifecycle import (
    LifecycleHooks,
    ToolCallContext,
    TurnContext,
)


def _ctx(**kw) -> TurnContext:
    return TurnContext(user_id="u", session_id="s", query="q", store=None, **kw)


@pytest.mark.asyncio
async def test_before_turn_joins_in_order_and_isolates_errors():
    h = LifecycleHooks()

    async def a(ctx):
        return "A"

    async def boom(ctx):
        raise RuntimeError("x")

    async def none(ctx):
        return None

    async def b(ctx):
        return "B"

    for f in (a, boom, none, b):
        h.on_before_turn(f)
    assert await h.before_turn(_ctx()) == "A\n\nB"


@pytest.mark.asyncio
async def test_before_turn_timeout_skips_handler():
    h = LifecycleHooks(timeout_seconds=0.05)

    async def slow(ctx):
        await asyncio.sleep(1)
        return "late"

    h.on_before_turn(slow)
    assert await h.before_turn(_ctx()) == ""


@pytest.mark.asyncio
async def test_after_turn_runs_in_background_and_swallows(caplog):
    seen = []
    h = LifecycleHooks()

    async def ok(ctx):
        seen.append(ctx.answer)

    async def boom(ctx):
        raise RuntimeError("after boom")

    h.on_after_turn(ok)
    h.on_after_turn(boom)
    with caplog.at_level(logging.WARNING):
        tasks = h.after_turn(_ctx(answer="done"))
        results = await asyncio.gather(*tasks, return_exceptions=True)
    assert seen == ["done"]
    assert all(r is None for r in results)
    assert any("after boom" in r.getMessage() or r.exc_info for r in caplog.records)


@pytest.mark.asyncio
async def test_after_turn_tasks_are_held_then_discarded():
    from src.internal.hooks import lifecycle

    h = LifecycleHooks()
    gate = asyncio.Event()

    async def wait(ctx):
        await gate.wait()

    h.on_after_turn(wait)
    (task,) = h.after_turn(_ctx())
    assert task in lifecycle._tasks
    gate.set()
    await task
    await asyncio.sleep(0)
    assert task not in lifecycle._tasks


@pytest.mark.asyncio
async def test_after_tool_call_receives_context_and_swallows():
    h = LifecycleHooks()
    seen: list[ToolCallContext] = []

    async def rec(ctx):
        seen.append(ctx)

    async def boom(ctx):
        raise RuntimeError("x")

    h.on_after_tool_call(boom)
    h.on_after_tool_call(rec)
    await h.after_tool_call(
        ToolCallContext(
            tool_name="t",
            arguments={"a": 1},
            result_summary="r" * 900,
            user_id="u",
            session_id="s",
        )
    )
    assert len(seen) == 1
    assert seen[0].tool_name == "t"
    assert seen[0].arguments == {"a": 1}
    assert len(seen[0].result_summary) == 500
