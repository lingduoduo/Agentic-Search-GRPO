"""Memory handlers for the in-process lifecycle hooks."""

from __future__ import annotations

import asyncio

from src.internal.hooks.lifecycle import LifecycleHooks, TurnContext
from src.internal.memory.service import memory_preamble


async def recall_handler(ctx: TurnContext) -> str | None:
    """Return the user's memory preamble; anonymous callers have none."""
    if ctx.user_id is None:
        return None
    return await asyncio.to_thread(
        memory_preamble,
        ctx.store,
        ctx.user_id,
        query=ctx.query,
        encoder=ctx.encoder,
    )


def register_memory_hooks(hooks: LifecycleHooks, settings) -> None:
    hooks.on_before_turn(recall_handler)
