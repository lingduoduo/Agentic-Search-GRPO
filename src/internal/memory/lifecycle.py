"""Memory handlers for the in-process lifecycle hooks."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading

from src.internal.hooks.lifecycle import LifecycleHooks, TurnContext
from src.internal.memory import consolidation
from src.internal.memory.capture import capture_turn
from src.internal.memory.service import memory_preamble

logger = logging.getLogger(__name__)

_busy: set[str] = set()
_busy_lock = threading.Lock()

# after_turn starts capture and consolidation together, and either can hold the
# user's guard for seconds of LLM calls, so each waits for it (bounded) instead
# of skipping whenever the other is running.
CONSOLIDATION_WAIT_SECONDS = 30.0
_GUARD_POLL_SECONDS = 0.5


@contextlib.contextmanager
def user_guard(user_id: str):
    """Process-local, thread-safe per-user mutex; yields False if already busy."""
    with _busy_lock:
        acquired = user_id not in _busy
        if acquired:
            _busy.add(user_id)
    try:
        yield acquired
    finally:
        if acquired:
            with _busy_lock:
                _busy.discard(user_id)


@contextlib.asynccontextmanager
async def acquire_user_guard(user_id: str, *, handler: str):
    """Hold ``user_guard``, polling up to CONSOLIDATION_WAIT_SECONDS for it.

    Yields False (after logging a warning) when the wait expires.
    """
    waited = 0.0
    while True:
        with user_guard(user_id) as free:
            if free:
                yield True
                return
        if waited >= CONSOLIDATION_WAIT_SECONDS:
            logger.warning(
                "%s skipped: memory busy for user %s after %.1fs",
                handler,
                user_id,
                waited,
            )
            yield False
            return
        await asyncio.sleep(_GUARD_POLL_SECONDS)
        waited += _GUARD_POLL_SECONDS


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


async def capture_handler(ctx: TurnContext) -> None:
    """Capture durable memories from a finished turn (background, never raises)."""
    if not ctx.user_id or ctx.llm is None or not ctx.answer:
        return
    async with acquire_user_guard(ctx.user_id, handler="capture_handler") as free:
        if not free:
            return
        try:
            await capture_turn(
                ctx.store,
                ctx.user_id,
                ctx.session_id,
                ctx.llm,
                ctx.query,
                ctx.answer,
                encoder=ctx.encoder,
            )
        except Exception:  # noqa: BLE001 - memory never fails a request
            logger.warning("per-turn memory capture failed", exc_info=True)


async def consolidation_handler(ctx: TurnContext) -> None:
    """Consolidate or refresh index summaries in the background (never raises)."""
    if ctx.user_id is None:
        return
    async with acquire_user_guard(ctx.user_id, handler="consolidation_handler") as free:
        if not free:
            return
        try:
            await asyncio.to_thread(
                consolidation.run_due,
                ctx.store,
                ctx.user_id,
                llm=ctx.llm,
                encoder=ctx.encoder,
            )
        except Exception:  # noqa: BLE001 - memory never fails a request
            logger.warning("memory consolidation failed", exc_info=True)


def register_memory_hooks(hooks: LifecycleHooks, settings) -> None:
    hooks.on_before_turn(recall_handler)
    if getattr(settings, "memory_auto_curate", False):
        hooks.on_after_turn(capture_handler)
    # Memory also grows via /api/memory/save and curation, so this does not
    # depend on the capture flag.
    hooks.on_after_turn(consolidation_handler)
