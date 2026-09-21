"""One agent run, independent of the transport carrying it.

The run loop behind ``/api/agent/stream`` was already transport-neutral without
being separable: its callbacks push plain dicts into a bounded queue, and SSE
framing is applied only at the point of yield. This is that loop lifted out of
the endpoint closure so a second transport can drive it without copying it --
the copy being what would let the two drift apart.

The driver owns the run: its id, its queue, its backpressure policy and its
task's lifecycle. A transport owns the socket or the response, and the two
lifetimes are deliberately not tied together.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from collections.abc import Callable
from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Bounded so a slow client cannot make the agent's memory grow without limit.
# Overflow drops trace and claim events rather than back-pressuring generation:
# the terminal answer still carries the full text, so a dropped claim costs
# liveness, not data.
MAX_QUEUED_EVENTS = 100

# How long to wait on the queue before re-checking whether the run finished.
_DRAIN_POLL_SECONDS = 0.05


class AgentRunDriver:
    """Drives one agent run and yields its events as plain dicts."""

    def __init__(self, run_id: str, *, max_queued: int = MAX_QUEUED_EVENTS) -> None:
        self.run_id = run_id
        self.queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=max_queued)
        self.dropped_trace_events = 0
        self.dropped_claim_events = 0
        self._loop = asyncio.get_running_loop()

    # -- callbacks handed to the agent loop --------------------------------

    async def on_turn(self, turn: int, tool_name: str | None, doc_count: int) -> None:
        text = f"{tool_name} · {doc_count} docs" if tool_name else "writing answer..."
        await self.queue.put({"type": "progress", "turn": turn, "text": text})

    def on_claim(self, text: str) -> None:
        """Publish a verified claim.

        Called from the generate_answer worker thread (AgenticRAGLoop.run
        offloads with asyncio.to_thread), so hop back to the loop before
        touching the queue -- put_nowait is not thread-safe.
        """
        self._loop.call_soon_threadsafe(
            self._offer_claim, {"type": "claim", "text": text}
        )

    async def on_trace(self, event: dict) -> None:
        try:
            self.queue.put_nowait({"type": "trace", "event": event})
        except asyncio.QueueFull:
            self.dropped_trace_events += 1

    def _offer_claim(self, item: dict) -> None:
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            self.dropped_claim_events += 1

    # -- the run -----------------------------------------------------------

    async def run(
        self,
        coro,
        *,
        finalize: Callable[[object], Iterable[dict]],
        on_error: Callable[[BaseException], dict],
    ) -> AsyncGenerator[dict, None]:
        """Drive `coro`, yielding queued events, then whatever finalize returns.

        `finalize` turns the run's result into its terminal events, and
        `on_error` turns a failure into one. Both stay with the caller because
        they speak in the caller's payload types; everything mechanical --
        draining, cancellation, cleanup, drop accounting -- lives here, so both
        transports inherit identical behaviour.
        """
        task = asyncio.create_task(coro)
        try:
            while not task.done():
                try:
                    item = await asyncio.wait_for(
                        self.queue.get(), timeout=_DRAIN_POLL_SECONDS
                    )
                    yield item
                except asyncio.TimeoutError:
                    continue
            while not self.queue.empty():
                yield self.queue.get_nowait()

            for event in finalize(task.result()):
                yield event
            self._log_drops()
        except BaseException as exc:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            if isinstance(exc, asyncio.CancelledError):
                return  # client disconnected
            yield on_error(exc)

    def _log_drops(self) -> None:
        if self.dropped_trace_events:
            logger.warning(
                "dropped %d live control-flow trace events", self.dropped_trace_events
            )
        if self.dropped_claim_events:
            logger.warning("dropped %d live claim events", self.dropped_claim_events)
