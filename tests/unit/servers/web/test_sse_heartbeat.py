"""A silent stream still has to put bytes on the wire.

A grounded answer can be quiet for a long time -- retrieval, then generation --
and a silent connection is one a proxy or load balancer is entitled to drop.
These pin that the keepalive fires, that it does not hurry the agent along, and
that it is invisible to every reader.
"""

from __future__ import annotations

import asyncio

import pytest

from src.internal.servers.sse import (
    SSE_COMMENT,
    _with_heartbeat,
    heartbeat_seconds,
    sse_frame,
)


async def _collect(generator) -> list[str]:
    return [frame async for frame in generator]


async def _silent_then(frames: list[str], quiet: float):
    await asyncio.sleep(quiet)
    for frame in frames:
        yield frame


@pytest.mark.asyncio
async def test_a_quiet_stream_emits_a_keepalive():
    body = _silent_then([sse_frame({"type": "answer"})], quiet=0.05)

    collected = await _collect(_with_heartbeat(body, interval=0.01))

    assert SSE_COMMENT in collected
    assert collected[-1] == sse_frame({"type": "answer"})


@pytest.mark.asyncio
async def test_a_busy_stream_emits_no_keepalive():
    async def chatty():
        for index in range(3):
            yield sse_frame({"type": "progress", "turn": index})

    collected = await _collect(_with_heartbeat(chatty(), interval=5))

    assert SSE_COMMENT not in collected
    assert len(collected) == 3


@pytest.mark.asyncio
async def test_the_keepalive_does_not_drop_the_event_it_waited_for():
    """The pending event is shielded: a heartbeat interrupts the wait, not the work."""
    body = _silent_then(
        [sse_frame({"type": "answer", "text": "kept"})],
        quiet=0.05,
    )

    collected = await _collect(_with_heartbeat(body, interval=0.01))

    payloads = [frame for frame in collected if frame != SSE_COMMENT]
    assert payloads == [sse_frame({"type": "answer", "text": "kept"})]


@pytest.mark.asyncio
async def test_an_empty_stream_terminates():
    async def nothing():
        return
        yield  # pragma: no cover - unreachable, makes this a generator

    assert await _collect(_with_heartbeat(nothing(), interval=0.01)) == []


def test_a_comment_frame_is_ignored_by_an_sse_reader():
    """Readers skip lines that are not `data:`; the browser's does the same."""
    assert SSE_COMMENT.startswith(":")
    assert SSE_COMMENT.endswith("\n\n")
    assert not any(
        line.strip().startswith("data:") for line in SSE_COMMENT.splitlines()
    )


def test_heartbeat_is_configurable_and_can_be_disabled(monkeypatch):
    from src.internal.configs.timeouts import reset_timeout_policies

    monkeypatch.setenv("AGENTIC_SEARCH_SSE_HEARTBEAT_SECONDS", "0")
    assert heartbeat_seconds() == 0

    # heartbeat_seconds() now reads the process-cached timeout policies, so a
    # second env change within the same test needs its own fresh read -- the
    # same reset the autouse fixture applies between tests.
    reset_timeout_policies()
    monkeypatch.setenv("AGENTIC_SEARCH_SSE_HEARTBEAT_SECONDS", "3.5")
    assert heartbeat_seconds() == 3.5


@pytest.mark.asyncio
async def test_a_client_that_stops_reading_closes_the_stream_underneath():
    """The risky path: the `finally` runs while GeneratorExit is in flight.

    A disconnecting client abandons the wrapper mid-yield, and an async
    generator that awaits carelessly during its own close either raises
    "ignored GeneratorExit" or wedges. Nothing else exercises this, because
    every other test drains to exhaustion.
    """
    closed = asyncio.Event()

    async def source():
        try:
            while True:
                yield sse_frame({"type": "progress"})
        finally:
            closed.set()

    wrapper = _with_heartbeat(source(), interval=5)
    async for _ in wrapper:
        break  # the client goes away after one frame
    await wrapper.aclose()

    assert closed.is_set(), "the underlying stream must be closed too"


@pytest.mark.asyncio
async def test_abandoning_a_quiet_stream_cancels_the_pending_event():
    """A heartbeat must not keep a dead client's work alive."""
    started = asyncio.Event()

    async def slow():
        started.set()
        await asyncio.sleep(30)
        yield sse_frame({"type": "answer"})  # pragma: no cover - never reached

    wrapper = _with_heartbeat(slow(), interval=0.01)
    frames = []
    async for frame in wrapper:
        frames.append(frame)
        break  # first heartbeat arrives, then the client leaves
    await wrapper.aclose()

    assert frames == [SSE_COMMENT]
    assert started.is_set()
