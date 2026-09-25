"""The one place Server-Sent Events are framed and returned.

Three endpoints stream SSE -- ``/api/agent/stream``, ``/chat/send-chat-message``
and ``/tool/send-tool-message`` -- and each used to carry its own copy of the
two-line frame encoder. Two of the three also forgot the anti-buffering headers,
which is the kind of omission a per-call-site copy invites: nginx buffers
proxied responses by default, and nothing in the local stack reproduces that,
so a missing header stays invisible until production.

Going through ``sse_response`` makes forgetting them impossible.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator

from fastapi.responses import StreamingResponse

from src.internal.configs.timeouts import get_timeout_policies

# ``no-cache`` keeps intermediaries from serving a stale stream; nginx reads
# ``X-Accel-Buffering: no`` as "pass this through as it arrives".
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


def sse_frame(data: dict) -> str:
    """Encode one event as an SSE frame.

    A frame is ``data: <json>`` closed by a blank line. The payload is kept to
    a single line -- ``json.dumps`` emits no newlines by default -- because a
    multi-line frame would need every line prefixed, and the browser-side
    reader does not reassemble those.
    """
    return f"data: {json.dumps(data)}\n\n"


# A comment frame: any line starting with ":" is ignored by every SSE reader,
# including the browser's EventSource and this repo's own readSSE. It exists to
# put bytes on an idle connection, nothing more.
SSE_COMMENT = ": keepalive\n\n"


def heartbeat_seconds() -> float:
    """How long a stream may be silent before it sends a keepalive.

    Default 15s, comfortably under the 30-60s idle timeouts intermediaries
    commonly apply. Zero disables it. Read from ``sse.heartbeat_seconds`` in
    the timeout policy file, overridable via
    ``AGENTIC_SEARCH_SSE_HEARTBEAT_SECONDS``.
    """
    return get_timeout_policies().sse.heartbeat_seconds


async def _with_heartbeat(
    generator: AsyncGenerator[str, None], interval: float
) -> AsyncGenerator[str, None]:
    """Emit a comment frame whenever the stream goes quiet for `interval`.

    A grounded answer can be silent for a long time -- retrieval, then
    generation -- and a silent connection is one a proxy or load balancer is
    entitled to drop. The agent is not hurried along: the pending event is
    shielded, so a heartbeat interrupts the wait, never the work.
    """
    iterator = generator.__aiter__()
    pending: asyncio.Future | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())
            try:
                frame = await asyncio.wait_for(asyncio.shield(pending), interval)
            except asyncio.TimeoutError:
                yield SSE_COMMENT  # still waiting; the future keeps running
                continue
            except StopAsyncIteration:
                pending = None
                return
            pending = None
            yield frame
    finally:
        # A client that leaves during a quiet stretch leaves `pending` mid-flight,
        # and closing a generator that is still running raises
        # "aclose(): asynchronous generator is already running". So cancel it and
        # *await* the cancellation: that unwinds the inner generator, after which
        # closing it is safe. Suppressing the RuntimeError instead would hide the
        # problem rather than fix it, and hide the next one too.
        if pending is not None and not pending.done():
            pending.cancel()
            with contextlib.suppress(BaseException):
                await pending
        await generator.aclose()


def sse_response(generator: AsyncGenerator[str, None]) -> StreamingResponse:
    """Return an SSE response that a buffering proxy cannot hold back."""
    interval = heartbeat_seconds()
    body = _with_heartbeat(generator, interval) if interval > 0 else generator
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers=dict(SSE_HEADERS),
    )
