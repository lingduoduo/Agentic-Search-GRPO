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

import json
from collections.abc import AsyncGenerator

from fastapi.responses import StreamingResponse

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


def sse_response(generator: AsyncGenerator[str, None]) -> StreamingResponse:
    """Return an SSE response that a buffering proxy cannot hold back."""
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers=dict(SSE_HEADERS),
    )
