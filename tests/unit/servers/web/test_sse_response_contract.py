"""Every SSE endpoint must go through the one helper that sets the headers.

nginx buffers proxied responses by default, so an SSE response without
``X-Accel-Buffering: no`` is delivered in one lump the moment the generator
finishes -- which is exactly not streaming. Nothing local reproduces that:
Vite's dev proxy does not buffer and ``TestClient`` does not proxy at all, so
the defect stays invisible until production. Two of the three endpoints shipped
without those headers for exactly that reason.

So the invariant is structural rather than per-endpoint: build the response
anywhere but ``sse.py`` and this fails, whether or not you remembered the
headers.
"""

from __future__ import annotations

import ast
from pathlib import Path

from src.internal.servers.sse import SSE_HEADERS, sse_frame, sse_response

SRC = Path("src")
HELPER = "src/internal/servers/sse.py"
SSE_ENDPOINT_MODULES = (
    "src/internal/servers/web/app.py",
    "src/internal/servers/query_and_chat/chat_backend.py",
    "src/internal/servers/query_and_chat/tool_backend.py",
)


def _callee_name(call: ast.Call) -> str | None:
    callee = call.func
    if isinstance(callee, ast.Attribute):
        return callee.attr
    if isinstance(callee, ast.Name):
        return callee.id
    return None


def _event_stream_constructions() -> list[str]:
    """Files under src/ that build a text/event-stream StreamingResponse."""
    found = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _callee_name(node) != "StreamingResponse":
                continue
            for kw in node.keywords:
                if (
                    kw.arg == "media_type"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value == "text/event-stream"
                ):
                    found.append(str(path))
    return found


def test_only_the_sse_helper_builds_event_stream_responses():
    offenders = sorted(set(_event_stream_constructions()) - {HELPER})

    assert not offenders, (
        "these build an SSE response directly instead of calling sse_response(): "
        + ", ".join(offenders)
    )


def test_the_helper_itself_is_still_the_one_that_builds_it():
    """Guard the guard: an invariant over an empty set passes for free."""
    assert HELPER in _event_stream_constructions()


def test_every_sse_endpoint_routes_through_the_helper():
    """And guard against satisfying the invariant by simply not streaming."""
    for module in SSE_ENDPOINT_MODULES:
        assert "sse_response(" in Path(module).read_text(), module


def test_sse_headers_are_the_ones_that_defeat_buffering():
    assert SSE_HEADERS["Cache-Control"] == "no-cache"
    assert SSE_HEADERS["X-Accel-Buffering"] == "no"


def test_sse_response_actually_carries_the_headers():
    """Declaring them in a constant is not the same as wiring them."""
    response = sse_response(iter([]))

    assert response.media_type == "text/event-stream"
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"


def test_sse_frame_is_one_data_line_closed_by_a_blank_line():
    frame = sse_frame({"type": "answer", "text": "hi"})

    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    assert frame.count("\n") == 2  # no newlines inside the payload
