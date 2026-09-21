"""Every SSE endpoint must defeat proxy buffering.

nginx buffers proxied responses by default, so an SSE response without
``X-Accel-Buffering: no`` is delivered to the client in one lump the moment the
generator finishes -- which is exactly not streaming. Nothing in the local
stack reproduces that: Vite's dev proxy does not buffer and ``TestClient`` does
not proxy at all, so the defect is invisible until production.

This walks the source instead of the wire, so an SSE endpoint added later is
covered the day it is written rather than the day someone remembers to test it.
"""

from __future__ import annotations

import ast
from pathlib import Path

REQUIRED_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

SRC = Path("src")


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _is_sse(call: ast.Call) -> bool:
    media_type = _keyword(call, "media_type")
    return isinstance(media_type, ast.Constant) and media_type.value == (
        "text/event-stream"
    )


def _headers_of(call: ast.Call) -> dict[str, str]:
    headers = _keyword(call, "headers")
    if not isinstance(headers, ast.Dict):
        return {}
    found: dict[str, str] = {}
    for key, value in zip(headers.keys, headers.values):
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
            found[str(key.value)] = str(value.value)
    return found


def _sse_responses() -> list[tuple[str, int, dict[str, str]]]:
    """Every StreamingResponse(media_type="text/event-stream") under src/."""
    found = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            name = (
                callee.attr
                if isinstance(callee, ast.Attribute)
                else (callee.id if isinstance(callee, ast.Name) else None)
            )
            if name == "StreamingResponse" and _is_sse(node):
                found.append((str(path), node.lineno, _headers_of(node)))
    return found


def test_the_sweep_finds_the_known_sse_endpoints():
    """Guard the guard: if this drops to zero the invariant below is vacuous."""
    paths = {path for path, _, _ in _sse_responses()}

    assert "src/internal/servers/web/app.py" in paths
    assert "src/internal/servers/query_and_chat/chat_backend.py" in paths
    assert "src/internal/servers/query_and_chat/tool_backend.py" in paths


def test_every_sse_response_defeats_proxy_buffering():
    offenders = []
    for path, lineno, headers in _sse_responses():
        missing = {
            key for key, want in REQUIRED_HEADERS.items() if headers.get(key) != want
        }
        if missing:
            offenders.append(f"{path}:{lineno} missing {sorted(missing)}")

    assert not offenders, "SSE responses without anti-buffering headers: " + "; ".join(
        offenders
    )
