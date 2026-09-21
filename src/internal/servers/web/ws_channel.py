"""The WebSocket control channel for agent sessions.

A transport, not a protocol of its own. It carries the same events
``/api/agent/stream`` already emits, driven by the same `AgentRunDriver`, and
resolves approvals through the same `ToolApprovalBroker` the HTTP endpoint
uses -- so the two transports cannot drift into different behaviour.

The socket does not own the run. Each run has a ``run_id`` and every event
carries it; the socket is that run's control channel. A v1 socket carries one
run at a time, which is a restriction on the transport rather than on the wire
format.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from collections.abc import Callable

from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from src.internal.servers.redis.redis_pool import retrieve_ws_token_data

logger = logging.getLogger(__name__)

# Close codes. 1008 is the RFC 6455 "policy violation" code, which is what a
# browser surfaces for a rejected handshake.
WS_CLOSE_UNAUTHENTICATED = 1008

CLIENT_EVENTS = frozenset(
    {"session.start", "approval.submit", "session.cancel", "ping"}
)


async def authenticate_ws(websocket: WebSocket) -> str | None:
    """Return the user id a single-use token belongs to, or None.

    The token is consumed atomically (GETDEL), so a replayed one authenticates
    nobody. Any Redis failure is a refusal, never a fallback: an in-process
    store would break single-use across workers.
    """
    token = websocket.query_params.get("token")
    if not token:
        return None
    try:
        data = await retrieve_ws_token_data(token)
    except Exception as exc:  # noqa: BLE001 - Redis down, any driver error
        logger.warning("ws auth failed: %s", exc)
        return None
    if not data:
        return None
    user_id = data.get("sub")
    return str(user_id) if user_id else None


class WsSession:
    """One authenticated socket and the single run it may be carrying."""

    def __init__(self, websocket: WebSocket, user_id: str) -> None:
        self._ws = websocket
        self.user_id = user_id
        self._run: "RunHandle | None" = None

    async def send(self, event: dict) -> None:
        await self._ws.send_json(event)

    async def error(self, detail: str, *, code: int | None = None) -> None:
        event: dict = {"type": "error", "detail": detail}
        if code is not None:
            event["code"] = code
        await self.send(event)


class RunHandle:
    """A run in flight, and the handle the socket cancels it by."""

    def __init__(self, run_id: str, task) -> None:
        self.run_id = run_id
        self.task = task

    def cancel(self) -> None:
        if not self.task.done():
            self.task.cancel()


async def serve(
    websocket: WebSocket,
    *,
    user_id: str,
    start_run: Callable[[WsSession, dict], Awaitable[None]],
    submit_approval: Callable[[WsSession, dict], Awaitable[None]],
) -> None:
    """Dispatch client events on an already-authenticated socket.

    Authentication happens in the route, not here, so that it is visible both
    to a reader and to the startup route audit, which scans an endpoint's own
    body for a recognised guard.

    `start_run` and `submit_approval` are injected because they need the app's
    broker, store and agent entry point; keeping them out of here is what lets
    this module be tested without standing up the whole application.
    """
    await websocket.accept()
    session = WsSession(websocket, user_id)
    try:
        while True:
            try:
                message = await websocket.receive_json()
            except WebSocketDisconnect:
                raise
            except Exception:  # noqa: BLE001 - malformed frame, not fatal
                await session.error("Malformed event", code=400)
                continue

            if not isinstance(message, dict):
                await session.error("Malformed event", code=400)
                continue

            event_type = message.get("type")
            if event_type not in CLIENT_EVENTS:
                # Unknown is an error, never a disconnect: a client that
                # speaks a newer vocabulary should degrade, not drop.
                await session.error(f"Unknown event: {event_type!r}", code=400)
                continue

            if event_type == "ping":
                await session.send({"type": "pong"})
            elif event_type == "session.start":
                await start_run(session, message)
            elif event_type == "approval.submit":
                await submit_approval(session, message)
            elif event_type == "session.cancel":
                await _cancel(session)
    except WebSocketDisconnect:
        pass
    finally:
        # A disconnect mid-run must not leave the agent running for nobody.
        if session._run is not None:
            session._run.cancel()


async def _cancel(session: WsSession) -> None:
    run = session._run
    if run is None:
        await session.error("No run in flight", code=409)
        return
    run.cancel()
    session._run = None
    await session.send({"type": "done", "run_id": run.run_id, "cancelled": True})
