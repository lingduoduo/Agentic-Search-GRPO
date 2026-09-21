"""The socket's lifecycle contract, tested at `serve()` rather than over HTTP.

Cancellation and leak-freedom are properties of the dispatcher, and asserting
them needs the server's own event loop -- which `TestClient` runs in another
thread, where `asyncio.all_tasks()` cannot see it. Driving `serve()` directly
with a fake socket keeps those assertions exact.
"""

from __future__ import annotations

import asyncio

import pytest
from starlette.websockets import WebSocketDisconnect

from src.internal.servers.web.ws_channel import RunHandle, WsSession, serve


class FakeWebSocket:
    """Replays a scripted client, and records what the server sent back."""

    def __init__(self, incoming: list) -> None:
        self._incoming = list(incoming)
        self.sent: list[dict] = []
        self.accepted = False
        self.close_code: int | None = None

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)

    async def receive_json(self):
        if not self._incoming:
            raise WebSocketDisconnect(1000)
        item = self._incoming.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self, code: int | None = None) -> None:
        self.close_code = code

    def types(self) -> list[str]:
        return [event["type"] for event in self.sent]


async def _noop(session, message) -> None:  # pragma: no cover - default stub
    pass


async def _serve(incoming: list, **kwargs) -> FakeWebSocket:
    websocket = FakeWebSocket(incoming)
    await serve(
        websocket,
        user_id="user-1",
        start_run=kwargs.get("start_run", _noop),
        submit_approval=kwargs.get("submit_approval", _noop),
    )
    return websocket


@pytest.mark.asyncio
async def test_ping_is_answered_and_the_socket_is_accepted():
    websocket = await _serve([{"type": "ping"}])

    assert websocket.accepted
    assert websocket.types() == ["pong"]


@pytest.mark.asyncio
async def test_unknown_event_does_not_end_the_session():
    websocket = await _serve([{"type": "session.update"}, {"type": "ping"}])

    assert websocket.types() == ["error", "pong"]
    assert websocket.sent[0]["code"] == 400


@pytest.mark.asyncio
async def test_a_non_object_frame_is_an_error():
    websocket = await _serve([["not", "an", "object"], {"type": "ping"}])

    assert websocket.types() == ["error", "pong"]


@pytest.mark.asyncio
async def test_cancel_stops_a_run_in_flight_and_reports_it():
    started = asyncio.Event()

    async def forever() -> None:
        started.set()
        await asyncio.Event().wait()

    async def start_run(session: WsSession, message: dict) -> None:
        task = asyncio.create_task(forever())
        await started.wait()
        session._run = RunHandle("run-1", task)
        await session.send({"type": "session.started", "run_id": "run-1"})

    websocket = await _serve(
        [{"type": "session.start"}, {"type": "session.cancel"}],
        start_run=start_run,
    )

    assert websocket.types() == ["session.started", "done"]
    assert websocket.sent[-1]["cancelled"] is True
    assert websocket.sent[-1]["run_id"] == "run-1"


@pytest.mark.asyncio
async def test_a_cancelled_run_leaves_no_task_behind():
    """Criterion 12: cancelling must not orphan the agent's task."""
    task_box: dict[str, asyncio.Task] = {}

    async def forever() -> None:
        await asyncio.Event().wait()

    async def start_run(session: WsSession, message: dict) -> None:
        task = asyncio.create_task(forever())
        task_box["task"] = task
        session._run = RunHandle("run-1", task)

    await _serve(
        [{"type": "session.start"}, {"type": "session.cancel"}], start_run=start_run
    )
    await asyncio.sleep(0)  # let the cancellation settle

    assert task_box["task"].cancelled() or task_box["task"].done()


@pytest.mark.asyncio
async def test_disconnect_mid_run_cancels_the_run():
    """A client that vanishes must not leave the agent running for nobody."""
    task_box: dict[str, asyncio.Task] = {}

    async def forever() -> None:
        await asyncio.Event().wait()

    async def start_run(session: WsSession, message: dict) -> None:
        task = asyncio.create_task(forever())
        task_box["task"] = task
        session._run = RunHandle("run-1", task)

    # No further frames: receive_json raises WebSocketDisconnect next.
    await _serve([{"type": "session.start"}], start_run=start_run)
    await asyncio.sleep(0)

    assert task_box["task"].cancelled() or task_box["task"].done()


@pytest.mark.asyncio
async def test_serving_a_socket_leaves_no_tasks_of_its_own():
    before = set(asyncio.all_tasks())

    await _serve([{"type": "ping"}])
    await asyncio.sleep(0)

    assert set(asyncio.all_tasks()) - before == set()


@pytest.mark.asyncio
async def test_cancel_after_a_run_finished_is_refused():
    """Criterion 11: a terminal event clears the run, so cancel has nothing left."""

    async def start_run(session: WsSession, message: dict) -> None:
        session._run = None  # as the pump's finally clause leaves it

    websocket = await _serve(
        [{"type": "session.start"}, {"type": "session.cancel"}], start_run=start_run
    )

    assert websocket.types() == ["error"]
    assert websocket.sent[0]["code"] == 409


# -- the vocabulary matches the code ----------------------------------------


def test_documented_server_events_are_exactly_the_ones_emitted():
    """Neither an undocumented event nor a documented one nothing sends.

    The spec originally listed `tool_call`, which this transport never emits --
    tool calls ride inside `done.tool_calls`. A list that drifts from the code
    is how a protocol begins advertising what it does not do.
    """
    import re
    from pathlib import Path

    from src.internal.servers.web.ws_channel import SERVER_EVENTS

    emitted: set[str] = set()
    for name in ("app.py", "run_driver.py", "ws_channel.py"):
        source = Path("src/internal/servers/web") / name
        emitted |= set(re.findall(r'"type": "([a-z_.]+)"', source.read_text()))

    assert emitted == set(SERVER_EVENTS)
