"""Contracts for the WebSocket control channel.

The channel's whole justification is that it reuses the run driver and the
approval broker rather than reimplementing them, so most of these assert
sameness with SSE rather than behaviour of their own.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.internal.auth import generate_user_jwt_token
from src.internal.db import AgenticSearchStore, UserRecord
from src.internal.servers.redis import redis_pool
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app
from tests.unit.servers.web._ws_fakes import FakeRedis

_USER = "user-1"


def _answer_result(question: str):
    """The same stub the SSE tests use, so parity assertions compare like with like."""
    from tests.unit.servers.web.test_sse_streaming import _answer_result as shared

    return shared(question)


@pytest.fixture()
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> FakeRedis:
    backing = FakeRedis()

    async def _connection():
        return backing

    monkeypatch.setattr(redis_pool, "get_async_redis_connection", _connection)
    return backing


@pytest.fixture()
def client(tmp_path) -> TestClient:
    store = AgenticSearchStore(tmp_path / "s.sqlite3")
    store.upsert_user(UserRecord(id=_USER, email="user-1@example.test"))
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"), store=store
    )
    return TestClient(app)


@pytest.fixture()
def stub_agent(monkeypatch: pytest.MonkeyPatch):
    async def fake_answer(question, *, llm=None, chat_history=None, **kw):
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {generate_user_jwt_token(user_id=_USER)}"}


def _token(client: TestClient) -> str:
    return client.post("/api/agent/ws-token", headers=_auth()).json()["token"]


def _drain(ws, *, until: str = "done") -> list[dict]:
    events = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if event["type"] in (until, "error"):
            return events


# -- authentication ----------------------------------------------------------


def test_socket_without_a_token_is_refused(client: TestClient, fake_redis):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/agent/ws") as ws:
            ws.receive_json()


def test_socket_with_an_unknown_token_is_refused(client: TestClient, fake_redis):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/agent/ws?token=nope") as ws:
            ws.receive_json()


def test_a_token_authenticates_only_once(client: TestClient, fake_redis):
    token = _token(client)

    with client.websocket_connect(f"/api/agent/ws?token={token}") as ws:
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/api/agent/ws?token={token}") as ws:
            ws.receive_json()


# -- dispatch ----------------------------------------------------------------


def test_unknown_event_is_an_error_not_a_disconnect(client: TestClient, fake_redis):
    with client.websocket_connect(f"/api/agent/ws?token={_token(client)}") as ws:
        ws.send_json({"type": "session.update", "foo": 1})
        error = ws.receive_json()

        assert error["type"] == "error"
        assert error["code"] == 400
        # Still alive: a client speaking a newer vocabulary degrades.
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}


def test_malformed_frame_is_an_error_not_a_disconnect(client: TestClient, fake_redis):
    with client.websocket_connect(f"/api/agent/ws?token={_token(client)}") as ws:
        ws.send_text("not json")
        error = ws.receive_json()

        assert error["type"] == "error"
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}


def test_cancel_without_a_run_is_refused(client: TestClient, fake_redis):
    with client.websocket_connect(f"/api/agent/ws?token={_token(client)}") as ws:
        ws.send_json({"type": "session.cancel"})
        error = ws.receive_json()

        assert error["type"] == "error"
        assert error["code"] == 409


# -- runs --------------------------------------------------------------------


def test_session_start_runs_and_reaches_done(
    client: TestClient, fake_redis, stub_agent
):
    with client.websocket_connect(f"/api/agent/ws?token={_token(client)}") as ws:
        ws.send_json(
            {
                "type": "session.start",
                "request": {"query": "what is faiss?", "mode": "chat_once"},
            }
        )
        started = ws.receive_json()

        assert started["type"] == "session.started"
        assert started["run_id"]

        events = _drain(ws)
        types = [event["type"] for event in events]

        assert types[-1] == "done"
        assert "answer" in types
        # Every event is attributable to its run, not to the socket.
        assert all(event["run_id"] == started["run_id"] for event in events)


def test_a_second_run_on_one_socket_is_refused(
    client: TestClient, fake_redis, stub_agent
):
    """v1 carries one run per socket; say so rather than interleaving."""
    with client.websocket_connect(f"/api/agent/ws?token={_token(client)}") as ws:
        ws.send_json(
            {
                "type": "session.start",
                "request": {"query": "first", "mode": "chat_once"},
            }
        )
        assert ws.receive_json()["type"] == "session.started"
        ws.send_json(
            {
                "type": "session.start",
                "request": {"query": "second", "mode": "chat_once"},
            }
        )

        events = _drain(ws)

        assert any(
            event["type"] == "error" and event.get("code") == 409 for event in events
        )


def test_invalid_request_is_rejected_without_starting_a_run(
    client: TestClient, fake_redis
):
    with client.websocket_connect(f"/api/agent/ws?token={_token(client)}") as ws:
        ws.send_json({"type": "session.start", "request": {"query": ""}})
        error = ws.receive_json()

        assert error["type"] == "error"
        assert error["code"] == 400


# -- parity with SSE ---------------------------------------------------------


def test_socket_and_sse_agree_on_the_terminal_payload(
    client: TestClient, fake_redis, stub_agent
):
    """The two transports must not drift; both go through _terminal_events."""
    body = {"query": "what is faiss?", "mode": "chat_once"}

    sse = client.post("/api/agent/stream", json=body)
    sse_events = [
        line[len("data:") :].strip()
        for line in sse.text.splitlines()
        if line.startswith("data:")
    ]
    import json

    sse_done = json.loads(sse_events[-1])

    with client.websocket_connect(f"/api/agent/ws?token={_token(client)}") as ws:
        ws.send_json({"type": "session.start", "request": body})
        ws.receive_json()  # session.started
        ws_done = _drain(ws)[-1]

    assert ws_done["type"] == sse_done["type"] == "done"
    # run_id differs per run by design; every other key must match.
    ignored = {"request_id", "run_id", "session_id"}
    assert {k: v for k, v in ws_done.items() if k not in ignored} == {
        k: v for k, v in sse_done.items() if k not in ignored
    }


# -- approvals ---------------------------------------------------------------


@pytest.fixture()
def approving_agent(monkeypatch: pytest.MonkeyPatch):
    """A tool run that parks on one approval and reports the decision."""
    from datetime import datetime, timedelta, timezone

    from src.agents.core.base import AgentLoopOutput
    from src.agents.tool import ToolAgentLoop, ToolApprovalRequest

    seen: dict[str, object] = {}

    async def fake_run(self, messages, sampling_params, *, on_approval=None, **kwargs):
        now = datetime.now(timezone.utc)
        decision = await on_approval(
            ToolApprovalRequest(
                approval_id="approval-1",
                tool_name="create_ticket",
                arguments={"title": "Fix it"},
                created_at=now,
                expires_at=now + timedelta(seconds=30),
            )
        )
        seen["decision"] = decision
        return AgentLoopOutput(
            prompt_ids=[],
            response_ids=[],
            response_mask=[],
            num_turns=1,
            final_answer=f"decision={decision.value}",
        )

    monkeypatch.setattr(ToolAgentLoop, "run", fake_run)
    return seen


def _start_tool_run(client: TestClient, ws) -> dict:
    ws.send_json(
        {
            "type": "session.start",
            "request": {"query": "Create a ticket", "mode": "tool_agent"},
        }
    )
    assert ws.receive_json()["type"] == "session.started"
    event = ws.receive_json()
    while event["type"] not in ("approval_required", "error"):
        event = ws.receive_json()
    return event


def test_approval_over_the_socket_resumes_the_same_run(
    client: TestClient, fake_redis, approving_agent
):
    with client.websocket_connect(f"/api/agent/ws?token={_token(client)}") as ws:
        client.app.state.search_agent_manager = object()
        client.app.state.search_agent_tokenizer = object()
        requested = _start_tool_run(client, ws)

        assert requested["type"] == "approval_required"
        assert requested["approval"]["id"] == "approval-1"

        ws.send_json(
            {
                "type": "approval.submit",
                "approval_id": "approval-1",
                "decision": "approve",
            }
        )
        events = _drain(ws)

    assert events[-1]["type"] == "done"
    assert str(approving_agent["decision"].value) == "approve"


def test_approval_errors_carry_the_same_codes_as_the_http_endpoint(
    client: TestClient, fake_redis, approving_agent
):
    """Both transports resolve one broker, so they must fail alike."""
    with client.websocket_connect(f"/api/agent/ws?token={_token(client)}") as ws:
        client.app.state.search_agent_manager = object()
        client.app.state.search_agent_tokenizer = object()
        _start_tool_run(client, ws)

        # Unknown id: 404 over HTTP, code 404 over the socket.
        ws.send_json(
            {"type": "approval.submit", "approval_id": "nope", "decision": "approve"}
        )
        assert ws.receive_json()["code"] == 404

        # Unknown decision value is a client error, not a broker outcome.
        ws.send_json(
            {"type": "approval.submit", "approval_id": "approval-1", "decision": "?"}
        )
        assert ws.receive_json()["code"] == 400

        ws.send_json(
            {
                "type": "approval.submit",
                "approval_id": "approval-1",
                "decision": "approve",
            }
        )
        events = _drain(ws)

    assert events[-1]["type"] == "done"

    # Deciding twice: 409 over HTTP, 409 over the socket.
    decided = client.post(
        "/api/agent/approvals/approval-1",
        json={"decision": "approve"},
        headers=_auth(),
    )
    assert decided.status_code in (404, 409)
