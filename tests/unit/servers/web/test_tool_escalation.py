"""The escalation broker, its event, the decide endpoint and the WebSocket message.

Escalation reuses the approval broker's mechanics through `DecisionBroker`, but
never its state: the two brokers are separate instances with separate pending
maps and counters.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.agents import EscalationDecision, ToolEscalationRequest
from src.internal.auth import generate_user_jwt_token
from src.internal.db import AgenticSearchStore, UserRecord
from src.internal.servers.redis import redis_pool
from src.internal.servers.web.app import (
    SearchExperienceSettings,
    _request_tool_escalation,
    create_web_app,
)
from src.internal.servers.web.tool_approval import (
    ApprovalConflict,
    ApprovalForbidden,
    ApprovalNotFound,
    ToolApprovalBroker,
    ToolEscalationBroker,
)
from tests.unit.servers.web._ws_fakes import FakeRedis


def _request(escalation_id="esc-1", expires_in=1.0):
    now = datetime.now(timezone.utc)
    return ToolEscalationRequest(
        escalation_id=escalation_id,
        tool_name="send_email",
        arguments={"to": "a@b.c", "password": "hunter2"},
        category="unknown",
        message="remote tool reported an error",
        attempts=1,
        created_at=now,
        expires_at=now + timedelta(seconds=expires_in),
    )


async def _pending(broker):
    for _ in range(100):
        if broker.pending_count:
            return
        await asyncio.sleep(0)
    raise AssertionError("not registered")


# -- the broker ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalation_round_trip_with_sanitized_view():
    broker = ToolEscalationBroker(timeout_seconds=5)
    views = []
    task = asyncio.create_task(
        broker.request("alice", _request(), on_registered=views.append)
    )
    await _pending(broker)
    assert "password" not in views[0].arguments
    assert (views[0].category, views[0].attempts) == ("unknown", 1)
    await broker.decide("esc-1", "alice", EscalationDecision.SKIP)
    assert await task is EscalationDecision.SKIP
    assert broker.counters["skip"] == 1


@pytest.mark.asyncio
async def test_escalation_ownership_and_double_decision():
    broker = ToolEscalationBroker(timeout_seconds=5)
    task = asyncio.create_task(broker.request("alice", _request()))
    await _pending(broker)
    with pytest.raises(ApprovalForbidden):
        await broker.decide("esc-1", "mallory", EscalationDecision.RETRY)
    await broker.decide("esc-1", "alice", EscalationDecision.RETRY)
    with pytest.raises((ApprovalConflict, ApprovalNotFound)):
        await broker.decide("esc-1", "alice", EscalationDecision.CANCEL)
    assert await task is EscalationDecision.RETRY


@pytest.mark.asyncio
async def test_escalation_times_out_as_expired():
    broker = ToolEscalationBroker(timeout_seconds=0.05)
    assert await broker.request("alice", _request()) is EscalationDecision.EXPIRED
    assert broker.counters["expired"] == 1


@pytest.mark.asyncio
async def test_expired_cannot_be_submitted_as_a_decision():
    broker = ToolEscalationBroker(timeout_seconds=5)
    task = asyncio.create_task(broker.request("alice", _request()))
    await _pending(broker)
    with pytest.raises(ApprovalConflict):
        await broker.decide("esc-1", "alice", EscalationDecision.EXPIRED)
    await broker.decide("esc-1", "alice", EscalationDecision.CANCEL)
    await task


@pytest.mark.asyncio
async def test_approval_and_escalation_brokers_share_no_state():
    approvals = ToolApprovalBroker(timeout_seconds=5)
    escalations = ToolEscalationBroker(timeout_seconds=5)
    task = asyncio.create_task(escalations.request("alice", _request()))
    await _pending(escalations)
    assert approvals.pending_count == 0
    await escalations.decide("esc-1", "alice", EscalationDecision.SKIP)
    await task
    assert approvals.counters["requested"] == 0


# -- the event -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_tool_escalation_publishes_one_event_and_returns_decision():
    broker = ToolEscalationBroker(timeout_seconds=5)
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=10)
    task = asyncio.create_task(
        _request_tool_escalation(broker, "alice", _request(), queue)
    )
    event = await asyncio.wait_for(queue.get(), timeout=1)
    assert event["type"] == "escalation_required"
    assert event["escalation"]["id"] == "esc-1"
    assert event["escalation"]["category"] == "unknown"
    assert "password" not in event["escalation"]["arguments"]
    await broker.decide("esc-1", "alice", EscalationDecision.RETRY)
    assert await task is EscalationDecision.RETRY
    assert queue.empty()


# -- the HTTP endpoint ----------------------------------------------------------


def _auth(user_id: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {generate_user_jwt_token(user_id=user_id)}"}


def _app(tmp_path):
    store = AgenticSearchStore(tmp_path / "s.sqlite3")
    store.upsert_user(UserRecord(id="user-1", email="user-1@example.test"))
    store.upsert_user(UserRecord(id="user-2", email="user-2@example.test"))
    return create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"), store=store
    )


def _wait_pending(broker) -> None:
    deadline = time.monotonic() + 5
    while broker.pending_count == 0:
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_escalation_endpoint_decides_and_maps_errors(tmp_path):
    app = _app(tmp_path)
    with TestClient(app) as client:
        broker = app.state.tool_escalation_broker
        pending = client.portal.start_task_soon(
            broker.request, "user-1", _request(expires_in=30)
        )
        _wait_pending(broker)

        forbidden = client.post(
            "/api/agent/escalations/esc-1",
            json={"decision": "skip"},
            headers=_auth("user-2"),
        )
        missing = client.post(
            "/api/agent/escalations/nope",
            json={"decision": "skip"},
            headers=_auth("user-1"),
        )
        invalid = client.post(
            "/api/agent/escalations/esc-1",
            json={"decision": "expired"},
            headers=_auth("user-1"),
        )
        decided = client.post(
            "/api/agent/escalations/esc-1",
            json={"decision": "skip"},
            headers=_auth("user-1"),
        )
        assert pending.result(timeout=5) is EscalationDecision.SKIP

        # The request resolved and left the map; a second post finds nothing
        # to decide. Both 404 and 409 mean "not decidable again".
        again = client.post(
            "/api/agent/escalations/esc-1",
            json={"decision": "cancel"},
            headers=_auth("user-1"),
        )

    assert forbidden.status_code == 403
    assert missing.status_code == 404
    assert invalid.status_code == 422
    assert decided.status_code == 200
    assert decided.json() == {"id": "esc-1", "decision": "skip"}
    assert again.status_code in (404, 409)
    # Escalations never touch the approval broker's counters.
    assert app.state.tool_approval_broker.counters["requested"] == 0
    assert app.state.tool_approval_broker.counters["errors"] == 0


def test_escalation_endpoint_requires_auth(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        response = client.post(
            "/api/agent/escalations/esc-1", json={"decision": "skip"}
        )
    assert response.status_code == 401


# -- the WebSocket message and disconnect ---------------------------------------


@pytest.fixture()
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> FakeRedis:
    backing = FakeRedis()

    async def _connection():
        return backing

    monkeypatch.setattr(redis_pool, "get_async_redis_connection", _connection)
    return backing


@pytest.fixture()
def escalating_agent(monkeypatch: pytest.MonkeyPatch):
    """A tool run that parks on one escalation and reports the decision."""
    from src.agents.core.base import AgentLoopOutput
    from src.agents.tool import ToolAgentLoop

    seen: dict[str, object] = {}

    async def fake_run(self, messages, sampling_params, *, on_escalation=None, **kw):
        seen["on_escalation"] = on_escalation
        try:
            decision = await on_escalation(_request(expires_in=30))
        except asyncio.CancelledError:
            seen["cancelled"] = True
            raise
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


def _ws_client(tmp_path) -> TestClient:
    client = TestClient(_app(tmp_path))
    client.app.state.search_agent_manager = object()
    client.app.state.search_agent_tokenizer = object()
    return client


def _ws_token(client: TestClient) -> str:
    return client.post("/api/agent/ws-token", headers=_auth("user-1")).json()["token"]


def _start_escalating_run(ws) -> dict:
    ws.send_json(
        {
            "type": "session.start",
            "request": {"query": "Send the email", "mode": "tool_agent"},
        }
    )
    assert ws.receive_json()["type"] == "session.started"
    event = ws.receive_json()
    while event["type"] not in ("escalation_required", "error"):
        event = ws.receive_json()
    return event


def test_escalation_over_the_socket_resumes_the_same_run(
    tmp_path, fake_redis, escalating_agent
):
    client = _ws_client(tmp_path)
    with client.websocket_connect(f"/api/agent/ws?token={_ws_token(client)}") as ws:
        requested = _start_escalating_run(ws)
        assert requested["type"] == "escalation_required"
        assert requested["escalation"]["id"] == "esc-1"
        assert "password" not in requested["escalation"]["arguments"]

        # An unknown decision value is a client error, not a broker outcome.
        ws.send_json(
            {"type": "escalation.submit", "escalation_id": "esc-1", "decision": "?"}
        )
        assert ws.receive_json()["code"] == 400
        # So is "expired": only the broker may decide that.
        ws.send_json(
            {
                "type": "escalation.submit",
                "escalation_id": "esc-1",
                "decision": "expired",
            }
        )
        assert ws.receive_json()["code"] == 409
        ws.send_json(
            {"type": "escalation.submit", "escalation_id": "nope", "decision": "retry"}
        )
        assert ws.receive_json()["code"] == 404

        ws.send_json(
            {
                "type": "escalation.submit",
                "escalation_id": "esc-1",
                "decision": "retry",
            }
        )
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] in ("done", "error"):
                break

    assert events[-1]["type"] == "done"
    assert escalating_agent["decision"] is EscalationDecision.RETRY


def test_socket_disconnect_during_escalation_cancels_it_and_never_completes(
    tmp_path, fake_redis, escalating_agent
):
    # A context-managed client keeps one event loop alive across the socket,
    # so the cancellation observed below is the channel's own, not loop
    # teardown cancelling every task.
    with _ws_client(tmp_path) as client:
        # The lifespan resets the model slots, so stub them after entering.
        client.app.state.search_agent_manager = object()
        client.app.state.search_agent_tokenizer = object()
        broker = client.app.state.tool_escalation_broker
        with client.websocket_connect(f"/api/agent/ws?token={_ws_token(client)}") as ws:
            assert _start_escalating_run(ws)["type"] == "escalation_required"
            assert broker.pending_count == 1

        deadline = time.monotonic() + 5
        while broker.counters["cancelled"] == 0 or "cancelled" not in escalating_agent:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert broker.pending_count == 0
        assert escalating_agent.get("cancelled") is True
        assert "decision" not in escalating_agent


def test_escalation_over_sse_resumes_the_same_run(tmp_path, escalating_agent):
    app = _app(tmp_path)
    with TestClient(app) as client:
        app.state.search_agent_manager = object()
        app.state.search_agent_tokenizer = object()
        broker = app.state.tool_escalation_broker
        result: dict[str, object] = {}

        def stream_request() -> None:
            result["response"] = client.post(
                "/api/agent/stream",
                json={"query": "Send the email", "mode": "tool_agent"},
                headers=_auth("user-1"),
            )

        thread = threading.Thread(target=stream_request)
        thread.start()
        _wait_pending(broker)
        decided = client.post(
            "/api/agent/escalations/esc-1",
            json={"decision": "cancel"},
            headers=_auth("user-1"),
        )
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert decided.status_code == 200
    events = [
        json.loads(line[len("data:") :].strip())
        for line in result["response"].text.splitlines()
        if line.startswith("data:") and line[len("data:") :].strip()
    ]
    escalation = next(e for e in events if e["type"] == "escalation_required")
    assert escalation["escalation"]["id"] == "esc-1"
    assert escalating_agent["decision"] is EscalationDecision.CANCEL


def test_tool_stream_emits_escalation_required(monkeypatch):
    from fastapi import FastAPI

    from src.internal.configs import load_app_settings
    from src.internal.servers.query_and_chat import tool_backend
    from src.internal.servers.query_and_chat.tool_backend import create_tool_router
    from src.internal.servers.web import tool_agent_runner

    captured: dict[str, object] = {}

    async def fake_run_tool_agent(query, *, on_escalation=None, **kw):
        captured["on_escalation"] = on_escalation
        captured["decision"] = await on_escalation(_request(expires_in=30))
        return ("done", [], [], "tool", {"tool_calls": [], "num_turns": 1})

    monkeypatch.setattr(tool_agent_runner, "_run_tool_agent", fake_run_tool_agent)

    class _User:
        id = "u1"
        is_anonymous = False
        email = "u@x"

    monkeypatch.setattr(tool_backend, "resolve_active_user", lambda *a, **k: _User())

    store = AgenticSearchStore(":memory:")
    store.upsert_user(UserRecord(id="u1", email="u@x"))
    app = FastAPI()
    app.include_router(
        create_tool_router(
            store, search_url="http://x/retrieve", resolved=load_app_settings()
        )
    )
    app.state.search_agent_manager = object()
    app.state.search_agent_tokenizer = object()
    app.state.tool_approval_broker = None
    app.state.tool_escalation_broker = ToolEscalationBroker(timeout_seconds=0.05)

    with TestClient(app).stream(
        "POST", "/tool/send-tool-message", json={"message": "go", "stream": True}
    ) as resp:
        events = [
            json.loads(line[len("data:") :].strip())
            for line in resp.iter_lines()
            if line.startswith("data:")
        ]

    assert any(
        e["type"] == "escalation_required" and e["escalation"]["id"] == "esc-1"
        for e in events
    )
    assert captured["decision"] is EscalationDecision.EXPIRED
