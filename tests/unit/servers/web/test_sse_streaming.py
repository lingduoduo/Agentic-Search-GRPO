"""Unit tests for POST /api/agent/stream (SSE)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
import pytest

from src.context.models import (
    AnswerGenerationResult,
    ContextDocument,
    PromptBundle,
    SearchContextBundle,
)
from src.internal.db import AgenticSearchStore
from src.internal.db import UserRecord
from src.internal.servers.web.app import (
    SearchExperienceSettings,
    _request_tool_approval,
    create_web_app,
)
from src.internal.servers.web.tool_approval import ToolApprovalBroker


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[len("data:") :].strip()
            if payload:
                events.append(json.loads(payload))
    return events


def _answer_result(question: str) -> AnswerGenerationResult:
    doc = ContextDocument(
        id="D1",
        title="T",
        content=f"[D1] Answer to {question}",
        url="https://t.test",
        score=0.9,
    )
    return AnswerGenerationResult(
        answer=f"[D1] Answer to {question}",
        citations=["D1"],
        context=SearchContextBundle(query=question, documents=[doc]),
        prompt=PromptBundle(system="", user="", messages=[]),
    )


def _approval_request(approval_id: str):
    from src.agents.tool import ToolApprovalRequest

    now = datetime.now(UTC)
    return ToolApprovalRequest(
        approval_id=approval_id,
        tool_name="create_ticket",
        arguments={"title": "Fix it"},
        created_at=now,
        expires_at=now + timedelta(seconds=30),
    )


@pytest.mark.asyncio
async def test_approval_publication_backpressures_without_evicting_approvals():
    from src.agents.tool import ApprovalDecision

    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=2)
    first = {"type": "approval_required", "approval": {"id": "first"}}
    second = {"type": "approval_required", "approval": {"id": "second"}}
    queue.put_nowait(first)
    queue.put_nowait(second)
    broker = ToolApprovalBroker()

    task = asyncio.create_task(
        _request_tool_approval(broker, "user-1", _approval_request("third"), queue)
    )
    while broker.pending_count == 0:
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not task.done()
    assert queue.get_nowait() == first
    await asyncio.sleep(0)
    assert queue.get_nowait() == second
    published = await queue.get()
    assert published["approval"]["id"] == "third"

    await broker.decide("third", "user-1", ApprovalDecision.APPROVE)
    assert await task is ApprovalDecision.APPROVE


@pytest.mark.asyncio
async def test_cancelled_approval_cleans_blocked_publication_task():
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=1)
    original = {"type": "approval_required", "approval": {"id": "first"}}
    queue.put_nowait(original)
    broker = ToolApprovalBroker()
    before = set(asyncio.all_tasks())

    task = asyncio.create_task(
        _request_tool_approval(broker, "user-1", _approval_request("second"), queue)
    )
    while broker.pending_count == 0:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert broker.pending_count == 0
    assert queue.get_nowait() == original
    assert set(asyncio.all_tasks()) - before == set()


def test_stream_endpoint_exists(tmp_path):
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"))
    client = TestClient(app)
    resp = client.post("/api/agent/stream", json={"query": "x", "mode": "chat_once"})
    assert resp.status_code != 404


def test_stream_chat_once_emits_answer_and_done(monkeypatch, tmp_path):
    async def fake_answer(question, *, llm=None, chat_history=None, **kw):
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"))
    client = TestClient(app)

    resp = client.post(
        "/api/agent/stream",
        json={"query": "What is FAISS?", "mode": "chat_once"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = _parse_sse(resp.text)
    types = [e["type"] for e in events]
    assert "answer" in types
    assert "done" in types

    answer_event = next(e for e in events if e["type"] == "answer")
    assert "[D1]" in answer_event["text"]

    done_event = next(e for e in events if e["type"] == "done")
    assert done_event["session_id"]
    assert "D1" in done_event["citations"]


def test_stream_done_event_includes_route(monkeypatch, tmp_path):
    """The auto-route done event carries the chosen route + degradation."""
    from src.internal.servers.web.intent import RouteDecision, RouteStrategy

    monkeypatch.setattr(
        "src.internal.servers.web.app.recognize_intent",
        lambda *a, **k: RouteDecision(RouteStrategy.CHAT),
    )

    async def fake_rag(query, **kw):
        return "grounded answer", ["[D1]"], [], "chat", {}

    monkeypatch.setattr("src.internal.servers.web.app._run_agentic_rag", fake_rag)

    class _LLM:
        def complete(self, messages, **_):
            return "unused"

    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"), llm=_LLM()
    )
    client = TestClient(app)

    resp = client.post("/api/agent/stream", json={"query": "explain FAISS"})
    assert resp.status_code == 200
    done_event = next(e for e in _parse_sse(resp.text) if e["type"] == "done")
    assert done_event["route"] == "chat"
    assert done_event["route_degraded"] is None
    assert done_event["intent"] == "chat"


def test_stream_chat_loop_emits_claim_events_before_answer(monkeypatch, tmp_path):
    """AgenticRAGLoop's on_claim callback (via _run_agentic_rag) surfaces as SSE
    claim events, arriving before the terminal answer event."""

    async def fake_rag(query, *, on_claim=None, **kw):
        if on_claim is not None:
            on_claim("FAISS is a vector search library. [D1]")
            on_claim("It supports approximate nearest neighbor search. [D2]")
        return "stub answer", ["D1", "D2"], [], "chat", {}

    monkeypatch.setattr("src.internal.servers.web.app._run_agentic_rag", fake_rag)

    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"))
    client = TestClient(app)

    resp = client.post(
        "/api/agent/stream",
        json={"query": "what is faiss?", "mode": "chat_loop"},
    )
    assert resp.status_code == 200

    events = _parse_sse(resp.text)
    types = [e["type"] for e in events]
    assert "claim" in types
    assert types.index("claim") < types.index("answer")
    assert all(isinstance(e["text"], str) for e in events if e["type"] == "claim")


def test_stream_claim_from_worker_thread_uses_the_threadsafe_hop(monkeypatch, tmp_path):
    """Finding 5: the real path calls on_claim from a worker thread.

    generate_answer runs inside asyncio.to_thread (AgenticRAGLoop.run, see
    src/agents/search/agentic_rag.py:328), so on_claim fires off the event
    loop thread and must hop back via loop.call_soon_threadsafe before
    touching the queue — asyncio.Queue.put_nowait is not thread-safe. The
    tests above call on_claim synchronously from inside the fake coroutine,
    i.e. from the loop thread itself, where that hop is a no-op — so they
    pass even if the hop is deleted. Timing-based assertions on a bare
    `_offer(...)` call are unreliable here: CPython's GIL keeps
    deque.append effectively atomic and the endpoint's own poll-then-drain
    consumer loop masks the delay, so the claim still arrives even without
    the hop. This test instead asserts the invariant directly: patch
    asyncio.Queue.put_nowait to record which thread called it, drive
    on_claim from a genuine asyncio.to_thread worker thread, and require
    every call to originate from the loop's own thread.
    """
    loop_thread_id: list[int] = []
    violations: list[int] = []
    real_put_nowait = asyncio.Queue.put_nowait

    def _guarded_put_nowait(self, item):
        tid = threading.get_ident()
        if loop_thread_id and tid != loop_thread_id[0]:
            violations.append(tid)
        return real_put_nowait(self, item)

    monkeypatch.setattr(asyncio.Queue, "put_nowait", _guarded_put_nowait)

    async def fake_rag(query, *, on_claim=None, **kw):
        loop_thread_id.append(threading.get_ident())

        def _emit_from_worker_thread() -> None:
            if on_claim is not None:
                on_claim("FAISS is a vector search library. [D1]")

        await asyncio.to_thread(_emit_from_worker_thread)
        return "stub answer", ["D1"], [], "chat", {}

    monkeypatch.setattr("src.internal.servers.web.app._run_agentic_rag", fake_rag)

    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"))
    client = TestClient(app)

    resp = client.post(
        "/api/agent/stream",
        json={"query": "what is faiss?", "mode": "chat_loop"},
    )
    assert resp.status_code == 200

    events = _parse_sse(resp.text)
    types = [e["type"] for e in events]
    assert "claim" in types
    assert types.index("claim") < types.index("answer")
    assert violations == []


def test_stream_auto_routed_chat_emits_claim_events_before_answer(
    monkeypatch, tmp_path
):
    """The real browser never sends `mode` (see web/src/pages/AssistPage.tsx), so
    every live request goes through the auto-routing path (`_run_auto_routed`),
    not the explicit `chat_loop` mode covered above. This pins on_claim's
    forwarding through that path's CHAT branch specifically."""
    from src.internal.servers.web.intent import RouteDecision, RouteStrategy

    monkeypatch.setattr(
        "src.internal.servers.web.app.recognize_intent",
        lambda *a, **k: RouteDecision(RouteStrategy.CHAT),
    )

    async def fake_rag(query, *, on_claim=None, **kw):
        if on_claim is not None:
            on_claim("FAISS is a vector search library. [D1]")
            on_claim("It supports approximate nearest neighbor search. [D2]")
        return "stub answer", ["D1", "D2"], [], "chat", {}

    monkeypatch.setattr("src.internal.servers.web.app._run_agentic_rag", fake_rag)

    class _LLM:
        def complete(self, messages, **_):
            return "unused"

    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"), llm=_LLM()
    )
    client = TestClient(app)

    # No "mode" key — exactly what AssistPage.tsx sends.
    resp = client.post("/api/agent/stream", json={"query": "what is faiss?"})
    assert resp.status_code == 200

    events = _parse_sse(resp.text)
    types = [e["type"] for e in events]
    assert "claim" in types
    assert types.index("claim") < types.index("answer")
    assert all(isinstance(e["text"], str) for e in events if e["type"] == "claim")


def test_stream_emits_error_event_on_failure(monkeypatch, tmp_path):
    async def bad_answer(question, **kw):
        raise RuntimeError("simulated backend failure")

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", bad_answer
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"))
    client = TestClient(app)

    resp = client.post(
        "/api/agent/stream",
        json={"query": "test", "mode": "chat_once"},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    error_events = [e for e in events if e["type"] == "error"]
    assert error_events


def test_stream_done_event_contains_documents(monkeypatch, tmp_path):
    async def fake_answer(question, **kw):
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"))
    client = TestClient(app)

    resp = client.post(
        "/api/agent/stream",
        json={"query": "test", "mode": "chat_once"},
    )
    events = _parse_sse(resp.text)
    done_event = next(e for e in events if e["type"] == "done")
    assert isinstance(done_event["documents"], list)
    assert len(done_event["documents"]) >= 1
    assert done_event["documents"][0]["title"] == "T"


def test_stream_and_json_return_identical_final_answer_data(monkeypatch, tmp_path):
    async def fake_answer(question, **kw):
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"))
    client = TestClient(app)
    payload = {"query": "same result", "mode": "chat_once"}

    json_data = client.post("/api/agent", json=payload).json()
    events = _parse_sse(client.post("/api/agent/stream", json=payload).text)
    answer = next(event for event in events if event["type"] == "answer")
    done = next(event for event in events if event["type"] == "done")

    assert answer["text"] == json_data["answer"]
    assert done["citations"] == json_data["citations"]
    assert done["documents"] == json_data["documents"]
    assert done["intent"] == json_data["intent"]
    assert done["tool_calls"] == json_data["tool_calls"]
    assert done["control_flow_trace"] == json_data["control_flow_trace"]


def test_stream_tool_approval_can_resume_same_request(monkeypatch, tmp_path):
    from src.agents.core.base import AgentLoopOutput
    from src.agents.tool import ApprovalDecision, ToolAgentLoop, ToolApprovalRequest
    from src.internal.auth import generate_user_jwt_token

    executions: list[str] = []

    async def fake_run(
        self, messages, sampling_params, *, on_turn=None, on_approval=None, **kwargs
    ):
        assert on_approval is not None
        assert on_turn is not None
        for turn in range(100):
            await on_turn(turn, "queued_tool", 0)
        now = datetime.now(UTC)
        decision = await on_approval(
            ToolApprovalRequest(
                approval_id="approval-1",
                tool_name="create_ticket",
                arguments={"title": "Fix it", "token": "hidden"},
                created_at=now,
                expires_at=now + timedelta(seconds=30),
            )
        )
        if decision is ApprovalDecision.APPROVE:
            executions.append("create_ticket")
        return AgentLoopOutput(
            prompt_ids=[],
            response_ids=[],
            response_mask=[],
            num_turns=1,
            final_answer="Ticket created.",
        )

    monkeypatch.setattr(ToolAgentLoop, "run", fake_run)
    store = AgenticSearchStore(tmp_path / "s.sqlite3")
    store.upsert_user(UserRecord(id="user-1", email="user-1@example.test"))
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"), store=store
    )
    token = generate_user_jwt_token(user_id="user-1")
    headers = {"Authorization": f"Bearer {token}"}
    result: dict[str, object] = {}

    with TestClient(app) as client:
        app.state.search_agent_manager = object()
        app.state.search_agent_tokenizer = object()

        def stream_request() -> None:
            result["response"] = client.post(
                "/api/agent/stream",
                json={"query": "Create a ticket", "mode": "tool_agent"},
                headers=headers,
            )

        thread = threading.Thread(target=stream_request)
        thread.start()
        deadline = time.monotonic() + 5
        while app.state.tool_approval_broker.pending_count == 0:
            assert time.monotonic() < deadline
            time.sleep(0.01)

        decision = client.post(
            "/api/agent/approvals/approval-1",
            json={"decision": "approve"},
            headers=headers,
        )
        thread.join(timeout=5)

    assert decision.status_code == 200
    assert not thread.is_alive()
    response = result["response"]
    assert response.status_code == 200
    events = _parse_sse(response.text)
    approval = next(event for event in events if event["type"] == "approval_required")
    assert approval["approval"]["arguments"] == {"title": "Fix it"}
    approval_index = events.index(approval)
    assert [event["turn"] for event in events[:approval_index]] == list(range(100))
    assert [event["type"] for event in events][-2:] == ["answer", "done"]
    assert executions == ["create_ticket"]


def test_stream_emits_progress_events_before_done(monkeypatch, tmp_path):
    """When SearchAgentLoop.run fires on_turn, progress SSE events appear before done."""
    from src.agents.search import SearchAgentLoop
    from src.agents.core.base import AgentLoopOutput
    from src.agents.core.control_flow_trace import ControlFlowEvent

    event = ControlFlowEvent(
        sequence=1,
        timestamp="2026-06-27T12:00:00.000Z",
        turn=1,
        component="planner",
        action="search_planned",
        status="decided",
        details={"decision": "search"},
    )

    async def fake_run(self, messages, sampling_params, *, on_turn=None, on_trace=None):
        if on_turn is not None:
            await on_turn(1, "search_routing_tool", 5)
            await on_turn(2, None, 0)
        if on_trace is not None:
            await on_trace(event)
        return AgentLoopOutput(
            prompt_ids=[],
            response_ids=[],
            response_mask=[],
            num_turns=2,
            final_answer="FAISS is fast.",
            action_trace=None,
            control_flow_trace=[event],
        )

    monkeypatch.setattr(SearchAgentLoop, "run", fake_run)

    class _MockMgr:
        async def generate(self, *a, **kw):
            return []

    class _MockTok:
        chat_template = "t"

        def apply_chat_template(self, *a, **kw):
            return [] if kw.get("tokenize", True) else ""

        def encode(self, t):
            return []

        def decode(self, ids, **kw):
            return "FAISS is fast."

    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"))
    app.state.search_agent_manager = _MockMgr()
    app.state.search_agent_tokenizer = _MockTok()

    client = TestClient(app)
    resp = client.post(
        "/api/agent/stream",
        json={"query": "What is FAISS?", "mode": "search_agent"},
    )
    assert resp.status_code == 200

    events = _parse_sse(resp.text)
    types = [e["type"] for e in events]
    assert "progress" in types, "Expected at least one progress event"
    assert "trace" in types
    assert "done" in types

    # Progress must arrive before done
    progress_indices = [i for i, e in enumerate(events) if e["type"] == "progress"]
    done_index = next(i for i, e in enumerate(events) if e["type"] == "done")
    assert all(p < done_index for p in progress_indices)

    progress_events = [e for e in events if e["type"] == "progress"]
    assert progress_events[0]["turn"] == 1
    assert "search_routing_tool" in progress_events[0]["text"]

    done_event = next(e for e in events if e["type"] == "done")
    assert "intent" in done_event  # done event now includes intent
    assert done_event["control_flow_trace"] == [event.to_dict()]

    trace_event = next(e for e in events if e["type"] == "trace")
    assert trace_event["event"] == event.to_dict()
    assert events.index(trace_event) < done_index

    session = client.get(f"/api/sessions/{done_event['session_id']}").json()
    assert session["messages"][-1]["metadata"]["control_flow_trace"] == [
        event.to_dict()
    ]
