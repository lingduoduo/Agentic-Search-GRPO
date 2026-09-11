import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.db import AgenticSearchStore
from src.internal.configs import load_app_settings
from src.internal.servers.query_and_chat.tool_backend import create_tool_router


def _make_app(*, with_model: bool) -> FastAPI:
    store = AgenticSearchStore(":memory:")
    app = FastAPI()
    app.include_router(
        create_tool_router(
            store, search_url="http://x/retrieve", resolved=load_app_settings()
        )
    )
    app.state.search_agent_manager = object() if with_model else None
    app.state.search_agent_tokenizer = object() if with_model else None
    app.state.tool_approval_broker = None
    app.state._store = store  # keep a handle for assertions
    return app


def test_send_tool_message_no_model_returns_400():
    client = TestClient(_make_app(with_model=False))
    resp = client.post(
        "/tool/send-tool-message", json={"message": "hi", "stream": False}
    )
    assert resp.status_code == 400
    assert "requires a local model" in resp.json()["detail"]


def test_tool_history_anonymous_is_empty():
    client = TestClient(_make_app(with_model=True))
    resp = client.get("/tool/tool-history")
    assert resp.status_code == 200
    assert resp.json() == {"sessions": []}


def test_send_tool_message_streams_progress_then_done(monkeypatch):
    from src.internal.servers.web import tool_agent_runner
    from src.internal.servers.web.tool_agent_runner import ToolCallView

    async def fake_run_tool_agent(query, *, on_turn=None, **kw):
        if on_turn is not None:
            await on_turn(1, "search", 3)
        tc = ToolCallView(
            tool_name="search",
            status="completed",
            arguments={"q": query},
            result_summary="3 items",
            latency_ms=12,
            error=None,
        )
        return ("the answer", [], [], "tool", {"tool_calls": [tc], "num_turns": 1})

    # tool_backend now imports _run_tool_agent function-locally (per-call) from
    # tool_agent_runner, so patch the source module rather than tool_backend's
    # (now nonexistent) module-level attribute.
    monkeypatch.setattr(tool_agent_runner, "_run_tool_agent", fake_run_tool_agent)

    client = TestClient(_make_app(with_model=True))
    with client.stream(
        "POST", "/tool/send-tool-message", json={"message": "find X", "stream": True}
    ) as resp:
        assert resp.status_code == 200
        events = [
            json.loads(line[len("data:") :].strip())
            for line in resp.iter_lines()
            if line.startswith("data:")
        ]

    types = [e["type"] for e in events]
    assert "progress" in types
    assert types[-1] == "done"
    tool_call = next(e for e in events if e["type"] == "tool_call")
    assert tool_call["tool_name"] == "search"
    done = events[-1]
    assert done["num_turns"] == 1 and done["session_id"]


def test_send_tool_message_emits_approval_required(monkeypatch):
    from dataclasses import dataclass

    from src.internal.servers.query_and_chat import tool_backend
    from src.internal.servers.web.tool_agent_runner import ToolCallView

    @dataclass
    class _View:
        id: str
        tool_name: str
        arguments: dict
        expires_at: str

    class _Broker:
        async def request(self, owner_user_id, approval_request, on_registered=None):
            if on_registered:
                on_registered(
                    _View(
                        id="ap1",
                        tool_name="web_search",
                        arguments={},
                        expires_at="2030-01-01T00:00:00Z",
                    )
                )
            return "approve"

    captured = {}

    async def fake_run_tool_agent(query, *, on_turn=None, on_approval=None, **kw):
        captured["on_approval"] = on_approval
        if on_approval is not None:
            await on_approval(object())
        tc = ToolCallView(
            tool_name="web_search",
            status="completed",
            arguments={},
            result_summary="ok",
            latency_ms=1,
            error=None,
        )
        return ("done", [], [], "tool", {"tool_calls": [tc], "num_turns": 1})

    from src.internal.servers.web import tool_agent_runner

    monkeypatch.setattr(tool_agent_runner, "_run_tool_agent", fake_run_tool_agent)

    class _User:
        id = "u1"
        is_anonymous = False
        email = "u@x"

    monkeypatch.setattr(tool_backend, "resolve_active_user", lambda *a, **k: _User())

    app = _make_app(with_model=True)
    app.state.tool_approval_broker = _Broker()
    from src.internal.db import UserRecord

    app.state._store.upsert_user(UserRecord(id="u1", email="u@x"))

    client = TestClient(app)
    with client.stream(
        "POST", "/tool/send-tool-message", json={"message": "go", "stream": True}
    ) as resp:
        events = [
            json.loads(line[len("data:") :].strip())
            for line in resp.iter_lines()
            if line.startswith("data:")
        ]

    assert captured["on_approval"] is not None
    assert any(
        e["type"] == "approval_required" and e["approval"]["id"] == "ap1"
        for e in events
    )


def test_no_broker_means_on_approval_none(monkeypatch):
    from src.internal.servers.web.tool_agent_runner import ToolCallView

    captured = {}

    async def fake_run_tool_agent(query, *, on_turn=None, on_approval=None, **kw):
        captured["on_approval"] = on_approval
        tc = ToolCallView(
            tool_name="x",
            status="completed",
            arguments={},
            result_summary="",
            latency_ms=1,
            error=None,
        )
        return ("done", [], [], "tool", {"tool_calls": [tc], "num_turns": 1})

    from src.internal.servers.web import tool_agent_runner

    monkeypatch.setattr(tool_agent_runner, "_run_tool_agent", fake_run_tool_agent)

    app = _make_app(with_model=True)  # broker is None by default
    client = TestClient(app)
    with client.stream(
        "POST", "/tool/send-tool-message", json={"message": "go", "stream": True}
    ) as resp:
        list(resp.iter_lines())
    assert captured["on_approval"] is None


def _make_app_with_memory(*, llm=None, memory_compression=False):
    store = AgenticSearchStore(":memory:")
    app = FastAPI()
    app.include_router(
        create_tool_router(
            store,
            search_url="http://x/retrieve",
            resolved=load_app_settings(),
            llm=llm,
            memory_compression=memory_compression,
        )
    )
    app.state.search_agent_manager = object()
    app.state.search_agent_tokenizer = object()
    app.state.tool_approval_broker = None
    app.state._store = store
    return app


def _seed_long(store, n=45):
    session = store.create_chat_session(title="long")
    records = [
        store.add_chat_message(
            session.id, role="user" if i % 2 == 0 else "assistant", content=f"m{i}"
        )
        for i in range(n)
    ]
    return session.id, records


def _capture_tool_agent(monkeypatch):
    from src.internal.servers.web import tool_agent_runner

    captured: list = []

    async def fake_run_tool_agent(query, *, history, **kw):
        captured.append(list(history))
        return ("ok", [], [], "tool", {"tool_calls": [], "num_turns": 1})

    monkeypatch.setattr(tool_agent_runner, "_run_tool_agent", fake_run_tool_agent)
    return captured


def test_send_tool_flag_off_hands_runner_last_forty(monkeypatch):
    from src.internal.cache.interface import InMemoryCache
    from src.internal.memory.working import SessionMemoryState, save_state

    cache = InMemoryCache()
    monkeypatch.setattr("src.internal.cache.interface._default_cache", cache)
    app = _make_app_with_memory()
    session_id, records = _seed_long(app.state._store)
    save_state(
        cache,
        session_id,
        SessionMemoryState(summary="S", summarized_through=records[4].id),
    )
    captured = _capture_tool_agent(monkeypatch)

    resp = TestClient(app).post(
        "/tool/send-tool-message",
        json={"message": "next", "session_id": session_id, "stream": False},
    )
    assert resp.status_code == 200
    assert len(captured[0]) == 40
    assert captured[0][0].role == "assistant"


def test_send_tool_flag_on_prepends_stored_summary(monkeypatch):
    from src.internal.cache.interface import InMemoryCache
    from src.internal.memory.working import (
        SUMMARY_PREFIX,
        SessionMemoryState,
        save_state,
    )

    cache = InMemoryCache()
    monkeypatch.setattr("src.internal.cache.interface._default_cache", cache)
    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.tool_backend.schedule_compression",
        lambda wm, **kw: None,
    )
    app = _make_app_with_memory(llm=object(), memory_compression=True)
    session_id, records = _seed_long(app.state._store)
    save_state(
        cache,
        session_id,
        SessionMemoryState(summary="S", summarized_through=records[4].id),
    )
    captured = _capture_tool_agent(monkeypatch)

    TestClient(app).post(
        "/tool/send-tool-message",
        json={"message": "next", "session_id": session_id, "stream": False},
    )
    assert len(captured[0]) == 41
    assert captured[0][0].role == "system"
    assert captured[0][0].content == SUMMARY_PREFIX + "S"


def test_send_tool_schedules_compression(monkeypatch):
    from src.internal.cache.interface import InMemoryCache
    from src.internal.servers.query_and_chat import tool_backend

    monkeypatch.setattr("src.internal.cache.interface._default_cache", InMemoryCache())
    sentinel = object()
    app = _make_app_with_memory(llm=sentinel, memory_compression=True)
    session_id, _ = _seed_long(app.state._store)
    _capture_tool_agent(monkeypatch)
    scheduled: list = []

    def fake_schedule(wm, **kw):
        scheduled.append((len(wm.pending), kw["enabled"], kw["llm"]))
        return None

    monkeypatch.setattr(tool_backend, "schedule_compression", fake_schedule)
    TestClient(app).post(
        "/tool/send-tool-message",
        json={"message": "next", "session_id": session_id, "stream": False},
    )
    assert scheduled == [(5, True, sentinel)]
