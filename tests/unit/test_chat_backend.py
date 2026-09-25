"""Unit tests for src.internal.servers.query_and_chat.chat_backend."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.auth import AuthenticatedUser
from src.internal.db import AgenticSearchStore, UserRecord
from src.internal.servers.query_and_chat.chat_backend import create_chat_router

_USER_ID = "u-test-1"
_USER = AuthenticatedUser(id=_USER_ID, email="test@example.com")
_ANON = AuthenticatedUser(id="anonymous", is_anonymous=True)


@pytest.fixture()
def store() -> AgenticSearchStore:
    s = AgenticSearchStore(":memory:")
    s.upsert_user(UserRecord(id=_USER_ID, email="test@example.com"))
    return s


@pytest.fixture()
def client(store: AgenticSearchStore, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with a real store and a patched-in authenticated user."""
    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.chat_backend.resolve_active_user",
        lambda _request, _store: _USER,
    )
    app = FastAPI()
    app.include_router(create_chat_router(store))
    return TestClient(app)


@pytest.fixture()
def anon_client(
    store: AgenticSearchStore, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    """TestClient where every request appears anonymous."""
    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.chat_backend.resolve_active_user",
        lambda _request, _store: _ANON,
    )
    app = FastAPI()
    app.include_router(create_chat_router(store))
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /chat/get-user-chat-sessions
# ---------------------------------------------------------------------------


def test_list_sessions_empty_for_new_user(client: TestClient):
    resp = client.get("/chat/get-user-chat-sessions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sessions"] == []
    assert body["has_more"] is False


def test_list_sessions_returns_created_sessions(
    client: TestClient, store: AgenticSearchStore
):
    store.create_chat_session(user_id=_USER_ID, title="First")
    store.create_chat_session(user_id=_USER_ID, title="Second")
    resp = client.get("/chat/get-user-chat-sessions")
    assert resp.status_code == 200
    titles = {s["title"] for s in resp.json()["sessions"]}
    assert titles == {"First", "Second"}


def test_list_sessions_anonymous_returns_empty(anon_client: TestClient):
    resp = anon_client.get("/chat/get-user-chat-sessions")
    assert resp.status_code == 200
    assert resp.json()["sessions"] == []


def test_list_sessions_has_more_flag(client: TestClient, store: AgenticSearchStore):
    for i in range(3):
        store.create_chat_session(user_id=_USER_ID, title=f"s{i}")
    resp = client.get("/chat/get-user-chat-sessions?page_size=2")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["sessions"]) == 2
    assert body["has_more"] is True


# ---------------------------------------------------------------------------
# GET /chat/get-chat-session/{session_id}
# ---------------------------------------------------------------------------


def test_get_session_not_found(client: TestClient):
    resp = client.get("/chat/get-chat-session/does-not-exist")
    assert resp.status_code == 404


def test_get_session_empty_messages(client: TestClient, store: AgenticSearchStore):
    session = store.create_chat_session(user_id=_USER_ID, title="My chat")
    resp = client.get(f"/chat/get-chat-session/{session.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == session.id
    assert body["title"] == "My chat"
    assert body["messages"] == []


def test_get_session_with_messages(client: TestClient, store: AgenticSearchStore):
    session = store.create_chat_session(user_id=_USER_ID)
    store.add_chat_message(session.id, role="user", content="Hello")
    store.add_chat_message(session.id, role="assistant", content="Hi!")
    resp = client.get(f"/chat/get-chat-session/{session.id}")
    assert resp.status_code == 200
    messages = resp.json()["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# POST /chat/create-chat-session
# ---------------------------------------------------------------------------


def test_create_session_returns_id(client: TestClient):
    resp = client.post("/chat/create-chat-session", json={"title": "New chat"})
    assert resp.status_code == 200
    body = resp.json()
    assert "chat_session_id" in body
    assert body["chat_session_id"]


def test_create_session_no_title(client: TestClient, store: AgenticSearchStore):
    resp = client.post("/chat/create-chat-session", json={})
    assert resp.status_code == 200
    session_id = resp.json()["chat_session_id"]
    session = store.get_chat_session(session_id)
    assert session is not None
    assert session.title is None


def test_create_session_anonymous_user(
    anon_client: TestClient, store: AgenticSearchStore
):
    resp = anon_client.post("/chat/create-chat-session", json={"title": "anon"})
    assert resp.status_code == 200
    session_id = resp.json()["chat_session_id"]
    session = store.get_chat_session(session_id)
    assert session is not None
    assert session.user_id is None  # anonymous → no user_id stored


# ---------------------------------------------------------------------------
# PUT /chat/rename-chat-session
# ---------------------------------------------------------------------------


def test_rename_session(client: TestClient, store: AgenticSearchStore):
    session = store.create_chat_session(user_id=_USER_ID, title="Old name")
    resp = client.put(
        "/chat/rename-chat-session",
        json={"chat_session_id": session.id, "name": "New name"},
    )
    assert resp.status_code == 200
    assert resp.json()["new_name"] == "New name"
    assert store.get_chat_session(session.id).title == "New name"


def test_rename_session_not_found(client: TestClient):
    resp = client.put(
        "/chat/rename-chat-session",
        json={"chat_session_id": "ghost", "name": "anything"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /chat/delete-chat-session/{session_id}
# ---------------------------------------------------------------------------


def test_delete_session(client: TestClient, store: AgenticSearchStore):
    session = store.create_chat_session(user_id=_USER_ID)
    resp = client.delete(f"/chat/delete-chat-session/{session.id}")
    assert resp.status_code == 200
    assert store.get_chat_session(session.id) is None


def test_delete_session_not_found(client: TestClient):
    resp = client.delete("/chat/delete-chat-session/ghost")
    assert resp.status_code == 404


def test_delete_cascades_messages(client: TestClient, store: AgenticSearchStore):
    session = store.create_chat_session(user_id=_USER_ID)
    store.add_chat_message(session.id, role="user", content="hi")
    client.delete(f"/chat/delete-chat-session/{session.id}")
    assert store.list_chat_messages(session.id) == []


# ---------------------------------------------------------------------------
# POST /chat/create-chat-message-feedback
# ---------------------------------------------------------------------------


def test_feedback_accepted(client: TestClient, store):
    session = store.create_chat_session()
    message = store.add_chat_message(session.id, role="assistant", content="answer")
    resp = client.post(
        "/chat/create-chat-message-feedback",
        json={"chat_message_id": message.id, "is_positive": True},
    )
    assert resp.status_code == 200
    assert store.get_chat_message(message.id).metadata["feedback"] == "like"


def test_feedback_negative(client: TestClient, store):
    session = store.create_chat_session()
    message = store.add_chat_message(session.id, role="assistant", content="answer")
    resp = client.post(
        "/chat/create-chat-message-feedback",
        json={
            "chat_message_id": message.id,
            "is_positive": False,
            "feedback_text": "wrong",
        },
    )
    assert resp.status_code == 200
    assert store.get_chat_message(message.id).metadata["feedback"] == "dislike"
    assert store.get_chat_message(message.id).metadata["feedback_text"] == "wrong"


# ---------------------------------------------------------------------------
# POST /chat/send-chat-message
# ---------------------------------------------------------------------------


def _make_app(*, with_model: bool) -> FastAPI:
    store = AgenticSearchStore(":memory:")
    app = FastAPI()
    app.include_router(create_chat_router(store))
    app.state.search_agent_manager = object() if with_model else None
    app.state.search_agent_tokenizer = object() if with_model else None
    return app


def test_send_chat_message_no_model_returns_400():
    client = TestClient(_make_app(with_model=False))
    resp = client.post(
        "/chat/send-chat-message", json={"message": "hi", "stream": False}
    )
    assert resp.status_code == 400
    assert "requires a local model" in resp.json()["detail"]


def test_send_chat_message_streams_answer_then_done(monkeypatch):
    from src.internal.servers.query_and_chat import chat_backend

    async def fake_run_plain_chat(message, *, on_turn=None, **kw):
        return f"echo: {message}"

    monkeypatch.setattr(chat_backend, "_run_plain_chat", fake_run_plain_chat)

    client = TestClient(_make_app(with_model=True))
    with client.stream(
        "POST", "/chat/send-chat-message", json={"message": "hello", "stream": True}
    ) as resp:
        assert resp.status_code == 200
        events = [
            json.loads(line[len("data:") :].strip())
            for line in resp.iter_lines()
            if line.startswith("data:")
        ]
    types = [e["type"] for e in events]
    assert types == ["answer", "done"]
    assert events[0]["text"] == "echo: hello"
    assert events[-1]["session_id"]


def test_chat_stream_sets_proxy_buffering_headers(monkeypatch):
    """A buffering reverse proxy would otherwise deliver the stream in one lump."""
    from src.internal.servers.query_and_chat import chat_backend

    async def fake_run_plain_chat(message, *, on_turn=None, **kw):
        return "ok"

    monkeypatch.setattr(chat_backend, "_run_plain_chat", fake_run_plain_chat)

    client = TestClient(_make_app(with_model=True))
    with client.stream(
        "POST", "/chat/send-chat-message", json={"message": "hi", "stream": True}
    ) as resp:
        assert resp.headers["cache-control"] == "no-cache"
        assert resp.headers["x-accel-buffering"] == "no"


def _seed_long(store, n=45):
    session = store.create_chat_session(user_id=_USER_ID, title="long")
    records = [
        store.add_chat_message(
            session.id, role="user" if i % 2 == 0 else "assistant", content=f"m{i}"
        )
        for i in range(n)
    ]
    return session.id, records


def _client_with(
    store,
    monkeypatch,
    *,
    llm=None,
    memory_compression=False,
    memory_auto_curate=False,
    memory_history_tokens=None,
):
    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.chat_backend.resolve_active_user",
        lambda _request, _store: _USER,
    )
    app = FastAPI()
    app.include_router(
        create_chat_router(
            store,
            llm=llm,
            memory_compression=memory_compression,
            memory_auto_curate=memory_auto_curate,
            memory_history_tokens=memory_history_tokens,
        )
    )
    app.state.search_agent_manager = object()
    app.state.search_agent_tokenizer = object()
    return TestClient(app)


def _capture_plain_chat(monkeypatch):
    captured: list = []

    async def fake_run(message, *, manager, tokenizer, history, on_turn=None, **kw):
        captured.append(list(history))
        return "ok"

    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.chat_backend._run_plain_chat", fake_run
    )
    return captured


def test_send_chat_flag_off_hands_runner_last_forty(store, monkeypatch):
    from src.internal.cache.interface import InMemoryCache
    from src.internal.memory.working import SessionMemoryState, save_state

    cache = InMemoryCache()
    monkeypatch.setattr("src.internal.cache.interface._default_cache", cache)
    session_id, records = _seed_long(store)
    save_state(
        cache,
        session_id,
        SessionMemoryState(summary="S", summarized_through=records[4].id),
    )
    captured = _capture_plain_chat(monkeypatch)

    client = _client_with(store, monkeypatch)
    resp = client.post(
        "/chat/send-chat-message",
        json={"message": "next", "session_id": session_id, "stream": False},
    )
    assert resp.status_code == 200
    assert len(captured[0]) == 40
    assert captured[0][0].role == "assistant"


def test_send_chat_flag_on_prepends_stored_summary(store, monkeypatch):
    from src.internal.cache.interface import InMemoryCache
    from src.internal.memory.working import (
        SUMMARY_PREFIX,
        SessionMemoryState,
        save_state,
    )

    cache = InMemoryCache()
    monkeypatch.setattr("src.internal.cache.interface._default_cache", cache)
    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.chat_backend.schedule_compression",
        lambda wm, **kw: None,
    )
    session_id, records = _seed_long(store)
    save_state(
        cache,
        session_id,
        SessionMemoryState(summary="S", summarized_through=records[4].id),
    )
    captured = _capture_plain_chat(monkeypatch)

    client = _client_with(store, monkeypatch, llm=object(), memory_compression=True)
    client.post(
        "/chat/send-chat-message",
        json={"message": "next", "session_id": session_id, "stream": False},
    )
    assert len(captured[0]) == 41
    assert captured[0][0].role == "system"
    assert captured[0][0].content == SUMMARY_PREFIX + "S"


def test_send_chat_schedules_compression(store, monkeypatch):
    from src.internal.cache.interface import InMemoryCache

    monkeypatch.setattr("src.internal.cache.interface._default_cache", InMemoryCache())
    session_id, _ = _seed_long(store)
    _capture_plain_chat(monkeypatch)
    scheduled: list = []

    def fake_schedule(wm, **kw):
        scheduled.append(
            (
                len(wm.pending),
                kw["enabled"],
                kw["llm"],
                kw["store"],
                kw["user_id"],
                kw["auto_curate"],
            )
        )
        return None

    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.chat_backend.schedule_compression",
        fake_schedule,
    )
    sentinel = object()
    client = _client_with(
        store,
        monkeypatch,
        llm=sentinel,
        memory_compression=True,
        memory_auto_curate=True,
    )
    client.post(
        "/chat/send-chat-message",
        json={"message": "next", "session_id": session_id, "stream": False},
    )
    assert scheduled == [(5, True, sentinel, store, _USER_ID, True)]


class _SummaryLLM:
    def complete(self, messages, **kwargs):
        return "S"


async def _await(task):
    return await task


def _seed_sized(store, n, chars=4000):
    session = store.create_chat_session(user_id=_USER_ID, title="long")
    for i in range(n):
        store.add_chat_message(
            session.id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"{i}:".ljust(chars, "x"),
        )
    return session.id


def test_send_chat_passes_the_token_budget(store, monkeypatch):
    session_id = _seed_sized(store, 10)
    captured = _capture_plain_chat(monkeypatch)
    client = _client_with(store, monkeypatch, memory_history_tokens=2500)
    client.post(
        "/chat/send-chat-message",
        json={"message": "next", "session_id": session_id, "stream": False},
    )
    assert len(captured[0]) == 2


@pytest.mark.parametrize("compression", [False, True])
def test_send_chat_compresses_only_when_direct_compression_is_on(
    store, monkeypatch, compression
):
    from src.internal.cache.interface import InMemoryCache
    from src.internal.servers.query_and_chat import chat_backend

    monkeypatch.setattr("src.internal.cache.interface._default_cache", InMemoryCache())
    session_id = _seed_sized(store, 10)
    _capture_plain_chat(monkeypatch)
    tasks: list = []
    real = chat_backend.schedule_compression

    def spy(wm, **kw):
        tasks.append(real(wm, **kw))
        return tasks[-1]

    monkeypatch.setattr(chat_backend, "schedule_compression", spy)
    with _client_with(
        store,
        monkeypatch,
        llm=_SummaryLLM(),
        memory_compression=compression,
        memory_history_tokens=2500,
    ) as client:
        client.post(
            "/chat/send-chat-message",
            json={"message": "next", "session_id": session_id, "stream": False},
        )
        assert (tasks[0] is not None) is compression
        if compression:
            assert client.portal.call(_await, tasks[0]) is True
