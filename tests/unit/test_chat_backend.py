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
