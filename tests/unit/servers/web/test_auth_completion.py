"""Regressions for authentication gaps found during the route audit."""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from src.internal.auth import generate_user_jwt_token
from src.internal.db import AgenticSearchStore, UserRecord
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


@pytest.fixture
def context(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DEV_ADMIN", "false")
    with AgenticSearchStore(":memory:") as store:
        for name in ("alice", "bob"):
            store.upsert_user(
                UserRecord(
                    id=name, metadata={"role": "admin" if name == "alice" else "basic"}
                )
            )
        app = create_web_app(SearchExperienceSettings(debug_panels=True), store=store)
        app.state.search_agent_manager = object()
        app.state.search_agent_tokenizer = object()
        yield TestClient(app), store


def _headers(user_id):
    return {
        "Authorization": "Bearer "
        + generate_user_jwt_token(
            user_id=user_id, extra={"role": "admin" if user_id == "alice" else "basic"}
        )
    }


@pytest.mark.parametrize("path", ["/chat/send-chat-message", "/tool/send-tool-message"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("caller", [None, "bob"])
def test_message_routes_refuse_other_users_sessions(context, path, stream, caller):
    client, store = context
    session = store.create_chat_session(user_id="alice")
    store.add_chat_message(session.id, role="user", content="private transcript")
    response = client.post(
        path,
        json={"message": "intrusion", "session_id": session.id, "stream": stream},
        headers=_headers(caller) if caller else {},
    )
    assert response.status_code == 404
    assert [m.content for m in store.list_chat_messages(session.id)] == [
        "private transcript"
    ]


@pytest.mark.parametrize("caller", [None, "bob", "alice"])
def test_session_creation_uses_caller_not_body_identity(context, caller):
    client, store = context
    response = client.post(
        "/api/sessions",
        json={"user_id": "alice"},
        headers=_headers(caller) if caller else {},
    )
    assert response.status_code == 200
    assert store.get_chat_session(response.json()["id"]).user_id == caller


def test_agent_session_attribution_cannot_be_forged(context):
    from src.internal.servers.web.app import AgentExperienceRequest, _ensure_session

    _, store = context
    session_id = _ensure_session(
        store, AgentExperienceRequest(query="hello", user_id="alice")
    )
    assert store.get_chat_session(session_id).user_id is None


@pytest.mark.parametrize("kind", ["message", "retrieval"])
@pytest.mark.parametrize("caller", [None, "bob", "alice"])
def test_feedback_respects_session_ownership(context, kind, caller):
    client, store = context
    session = store.create_chat_session(user_id="alice")
    message = store.add_chat_message(session.id, role="assistant", content="private")
    if kind == "message":
        path = "/chat/create-chat-message-feedback"
        body = {"chat_message_id": message.id, "is_positive": True}
    else:
        path = "/api/feedback"
        body = {"session_id": session.id, "signal": "thumbs_up"}
    response = client.post(path, json=body, headers=_headers(caller) if caller else {})
    assert response.status_code == (200 if caller == "alice" else 404)
    if caller != "alice":
        assert store.list_chat_messages(session.id)[0].metadata == {}
        assert store.list_retrieval_feedback() == []


@pytest.mark.parametrize(
    "path", ["/api/debug/requests", "/api/debug/tools", "/api/debug/latency"]
)
@pytest.mark.parametrize("caller,status", [(None, 401), ("bob", 403), ("alice", 200)])
def test_debug_routes_require_admin(context, path, caller, status):
    client, _ = context
    assert (
        client.get(path, headers=_headers(caller) if caller else {}).status_code
        == status
    )


@pytest.mark.parametrize(
    "path", ["/me", "/me/permissions", "/manage/user-groups", "/scim/v2/tokens"]
)
def test_deactivated_account_cannot_use_existing_token(context, path):
    client, store = context
    record = store.get_user("alice")
    store.upsert_user(replace(record, metadata={**record.metadata, "is_active": False}))
    assert client.get(path, headers=_headers("alice")).status_code == 401


def test_demoted_admin_cannot_use_stale_role_claim(context):
    client, store = context
    record = store.get_user("alice")
    store.upsert_user(replace(record, metadata={"role": "basic"}))
    assert client.get("/scim/v2/tokens", headers=_headers("alice")).status_code == 403


def test_deleted_admin_cannot_use_existing_token(context):
    client, store = context
    store.delete_user("alice")
    assert client.get("/scim/v2/tokens", headers=_headers("alice")).status_code == 401


def test_deactivated_user_cannot_create_owned_session(context):
    client, store = context
    record = store.get_user("alice")
    store.upsert_user(replace(record, metadata={**record.metadata, "is_active": False}))
    response = client.post(
        "/chat/create-chat-session", json={}, headers=_headers("alice")
    )
    assert response.status_code == 200
    assert store.get_chat_session(response.json()["chat_session_id"]).user_id is None


def test_strict_memory_rejects_a_deactivated_web_account(context):
    _, store = context
    app = create_web_app(
        SearchExperienceSettings(memory_require_auth=True), store=store
    )
    client = TestClient(app)
    assert client.get("/api/memory/list", headers=_headers("alice")).status_code == 200
    record = store.get_user("alice")
    store.upsert_user(replace(record, metadata={**record.metadata, "is_active": False}))
    assert client.get("/api/memory/list", headers=_headers("alice")).status_code == 401


@pytest.mark.parametrize("path", ["/me", "/me/permissions", "/scim/v2/tokens"])
def test_scim_deprovisioning_revokes_existing_tokens(context, path):
    from src.internal.servers.scim.dal import ScimDAL

    client, store = context
    dal = ScimDAL(store)
    dal.deactivate_user(dal.get_user("alice"))
    assert client.get(path, headers=_headers("alice")).status_code == 401


def test_scim_deactivation_prevents_login_until_admin_reactivates(context):
    from src.internal.servers.scim.dal import ScimDAL

    client, store = context
    credentials = {"email": "carol@example.test", "password": "test-only-password"}
    registered = client.post("/auth/register", json=credentials).json()
    dal = ScimDAL(store)
    dal.deactivate_user(dal.get_user(registered["id"]))
    form = {"username": credentials["email"], "password": credentials["password"]}
    assert client.post("/auth/login", data=form).status_code == 403
    response = client.patch(
        "/manage/admin/activate-user",
        json={"user_email": credentials["email"]},
        headers=_headers("alice"),
    )
    assert response.status_code == 200
    assert response.json()["is_active"] is True
    assert client.post("/auth/login", data=form).status_code == 200
    assert client.get("/me").status_code == 200


def test_scim_http_delete_revokes_web_identity(context):
    client, store = context
    store.create_scim_user_mapping("bob", external_id="idp-bob")
    minted = client.post(
        "/scim/v2/tokens", json={"name": "test-provisioner"}, headers=_headers("alice")
    )
    assert minted.status_code == 201
    scim_headers = {"Authorization": "Bearer " + minted.json()["raw_token"]}
    assert client.delete("/scim/v2/Users/bob", headers=scim_headers).status_code == 204
    assert client.get("/me", headers=_headers("bob")).status_code == 401


def test_user_listing_filters_on_scim_deactivation(context):
    client, store = context
    store.set_user_active("bob", False)
    response = client.get(
        "/manage/users/accepted",
        params={"is_active": "false"},
        headers=_headers("alice"),
    )
    assert response.status_code == 200
    assert [u["id"] for u in response.json()["items"]] == ["bob"]
    assert response.json()["items"][0]["is_active"] is False


@pytest.mark.parametrize("path", ["/chat/send-chat-message", "/tool/send-tool-message"])
@pytest.mark.parametrize("owner", [None, "alice"])
def test_authorized_callers_can_continue_sessions(context, monkeypatch, path, owner):
    client, store = context
    session = store.create_chat_session(user_id=owner)
    store.add_chat_message(session.id, role="user", content="previous question")

    async def chat_answer(*args, **kwargs):
        return "continued answer"

    async def tool_answer(*args, **kwargs):
        return "continued answer", [], [], None, {}

    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.chat_backend._run_plain_chat", chat_answer
    )
    monkeypatch.setattr(
        "src.internal.servers.web.tool_agent_runner._run_tool_agent", tool_answer
    )
    response = client.post(
        path,
        json={"message": "follow up", "session_id": session.id, "stream": False},
        headers=_headers(owner) if owner else {},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "continued answer"
    assert [m.content for m in store.list_chat_messages(session.id)] == [
        "previous question",
        "follow up",
        "continued answer",
    ]
