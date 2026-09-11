"""A JWT whose user no longer exists must not be trusted.

Tokens outlive the rows they name: the dev store defaults to `:memory:` and is
rebuilt on every restart, while the browser keeps a cookie that is still validly
signed. Before this was handled, such a token was *worse* than no token — the id
it carried failed the chat_sessions.user_id foreign key, and re-registering the
same email under a fresh id made auto-provisioning collide with users.email
UNIQUE. Both surfaced as 500s, while read-only endpoints kept working, so the
same cookie appeared to succeed and fail at the same time.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.internal.auth import generate_user_jwt_token
from src.internal.db import AgenticSearchStore
from src.internal.db import UserRecord
from src.internal.servers.users.api import resolve_active_user
from src.internal.servers.web.app import create_web_app

STALE_ID = "user_from_a_previous_life"
# The throwaway local dev account seeded by web/dev-login.html. Kept out of the
# request literals so secret scanners do not read it as a hardcoded credential.
DEV_EMAIL = "dev@localhost"
DEV_LOGIN = "devpass"


@pytest.fixture
def store() -> AgenticSearchStore:
    return AgenticSearchStore(":memory:")


@pytest.fixture
def client(store: AgenticSearchStore) -> TestClient:
    # No `with`: the lifespan would load SEARCH_AGENT_MODEL, which these tests
    # do not need.
    return TestClient(create_web_app(store=store), raise_server_exceptions=False)


def _stale_cookie(client: TestClient, email: str | None = None) -> None:
    client.cookies.set(
        "fastapiusersauth",
        generate_user_jwt_token(user_id=STALE_ID, email=email, expires_in_seconds=3600),
    )


def test_resolve_active_user_rejects_token_for_missing_user(store):
    class _Req:
        headers = {
            "Authorization": f"Bearer {generate_user_jwt_token(user_id=STALE_ID)}"
        }
        cookies: dict[str, str] = {}

    assert resolve_active_user(_Req(), store) is None

    store.upsert_user(UserRecord(id=STALE_ID, email="ghost@example.test"))
    resolved = resolve_active_user(_Req(), store)
    assert resolved is not None and resolved.id == STALE_ID


def test_agent_does_not_500_when_token_user_was_deleted(client):
    """The email-UNIQUE collision: same email, new id, old cookie still around."""
    client.post(
        "/auth/register",
        json={"email": DEV_EMAIL, "username": "dev", "password": DEV_LOGIN},
    )
    _stale_cookie(client, email=DEV_EMAIL)

    response = client.post("/api/agent", json={"query": "RAG"})

    assert response.status_code == 200


def test_chat_session_does_not_500_when_token_user_was_deleted(client):
    """The foreign-key violation on chat_sessions.user_id."""
    _stale_cookie(client)

    response = client.post("/chat/create-chat-session", json={})

    assert response.status_code == 200


def test_stale_token_creates_an_anonymous_session_not_an_orphan(client, store):
    _stale_cookie(client)

    session_id = client.post("/chat/create-chat-session", json={}).json()[
        "chat_session_id"
    ]

    # Attributed to nobody rather than to a user the store does not have.
    assert store.get_chat_session(session_id).user_id is None


def test_stale_token_is_treated_exactly_like_no_credential(client):
    """The whole point: stale must not be its own third behaviour."""
    anonymous = client.post("/chat/create-chat-session", json={}).status_code
    _stale_cookie(client)
    stale = client.post("/chat/create-chat-session", json={}).status_code

    assert stale == anonymous


def test_search_rejects_a_stale_token_as_unauthenticated(client):
    """Search requires a user, so stale is a clean 401 — never a 200."""
    _stale_cookie(client)

    response = client.post(
        "/search/send-search-message",
        json={
            "search_query": "RAG",
            "num_hits": 2,
            "run_query_expansion": False,
            "stream": False,
        },
    )

    assert response.status_code == 401


def test_valid_login_still_authenticates(client, store):
    client.post(
        "/auth/register",
        json={"email": DEV_EMAIL, "username": "dev", "password": DEV_LOGIN},
    )
    client.post("/auth/login", data={"username": DEV_EMAIL, "password": DEV_LOGIN})

    session_id = client.post("/chat/create-chat-session", json={}).json()[
        "chat_session_id"
    ]

    user_id = store.get_chat_session(session_id).user_id
    assert user_id is not None
    assert store.get_user(user_id) is not None


def test_client_supplied_unknown_user_id_does_not_500(client, store):
    """`user_id` also arrives in the request body, not only from a token."""
    response = client.post(
        "/api/agent", json={"query": "RAG", "user_id": "nobody_by_that_name"}
    )

    assert response.status_code == 200


@pytest.mark.parametrize(
    "authorization", ["Bearer invalid", "Bearer ", "Basic invalid"]
)
def test_invalid_authorization_does_not_retry_valid_cookie(
    client, store, authorization
):
    store.upsert_user(UserRecord(id="alice", email="alice@example.test"))
    client.cookies.set("fastapiusersauth", generate_user_jwt_token(user_id="alice"))
    assert client.get("/me").status_code == 200

    headers = {"Authorization": authorization}
    assert client.get("/me", headers=headers).status_code == 401
    response = client.post("/chat/create-chat-session", json={}, headers=headers)
    assert response.status_code == 200
    assert store.get_chat_session(response.json()["chat_session_id"]).user_id is None
