"""Contracts for minting the WebSocket channel's single-use token.

The token is the whole of the socket's authentication, so these exercise the
real `store_ws_token` logic -- rate limit, TTL, atomic single-use consumption --
against an in-memory double, rather than stubbing the functions under test.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from src.internal.auth import generate_user_jwt_token
from src.internal.db import AgenticSearchStore, UserRecord
from src.internal.servers.redis import redis_pool
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app
from tests.unit.servers.web._ws_fakes import FakeRedis

_USER = "user-1"


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


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {generate_user_jwt_token(user_id=_USER)}"}


def test_anonymous_caller_cannot_mint_a_token(client: TestClient, fake_redis):
    response = client.post("/api/agent/ws-token")

    assert response.status_code == 401


def test_authenticated_caller_gets_a_token_and_its_lifetime(
    client: TestClient, fake_redis
):
    response = client.post("/api/agent/ws-token", headers=_auth())

    assert response.status_code == 200
    body = response.json()
    assert body["token"]
    assert body["expires_in"] == redis_pool.WS_TOKEN_TTL_SECONDS


def test_token_is_single_use(client: TestClient, fake_redis):
    token = client.post("/api/agent/ws-token", headers=_auth()).json()["token"]

    first = asyncio.run(redis_pool.retrieve_ws_token_data(token))
    second = asyncio.run(redis_pool.retrieve_ws_token_data(token))

    assert first == {"sub": _USER}
    assert second is None, "a replayed token must not authenticate"


def test_an_unknown_token_does_not_authenticate(client: TestClient, fake_redis):
    assert asyncio.run(redis_pool.retrieve_ws_token_data("never-minted")) is None


def test_minting_is_rate_limited_per_user(client: TestClient, fake_redis):
    for _ in range(redis_pool.WS_TOKEN_RATE_LIMIT_MAX):
        assert client.post("/api/agent/ws-token", headers=_auth()).status_code == 200

    refused = client.post("/api/agent/ws-token", headers=_auth())

    assert refused.status_code == 429


def test_redis_unavailable_reports_the_transport_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """503, not 500: the socket is unavailable, and SSE is unaffected."""

    async def _boom():
        raise ConnectionError("redis down")

    monkeypatch.setattr(redis_pool, "get_async_redis_connection", _boom)

    response = client.post("/api/agent/ws-token", headers=_auth())

    assert response.status_code == 503


def test_token_does_not_survive_its_ttl(client: TestClient, fake_redis: FakeRedis):
    """A token past WS_TOKEN_TTL_SECONDS authenticates nobody."""
    token = client.post("/api/agent/ws-token", headers=_auth()).json()["token"]

    fake_redis.advance(redis_pool.WS_TOKEN_TTL_SECONDS + 1)

    assert asyncio.run(redis_pool.retrieve_ws_token_data(token)) is None


def test_an_expired_token_is_refused_by_the_socket(
    client: TestClient, fake_redis: FakeRedis
):
    from starlette.websockets import WebSocketDisconnect

    token = client.post("/api/agent/ws-token", headers=_auth()).json()["token"]
    fake_redis.advance(redis_pool.WS_TOKEN_TTL_SECONDS + 1)

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/api/agent/ws?token={token}") as ws:
            ws.receive_json()
