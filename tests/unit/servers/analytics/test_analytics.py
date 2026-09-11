"""Tests for src/servers/analytics and AgenticSearchStore analytics queries."""

from __future__ import annotations


from fastapi.testclient import TestClient

from src.internal.auth import generate_user_jwt_token
from src.internal.configs import AppSettings
from src.internal.configs import AuthSettings
from src.internal.db import AgenticSearchStore
from src.internal.db.models import UserRecord
from src.internal.servers.web.app import SearchExperienceSettings
from src.internal.servers.web.app import create_web_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DAY1 = "2024-03-01T10:00:00+00:00"
_DAY1_END = "2024-03-01T23:59:00+00:00"
_DAY2 = "2024-03-02T09:00:00+00:00"
_RANGE_START = "2024-02-29T00:00:00+00:00"
_RANGE_END = "2024-03-03T00:00:00+00:00"

_ADMIN_ID = "admin-user"
_USER_A = "user-a"
_USER_B = "user-b"


def _make_store(tmp_path) -> AgenticSearchStore:
    store = AgenticSearchStore(tmp_path / "test.sqlite3")
    # Register users so the FK constraint on chat_sessions.user_id is satisfied.
    for uid, email in (
        (_ADMIN_ID, "admin@test.local"),
        (_USER_A, "a@test.local"),
        (_USER_B, "b@test.local"),
    ):
        store.upsert_user(UserRecord(id=uid, email=email))
    # Session 1: admin on day 1, 2 messages
    s1 = store.create_chat_session(user_id=_ADMIN_ID)
    store._conn.execute(
        "UPDATE chat_sessions SET created_at = ? WHERE id = ?", (_DAY1, s1.id)
    )
    store.add_chat_message(s1.id, role="user", content="q1")
    store.add_chat_message(s1.id, role="assistant", content="a1")
    store._conn.execute(
        "UPDATE chat_messages SET created_at = ? WHERE session_id = ?",
        (_DAY1, s1.id),
    )
    # Session 2: user_a on day 1, 1 message
    s2 = store.create_chat_session(user_id=_USER_A)
    store._conn.execute(
        "UPDATE chat_sessions SET created_at = ? WHERE id = ?", (_DAY1, s2.id)
    )
    store.add_chat_message(s2.id, role="user", content="q2")
    store._conn.execute(
        "UPDATE chat_messages SET created_at = ? WHERE session_id = ?",
        (_DAY1, s2.id),
    )
    # Session 3: user_b on day 2, no messages
    s3 = store.create_chat_session(user_id=_USER_B)
    store._conn.execute(
        "UPDATE chat_sessions SET created_at = ? WHERE id = ?", (_DAY2, s3.id)
    )
    # Session 4: anonymous on day 1 (no user_id)
    s4 = store.create_chat_session(user_id=None)
    store._conn.execute(
        "UPDATE chat_sessions SET created_at = ? WHERE id = ?", (_DAY1, s4.id)
    )
    store._conn.commit()
    return store


# ---------------------------------------------------------------------------
# Store-level analytics tests
# ---------------------------------------------------------------------------


def test_query_session_analytics_counts_sessions_and_messages(tmp_path):
    store = _make_store(tmp_path)
    rows = store.query_session_analytics(_RANGE_START, _RANGE_END)
    by_day = {day: (sessions, messages) for day, sessions, messages in rows}

    # Day 1: 4 sessions (admin, user_a, user_b on day2, anonymous on day1)
    # Actually: 3 sessions on day1 (admin, user_a, anonymous)
    assert by_day["2024-03-01"][0] == 3  # sessions
    assert by_day["2024-03-01"][1] == 3  # messages (2 + 1 + 0)
    # Day 2: 1 session, 0 messages
    assert by_day["2024-03-02"][0] == 1
    assert by_day["2024-03-02"][1] == 0
    store.close()


def test_query_session_analytics_respects_date_range(tmp_path):
    store = _make_store(tmp_path)
    # Only day 2
    rows = store.query_session_analytics(_DAY2, _RANGE_END)
    assert len(rows) == 1
    assert rows[0][0] == "2024-03-02"
    store.close()


def test_user_session_analytics_counts_distinct_users(tmp_path):
    store = _make_store(tmp_path)
    rows = store.user_session_analytics(_RANGE_START, _RANGE_END)
    by_day = {day: active for day, active in rows}

    # Day 1: admin + user_a = 2 (anonymous session excluded)
    assert by_day["2024-03-01"] == 2
    # Day 2: user_b = 1
    assert by_day["2024-03-02"] == 1
    store.close()


def test_analytics_returns_empty_for_out_of_range(tmp_path):
    store = _make_store(tmp_path)
    rows = store.query_session_analytics(
        "2020-01-01T00:00:00+00:00", "2020-12-31T00:00:00+00:00"
    )
    assert rows == []
    store.close()


# ---------------------------------------------------------------------------
# Analytics API endpoint tests
# ---------------------------------------------------------------------------


def _admin_app(tmp_path):
    store = _make_store(tmp_path)
    # super_users uses the default dev secret so user_from_headers can validate tokens.
    settings = AppSettings(auth=AuthSettings(super_users=(_ADMIN_ID,)))
    web = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "test.sqlite3"),
        app_settings=settings,
        store=store,
    )
    return TestClient(web), store


def _bearer(user_id: str) -> dict:
    token = generate_user_jwt_token(user_id=user_id)
    return {"Authorization": f"Bearer {token}"}


def test_analytics_query_endpoint_returns_daily_rows(tmp_path):
    client, store = _admin_app(tmp_path)
    response = client.get(
        "/analytics/query",
        params={"start": _RANGE_START, "end": _RANGE_END},
        headers=_bearer(_ADMIN_ID),
    )
    assert response.status_code == 200
    data = response.json()
    assert any(row["date"] == "2024-03-01" for row in data)
    day1 = next(r for r in data if r["date"] == "2024-03-01")
    assert day1["total_sessions"] == 3
    assert day1["total_messages"] == 3
    store.close()


def test_analytics_user_endpoint_returns_daily_active_users(tmp_path):
    client, store = _admin_app(tmp_path)
    response = client.get(
        "/analytics/user",
        params={"start": _RANGE_START, "end": _RANGE_END},
        headers=_bearer(_ADMIN_ID),
    )
    assert response.status_code == 200
    data = response.json()
    day1 = next(r for r in data if r["date"] == "2024-03-01")
    assert day1["total_active_users"] == 2
    store.close()


def test_analytics_rejects_unauthenticated_request(tmp_path):
    client, store = _admin_app(tmp_path)
    assert client.get("/analytics/query").status_code == 401
    store.close()


def test_analytics_rejects_non_admin_user(tmp_path):
    client, store = _admin_app(tmp_path)
    response = client.get("/analytics/query", headers=_bearer(_USER_A))
    assert response.status_code == 403
    store.close()


def test_analytics_uses_default_30_day_window_when_no_params(tmp_path):
    store = AgenticSearchStore(tmp_path / "live.sqlite3")
    store.upsert_user(UserRecord(id=_ADMIN_ID))
    settings = AppSettings(auth=AuthSettings(super_users=(_ADMIN_ID,)))
    web = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "live.sqlite3"),
        app_settings=settings,
        store=store,
    )
    client = TestClient(web)
    # No start/end — should not raise, just return an empty list for a fresh store.
    response = client.get("/analytics/query", headers=_bearer(_ADMIN_ID))
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    store.close()


def test_analytics_by_llm_groups_sessions(tmp_path):
    client, store = _admin_app(tmp_path)
    response = client.get(
        "/analytics/by-llm",
        params={"start": _RANGE_START, "end": _RANGE_END},
        headers=_bearer(_ADMIN_ID),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["dimension"] == "llm"
    # All 4 sessions have NULL llm_name → coalesced to 'unknown'
    assert data["total_sessions"] == 4
    assert len(data["items"]) == 1
    assert data["items"][0]["label"] == "unknown"
    assert data["items"][0]["session_count"] == 4
    store.close()


def test_analytics_by_persona_groups_sessions(tmp_path):
    client, store = _admin_app(tmp_path)
    response = client.get(
        "/analytics/by-persona",
        params={"start": _RANGE_START, "end": _RANGE_END},
        headers=_bearer(_ADMIN_ID),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["dimension"] == "persona"
    # All 4 sessions have NULL persona_id → coalesced to 'default'
    assert data["total_sessions"] == 4
    assert len(data["items"]) == 1
    assert data["items"][0]["label"] == "default"
    assert data["items"][0]["session_count"] == 4
    store.close()


def test_analytics_by_flow_groups_sessions(tmp_path):
    client, store = _admin_app(tmp_path)
    response = client.get(
        "/analytics/by-flow",
        params={"start": _RANGE_START, "end": _RANGE_END},
        headers=_bearer(_ADMIN_ID),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["dimension"] == "flow"
    # All 4 sessions have NULL flow_type → coalesced to 'chat'
    assert data["total_sessions"] == 4
    assert len(data["items"]) == 1
    assert data["items"][0]["label"] == "chat"
    assert data["items"][0]["session_count"] == 4
    store.close()
