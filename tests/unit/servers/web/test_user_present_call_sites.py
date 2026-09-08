"""`user_present` is entitlement, so it may only come from the authenticated user.

`request.user_id` is bookkeeping — a client picks it and it names a session.
Deriving "is there a user?" from it would let an unauthenticated caller type
someone else's id and be handed that user's memory tools.

The predicate used to be spelled out by hand at each call site
(`auth_user is not None and not auth_user.is_anonymous`), which is one edit away
from `user_id is not None` and reads almost the same. `capabilities.user_present`
is now the single spelling; these tests pin the behaviour at the routes so the
shortcut fails the suite rather than the review.
"""

from __future__ import annotations

import types

import pytest
from fastapi.testclient import TestClient

import src.internal.servers.web.app as web_app
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


@pytest.fixture
def seen() -> dict:
    return {}


@pytest.fixture
def patched_tool_agent(monkeypatch, seen):
    async def _fake_tool_agent(query, **kwargs):
        seen["user_present"] = kwargs.get("user_present")
        return ("tool answer", [], [], "tool", {})

    monkeypatch.setattr(web_app, "_run_tool_agent", _fake_tool_agent)
    monkeypatch.setattr(
        "src.internal.servers.web.tool_agent_runner._run_tool_agent", _fake_tool_agent
    )
    return _fake_tool_agent


def _client(tmp_path, name: str) -> TestClient:
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / f"{name}.sqlite3"))
    app.state.search_agent_manager = types.SimpleNamespace()
    app.state.search_agent_tokenizer = types.SimpleNamespace()
    return TestClient(app)


def test_tool_agent_mode_withholds_user_scoped_tools_from_a_claimed_user_id(
    patched_tool_agent, seen, tmp_path
):
    response = _client(tmp_path, "tool_mode").post(
        "/api/agent",
        json={
            "query": "remember this",
            "mode": "tool_agent",
            # Unauthenticated, but naming a user. Nothing about entitlement may
            # follow from this field.
            "user_id": "victim",
        },
    )

    assert response.status_code == 200, response.text
    assert seen["user_present"] is False


def test_auto_route_withholds_user_scoped_tools_from_a_claimed_user_id(
    monkeypatch, patched_tool_agent, seen, tmp_path
):
    from src.internal.servers.web.intent import RouteDecision, RouteStrategy

    monkeypatch.setattr(
        web_app,
        "recognize_intent",
        lambda *a, **k: RouteDecision(RouteStrategy.TOOL),
    )

    response = _client(tmp_path, "auto_route").post(
        "/api/agent",
        json={"query": "remember this", "user_id": "victim"},
    )

    assert response.status_code == 200, response.text
    assert seen["user_present"] is False


def test_send_tool_message_withholds_user_scoped_tools_when_anonymous(
    patched_tool_agent, seen, tmp_path
):
    # The tool surface has its own router; it resolves the caller itself, so it
    # needs its own proof rather than inheriting /api/agent's.
    response = _client(tmp_path, "tool_router").post(
        "/tool/send-tool-message",
        json={"message": "remember this", "stream": False, "run_search_tool": False},
    )

    assert response.status_code == 200, response.text
    assert seen["user_present"] is False
