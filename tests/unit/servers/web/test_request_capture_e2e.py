"""End-to-end coverage: Dev Console request capture through the real /api/agent
endpoint. Verifies the flag-on path records a run (with a `final` stage
reachable via GET /api/debug/request/{id}) and the flag-off path exposes no
debug surface at all (the debug router itself isn't mounted).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.agents.search import AgenticRAGResult
from src.context.models import SearchContextBundle
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app
from src.internal.servers.web.intent import RouteDecision, RouteStrategy
from src.internal.auth import generate_user_jwt_token
from src.internal.db import UserRecord


def _stub_agentic_rag_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub AgenticRAGLoop.run so the pipeline completes with no local model
    and no network call (mirrors test_auto_route_agentic_rag_for_chat in
    test_web_experience_app.py)."""
    fake_result = AgenticRAGResult(
        answer="Grounded answer [D1]",
        citations=[],
        context=SearchContextBundle(query="vector database", documents=[]),
        rounds_used=1,
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.AgenticRAGLoop.run",
        AsyncMock(return_value=fake_result),
    )


def _route_to_chat_with_fake_rag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the CHAT route and stub AgenticRAGLoop.run so the pipeline
    completes with no local model and no network call (mirrors
    test_auto_route_agentic_rag_for_chat in test_web_experience_app.py)."""
    monkeypatch.setattr(
        "src.internal.servers.web.app.recognize_intent",
        lambda *a, **k: RouteDecision(RouteStrategy.CHAT),
    )
    _stub_agentic_rag_run(monkeypatch)


@pytest.fixture
def web_client_debug_on(monkeypatch, tmp_path):
    # Exercise the real recognize_intent -> classify_route cascade so the `intent` stage
    # (emitted by classify_route via record_stage) is captured end-to-end
    # alongside `final`, instead of bypassing the classifier entirely.
    _stub_agentic_rag_run(monkeypatch)
    fake_llm = MagicMock()
    fake_llm.complete.return_value = "chat"
    app = create_web_app(
        SearchExperienceSettings(
            db_path=tmp_path / "debug_on.sqlite3", debug_panels=True
        ),
        llm=fake_llm,
    )
    with TestClient(app) as client:
        app.state.auth_store.upsert_user(
            UserRecord(id="debug-admin", metadata={"role": "admin"})
        )
        client.cookies.set(
            "fastapiusersauth", generate_user_jwt_token(user_id="debug-admin")
        )
        yield client


@pytest.fixture
def web_client_debug_off(monkeypatch, tmp_path):
    _route_to_chat_with_fake_rag(monkeypatch)
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "debug_off.sqlite3"),
        llm=MagicMock(),
    )
    with TestClient(app) as client:
        yield client


def test_auto_routed_request_is_captured_when_flag_on(web_client_debug_on):
    # A multi-word, non-bare-lookup query with no tool/search/chat-start cues
    # (not decided by the deterministic _regex_route pass) reaches the real
    # classify_route, whose fake "chat" completion routes to CHAT.
    response = web_client_debug_on.post(
        "/api/agent",
        json={"query": "vector databases and embedding storage internals"},
    )
    assert response.status_code == 200

    listed = web_client_debug_on.get("/api/debug/requests").json()["requests"]
    assert listed, "expected a captured run"

    snap = web_client_debug_on.get(
        f"/api/debug/request/{listed[0]['request_id']}"
    ).json()
    stages = {s["stage"] for s in snap["stages"]}
    assert "intent" in stages
    assert "final" in stages


def test_no_capture_when_flag_off(web_client_debug_off):
    response = web_client_debug_off.post(
        "/api/agent", json={"query": "vector database"}
    )
    assert response.status_code == 200

    # debug_panels=False means the debug router is never mounted (see
    # test_debug_panels_gate.py), so the endpoint doesn't exist at all rather
    # than returning an empty list.
    assert web_client_debug_off.get("/api/debug/requests").status_code == 404
