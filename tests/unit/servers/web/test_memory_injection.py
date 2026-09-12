"""End-to-end: the /api/agent dispatch threads a signed-in caller's memory into
the answer path, and only a signed-in caller's.

AGENTIC_SEARCH_MEMORY_INJECTION and SearchExperienceSettings.memory_injection
are gone: memory is no longer an opt-in flag. It is built by
`resolve_capabilities` whenever an *authenticated* user resolves — that is
what signing in is for. A client-supplied `user_id` in the request body is not
authentication: an anonymous caller who names another user's id in the body
must not receive that user's memory, and must not receive a widened access
ACL either. (A prior version of this wiring used
`request.user_id or capabilities.user_id` for entitlement, which let an
unauthenticated caller name any user and receive that user's access — see
src/internal/access/capabilities.py.)
"""

from fastapi.testclient import TestClient

from src.context.models import (
    AnswerGenerationResult,
    PromptBundle,
    SearchContextBundle,
    SearchFilters,
)
from src.internal.auth import generate_user_jwt_token
from src.internal.db.models import UserRecord
from src.internal.db.store import AgenticSearchStore
from src.internal.servers.web import app as web_app
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


def _capturing_awr(captured: dict):
    async def fake_answer_with_retrieval(question, **kwargs):
        captured["user_memory"] = kwargs.get("user_memory")
        captured["filters"] = kwargs.get("filters")
        ctx = SearchContextBundle(query=question, documents=[])
        return AnswerGenerationResult(
            answer="ok",
            citations=[],
            context=ctx,
            prompt=PromptBundle(system="", user="", messages=[]),
        )

    return fake_answer_with_retrieval


def test_memory_injected_for_signed_in_user(monkeypatch):
    store = AgenticSearchStore(":memory:")
    store.upsert_user(UserRecord(id="u1"))
    store.add_user_memory("u1", "User is allergic to peanuts")
    captured: dict = {}
    monkeypatch.setattr(web_app, "answer_with_retrieval", _capturing_awr(captured))

    app = create_web_app(SearchExperienceSettings(), store=store)
    client = TestClient(app)
    client.cookies.set("fastapiusersauth", generate_user_jwt_token(user_id="u1"))
    resp = client.post(
        "/api/agent",
        json={"query": "Recommend Thai food", "mode": "chat_once"},
    )

    assert resp.status_code == 200
    assert captured["user_memory"] is not None
    assert "allergic to peanuts" in captured["user_memory"]
    store.close()


def test_memory_not_injected_for_anonymous(monkeypatch):
    store = AgenticSearchStore(":memory:")
    captured: dict = {}
    monkeypatch.setattr(web_app, "answer_with_retrieval", _capturing_awr(captured))

    app = create_web_app(SearchExperienceSettings(), store=store)
    resp = TestClient(app).post(
        "/api/agent",
        json={"query": "Recommend Thai food", "mode": "chat_once"},
    )

    assert resp.status_code == 200
    assert captured["user_memory"] is None
    store.close()


def test_client_supplied_user_id_without_auth_gets_no_memory_or_acl(monkeypatch):
    """The hole this task closes.

    A request names a real user's id in the body but carries no
    authentication (no cookie, no bearer token). Entitlement must derive only
    from the authenticated caller — here, nobody — so this request gets no
    memory and no widened ACL, exactly like any other anonymous request, even
    though `u1` genuinely exists and has stored memories.
    """
    store = AgenticSearchStore(":memory:")
    store.upsert_user(UserRecord(id="u1"))
    store.add_user_memory("u1", "User is allergic to peanuts")
    captured: dict = {}
    monkeypatch.setattr(web_app, "answer_with_retrieval", _capturing_awr(captured))

    app = create_web_app(SearchExperienceSettings(), store=store)
    resp = TestClient(app).post(
        "/api/agent",
        json={
            "query": "Recommend Thai food",
            "mode": "chat_once",
            "user_id": "u1",
        },
    )

    assert resp.status_code == 200
    assert captured["user_memory"] is None
    assert captured["filters"] == SearchFilters(access_acl=["public"])
    store.close()


def test_assist_passes_query_and_app_state_encoder_to_capabilities(monkeypatch):
    store = AgenticSearchStore(":memory:")
    store.upsert_user(UserRecord(id="u1"))
    captured: dict = {}
    monkeypatch.setattr(web_app, "answer_with_retrieval", _capturing_awr(captured))
    seen: dict = {}
    real = web_app.resolve_capabilities

    def recording(user, db, *, query=None, encoder=None):
        seen.update(query=query, encoder=encoder)
        return real(user, db, query=query, encoder=encoder)

    monkeypatch.setattr(web_app, "resolve_capabilities", recording)
    app = create_web_app(SearchExperienceSettings(), store=store)
    sentinel = object()
    app.state.memory_encoder = sentinel
    client = TestClient(app)
    client.cookies.set("fastapiusersauth", generate_user_jwt_token(user_id="u1"))
    client.post(
        "/api/agent", json={"query": "Recommend Thai food", "mode": "chat_once"}
    )
    assert seen == {"query": "Recommend Thai food", "encoder": sentinel}


def test_lifespan_builds_the_memory_encoder_once(monkeypatch):
    calls: list = []
    sentinel = object()

    def fake_build():
        calls.append(1)
        return sentinel

    monkeypatch.setattr(web_app, "maybe_build_encoder", fake_build)
    app = create_web_app(
        SearchExperienceSettings(), store=AgenticSearchStore(":memory:")
    )
    with TestClient(app):
        assert app.state.memory_encoder is sentinel
    assert calls == [1]
