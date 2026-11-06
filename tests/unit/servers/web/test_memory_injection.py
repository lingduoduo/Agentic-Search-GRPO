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


def test_recall_sees_rewritten_query_session_and_app_state_encoder(
    monkeypatch, tmp_path
):
    import json

    from src.internal.hooks import HookConfig, HookPoint, HookRegistry

    class FakeHookResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"query": "rewritten question"}).encode()

    store = AgenticSearchStore(":memory:")
    store.upsert_user(UserRecord(id="u1"))
    captured: dict = {}
    monkeypatch.setattr(web_app, "answer_with_retrieval", _capturing_awr(captured))
    monkeypatch.setattr(
        "src.internal.hooks.executor.urllib.request.urlopen",
        lambda request, timeout: FakeHookResponse(),
    )
    registry = HookRegistry(
        [
            HookConfig(
                hook_point=HookPoint.QUERY_PROCESSING,
                endpoint_url="https://hooks.test/query",
            )
        ]
    )
    app = create_web_app(
        SearchExperienceSettings(), store=store, hook_registry=registry
    )
    sentinel = object()
    app.state.memory_encoder = sentinel
    seen: list = []

    async def recording(ctx):
        seen.append((ctx.query, ctx.session_id, ctx.encoder, ctx.user_id))
        return None

    app.state.lifecycle_hooks.on_before_turn(recording)
    client = TestClient(app)
    client.cookies.set("fastapiusersauth", generate_user_jwt_token(user_id="u1"))
    resp = client.post(
        "/api/agent", json={"query": "Recommend Thai food", "mode": "chat_once"}
    )
    assert resp.status_code == 200
    assert len(seen) == 1
    query, session_id, encoder, user_id = seen[0]
    assert query == "rewritten question"
    assert session_id is not None
    assert encoder is sentinel
    assert user_id == "u1"
    store.close()


def test_raising_before_turn_hook_does_not_fail_the_request(monkeypatch):
    store = AgenticSearchStore(":memory:")
    captured: dict = {}
    monkeypatch.setattr(web_app, "answer_with_retrieval", _capturing_awr(captured))
    app = create_web_app(SearchExperienceSettings(), store=store)

    async def boom(ctx):
        raise RuntimeError("hook exploded")

    app.state.lifecycle_hooks.on_before_turn(boom)
    resp = TestClient(app).post(
        "/api/agent", json={"query": "Recommend Thai food", "mode": "chat_once"}
    )
    assert resp.status_code == 200
    assert resp.json()["answer"] == "ok"
    store.close()


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


def _after_turn_app(monkeypatch, *, auto_curate=True):
    from src.internal.hooks.lifecycle import LifecycleHooks

    store = AgenticSearchStore(":memory:")
    store.upsert_user(UserRecord(id="u1"))
    monkeypatch.setattr(web_app, "answer_with_retrieval", _capturing_awr({}))
    compress_calls: list[dict] = []
    monkeypatch.setattr(
        web_app, "schedule_compression", lambda *a, **kw: compress_calls.append(kw)
    )
    seen: list = []

    async def spy(ctx):
        seen.append((ctx.user_id, ctx.answer))

    orig = LifecycleHooks.__init__

    def init(self, *a, **kw):
        orig(self, *a, **kw)
        self.on_after_turn(spy)

    monkeypatch.setattr(LifecycleHooks, "__init__", init)
    app = create_web_app(
        SearchExperienceSettings(memory_auto_curate=auto_curate), store=store
    )
    return app, store, seen, compress_calls


def test_after_turn_runs_once_and_disables_compression_curation(monkeypatch):
    app, store, seen, compress_calls = _after_turn_app(monkeypatch)
    client = TestClient(app)
    client.cookies.set("fastapiusersauth", generate_user_jwt_token(user_id="u1"))
    resp = client.post(
        "/api/agent", json={"query": "I prefer tea", "mode": "chat_once"}
    )
    assert resp.status_code == 200
    assert seen == [("u1", "ok")]
    assert compress_calls and compress_calls[0]["auto_curate"] is False
    store.close()


def test_after_turn_anonymous_schedules_no_capture(monkeypatch):
    app, store, seen, _ = _after_turn_app(monkeypatch)
    resp = TestClient(app).post(
        "/api/agent", json={"query": "I prefer tea", "mode": "chat_once"}
    )
    assert resp.status_code == 200
    from src.internal.memory.lifecycle import capture_handler

    assert capture_handler in app.state.lifecycle_hooks._after_turn
    assert seen == [(None, "ok")]  # hook fires, but capture_handler no-ops (u=None)
    assert store.get_user_memory_records("u1") == []
    store.close()
