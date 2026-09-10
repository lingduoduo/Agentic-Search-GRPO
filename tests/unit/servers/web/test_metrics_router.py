"""GET /api/admin/metrics: the process's route and stage windows plus the
feedback split, readable in production by an admin — not only behind the
dev-only debug router."""

from __future__ import annotations

import dataclasses

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.configs import load_app_settings
from src.internal.db.store import AgenticSearchStore
from src.internal.observability import stage_metrics as sm
from src.internal.servers.web.metrics_router import create_metrics_router


def _client(db: AgenticSearchStore, *, admin: bool) -> TestClient:
    settings = load_app_settings({})
    if admin:
        settings = dataclasses.replace(
            settings, auth=dataclasses.replace(settings.auth, dev_admin_bypass=True)
        )
    app = FastAPI()
    app.include_router(create_metrics_router(db, settings))
    return TestClient(app)


def test_requires_admin():
    resp = _client(AgenticSearchStore(":memory:"), admin=False).get(
        "/api/admin/metrics"
    )
    assert resp.status_code == 401


def test_returns_routes_stages_and_feedback_by_target():
    db = AgenticSearchStore(":memory:")
    db.save_retrieval_feedback("s1", "thumbs_down", target="retrieval")
    token = sm.start_request()
    sm.note_retrieval(elapsed_ms=3.0, docs=2)
    sm.note_generation(elapsed_ms=30.0, prompt_tokens=50, completion_tokens=5)
    sm.STAGE_LATENCY.record(sm.finish_request(token))

    resp = _client(db, admin=True).get("/api/admin/metrics")

    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["routes"], list)
    assert body["stages"]["retrieval"]["count"] >= 1
    assert body["stages"]["generation"]["count"] >= 1
    assert body["feedback"]["by_target"]["retrieval"] == {
        "rated": 1,
        "thumbs_up_rate": 0.0,
    }
