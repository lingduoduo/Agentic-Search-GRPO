"""GET /api/retrieval/status: which retrieval server answers, and is it up."""

from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from src.internal.configs import load_app_settings
from src.internal.db import AgenticSearchStore
from src.internal.servers.web import readiness
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


def _stub_retrieval(monkeypatch, status_code):
    def factory(timeout):
        return httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(status_code)),
            timeout=timeout,
        )

    monkeypatch.setattr(readiness, "_retrieval_client", factory)


def _signed_in(monkeypatch, search_url, env=None):
    monkeypatch.setenv("AGENTIC_SEARCH_DEV_ADMIN", "false")
    store = AgenticSearchStore(":memory:")
    client = TestClient(
        create_web_app(
            SearchExperienceSettings(db_path=":memory:", search_url=search_url),
            app_settings=load_app_settings(env or {}),
            store=store,
        )
    )
    credentials = {"email": "alice@example.test", "password": "test-only-password"}
    client.post("/auth/register", json=credentials)
    client.post(
        "/auth/login",
        data={"username": credentials["email"], "password": credentials["password"]},
    )
    return client


def test_reports_the_retrieval_host_without_credentials_or_path(monkeypatch):
    _stub_retrieval(monkeypatch, 200)
    client = _signed_in(monkeypatch, "http://user:secret@retrieval:8001/retrieve?k=1")

    response = client.get("/api/retrieval/status")

    assert response.status_code == 200
    assert response.json() == {
        "url": "http://retrieval:8001",
        "embedded": False,
        "ok": True,
    }


def test_reports_down_when_the_health_probe_fails(monkeypatch):
    _stub_retrieval(monkeypatch, 503)
    client = _signed_in(monkeypatch, "http://localhost:8001/retrieve")

    assert client.get("/api/retrieval/status").json()["ok"] is False


def test_flags_embedded_retrieval(monkeypatch):
    _stub_retrieval(monkeypatch, 200)
    client = _signed_in(
        monkeypatch,
        "http://127.0.0.1:7860/retrieval/retrieve",
        {"AGENTIC_SEARCH_EMBEDDED_RETRIEVAL": "demo"},
    )

    assert client.get("/api/retrieval/status").json() == {
        "url": "http://127.0.0.1:7860",
        "embedded": True,
        "ok": True,
    }


def test_requires_sign_in(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DEV_ADMIN", "false")
    client = TestClient(create_web_app(store=AgenticSearchStore(":memory:")))

    assert client.get("/api/retrieval/status").status_code == 401
