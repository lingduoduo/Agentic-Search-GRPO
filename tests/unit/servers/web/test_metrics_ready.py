"""GET /metrics (opt-in Prometheus exposition) and GET /ready (readiness)."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from src.internal.configs import AppSettings, AuthSettings
from src.internal.configs.license_enforcement_config import (
    is_license_enforcement_exempt,
)
from src.internal.configs.timeouts import load_timeout_policies, use_timeout_policies
from src.internal.db import AgenticSearchStore
from src.internal.servers.web import readiness
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


def _app(tmp_path, **settings):
    return create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3", **settings),
        app_settings=AppSettings(auth=AuthSettings()),
    )


def test_metrics_is_404_when_the_flag_is_off(tmp_path):
    assert TestClient(_app(tmp_path)).get("/metrics").status_code == 404


def test_metrics_flag_defaults_off_and_reads_the_env(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_METRICS_ENABLED", raising=False)
    assert SearchExperienceSettings.from_app_settings().metrics_enabled is False
    monkeypatch.setenv("AGENTIC_SEARCH_METRICS_ENABLED", "true")
    assert SearchExperienceSettings.from_app_settings().metrics_enabled is True


def test_metrics_is_public_and_counts_health(tmp_path):
    app = _app(tmp_path, metrics_enabled=True)
    # `with` runs the lifespan, so check_router_auth audits /metrics too.
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert (
        'agentic_search_http_requests_total{method="GET",route="/health",status="2xx"}'
        in response.text
    )


def test_metrics_and_ready_are_license_exempt():
    assert is_license_enforcement_exempt("/metrics")
    assert is_license_enforcement_exempt("/ready")


def _stub_retrieval(monkeypatch, handler, seen_timeouts=None):
    def factory(timeout):
        if seen_timeouts is not None:
            seen_timeouts.append(timeout)
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler), timeout=timeout
        )

    monkeypatch.setattr(readiness, "_retrieval_client", factory)


def _ok(request):
    assert request.url.path == "/health"
    return httpx.Response(200, json={"status": "ok"})


def _ready_app(tmp_path, store=None):
    return create_web_app(
        SearchExperienceSettings(
            db_path=tmp_path / "state.sqlite3",
            search_url="http://retrieval.internal:8001/retrieve",
        ),
        app_settings=AppSettings(auth=AuthSettings()),
        store=store,
    )


@pytest.mark.parametrize(
    ("search_url", "expected"),
    [
        ("http://localhost:8001/retrieve", "http://localhost:8001/health"),
        ("http://h:8001/retrieve?x=1#frag", "http://h:8001/health"),
        ("https://h/api/v1/retrieve", "https://h/health"),
    ],
)
def test_retrieval_health_url_replaces_path_and_drops_query(search_url, expected):
    assert readiness.retrieval_health_url(search_url) == expected


def test_ready_200_when_store_and_retrieval_are_fine(tmp_path, monkeypatch):
    _stub_retrieval(monkeypatch, _ok)
    response = TestClient(_ready_app(tmp_path)).get("/ready")  # no auth header
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {
            "store": {"ok": True, "error": None},
            "retrieval": {"ok": True, "error": None},
        },
    }


def test_ready_503_when_retrieval_unreachable_without_leaking_url(
    tmp_path, monkeypatch
):
    def refuse(request):
        raise httpx.ConnectError(f"refused {request.url} user:secret", request=request)

    _stub_retrieval(monkeypatch, refuse)
    response = TestClient(_ready_app(tmp_path)).get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["retrieval"] == {"ok": False, "error": "ConnectError"}
    assert body["checks"]["store"]["ok"] is True
    assert "retrieval.internal" not in response.text
    assert "secret" not in response.text


def test_ready_503_when_retrieval_health_is_not_2xx(tmp_path, monkeypatch):
    _stub_retrieval(monkeypatch, lambda request: httpx.Response(503, text="down"))
    response = TestClient(_ready_app(tmp_path)).get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["retrieval"] == {
        "ok": False,
        "error": "HTTP 503",
    }


def test_ready_503_when_store_ping_raises(tmp_path, monkeypatch):
    _stub_retrieval(monkeypatch, _ok)
    store = AgenticSearchStore(tmp_path / "s.sqlite3")

    def broken():
        raise RuntimeError("disk gone")

    monkeypatch.setattr(store, "ping", broken)
    response = TestClient(_ready_app(tmp_path, store=store)).get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["store"] == {
        "ok": False,
        "error": "RuntimeError",
    }
    store.close()


def test_ready_503_when_store_is_closed(tmp_path, monkeypatch):
    _stub_retrieval(monkeypatch, _ok)
    store = AgenticSearchStore(tmp_path / "s.sqlite3")
    app = _ready_app(tmp_path, store=store)
    store.close()
    response = TestClient(app).get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["store"] == {
        "ok": False,
        "error": "ProgrammingError",
    }


def test_store_ping_succeeds_on_an_open_store(tmp_path):
    store = AgenticSearchStore(tmp_path / "s.sqlite3")
    assert store.ping() is None
    store.close()


def test_retrieval_probe_timeout_follows_the_policy(tmp_path, monkeypatch):
    seen: list[float] = []
    _stub_retrieval(monkeypatch, _ok, seen)
    policies = load_timeout_policies(
        {}, overrides={"readiness": {"probe_timeout_seconds": 0.7}}
    )
    with use_timeout_policies(policies):
        assert TestClient(_ready_app(tmp_path)).get("/ready").status_code == 200
    assert seen == [0.7]


def test_ready_passes_the_route_auth_audit(tmp_path, monkeypatch):
    _stub_retrieval(monkeypatch, _ok)
    # `with` runs the lifespan, and the lifespan runs check_router_auth.
    with TestClient(_ready_app(tmp_path)) as client:
        assert client.get("/ready").status_code == 200
