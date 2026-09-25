"""GET /metrics (opt-in Prometheus exposition) and GET /ready (readiness)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.internal.configs import AppSettings, AuthSettings
from src.internal.configs.license_enforcement_config import (
    is_license_enforcement_exempt,
)
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
