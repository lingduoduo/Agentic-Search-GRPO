"""CORS on the web app: off by default, on only for listed origins.

The bundled UI calls the backend same-origin through a proxy; a frontend served
from another origin is allowed only via AGENTIC_SEARCH_CORS_ORIGINS.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.internal.servers.web.app import SearchExperienceSettings, create_web_app

ORIGIN = "http://127.0.0.1:5173"
PREFLIGHT = {
    "Origin": ORIGIN,
    "Access-Control-Request-Method": "POST",
    "Access-Control-Request-Headers": "content-type",
}


def _client(tmp_path) -> TestClient:
    return TestClient(
        create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    )


def test_cors_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_CORS_ORIGINS", raising=False)
    response = _client(tmp_path).get("/health", headers={"Origin": ORIGIN})
    assert "access-control-allow-origin" not in response.headers


def test_listed_origin_gets_credentialed_cors(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "AGENTIC_SEARCH_CORS_ORIGINS", f" {ORIGIN}/ , https://ui.example"
    )
    client = _client(tmp_path)

    preflight = client.options("/auth/login", headers=PREFLIGHT)
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == ORIGIN
    assert preflight.headers["access-control-allow-credentials"] == "true"

    response = client.get("/health", headers={"Origin": ORIGIN})
    assert response.headers["access-control-allow-origin"] == ORIGIN


def test_unlisted_origin_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CORS_ORIGINS", "https://ui.example")
    client = _client(tmp_path)
    assert client.options("/auth/login", headers=PREFLIGHT).status_code == 400
    response = client.get("/health", headers={"Origin": ORIGIN})
    assert "access-control-allow-origin" not in response.headers


def test_wildcard_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CORS_ORIGINS", "*")
    with pytest.raises(ValueError, match="not '\\*'"):
        _client(tmp_path)
