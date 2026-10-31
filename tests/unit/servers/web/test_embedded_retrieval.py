"""With AGENTIC_SEARCH_EMBEDDED_RETRIEVAL set, the API serves /retrieval itself.

It reuses the demo TF-IDF server's routes, so the request and response shapes
are the standalone server's. Like that server, which binds 127.0.0.1, the routes
are internal: /retrieve honours caller-supplied ACL filters, so an outside
caller who omits them would read every document. Only loopback clients -- the
API calling itself -- get through.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from src.internal.configs import load_app_settings
from src.internal.servers._auth import require_loopback_client
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app
from src.internal.servers.web.auth_check import (
    PUBLIC_ENDPOINT_SPECS,
    check_router_auth,
)


def _app(tmp_path, env: dict[str, str]):
    return create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "s.sqlite3"),
        app_settings=load_app_settings(env),
    )


def _loopback_client(tmp_path, env: dict[str, str]) -> TestClient:
    """TestClient's peer is "testclient", so stand in for the loopback check."""
    app = _app(tmp_path, env)
    app.dependency_overrides[require_loopback_client] = lambda: None
    return TestClient(app)


def test_the_registered_corpus_is_served_in_process(tmp_path):
    client = _loopback_client(tmp_path, {"AGENTIC_SEARCH_EMBEDDED_RETRIEVAL": "demo"})

    response = client.post(
        "/retrieval/retrieve", json={"queries": ["What is FAISS?"], "topk": 1}
    )

    assert response.status_code == 200
    [[top]] = response.json()["results"]
    assert "FAISS" in top["title"]
    assert client.get("/retrieval/health").json() == {"status": "ok"}


def test_a_corpus_file_path_is_accepted(tmp_path):
    corpus = tmp_path / "c.jsonl"
    corpus.write_text('{"id": "x1", "title": "Zebra facts", "contents": "zebra"}\n')
    client = _loopback_client(
        tmp_path, {"AGENTIC_SEARCH_EMBEDDED_RETRIEVAL": str(corpus)}
    )

    response = client.post("/retrieval/retrieve", json={"query": "zebra", "topk": 1})

    assert response.json()["results"][0]["id"] == "x1"


def test_a_non_loopback_caller_is_refused(tmp_path):
    client = TestClient(_app(tmp_path, {"AGENTIC_SEARCH_EMBEDDED_RETRIEVAL": "demo"}))

    response = client.post("/retrieval/retrieve", json={"query": "FAISS"})

    assert response.status_code == 403


@pytest.mark.parametrize(
    ("host", "allowed"),
    [
        ("127.0.0.1", True),
        ("127.0.0.53", True),
        ("::1", True),
        ("::ffff:127.0.0.1", True),
        ("10.0.0.5", False),
        ("172.18.0.3", False),
        ("testclient", False),
        (None, False),
    ],
)
def test_loopback_check(host, allowed):
    scope = {"type": "http", "client": (host, 1234) if host else None}
    if allowed:
        require_loopback_client(Request(scope))
    else:
        with pytest.raises(HTTPException) as exc:
            require_loopback_client(Request(scope))
        assert exc.value.status_code == 403


def test_the_routes_pass_the_startup_auth_audit_without_being_public(tmp_path):
    app = _app(tmp_path, {"AGENTIC_SEARCH_EMBEDDED_RETRIEVAL": "demo"})

    check_router_auth(app, PUBLIC_ENDPOINT_SPECS)  # raises on an unguarded route

    assert not any(path.startswith("/retrieval") for path, _ in PUBLIC_ENDPOINT_SPECS)
    paths = {getattr(route, "path", "") for route in app.routes}
    assert {"/retrieval/retrieve", "/retrieval/health"} <= paths
    assert not any(p.startswith("/retrieval/docs") for p in paths)


def test_not_mounted_unless_enabled(tmp_path):
    client = TestClient(_app(tmp_path, {}))

    assert client.post("/retrieval/retrieve", json={"query": "x"}).status_code == 404
