"""Tests for the new retrieval service FastAPI app (server.py)."""

from __future__ import annotations

import os
import pytest
from src.internal.auth.users import generate_user_jwt_token
from src.internal.configs import AppSettings, AuthSettings

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.service import RetrievalService
from src.internal.servers.retrieval.server import create_app


def _make_service(
    results: list[RetrievalResult], mode: str = "sparse"
) -> RetrievalService:
    svc = MagicMock(spec=RetrievalService)
    svc.search.return_value = (results, mode)
    return svc


def _result(doc_id: str = "d1", score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id, title="Title", text="body", url="https://x.com", score=score
    )


def test_health_returns_ok():
    client = TestClient(create_app(_make_service([])))
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "backend" in data


def test_search_returns_results():
    svc = _make_service([_result("d1", 0.9), _result("d2", 0.7)])
    client = TestClient(create_app(svc))

    resp = client.post("/search", json={"query": "procurement", "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["retrieval_mode"] == "sparse"
    assert data["executed_queries"] == ["procurement"]
    assert len(data["results"]) == 2
    assert data["results"][0]["doc_id"] == "d1"
    assert "latency_ms" in data


def test_search_calls_service_with_top_k():
    svc = _make_service([])
    client = TestClient(create_app(svc))

    client.post("/search", json={"query": "vector search", "top_k": 10})
    svc.search.assert_called_once_with("vector search", top_k=10, filters=None)


def test_search_rejects_empty_query():
    client = TestClient(create_app(_make_service([])))
    resp = client.post("/search", json={"query": "", "top_k": 5})
    assert resp.status_code == 422


def test_search_default_top_k_is_5():
    svc = _make_service([])
    client = TestClient(create_app(svc))

    client.post("/search", json={"query": "anything"})
    svc.search.assert_called_once_with("anything", top_k=5, filters=None)


def test_health_response_has_api_version_header():
    client = TestClient(create_app(_make_service([])))
    resp = client.get("/health")
    assert resp.headers.get("Retrieval-API-Version") == "1.0"


def test_search_response_has_api_version_header():
    client = TestClient(create_app(_make_service([_result()])))
    resp = client.post("/search", json={"query": "anything"})
    assert resp.headers.get("Retrieval-API-Version") == "1.0"


def _client(svc=None, *, bypass: bool = False) -> TestClient:
    settings = AppSettings(auth=AuthSettings(dev_admin_bypass=bypass))
    return TestClient(create_app(svc or _make_service([]), app_settings=settings))


def _bearer(**extra) -> dict[str, str]:
    token = generate_user_jwt_token(user_id="someone", extra=extra or None)
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/api/admin/retrieval/stats", None),
        ("patch", "/api/admin/retrieval/config", {"result_cache_ttl": 5}),
        ("post", "/internal/search/sparse", {"query": "q", "top_k": 5}),
        ("post", "/internal/optimize/bm25-tune", {"qa_pairs_path": "/etc/passwd"}),
    ],
)
def test_admin_routes_require_a_token(method, path, body):
    kwargs = {"json": body} if body is not None else {}
    assert getattr(_client(), method)(path, **kwargs).status_code == 401


def test_non_admin_token_is_forbidden():
    assert (
        _client().get("/api/admin/retrieval/stats", headers=_bearer()).status_code
        == 403
    )


def test_admin_token_reads_stats():
    resp = _client().get("/api/admin/retrieval/stats", headers=_bearer(role="admin"))
    assert resp.status_code == 200


def test_dev_bypass_opens_admin_routes():
    assert _client(bypass=True).get("/api/admin/retrieval/stats").status_code == 200


def test_patch_applies_result_cache_ttl():
    svc = _make_service([])
    svc._result_cache = MagicMock()
    svc._result_cache._ttl = 300
    resp = _client(svc).patch(
        "/api/admin/retrieval/config",
        json={"result_cache_ttl": 5},
        headers=_bearer(role="admin"),
    )
    assert resp.status_code == 200
    assert resp.json() == {"applied": ["result_cache_ttl"]}
    assert svc._result_cache._ttl == 5


@pytest.mark.parametrize(
    "body",
    [{"rrf_k": 80}, {"mmr_lambda": 0.4}, {"nprobe": 96}, {"result_cache_ttl": -1}],
)
def test_patch_rejects_unapplied_fields(body, monkeypatch):
    monkeypatch.delenv("RRF_K", raising=False)
    monkeypatch.delenv("MMR_LAMBDA", raising=False)
    resp = _client().patch(
        "/api/admin/retrieval/config", json=body, headers=_bearer(role="admin")
    )
    assert resp.status_code == 422
    assert "RRF_K" not in os.environ and "MMR_LAMBDA" not in os.environ


def test_health_and_search_stay_open():
    client = _client()
    assert client.get("/health").status_code == 200
    assert client.post("/search", json={"query": "q"}).status_code == 200
