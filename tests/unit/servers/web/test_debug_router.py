"""Tests for the dev-console debug router (retrieval proxy)."""

from __future__ import annotations

import json

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.servers.web.debug_router import create_debug_router


def _client(handler, db=None) -> TestClient:
    """Mount the debug router with an injected httpx client backed by *handler*."""
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    app = FastAPI()
    app.include_router(
        create_debug_router(
            search_url="http://retrieval:8001/retrieve",
            http_client=http_client,
            db=db,
        )
    )
    return TestClient(app)


def test_workers_endpoint_is_gone():
    # It always returned {"metrics": null} and nothing read it.
    client = _client(lambda r: httpx.Response(200, json={}), db=None)
    assert client.get("/api/debug/workers").status_code == 404


class _StubPipeline:
    def __init__(self, bundle):
        self._bundle = bundle

    def transform(self, query, filters=None):
        return self._bundle


def test_query_transform_returns_variants_and_filters(monkeypatch):
    import src.internal.servers.web.debug_router as mod
    from src.context.query_transform import TransformedQueryBundle

    bundle = TransformedQueryBundle(
        original="vector db",
        sub_queries=["what is a vector database", "how do embeddings index"],
        merged_filters={"year": 2024},
    )
    monkeypatch.setattr(
        mod,
        "build_query_transform_pipeline_from_env",
        lambda llm: _StubPipeline(bundle),
    )
    client = _client(lambda r: httpx.Response(200, json={}))
    resp = client.post("/api/debug/query-transform", json={"query": "vector db"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] is True
    assert "what is a vector database" in body["variants"]
    assert body["merged_filters"] == {"year": 2024}
    assert body["legs"]["sub_queries"] == [
        "what is a vector database",
        "how do embeddings index",
    ]


def test_query_transform_no_pipeline_is_inactive(monkeypatch):
    import src.internal.servers.web.debug_router as mod

    monkeypatch.setattr(
        mod, "build_query_transform_pipeline_from_env", lambda llm: None
    )
    client = _client(lambda r: httpx.Response(200, json={}))
    resp = client.post("/api/debug/query-transform", json={"query": "vector db"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] is False
    assert body["variants"] == ["vector db"]


def test_health_reports_retrieval_up_and_web_self():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"})

    client = _client(handler)
    resp = client.get("/api/debug/health")
    assert resp.status_code == 200
    servers = {s["name"]: s for s in resp.json()["servers"]}
    assert servers["retrieval"]["status"] == "up"
    assert servers["web"]["status"] == "up"


def test_health_reports_retrieval_down_when_unreachable_no_500():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _client(handler)
    resp = client.get("/api/debug/health")
    assert resp.status_code == 200  # never raises
    servers = {s["name"]: s for s in resp.json()["servers"]}
    assert servers["retrieval"]["status"] == "down"
    assert servers["web"]["status"] == "up"


def test_health_reports_retrieval_down_on_non_200():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "unhealthy"})

    client = _client(handler)
    resp = client.get("/api/debug/health")
    servers = {s["name"]: s for s in resp.json()["servers"]}
    assert servers["retrieval"]["status"] == "down"


def test_retrieval_proxy_forwards_to_internal_search_and_returns_results():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "results": [{"doc_id": "d1", "title": "T", "text": "b", "score": 0.9}],
                "retrieval_mode": "sparse",
                "executed_queries": ["vector database"],
                "latency_ms": 1.2,
            },
        )

    client = _client(handler)
    resp = client.post(
        "/api/debug/retrieval/sparse",
        json={"query": "vector database", "top_k": 5},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieval_mode"] == "sparse"
    assert body["results"][0]["doc_id"] == "d1"
    # Proxy must derive the per-mode endpoint from the /retrieve base URL.
    assert captured["url"] == "http://retrieval:8001/internal/search/sparse"


def test_hybrid_proxy_forwards_tuning_knobs():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"retrieval_mode": "hybrid", "results": []})

    client = _client(handler)
    resp = client.post(
        "/api/debug/retrieval/hybrid",
        json={
            "query": "q",
            "top_k": 5,
            "rrf_k": 30,
            "mmr_lambda": 0.7,
            "over_fetch": 3,
        },
    )

    assert resp.status_code == 200
    assert captured["body"] == {
        "query": "q",
        "top_k": 5,
        "rerank": False,
        "rrf_k": 30,
        "mmr_lambda": 0.7,
        "over_fetch": 3,
    }


def test_proxy_forwards_rerank_flag():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"retrieval_mode": "sparse", "results": []})

    client = _client(handler)
    resp = client.post(
        "/api/debug/retrieval/sparse",
        json={"query": "q", "top_k": 5, "rerank": True},
    )

    assert resp.status_code == 200
    assert captured["body"]["rerank"] is True


def test_proxy_rerank_defaults_false():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"retrieval_mode": "sparse", "results": []})

    client = _client(handler)
    client.post("/api/debug/retrieval/sparse", json={"query": "q", "top_k": 5})
    assert captured["body"]["rerank"] is False


def test_proxy_passes_through_503_when_dense_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "Dense search not configured"})

    client = _client(handler)
    resp = client.post("/api/debug/retrieval/dense", json={"query": "q", "top_k": 5})

    assert resp.status_code == 503
    assert "Dense" in resp.json()["detail"]


def test_proxy_passes_through_404_when_endpoint_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    client = _client(handler)
    resp = client.post("/api/debug/retrieval/sparse", json={"query": "q", "top_k": 5})

    assert resp.status_code == 404


def test_unknown_mode_rejected_without_calling_upstream():
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    client = _client(handler)
    resp = client.post("/api/debug/retrieval/bogus", json={"query": "q", "top_k": 5})

    assert resp.status_code == 404
    assert called is False


def _ok(request):  # trivial httpx handler; the eval-results endpoint ignores http
    return httpx.Response(200, json={})


def test_eval_results_lists_numeric_metrics(tmp_path, monkeypatch):
    (tmp_path / "beir.json").write_text(
        json.dumps({"recall@10": 0.5, "ndcg@10": 0.4, "_note": "x"})
    )
    (tmp_path / "notjson.txt").write_text("nope")
    monkeypatch.setenv("AGENTIC_SEARCH_EVAL_RESULTS_DIR", str(tmp_path))

    client = _client(_ok)
    resp = client.get("/api/debug/eval-results")
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert [r["name"] for r in results] == ["beir.json"]
    assert results[0]["metrics"] == {"recall@10": 0.5, "ndcg@10": 0.4}  # _note dropped


def test_eval_results_missing_dir_returns_empty(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_EVAL_RESULTS_DIR", "/nonexistent/xyz-eval")
    client = _client(_ok)
    resp = client.get("/api/debug/eval-results")
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_eval_results_groups_retrieval_apart_from_generation(tmp_path, monkeypatch):
    # A Bamboogle summary and a reranked eval_runner report side by side: the
    # flat `metrics` keeps the old keys (plus dotted nested ones) and `groups`
    # files each number under the taxonomy so recall and EM are never peers.
    (tmp_path / "bamboogle.summary.json").write_text(
        json.dumps(
            {
                "num_examples": 125,
                "exact_match": 0.4,
                "contains_match": 0.6,
                "avg_reward": 0.3,
                "avg_reward_retrieval": 0.1,
                "avg_reward_generation": 0.2,
            }
        )
    )
    (tmp_path / "retrieval.json").write_text(
        json.dumps(
            {
                "retrieval": {"recall@10": 0.5, "mrr": 0.4, "num_queries": 20},
                "reranked": {"recall@10": 0.6},
                "latency_ms": {"p99": 12.5},
            }
        )
    )
    monkeypatch.setenv("AGENTIC_SEARCH_EVAL_RESULTS_DIR", str(tmp_path))

    results = {
        r["name"]: r
        for r in _client(_ok).get("/api/debug/eval-results").json()["results"]
    }

    bamboogle = results["bamboogle.summary.json"]
    assert bamboogle["metrics"]["exact_match"] == 0.4
    assert bamboogle["groups"] == {
        "generation": {"exact_match": 0.4, "contains_match": 0.6},
        "reward": {
            "avg_reward": 0.3,
            "avg_reward_retrieval": 0.1,
            "avg_reward_generation": 0.2,
        },
        "other": {"num_examples": 125},
    }
    retrieval = results["retrieval.json"]
    assert retrieval["metrics"]["reranked.recall@10"] == 0.6
    assert retrieval["groups"] == {
        "retrieval": {
            "retrieval.recall@10": 0.5,
            "retrieval.mrr": 0.4,
            "retrieval.num_queries": 20,
            "reranked.recall@10": 0.6,
        },
        "latency": {"latency_ms.p99": 12.5},
    }


def test_latency_endpoint_reports_stages_beside_routes(monkeypatch):
    import src.internal.servers.web.debug_router as mod
    from src.internal.observability import stage_metrics as sm

    stats = sm.StageLatencyStats()
    monkeypatch.setattr(mod, "STAGE_LATENCY", stats)
    token = sm.start_request()
    sm.note_retrieval(elapsed_ms=4.0, docs=3)
    sm.note_generation(
        elapsed_ms=40.0, prompt_tokens=100, completion_tokens=10, kind="answer"
    )
    stats.record(sm.finish_request(token))

    body = _client(_ok).get("/api/debug/latency").json()
    assert "routes" in body
    assert body["stages"]["retrieval"] == {
        "count": 1,
        "p50_ms": 4.0,
        "p95_ms": 4.0,
        "max_ms": 4.0,
        "avg_docs": 3.0,
        "cache_hit_rate": 0.0,
    }
    assert body["stages"]["generation"]["avg_completion_tokens"] == 10.0
    assert body["stages"]["auxiliary"] == {"count": 0}


def test_eval_results_drops_non_finite_metrics(tmp_path, monkeypatch):
    (tmp_path / "nan.json").write_text(json.dumps({"good": 0.5, "bad": float("nan")}))
    monkeypatch.setenv("AGENTIC_SEARCH_EVAL_RESULTS_DIR", str(tmp_path))

    client = _client(_ok)
    resp = client.get("/api/debug/eval-results")
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results[0]["metrics"] == {"good": 0.5}
