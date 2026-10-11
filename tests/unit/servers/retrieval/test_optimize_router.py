"""Tests for /internal/optimize/* admin endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.servers.retrieval.optimize_router import create_optimize_router
from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.service import RetrievalService


def _make_app() -> TestClient:
    app = FastAPI()
    router = create_optimize_router()
    app.include_router(router)
    return TestClient(app)


def _write_qa(path: Path) -> None:
    path.write_text(json.dumps({"query": "q", "relevant_doc_ids": ["d1"]}) + "\n")


def test_bm25_tune_returns_params(tmp_path):
    qa = tmp_path / "qa.jsonl"
    _write_qa(qa)
    client = _make_app()

    mock_svc = MagicMock()
    mock_svc.search.return_value = ([MagicMock(doc_id="d1")], "hybrid")

    with patch(
        "src.internal.servers.retrieval.optimize_router._make_service",
        return_value=mock_svc,
    ):
        resp = client.post(
            "/internal/optimize/bm25-tune",
            json={"qa_pairs_path": str(qa), "k1_range": [1.2], "b_range": [0.75]},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "k1" in data and "b" in data and "score" in data


def test_fusion_weights_endpoint(tmp_path):
    qa = tmp_path / "qa.jsonl"
    _write_qa(qa)
    client = _make_app()

    mock_svc = MagicMock()
    mock_svc.search.return_value = ([MagicMock(doc_id="d1")], "hybrid")

    with patch(
        "src.internal.servers.retrieval.optimize_router._make_service",
        return_value=mock_svc,
    ):
        resp = client.post(
            "/internal/optimize/fusion-weights",
            json={"qa_pairs_path": str(qa), "w_sparse_range": [0.4, 0.6]},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "w_sparse" in data and "w_dense" in data
    assert abs(data["w_sparse"] + data["w_dense"] - 1.0) < 1e-4


def _result(doc_id):
    return RetrievalResult(doc_id=doc_id, title="T", text="text", url=None, score=1.0)


def test_bm25_endpoint_evaluates_real_candidate_configuration(tmp_path):
    qa = tmp_path / "qa.jsonl"
    _write_qa(qa)
    seen = []

    def backend_factory(**kwargs):
        seen.append(kwargs)
        backend = MagicMock()
        backend.search_sparse.return_value = [
            _result("d1" if kwargs.get("bm25_k1") == 2.0 else "wrong")
        ]
        backend.search_dense.side_effect = NotImplementedError
        return backend

    with (
        patch(
            "src.internal.retrieval.service._build_backend", side_effect=backend_factory
        ),
        patch.object(
            RetrievalService, "from_env", wraps=RetrievalService.from_env
        ) as factory,
    ):
        response = _make_app().post(
            "/internal/optimize/bm25-tune",
            json={"qa_pairs_path": str(qa), "k1_range": [1.0, 2.0], "b_range": [0.75]},
        )
    assert response.status_code == 200
    assert response.json() == {"k1": 2.0, "b": 0.75, "score": 1.0}
    assert seen == [{"bm25_k1": 1.0, "bm25_b": 0.75}, {"bm25_k1": 2.0, "bm25_b": 0.75}]
    assert all(
        call.kwargs["use_result_cache"] is False for call in factory.call_args_list
    )


def test_fusion_endpoint_selects_nonfirst_weight_using_real_service(tmp_path):
    qa = tmp_path / "qa.jsonl"
    _write_qa(qa)
    backend = MagicMock()
    backend.search_sparse.return_value = [_result(f"bad{i}") for i in range(12)]
    backend.search_dense.return_value = [_result("d1")]
    with (
        patch("src.internal.retrieval.service._build_backend", return_value=backend),
        patch.object(
            RetrievalService, "from_env", wraps=RetrievalService.from_env
        ) as factory,
    ):
        response = _make_app().post(
            "/internal/optimize/fusion-weights",
            json={"qa_pairs_path": str(qa), "w_sparse_range": [1.0, 0.0]},
        )
    assert response.status_code == 200
    assert response.json() == {"w_sparse": 0.0, "w_dense": 1.0}
    assert all(
        call.kwargs["use_result_cache"] is False for call in factory.call_args_list
    )
