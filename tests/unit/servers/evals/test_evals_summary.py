"""Tests for GET /api/admin/evals/summary."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.configs import load_app_settings
from src.internal.db.store import AgenticSearchStore
from src.internal.servers.evals.api import create_evals_router


def _no_op_admin():
    return MagicMock()


def _client(db: AgenticSearchStore | None) -> TestClient:
    settings = load_app_settings()
    app = FastAPI()
    app.include_router(create_evals_router(settings, db=db, require_admin=_no_op_admin))
    return TestClient(app)


def test_evals_summary_returns_zeros_when_empty():
    db = AgenticSearchStore(":memory:")
    client = _client(db)

    resp = client.get("/api/admin/evals/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["thumbs_up_rate"] == 0.0
    assert data["ctr"] == 0.0
    assert data["rated_queries"] == 0


def test_evals_summary_with_no_db_returns_zeros():
    client = _client(db=None)

    resp = client.get("/api/admin/evals/summary")
    assert resp.status_code == 200
    assert resp.json()["rated_queries"] == 0


def test_evals_summary_splits_feedback_by_target():
    db = AgenticSearchStore(":memory:")
    db.save_retrieval_feedback("s1", "thumbs_down", target="retrieval")
    db.save_retrieval_feedback("s2", "thumbs_up", target="generation")
    db.save_retrieval_feedback("s3", "thumbs_up")
    client = _client(db)

    data = client.get("/api/admin/evals/summary").json()
    assert data["by_target"] == {
        "retrieval": {"rated": 1, "thumbs_up_rate": 0.0},
        "generation": {"rated": 1, "thumbs_up_rate": 1.0},
        "overall": {"rated": 1, "thumbs_up_rate": 1.0},
    }


def test_evals_summary_reflects_stored_feedback():
    db = AgenticSearchStore(":memory:")
    db.save_retrieval_feedback("s1", "thumbs_up")
    db.save_retrieval_feedback("s2", "thumbs_up")
    db.save_retrieval_feedback("s3", "thumbs_down")
    client = _client(db)

    resp = client.get("/api/admin/evals/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["rated_queries"] == 3
    assert abs(data["thumbs_up_rate"] - 2 / 3) < 0.01
