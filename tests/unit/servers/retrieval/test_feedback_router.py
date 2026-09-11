"""Tests for POST /api/feedback router."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.db.store import AgenticSearchStore
from src.internal.servers.retrieval.feedback_router import create_feedback_router


def _store(*session_ids: str) -> AgenticSearchStore:
    """A store holding anonymous sessions *session_ids*.

    Feedback is only accepted for a session that exists: the row feeds the
    training loaders, so the endpoint never writes one for a made-up id.
    """
    db = AgenticSearchStore(":memory:")
    for session_id in session_ids:
        db.create_chat_session(session_id=session_id, user_id=None)
    return db


def _app(db: AgenticSearchStore) -> TestClient:
    app = FastAPI()
    app.include_router(create_feedback_router(db))
    return TestClient(app)


def test_feedback_thumbs_up_persisted():
    db = _store("s1")
    client = _app(db)

    resp = client.post(
        "/api/feedback", json={"session_id": "s1", "signal": "thumbs_up"}
    )

    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert db.get_feedback_summary()["rated_queries"] == 1


def test_feedback_thumbs_down_persisted():
    db = _store("s1")
    client = _app(db)

    resp = client.post(
        "/api/feedback", json={"session_id": "s1", "signal": "thumbs_down"}
    )

    assert resp.status_code == 200
    summary = db.get_feedback_summary()
    assert summary["thumbs_up_rate"] == 0.0


def test_feedback_for_an_unknown_session_is_refused_and_nothing_is_written():
    # Before this, an unauthenticated caller could write a training-feedback
    # row against any session id that did not exist locally.
    db = _store("s1")
    client = _app(db)

    resp = client.post(
        "/api/feedback", json={"session_id": "no-such-session", "signal": "thumbs_up"}
    )

    assert resp.status_code == 404
    assert db.list_retrieval_feedback() == []


def test_feedback_target_is_persisted_and_split_in_the_summary():
    db = _store("s1")
    client = _app(db)

    for target, signal in (
        ("retrieval", "thumbs_down"),
        ("generation", "thumbs_up"),
        ("generation", "thumbs_down"),
    ):
        resp = client.post(
            "/api/feedback",
            json={"session_id": "s1", "signal": signal, "target": target},
        )
        assert resp.status_code == 200

    rows = db.list_retrieval_feedback()
    assert [r["metadata"]["target"] for r in rows] == [
        "retrieval",
        "generation",
        "generation",
    ]
    by_target = db.get_feedback_summary()["by_target"]
    assert by_target["retrieval"] == {"rated": 1, "thumbs_up_rate": 0.0}
    assert by_target["generation"] == {"rated": 2, "thumbs_up_rate": 0.5}
    assert by_target["overall"] == {"rated": 0, "thumbs_up_rate": 0.0}


def test_feedback_target_defaults_to_overall():
    db = _store("s1")
    client = _app(db)
    client.post("/api/feedback", json={"session_id": "s1", "signal": "thumbs_up"})
    assert db.list_retrieval_feedback()[0]["metadata"]["target"] == "overall"


def test_feedback_invalid_target_rejected():
    db = _store("s1")
    client = _app(db)
    resp = client.post(
        "/api/feedback",
        json={"session_id": "s1", "signal": "thumbs_up", "target": "vibes"},
    )
    assert resp.status_code == 422


def test_feedback_invalid_signal_rejected():
    db = _store("s1")
    client = _app(db)

    resp = client.post("/api/feedback", json={"session_id": "s1", "signal": "meh"})

    assert resp.status_code == 422


def test_feedback_multiple_signals_accumulate():
    db = _store("s1", "s2", "s3")
    client = _app(db)

    client.post("/api/feedback", json={"session_id": "s1", "signal": "thumbs_up"})
    client.post("/api/feedback", json={"session_id": "s2", "signal": "thumbs_up"})
    client.post("/api/feedback", json={"session_id": "s3", "signal": "thumbs_down"})

    summary = db.get_feedback_summary()
    assert summary["rated_queries"] == 3


def test_feedback_router_accepts_optional_metadata():
    db = _store("s1")
    client = _app(db)

    resp = client.post(
        "/api/feedback",
        json={
            "session_id": "s1",
            "signal": "thumbs_down",
            "note": "token=secret-value",
            "source": "answer_panel",
            "correlation_id": "turn-1",
        },
    )

    assert resp.status_code == 200
    row = db.list_retrieval_feedback()[0]
    assert row["metadata"]["note"] == "token=[REDACTED]"
    assert row["metadata"]["source"] == "answer_panel"
