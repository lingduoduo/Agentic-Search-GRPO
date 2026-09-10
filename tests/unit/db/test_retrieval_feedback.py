"""Tests for retrieval_feedback table and AgenticSearchStore methods."""

from __future__ import annotations

from src.internal.db.store import AgenticSearchStore


def _store() -> AgenticSearchStore:
    return AgenticSearchStore(path=":memory:")


def test_save_and_count_feedback():
    db = _store()
    db.save_retrieval_feedback("s1", "thumbs_up")
    db.save_retrieval_feedback("s1", "thumbs_down")
    summary = db.get_feedback_summary()
    assert summary["rated_queries"] == 2


def test_thumbs_up_rate_correct():
    db = _store()
    db.save_retrieval_feedback("s1", "thumbs_up")
    db.save_retrieval_feedback("s2", "thumbs_up")
    db.save_retrieval_feedback("s3", "thumbs_down")
    summary = db.get_feedback_summary()
    assert summary["thumbs_up_rate"] == pytest.approx(2 / 3, abs=1e-4)


def test_empty_summary_returns_zeros():
    db = _store()
    summary = db.get_feedback_summary()
    assert summary["thumbs_up_rate"] == 0.0
    assert summary["ctr"] == 0.0
    assert summary["rated_queries"] == 0


def test_rows_without_a_target_read_as_overall():
    db = _store()
    db.save_retrieval_feedback("s1", "thumbs_up")  # legacy row, no target
    db.save_retrieval_feedback("s2", "thumbs_down", target="retrieval")
    by_target = db.get_feedback_summary()["by_target"]
    assert by_target["overall"] == {"rated": 1, "thumbs_up_rate": 1.0}
    assert by_target["retrieval"] == {"rated": 1, "thumbs_up_rate": 0.0}
    assert by_target["generation"] == {"rated": 0, "thumbs_up_rate": 0.0}


def test_ctr_uses_distinct_sessions():
    db = _store()
    # One session with multiple ratings — still counts as 1 rated session
    db.save_retrieval_feedback("s1", "thumbs_up")
    db.save_retrieval_feedback("s1", "thumbs_down")
    db.save_retrieval_feedback("s2", "thumbs_up")
    summary = db.get_feedback_summary()
    # 2 distinct sessions rated out of however many total exist
    assert summary["rated_queries"] == 3


def test_session_id_can_be_none():
    db = _store()
    db.save_retrieval_feedback(None, "thumbs_up")
    summary = db.get_feedback_summary()
    assert summary["rated_queries"] == 1


def test_feedback_metadata_is_redacted_and_queryable():
    db = _store()
    feedback_id = db.save_retrieval_feedback(
        "s1",
        "thumbs_down",
        note="failed with Bearer abc.def",
        source="answer_panel",
        correlation_id="turn-1",
    )
    rows = db.list_retrieval_feedback()
    assert rows[0]["id"] == feedback_id
    assert rows[0]["metadata"]["note"] == "failed with Bearer [REDACTED]"
    assert rows[0]["metadata"]["source"] == "answer_panel"
    assert rows[0]["correlation_id"] == "turn-1"
    assert db.get_feedback_summary()["rated_queries"] == 1


import pytest  # noqa: E402 (needed after test functions that reference it)
