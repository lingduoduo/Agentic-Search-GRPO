"""Tests for load_feedback_examples."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from src.internal.db import AgenticSearchStore
from src.model.post_training.data import PromptTrainingExample, load_feedback_examples


def _seed_store(db_path: str, rows: list[dict]) -> None:
    """Seed feedback and chat messages using AgenticSearchStore public API."""
    with AgenticSearchStore(db_path) as store:
        for r in rows:
            session_id = r["session_id"]
            store.create_chat_session(session_id=session_id, user_id=None)
            if r.get("message"):
                store.add_chat_message(
                    session_id=session_id, role="user", content=r["message"]
                )
            store.save_retrieval_feedback(
                session_id, r["signal"], target=r.get("target")
            )


def test_target_reaches_the_example_metadata(tmp_path):
    db = str(tmp_path / "test.sqlite3")
    _seed_store(
        db,
        [
            {
                "session_id": "s1",
                "signal": "thumbs_down",
                "message": "Q1",
                "target": "retrieval",
            },
            {"session_id": "s2", "signal": "thumbs_up", "message": "Q2"},
        ],
    )
    examples = sorted(
        load_feedback_examples(db, min_ratings=1), key=lambda e: e.question
    )
    assert examples[0].metadata == {
        "human_signal": -1.0,
        "human_signal_target": "retrieval",
    }
    # A row recorded before targets existed reads as "overall".
    assert examples[1].metadata["human_signal_target"] == "overall"


def test_thumbs_up_sets_positive_signal(tmp_path):
    db = str(tmp_path / "test.sqlite3")
    _seed_store(db, [{"session_id": "s1", "signal": "thumbs_up", "message": "Q1"}])
    examples = load_feedback_examples(db, min_ratings=1)
    assert len(examples) == 1
    assert examples[0].question == "Q1"
    assert examples[0].ground_truth == ""
    assert examples[0].metadata["human_signal"] == 1.0


def test_thumbs_down_sets_negative_signal(tmp_path):
    db = str(tmp_path / "test.sqlite3")
    _seed_store(db, [{"session_id": "s1", "signal": "thumbs_down", "message": "Q1"}])
    examples = load_feedback_examples(db, min_ratings=1)
    assert examples[0].metadata["human_signal"] == -1.0


def test_session_without_chat_messages_is_skipped(tmp_path):
    db = str(tmp_path / "test.sqlite3")
    _seed_store(db, [{"session_id": "s1", "signal": "thumbs_up"}])
    with pytest.raises(ValueError, match="Only 0 rated sessions"):
        load_feedback_examples(db, min_ratings=1)


def test_raises_when_below_min_ratings(tmp_path):
    db = str(tmp_path / "test.sqlite3")
    _seed_store(db, [{"session_id": "s1", "signal": "thumbs_up", "message": "Q"}])
    with pytest.raises(
        ValueError, match="Only 1 rated sessions found; need at least 5"
    ):
        load_feedback_examples(db, min_ratings=5)


def test_multiple_sessions_all_returned(tmp_path):
    db = str(tmp_path / "test.sqlite3")
    rows = [
        {"session_id": f"s{i}", "signal": "thumbs_up", "message": f"Q{i}"}
        for i in range(3)
    ]
    _seed_store(db, rows)
    examples = load_feedback_examples(db, min_ratings=3)
    assert len(examples) == 3
    assert all(ex.metadata["human_signal"] == 1.0 for ex in examples)


def test_returns_prompt_training_example_instances(tmp_path):
    db = str(tmp_path / "test.sqlite3")
    _seed_store(db, [{"session_id": "s1", "signal": "thumbs_up", "message": "Q"}])
    examples = load_feedback_examples(db, min_ratings=1)
    assert all(isinstance(ex, PromptTrainingExample) for ex in examples)
