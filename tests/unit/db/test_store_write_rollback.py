# tests/unit/db/test_store_write_rollback.py
"""A public store method that raises must not leave its writes for the next
commit to publish (spec: 2026-09-25-store-write-rollback-design.md)."""

from __future__ import annotations

import pytest

from src.internal.db import AgenticSearchStore, UserRecord
from src.internal.db.store import _synchronized


class _ExplodingEntry(dict):
    def get(self, key, default=None):  # noqa: ARG002
        raise RuntimeError("bad entry")


def _store(tmp_path) -> AgenticSearchStore:
    store = AgenticSearchStore(tmp_path / "state.sqlite3")
    store.upsert_user(UserRecord(id="u1"))
    return store


def test_failed_profile_replace_is_not_committed_by_a_later_write(tmp_path):
    store = _store(tmp_path)
    store.replace_user_profile(
        "u1", [{"topic": "work", "subtopic": "role", "content": "engineer"}]
    )
    session = store.create_chat_session(user_id="u1", title="t")

    with pytest.raises(RuntimeError, match="bad entry"):
        store.replace_user_profile(
            "u1",
            [{"topic": "home", "content": "Paris"}, _ExplodingEntry()],
        )
    # A committing write on the request path, the one that used to publish the DELETE.
    store.add_chat_message(session.id, role="user", content="hi")

    assert [e.content for e in store.get_user_profile("u1")] == ["engineer"]
    store.close()


def test_failed_write_leaves_no_open_transaction(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(RuntimeError):
        store.replace_user_profile("u1", [_ExplodingEntry()])
    assert store._conn.in_transaction is False
    store.close()


@_synchronized
class _NestingStore(AgenticSearchStore):
    def outer_swallows_inner_failure(self) -> None:
        self._conn.execute(
            "INSERT INTO user_profiles (id, user_id, topic, subtopic, content,"
            " created_at, updated_at) VALUES ('outer', 'u1', 't', '', 'c', 'x', 'x')"
        )
        try:
            self.inner_fails()
        except RuntimeError:
            pass
        self._conn.commit()

    def inner_fails(self) -> None:
        self._conn.execute(
            "INSERT INTO user_profiles (id, user_id, topic, subtopic, content,"
            " created_at, updated_at) VALUES ('inner', 'u1', 't', '', 'c', 'x', 'x')"
        )
        raise RuntimeError("inner")


def test_inner_failure_does_not_roll_back_the_outer_call(tmp_path):
    store = _NestingStore(tmp_path / "state.sqlite3")
    store.outer_swallows_inner_failure()
    ids = {e.id for e in store.get_user_profile("u1")}
    # The outer call owns the transaction: its row survives, and so does the
    # inner row it chose to commit after catching the error.
    assert ids == {"outer", "inner"}
    store.close()
