# Store Write Rollback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** If a public `AgenticSearchStore` method raises part-way through, its
uncommitted statements are rolled back, so a later commit cannot publish them.

**Architecture:** One change to the `_synchronized` wrapper in
`src/internal/db/store.py`. The outermost wrapped call rolls back the open
transaction on any exception. No method bodies change.

**Tech Stack:** Python 3.10+, stdlib `sqlite3`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-store-write-rollback-design.md`

## Global Constraints

- Python floor is 3.10. Do not use 3.11+-only syntax.
- Only the depth-1 (outermost) wrapped call may roll back.
- Catch `BaseException`, not `Exception`, and always re-raise.
- No changes to individual store method bodies.

## Review Focus

- A read-only method that raises: must not raise a second error from
  `rollback()`. It is guarded by `in_transaction` (Task 1, test 2 covers the
  state).
- An inner public call that fails inside an outer call which catches the
  error: the outer work must survive (Task 1, test 3).
- A rollback on a file-backed WAL database, not only `:memory:`: the tests use
  `tmp_path` files.

---

### Task 1: Roll back in `_synchronized`

**Files:**
- Modify: `src/internal/db/store.py` (`_synchronized.wrap.guarded` ~:111-117;
  `__init__` ~:142-150)
- Test: `tests/unit/db/test_store_write_rollback.py` (create)

**Interfaces:**
- Consumes: `AgenticSearchStore`, `_synchronized`, `UserRecord` from
  `src.internal.db`.
- Produces: nothing new. Behaviour only.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `pytest tests/unit/db/test_store_write_rollback.py -v`
Expected: `test_failed_profile_replace_is_not_committed_by_a_later_write`
fails (profile is `["home"]` or `[]`), and
`test_failed_write_leaves_no_open_transaction` fails (`in_transaction` is
True). The nesting test passes, and it must keep passing after the change.

- [ ] **Step 3: Implement**

In `_synchronized.wrap`:

```python
    def wrap(method):
        @functools.wraps(method)
        def guarded(self, *args, **kwargs):
            with self._lock:
                self._call_depth += 1
                try:
                    return method(self, *args, **kwargs)
                except BaseException:
                    # A method is the transaction unit (see above). If it
                    # raises before commit(), sqlite3 leaves its implicit
                    # transaction open, and the next method's commit() would
                    # publish the half-done write. Only the outermost call
                    # decides: an inner failure an outer method catches is
                    # still the outer method's transaction.
                    if self._call_depth == 1 and self._conn.in_transaction:
                        self._conn.rollback()
                    raise
                finally:
                    self._call_depth -= 1

        return guarded
```

In `__init__`, right after `self._lock = threading.RLock()`:

```python
        self._call_depth = 0
```

Add one sentence to the `_synchronized` docstring: "A method that raises is
rolled back at the outermost call, so a failed write can never be published by
a later commit."

- [ ] **Step 4: Run the tests and verify they pass**

Run: `pytest tests/unit/db/ tests/unit/test_db_store.py tests/unit/memory/ -q`
Expected: all pass.

- [ ] **Step 5: Mutation check**

Comment out the `self._conn.rollback()` line and run
`pytest tests/unit/db/test_store_write_rollback.py -q`. Expected: 2 failures.
Restore the line, then delete stale bytecode: `find src tests -name
'__pycache__' -path '*db*' -exec rm -rf {} +`. Re-run and expect a pass.

- [ ] **Step 6: Full suite, lint, commit**

Run: `pytest -q -x -p no:cacheprovider` and `ruff check . && ruff format --check .`

```bash
git add src/internal/db/store.py tests/unit/db/test_store_write_rollback.py
git commit -m "Store writes roll back on failure instead of leaking into the next commit"
```
