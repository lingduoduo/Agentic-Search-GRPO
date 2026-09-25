# Store writes roll back on failure: design

## Problem

`AgenticSearchStore` (`src/internal/db/store.py`) shares one `sqlite3`
connection. About 39 public methods run one or more statements and then call
`self._conn.commit()`, with no `try`/`rollback`. Python's `sqlite3` opens an
implicit transaction at the first DML statement. So when a method raises
between its first statement and its `commit()`, the transaction **stays
open**. The next public method to call `commit()` commits it, whatever it
contains.

Concrete damage:

- **`replace_user_profile`** (:2074) is documented "Atomically replace". It
  runs `DELETE FROM user_profiles WHERE user_id = ?` and then one `INSERT` per
  entry. Suppose an entry raises part-way through. The next unrelated write
  (for example `add_chat_message`, which also runs on the request path) then
  commits the DELETE, and the user's profile is gone.
- **`add_chat_message`** (:690) runs `INSERT` and then `UPDATE chat_sessions`.
  If the UPDATE fails, the message row is committed later anyway, without the
  session timestamp that was supposed to go with it.

Only `register_user` and `upgrade_password_hash` use `with self._conn:`, which
rolls back on error.

## Decision

Fix the pattern once, in `_synchronized`, instead of editing 39 methods.
`_synchronized` already wraps every public method in the instance's `RLock`,
and its docstring says the lock spans whole methods *because* a method is the
transaction unit. Rolling back in that same wrapper makes the unit atomic on
failure as well.

```python
def guarded(self, *args, **kwargs):
    with self._lock:
        self._call_depth += 1
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            if self._call_depth == 1 and self._conn.in_transaction:
                self._conn.rollback()
            raise
        finally:
            self._call_depth -= 1
```

- **Outermost call only.** Public methods call other public methods (that is
  why the lock is reentrant). Suppose an inner call raises and the outer method
  catches the error and carries on. Then the outer method owns the
  transaction, and rolling it back from the inner frame would discard the
  outer method's earlier statements. Only the depth-1 frame decides.
- **`BaseException`**, so a `KeyboardInterrupt`, or a `CancelledError`
  propagating through a thread, cannot leave a half-written transaction for
  the next caller to commit.
- **`in_transaction` guard**, so a read-only method that raises does not issue
  a pointless `ROLLBACK`.
- **`self._call_depth = 0`** is set in `__init__` before `_init_schema`.
  `_init_schema` is private, so it is not wrapped and the counter is not
  touched. It is safe to keep the counter on the instance: the lock is held
  whenever the counter changes, so only one thread can be inside at a time.

`replace_user_profile`'s docstring becomes true. No method bodies change.

## Out of scope

- Rewriting methods to use `with self._conn:`, which would be redundant with
  the wrapper.
- Schema versioning and down-migrations, raised in the same investigation.
- Direct `store._conn` users outside the class (offline training data
  loaders). The class docstring already names these as outside the lock's
  guarantee.
- The orphaned user message left when `/api/agent` dispatch fails. That
  message was committed successfully on its own; it is a request-level
  compensation question, not a transaction bug.

## Testing

New file: `tests/unit/db/test_store_write_rollback.py`.

1. **Profile survives a failed replace.** Store profile A. Call
   `replace_user_profile` with one good entry followed by an entry whose
   `.get` raises. Assert it raises. Then call `add_chat_message` on a real
   session (a committing write). Then assert `get_user_profile` still returns
   A.
2. **The failed replace leaves no open transaction.** After (1)'s failure,
   `store._conn.in_transaction` is `False`.
3. **Nested call: the inner failure does not roll back the outer.** Define a
   `_synchronized` subclass in the test. Its outer method inserts a row, calls
   an inner public method that inserts and raises, catches that error, and
   commits. The outer row persists. The inner row persists as well, which is
   the documented cost of outer ownership, and the test asserts it explicitly
   so the behaviour is pinned.
4. **Mutation check.** Delete the `rollback()` line and watch test 1 fail.
   This is done by hand and recorded in the PR.

The existing suites `tests/unit/test_db_store.py`, `tests/unit/db/` and
`tests/unit/memory/` must stay green.
