# Store Schema Versioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The SQLite store records a schema version and a min-reader version.
It applies numbered migrations transactionally, and an older build refuses
only a database it cannot read.

**Architecture:** `_ensure_schema()` in `AgenticSearchStore.__init__` works
from `PRAGMA user_version`, a `schema_meta` table, and the class attributes
`SCHEMA_VERSION` and `_MIGRATIONS`.

**Tech Stack:** Python 3.10+ stdlib `sqlite3`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-store-schema-versioning-design.md`

## Global Constraints

- Version 1 is today's schema. No existing table changes.
- A compatible rollback writes nothing to the schema.
- Each migration is one transaction, covering its statements, the version
  bump and the meta update.

## Review Focus

- `_init_schema` uses `executescript`, which commits implicitly. It must never
  run inside a migration transaction.
- `_synchronized` wraps only public methods. `_ensure_schema` is private and
  runs in `__init__` before any concurrency, so it needs no lock.
- An existing on-disk database from before this change (v0 with data) is
  stamped without data loss.

---

### Task 1: Tests (red)

- [x] **Write the tests.** Write
  `tests/unit/db/test_store_schema_versioning.py` covering every case in the
  spec's Testing section. Use a subclass factory:
  `_store_class(version, migrations)` returns a subclass with those class
  attributes. Helpers read `PRAGMA user_version` and `schema_meta` through a
  fresh `sqlite3.connect(path)`.
- [x] **Run them.** Expect failures: `Migration`, `SchemaVersionError` and
  `schema_meta` do not exist yet.

### Task 2: Implement (green)

- [x] **Implement.** In `store.py`, add:
  - `Migration`, `SchemaVersionError`;
  - `SCHEMA_VERSION = 1`, `_MIGRATIONS = {}`;
  - `_ensure_schema`, which replaces `self._init_schema()` in `__init__`;
  - `_read_min_reader()` and `_write_meta()`.
- [x] **Run them.** Run the new tests and every existing store test under
  `tests/unit/db` and `tests/unit/test_db_store.py`. Expect them to pass.
- [x] **Mutation checks.** These are in the spec. Restore and purge
  `__pycache__` after each.

### Task 3: Docs and verification

- [x] **Docs.** Add a short "Schema versioning" note in the store's module
  docstring or in `docs/architecture.md`, whichever already describes the
  store. Cover how to add a migration and when to set
  `breaks_older_readers`.
- [x] **Verify.** Run the full unit suite, then ruff, then `git diff --check`.
  Open the PR.
