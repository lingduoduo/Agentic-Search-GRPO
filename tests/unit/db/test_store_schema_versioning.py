"""Store schema versioning with a min-reader version
(spec: 2026-09-25-store-schema-versioning-design.md)."""

from __future__ import annotations

import sqlite3

import pytest

from src.internal.db import UserRecord
from src.internal.db.store import AgenticSearchStore, Migration, SchemaVersionError


def _store_class(version: int, migrations: dict[int, Migration]):
    class _Store(AgenticSearchStore):
        SCHEMA_VERSION = version
        _MIGRATIONS = migrations

    return _Store


def _version(path) -> int:
    with sqlite3.connect(path) as conn:
        return conn.execute("PRAGMA user_version").fetchone()[0]


def _min_reader(path) -> int | None:
    with sqlite3.connect(path) as conn:
        try:
            row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'min_reader_version'"
            ).fetchone()
        except sqlite3.OperationalError:
            return None
    return row[0] if row else None


def _columns(path, table) -> set[str]:
    with sqlite3.connect(path) as conn:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


ADD_NICKNAME = Migration(
    statements=("ALTER TABLE users ADD COLUMN nickname TEXT",),
    breaks_older_readers=False,
)
ADD_AUDIT_TABLE = Migration(
    statements=("CREATE TABLE audit_marks (id TEXT PRIMARY KEY)",),
    breaks_older_readers=False,
)
RESHAPE = Migration(
    statements=("CREATE TABLE reshaped (id TEXT PRIMARY KEY)",),
    breaks_older_readers=True,
)


# --- stamping --------------------------------------------------------------


def test_a_new_database_is_stamped_at_the_current_version(tmp_path):
    path = tmp_path / "s.sqlite3"
    AgenticSearchStore(path).close()
    assert _version(path) == AgenticSearchStore.SCHEMA_VERSION == 1
    assert _min_reader(path) == 1


def test_an_in_memory_store_opens_at_the_current_version():
    store = AgenticSearchStore(":memory:")
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == 1
    store.close()


def test_a_legacy_unversioned_database_is_stamped_and_keeps_its_rows(tmp_path):
    path = tmp_path / "s.sqlite3"
    store = AgenticSearchStore(path)
    store.upsert_user(UserRecord(id="alice"))
    store.close()
    with sqlite3.connect(path) as conn:  # what a pre-versioning build left
        conn.execute("PRAGMA user_version = 0")
        conn.execute("DROP TABLE schema_meta")

    store = AgenticSearchStore(path)
    assert store.get_user("alice") is not None
    store.close()
    assert _version(path) == 1
    assert _min_reader(path) == 1


# --- upgrades --------------------------------------------------------------


def test_migrations_apply_in_order(tmp_path):
    path = tmp_path / "s.sqlite3"
    AgenticSearchStore(path).close()

    v3 = _store_class(3, {2: ADD_NICKNAME, 3: ADD_AUDIT_TABLE})
    v3(path).close()

    assert _version(path) == 3
    assert "nickname" in _columns(path, "users")
    assert _columns(path, "audit_marks") == {"id"}
    assert _min_reader(path) == 1  # both additive: v1 builds can still read


def test_a_breaking_migration_raises_the_min_reader(tmp_path):
    path = tmp_path / "s.sqlite3"
    AgenticSearchStore(path).close()

    _store_class(3, {2: ADD_NICKNAME, 3: RESHAPE})(path).close()

    assert _version(path) == 3
    assert _min_reader(path) == 3


def test_a_failed_migration_stops_at_the_last_good_version(tmp_path):
    path = tmp_path / "s.sqlite3"
    AgenticSearchStore(path).close()
    broken = Migration(
        statements=(
            "CREATE TABLE half_done (id TEXT)",
            "ALTER TABLE no_such_table ADD COLUMN x TEXT",
        ),
        breaks_older_readers=True,
    )

    with pytest.raises(sqlite3.OperationalError):
        _store_class(3, {2: ADD_NICKNAME, 3: broken})(path)

    assert _version(path) == 2
    assert _min_reader(path) == 1
    with sqlite3.connect(path) as conn:  # the failed step left nothing behind
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    assert "half_done" not in tables


def test_a_gap_in_the_migrations_raises(tmp_path):
    path = tmp_path / "s.sqlite3"
    AgenticSearchStore(path).close()

    with pytest.raises(RuntimeError, match="migration to schema v2"):
        _store_class(3, {3: ADD_AUDIT_TABLE})(path)
    assert _version(path) == 1


# --- rollback --------------------------------------------------------------


def test_an_older_build_opens_a_compatible_newer_database_read_only(tmp_path):
    path = tmp_path / "s.sqlite3"
    AgenticSearchStore(path).close()
    _store_class(3, {2: ADD_NICKNAME, 3: ADD_AUDIT_TABLE})(path).close()

    store = AgenticSearchStore(path)  # rolled back to a v1 build
    store.upsert_user(UserRecord(id="bob"))
    assert store.get_user("bob") is not None
    store.close()

    assert _version(path) == 3  # the rollback wrote no schema change
    assert _min_reader(path) == 1


def test_an_older_build_refuses_a_database_it_cannot_read(tmp_path):
    path = tmp_path / "s.sqlite3"
    AgenticSearchStore(path).close()
    _store_class(3, {2: ADD_NICKNAME, 3: RESHAPE})(path).close()

    with pytest.raises(SchemaVersionError, match="schema >= 3"):
        AgenticSearchStore(path)
    assert _version(path) == 3


def test_a_new_database_on_a_later_build_runs_every_migration(tmp_path):
    """A fresh database starts from the v1 baseline; it must not be stamped at
    v3 without the v2 and v3 changes."""
    path = tmp_path / "s.sqlite3"
    _store_class(3, {2: ADD_NICKNAME, 3: RESHAPE})(path).close()

    assert _version(path) == 3
    assert "nickname" in _columns(path, "users")
    assert _columns(path, "reshaped") == {"id"}
    assert _min_reader(path) == 3
