# Store schema versioning with a min-reader version: design

## Problem

`AgenticSearchStore` (`src/internal/db/store.py`, SQLite) has no schema
version:

- `_init_schema` runs `CREATE TABLE IF NOT EXISTS`.
- `_migrate_schema` runs `ALTER TABLE … ADD COLUMN` and swallows
  `OperationalError`.

So a newer build's schema changes are applied silently. An older build, after
a rollback, has no way to tell whether it can still read the database. Rollback
safety depends on every past change having happened to be additive, and nothing
records or checks that.

## Decision (approved by the user: min-reader version)

### Two numbers per database

- **Version.** The schema version the database is at, stored in
  `PRAGMA user_version`.
- **Min reader.** The oldest code schema version that can still read the
  database. It is stored in a one-row table, `schema_meta(key TEXT PRIMARY KEY,
  value INTEGER NOT NULL)`, under the key `min_reader_version`. The model is
  SQLite's own read/write file-format versions.

### Code side

- `AgenticSearchStore.SCHEMA_VERSION = 1`. **Version 1 is today's schema**:
  everything `_init_schema` (including `_migrate_schema`) produces.
- `AgenticSearchStore._MIGRATIONS: dict[int, Migration]` is empty today.
  `Migration` is a frozen dataclass:
  - `statements: tuple[str, ...]`;
  - `breaks_older_readers: bool`.

  `_MIGRATIONS[n]` upgrades the database from `n-1` to `n`.
- Both are class attributes, so tests can subclass them to exercise upgrades.

### Opening a database (`_ensure_schema`, which replaces the direct `_init_schema()` call in `__init__`)

Let `v = PRAGMA user_version`, `S = SCHEMA_VERSION`, and `m` = the stored min
reader, or 0 when absent.

| Case | What happens |
|---|---|
| `v == 0` (new, or older than versioning) | Run `_init_schema()` (idempotent, so it also completes a legacy database). Then set `user_version = S` and `min_reader_version = S`. |
| `0 < v < S` | Run `_init_schema()`. Then for each `k` in `v+1..S`, apply `_MIGRATIONS[k]` **in one transaction**: its statements, `user_version = k`, and when `breaks_older_readers` is set, `min_reader_version = k`. A failed migration rolls back its own step and raises. Earlier steps stay applied. |
| `v == S` | Run `_init_schema()`, exactly as today. |
| `v > S` and `S >= m` | Compatible rollback. Open **without** writing any schema: no `_init_schema`, and no version or meta changes. Log WARNING: "database is at schema v{v}; this build is v{S} and is compatible". |
| `v > S` and `S < m` | Raise `SchemaVersionError` (a new `RuntimeError` subclass in `store.py`). The message says what the database needs ("requires a build with schema ≥ m"), and that the fix is to deploy a newer build or restore a backup taken before the upgrade. |

- **Missing migrations.** A gap, meaning a `_MIGRATIONS` entry missing for
  some `k` in `v+1..S`, raises at open with a clear message. It never silently
  skips a step.
- **`:memory:` databases** always start at `v == 0`.

### Out of scope

- Down-migrations.
- Automatic backups.
- The Postgres/SQLAlchemy engine code, which is unused.
- Changing any existing table.

## Testing

`tests/unit/db/test_store_schema_versioning.py`:

- **Stamping.**
  - A new database (file and `:memory:`) gets `user_version == 1` and
    `min_reader == 1`.
  - A legacy database has tables but `user_version == 0`. Build it by
    creating a store, then running `PRAGMA user_version = 0` and dropping
    `schema_meta`. Reopening it stamps it and keeps its rows.
- **Upgrades.** With a subclass at `SCHEMA_VERSION = 3` and two migrations:
  - both apply in order, the version is 3, and the new column or table
    exists;
  - `min_reader` becomes 3 only when migration 3 `breaks_older_readers`, and
    stays 1 when neither breaks;
  - a migration whose statement fails leaves the database at the last good
    version and raises;
  - a gap in `_MIGRATIONS` raises.
- **Rollback.**
  - A database at v3 with min reader 1, opened by the v1 class: opens, and
    the version and min reader are unchanged.
  - A database at v3 with min reader 3, opened by the v1 class: raises
    `SchemaVersionError`, whose message names the required version.
- **Regression.** All existing store tests pass unchanged.
- **Mutation checks.**
  - Skip the `S < m` check, and the refusal test goes red.
  - Always set `min_reader`, and the compatible-rollback test goes red.
  - Drop the transaction, and the failed-migration test goes red.
