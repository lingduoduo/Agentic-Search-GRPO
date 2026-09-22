# Docker Stack Startability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the containerised stack startable, and guard the class of mistake with unit-speed contract tests.

**Architecture:** Two artifact fixes (install order, corpus location) plus a test module that exercises the Docker artifacts against the real app and build context.

**Tech Stack:** Docker, docker compose, Python 3.10+, pytest, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-22-docker-stack-can-start-design.md`

## Global Constraints

- **Exercise the artifact, never grep it.** Two claims in the investigation were wrong from greps: `/health` looked missing because `create_base_app` registers it, and the `mlx-lm` build looked broken because `mlx` has no linux wheel while `mlx-lm` conditions it on Darwin itself. Instantiate the app; read the wheel metadata.
- **Do not add a non-root `USER`.** Named volumes are root-owned and the app writes sqlite to `/data`; adding it without volume ownership turns a startable stack into one that cannot write, with no CI to catch it.
- **Do not change the image's Python version.** 3.11 vs CI's 3.10/3.12 is a real inconsistency, but not one to change blind on an unbuilt image.
- **Declare what a test imports.** `PyYAML` was transitive only; that is how `httpx` went undeclared in #607. Declare it and import it directly — an `importorskip` would let the whole guard vanish silently.
- **Commit before any `git checkout`-based mutation script.** This discarded uncommitted Dockerfile work mid-task, for the third time this session.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: Establish what is actually broken

- [x] **Step 1: Exercise the retrieval app's routes** with a TestClient rather than grepping decorators. Result: `/health` exists and returns 200; the healthcheck claim was wrong.
- [x] **Step 2: Reproduce the editable install** in a clean venv with only `pyproject.toml` present. Result: empty finder MAPPING; `import src` works from `/app` only.
- [x] **Step 3: Run `resolve_corpus_docs` on the compose path.** Result: `ValueError: Unknown corpus spec`.
- [x] **Step 4: Check `mlx-lm`'s wheel metadata** before claiming a build failure. Result: `mlx` is conditioned on Darwin; claim withdrawn.

---

### Task 2: Guard the claims

**Files:**
- Add: `tests/unit/test_docker_stack_contract.py`
- Modify: `requirements-unit-test.txt`

- [x] **Step 1: Write the healthcheck tests** — the compose path must be a route the app serves, and must return 200. These pass, documenting the withdrawn claim.
- [x] **Step 2: Write the corpus test** — not under a volume mount, present in the repo, not `.dockerignore`d. Watch it fail.
- [x] **Step 3: Write the install-order test.** Watch it fail.
- [x] **Step 4: Declare `PyYAML`** and import it directly.

---

### Task 3: Fix the artifacts

**Files:**
- Modify: `Dockerfile`, `.dockerignore`, `docker/docker-compose.yml`

- [x] **Step 1: Move `pip install -e .` after `COPY . .`**, keeping `requirements.txt` installed first for layer caching; `--no-deps` since requirements pins everything.
- [x] **Step 2: Point the retrieval command at `/app/data/corpus.jsonl`** and add `.dockerignore` negations for the two tracked data files.
- [x] **Step 3: Drop the obsolete `version: "3.8"`** key.

---

### Task 4: Document the single-worker constraint

- [x] **Step 1: Comment the `CMD` line** — where someone would add `--workers` — naming each piece of per-process `app.state` and the Redis prerequisite.
- [x] **Step 2: Add a guard test** that fails on `--workers` as a directive. Mutation-check it.
- [x] **Step 3: Ignore comment lines in that check** — the first run failed on the explanatory comment itself, which names the flag.
- [x] **Step 4: Assert the `app.state` attributes still exist**, so the constraint is re-examined rather than silently outliving its cause.

---

### Task 5: Verify

- [x] `ruff check . --fix && ruff format .`
- [x] `pytest tests/unit` — 4347 passed, 1 skipped.
- [x] Spec and plan committed on the branch.
- [ ] **Not done: a real `docker build`.** No daemon available in this environment, so the artifacts are verified by contract tests and a venv reproduction, not by a build. Flagged in the PR.
