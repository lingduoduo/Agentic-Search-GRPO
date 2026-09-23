# Retire Indexing Integration Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve #622's open question by establishing whether the repaired indexing tests can pass, and remove them once it is clear they cannot.

**Architecture:** Deletion, justified by probing the running application's routes rather than reading the tests.

**Tech Stack:** Python 3.10+, pytest.

**Spec:** `docs/superpowers/specs/2026-09-23-retire-indexing-integration-tests-design.md`

## Global Constraints

- **Ask the app for its routes; do not grep for them.** Grepping `@app.` missed a route registered by a shared factory in #624 and produced a wrong claim that shipped.
- **Probe by prefix, not just exact match.** An endpoint can exist under a different parameter spelling; `/manage/admin/cc-pair` has zero prefix matches, which is what makes the conclusion safe.
- **Do not extend the deletion to the rest of the suite.** 69 of 122 files reference removed endpoints, but that is the user's decision about a 122-file suite.
- **State the extraction's limits.** The endpoint census is a regex over test sources and produced a truncated capture; quote the direction, not the decimal.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: Settle whether a live stack would help

- [x] **Step 1: Note that #629 supplies the stated precondition** — a live Postgres/Redis stack in CI.
- [x] **Step 2: Extract the endpoints the two tests reach** through `CCPairManager` and `IndexAttemptManager`.
- [x] **Step 3: Probe a real `create_web_app()`** for those paths, exact and by prefix. Result: none served; the only "connector" route is an unrelated OAuth callback.
- [x] **Step 4: Cross-check against the repo's own record** — CLAUDE.md documents the indexing pipeline's removal.

---

### Task 2: Remove what cannot run

- [x] **Step 1: Delete `tests/integration/tests/indexing/`** — both repaired tests, the file-connector test using the same removed API, its fixtures, and the mock-connector conftest.

---

### Task 3: Record the wider finding without acting on it

- [x] **Step 1: Census the suite** — 180 endpoints referenced, 26 served, 69 of 122 files touching a removed one.
- [x] **Step 2: Calibrate the census** against known routes before quoting it.
- [x] **Step 3: Write it into the spec as a decision for the user**, not a change.

---

### Task 4: Verify

- [x] `ruff check . --fix && ruff format .`
- [x] `pytest tests/unit` — 4357 passed, 1 skipped, unchanged — the integration tree is outside `testpaths`.
- [x] Spec and plan committed on the branch.
