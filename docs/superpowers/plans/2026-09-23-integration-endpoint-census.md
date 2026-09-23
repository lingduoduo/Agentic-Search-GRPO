# Integration Endpoint Census Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a census of the integration suite's endpoint drift that is accurate enough to decide 119 files on, and ship it as a re-runnable tool.

**Architecture:** AST extraction of endpoint templates, normalised against the running app's routes, reported in two attribution modes that bracket the answer.

**Tech Stack:** Python 3.10+ (`ast`), the real `create_web_app()`.

**Spec:** `docs/superpowers/specs/2026-09-23-integration-endpoint-census-design.md`

## Global Constraints

- **Do not delete anything.** This measures; #630 already retired the one directory a route probe had condemned. The rest is the user's decision.
- **Never regex an f-string for a path.** `f"{API_SERVER_URL}/admin/api-key/{api_key}"` truncates at the brace and yields a path that matches nothing for the wrong reason — the original defect.
- **Strip query strings before comparing.** Six endpoints were reported removed purely because of a `?`.
- **Report two attribution modes.** Direct-only under-counts (tests call managers); helper-aware over-counts "mixed" (importing a manager is not calling it). Quote the bracket, not a single number.
- **Verify "gone" by prefix, not exact match**, before believing it.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: Replace the broken census

**Files:**
- Add: `examples/audit_integration_endpoints.py`

- [x] **Step 1: Walk the AST** for f-strings and concatenations referencing a server URL; rebuild templates with `{}` per interpolation.
- [x] **Step 2: Normalise both sides** — strip query, collapse parameter names.
- [x] **Step 3: Compare against a real `create_web_app()`**, not a grep of route decorators.
- [x] **Step 4: Fix the query-string false positives** found on the first run (`/admin/chat-session-history?{}`, `/?transportType=...`).

---

### Task 2: Report the decision-relevant split

- [x] **Step 1: Bucket per file** — every endpoint gone / mixed / fully served / unattributable.
- [x] **Step 2: Add helper attribution**, since two thirds of tests build no URL themselves.
- [x] **Step 3: Add `--direct-only`** and document why neither mode is exact.
- [x] **Step 4: Add `--show-files` and `--format json`** so the output can drive a decision rather than just inform one.

---

### Task 3: Verify the tool itself

- [x] **Step 1: Spot-check three "gone" families by prefix** against the live app — all zero.
- [x] **Step 2: Confirm the query fix moved endpoints** from gone to served (26 → 35).
- [x] **Step 3: Check both invocation forms** — `python -m` and direct.
- [x] `ruff check . --fix && ruff format .`
- [x] `pytest tests/unit` — 4362 passed, 1 skipped, unaffected.
- [x] Spec and plan committed on the branch.
