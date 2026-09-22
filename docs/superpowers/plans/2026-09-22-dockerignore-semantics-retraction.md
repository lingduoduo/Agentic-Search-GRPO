# Dockerignore Semantics Retraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the false `.dockerignore` rule #623 merged, and add the build-context CI job that would have caught both it and the original defect.

**Architecture:** Helper returns to last-match-wins, pinned by a parametrised test over three measured forms; a lightweight contract Dockerfile runs in CI.

**Tech Stack:** Docker (buildkit), GitHub Actions, Python 3.10+, pytest, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-22-dockerignore-semantics-retraction-design.md`

## Global Constraints

- **Reset the checkout before measuring anything about the repo.** `git fetch` alone left a stale `main`, which is what produced the false observation behind #623.
- **Never grep `--progress=plain` output for a string the RUN command itself echoes.** That reported all three forms as failing and nearly buried the retraction. Match only timestamped output lines.
- **One form per `--no-cache` build.** Buildkit cached `COPY . .` across context changes and made a mutation check pass that should have failed.
- **A test that models an external tool must be parametrised over measured cases**, or it agrees with the author's assumption — the failure mode in both #622 and #623.
- **Do not revert `.dockerignore` to `data`.** Both negated forms work; changing it back is churn.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: Measure the three forms properly

- [x] **Step 1: Reset to `origin/main`** so the file under test is the merged one.
- [x] **Step 2: Write each form as a complete `.dockerignore`**, printing the active rules before each build.
- [x] **Step 3: Build each with `--no-cache`**, matching only timestamped `OK`/`FAIL` lines.
- [x] **Step 4: Record** — only `data` with no negations keeps the corpus out.

---

### Task 2: Retract the false rule

**Files:**
- Modify: `tests/unit/test_docker_stack_contract.py`, `.dockerignore`

- [x] **Step 1: Restore last-match-wins** in `_excluded_by`; split rule parsing into `_dockerignore_rules`.
- [x] **Step 2: Replace the semantics test** with a parametrised one over the three measured forms, noting how they were measured.
- [x] **Step 3: Keep a test** that unnegated siblings under `data/` stay excluded.
- [x] **Step 4: Fix the `.dockerignore` comment**, which asserted the retracted rule.

---

### Task 3: Put a real build in CI

**Files:**
- Add: `docker/Dockerfile.contract`
- Modify: `.github/workflows/ci.yml`

- [x] **Step 1: Write the contract** — corpus and registry present, nothing else from `data/`, editable install registers `src`, `src` resolves from `/`.
- [x] **Step 2: Add the `Docker build context` job.**
- [x] **Step 3: Validate both ways** — passes on this branch; fails at the first step against the pre-#622 form.
- [x] **Step 4: Correct the contract's own comments**, which repeated the retracted claim.

---

### Task 4: Verify and clean up

- [x] `ruff check . --fix && ruff format .`
- [x] `pytest tests/unit` — 4351 passed, 1 skipped.
- [ ] Prune the local build cache and stop colima.
- [x] Spec and plan committed on the branch.
