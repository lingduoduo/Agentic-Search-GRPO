# Train Loop Failed-Run Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `train_loop` must not return an empty history and no error when every step failed, while keeping the skip that protects long runs from isolated failures.

**Architecture:** Two raise conditions on one error type — a consecutive-failure streak (fails fast) and a completed-loop-with-no-successful-step check (catches short runs below the threshold).

**Tech Stack:** Python 3.10+, pytest, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-22-train-loop-reports-failed-runs-design.md`

## Global Constraints

- **Do not remove the skip.** `step_async` drives live search rollouts and can legitimately hang or blip; `step_timeout_s` exists because of it. Aborting on the first failure re-breaks what the `continue` was added to fix.
- **Count consecutive failures, not cumulative.** Reset on every success, or an alternating failure pattern trips a guard meant for broken trainers.
- **Both raise conditions are needed.** The streak alone lets a `max_steps` below the threshold slip through; the empty-history check alone still burns the whole budget. Each was mutation-checked separately.
- **Keep `max_consecutive_failures = 0` as an escape hatch** for a caller that wants the old unbounded behaviour.
- **Chain with `from exc`** so the original traceback survives.
- **Commit before any `git checkout`-based mutation script.** Also note the pre-commit `ruff-format` hook aborts the commit when it reformats — re-stage and commit again, and do not let an `&&`-chained mutation run against an un-committed tree.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: Report a run that never trained

**Files:**
- Modify: `src/model/post_training/grpo/training.py`
- Test: `tests/unit/test_train_loop.py`

- [x] **Step 1: Add an `AlwaysFailingTrainer` double** whose every step raises.
- [x] **Step 2: Write three failing tests** — all steps fail raises; a broken run stops well before `max_steps=500`; a `max_steps=1` total failure still raises. Watch all three fail with `DID NOT RAISE`.
- [x] **Step 3: Write three tolerance controls** — an isolated failure is still skipped, alternating failures never trip the threshold, `max_consecutive_failures=0` restores unbounded skipping. These pass from the start.
- [x] **Step 4: Add `TrainingStepsFailedError`** with a docstring recording why the unbounded skip was wrong and how to opt out.
- [x] **Step 5: Add `TrainLoopConfig.max_consecutive_failures = 3`** with a comment on why consecutive rather than cumulative.
- [x] **Step 6: Track the streak** — increment on failure, reset on success, raise at the threshold chained from the last exception.
- [x] **Step 7: Add the empty-history check** after the loop, for runs shorter than the threshold.
- [x] **Step 8: Confirm the two pre-existing skip tests pass unmodified.**

**Verify:** `pytest tests/unit/test_train_loop.py` → 11 passed.

---

### Task 2: Mutation-check each half separately

- [x] **Step 1: Remove the streak guard** → `test_a_broken_run_stops_early_instead_of_burning_the_budget` fails (500 attempts).
- [x] **Step 2: Remove the empty-history check** → `test_a_short_run_that_wholly_fails_still_raises` fails.
- [x] **Step 3: Restore and confirm an empty `git diff`** plus a clean `__pycache__`.

---

### Task 3: Full verification

- [x] `ruff check . --fix && ruff format .`
- [x] `pytest tests/unit` — 4335 passed, 1 skipped.
- [x] Spec and plan committed on the branch.
- [x] Note the #616 rebase dependency in the PR body.
