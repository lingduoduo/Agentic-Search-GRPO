# LoopSnapshot Redundant Field Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether `LoopSnapshot.model_emitted_answer` should participate in control flow, and act on it.

**Architecture:** Pure removal plus a guard test and a stated invariant on `LoopSnapshot`.

**Tech Stack:** Python 3.10+, pytest, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-22-loop-snapshot-redundant-field-design.md`

## Global Constraints

- **Decide from the call sites, not the comment.** `# reserved for Phase 2` invites keeping it; the call graph settles it.
- **Do not merge the two controller decisions.** If they ever become one method, the flag's argument returns with the design that needs it.
- **Mutation-check the absence guard.** A test asserting a field is gone passes trivially either way.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: Settle the question

- [x] **Step 1: Trace both construction sites** against the method called immediately after each.
- [x] **Step 2: Conclude** — `final_answer_decision` only ever sees `True`, `should_continue_searching` only ever sees `False`; the flag cannot disambiguate anything.

---

### Task 2: Remove it

**Files:**
- Modify: `src/agents/components/loop_controller.py`, `src/agents/search/search.py`, `tests/unit/test_loop_controller.py`

- [x] **Step 1: Write the failing guard test** asserting `LoopSnapshot`'s exact field set. Watch it fail on the field's presence.
- [x] **Step 2: Remove the field** and both construction sites, and drop it from the `_snap` test helper.
- [x] **Step 3: Add a `LoopSnapshot` docstring** stating that every field must be one a decision reads, and why this one was not.
- [x] **Step 4: Add a second test** that both decisions still resolve correctly, so the change is not only absence-checked.

---

### Task 3: Verify

- [x] **Step 1: Mutation-check** — re-adding the field fails the guard. Restore; confirm no `MUTANT` residue.
- [x] `ruff check . --fix && ruff format .`
- [x] `pytest tests/unit` — 4330 passed, 1 skipped.
- [x] Spec and plan committed on the branch.
