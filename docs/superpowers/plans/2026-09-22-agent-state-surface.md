# Agent State Surface Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut `src/agents/core/state.py` back to the state the loops actually maintain, and stop `from src import RouteDecision` resolving to the dead half of a name collision.

**Architecture:** Pure removal plus guard tests. Nine re-export-only types, eight unwritten `AgentState` fields, three test-only methods, and three unreachable `TaskStatus` members.

**Tech Stack:** Python 3.10+, pytest, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-22-agent-state-surface-design.md`

## Global Constraints

- **Resolve reachability with AST, not grep.** Four of these names exist twice in the tree (`RouteDecision`, `RetrievedDocument`, `ToolCall`, `ToolResult`); a name grep cannot tell the live class from the dead one and will report everything as live.
- **Commit before running any checkout-based mutation script.** `git checkout -- <file>` restores to HEAD, which silently discards uncommitted work. This bit during implementation.
- **Do not re-point the `RouteDecision` export** at the internal routing class — promoting an internal type to the public API contradicts the PR's thesis. Remove it so the import fails loudly.
- **Keep `ToolExecutionResult.success`.** The cut is "surface implying an orchestration layer", not "every accessor without a caller".
- **Mutation-check every absence guard.** A test asserting something is gone passes trivially whether or not it is testing anything.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: Establish true reachability

**Files:** none (analysis)

- [x] **Step 1: Walk every `ImportFrom` in `src/`, `tests/`, `examples/`** and record which module each name is actually imported *from*, so same-name classes are separated.
- [x] **Step 2: Classify** each of the sixteen `state.py` types as live or re-export-only. Result: nine re-export-only.
- [x] **Step 3: Trace the cascade** — `to_tool_result` (fed `add_tool_result`), `TaskStatus.PENDING` (only `TaskNode`'s default).

---

### Task 2: Remove the dead surface

**Files:**
- Modify: `src/agents/core/state.py`, `src/__init__.py`, `src/agents/__init__.py`, `src/agents/core/__init__.py`

- [x] **Step 1: Rewrite `state.py`** keeping the seven live types; drop the nine dead ones, the eight unwritten fields, `record_trace`, `add_tool_result`, `to_tool_result`, and `TaskStatus.PENDING`/`RUNNING`/`RETRYING`. Add a scope note saying re-adding a type is a claim a loop maintains it.
- [x] **Step 2: Strip nine re-export lines** from each of the three `__init__.py` chains. Confirm `src.__all__` (computed from `globals()`) follows automatically.
- [x] **Step 3: Assert by import** that all nine names are gone from `src` and the live seven remain.

**Verify:** 269 → 150 lines in `state.py`; `python -c "import src"` clean.

---

### Task 3: Pin the removals

**Files:**
- Modify: `tests/unit/test_agent_state.py`, `tests/unit/test_state_models.py`

- [x] **Step 1: Add four guards** to `test_agent_state.py` — names stay removed, `RouteDecision` cannot resolve to the unused class, `TaskStatus` has only terminal outcomes, `AgentState` has exactly its written fields.
- [x] **Step 2: Mutation-check them.** Re-add `ToolType`, `TaskStatus.RETRYING` and `AgentState.trace`; confirm three guards fail. Restore — **by rewriting the file, not `git checkout`**, if the work is uncommitted.
- [x] **Step 3: Trim `test_state_models.py`** to what it uniquely covers (slotted + `asdict`-able); delete the test that only exercised the dead surface and the `to_tool_result` test.

**Verify:** `pytest tests/unit/test_agent_state.py tests/unit/test_state_models.py` → 16 passed.

---

### Task 4: Full verification

- [x] `ruff check . --fix && ruff format .`
- [x] `pytest tests/unit` — 4332 passed, 1 skipped.
- [x] Spec and plan committed on the branch.
