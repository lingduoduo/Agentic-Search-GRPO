# Retire Vestigial Agent Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop maintaining a second agent framework with no agents and a connector-checkpoint type with no connectors, and leave no known-broken test behind.

**Architecture:** Removal plus guard tests. The live framework (`AgentLoopBase` + registry) is untouched.

**Tech Stack:** Python 3.10+, pytest, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-22-retire-vestigial-agent-infra-design.md`

## Global Constraints

- **Keep `TenantRedisClient`.** The `redis_*` modules under `src/internal/servers/redis/` use it independently of the queue manager.
- **Do not change `bamboogle.py` behaviour.** `evaluate_bamboogle` is duck-typed on `invoke(state) -> Any`; only its docstrings name the deleted class.
- **Repair a broken test only when the repair can be justified without running it.** Where it cannot, delete — inventing assertions for an unrunnable subsystem is worse than removing the file.
- **Remove blanket `# noqa: F821` rather than preserving it.** It is what hid the breakage; taking it off is how the rest of the breakage surfaces.
- **Pin every removal with a guard test.** "Nothing imports it" is the condition that let these drift unnoticed.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: Remove the second agent framework

**Files:**
- Delete: `src/agents/core/graph_base.py`, `src/internal/chat/queue_manager.py`, `tests/unit/test_graph_base.py`, `tests/unit/test_queue_manager.py`
- Modify: `src/agents/core/__init__.py`, `src/internal/configs/constants.py`, `src/model/post_training/eval/bamboogle.py`

- [x] **Step 1: Confirm `AgentQueueManager` has no consumer** besides `graph_base` and its own test.
- [x] **Step 2: Confirm `TenantRedisClient` has independent consumers** so it is not swept up.
- [x] **Step 3: Delete the four files.**
- [x] **Step 4: Remove `InvokeFrom`**, orphaned once the queue is gone.
- [x] **Step 5: Drop the `core/__init__.py` paragraph** explaining the `AgentState` collision — the collision is gone with the module.
- [x] **Step 6: Rewrite the three `bamboogle.py` docstring references** to describe the duck-typed `invoke()` contract instead of naming a deleted class.

**Verify:** no residual references to `graph_base`, `BaseAgent`, `AgentQueueManager`, `QueueEvent`, `InvokeFrom`.

---

### Task 2: Remove the connector-checkpoint surface

**Files:**
- Modify: `src/internal/connectors/models.py`, `src/internal/connectors/__init__.py`, `src/__init__.py`

- [x] **Step 1: Delete `ConnectorCheckpoint`** and both re-exports.
- [x] **Step 2: Confirm the live models survive** — `Document`, `SlimDocument`, `HierarchyNode`, `ConnectorFailure`.

---

### Task 3: Leave no known-broken test

**Files:**
- Delete: `tests/integration/tests/indexing/test_checkpointing.py`, `tests/integration/tests/indexing/test_polling.py`
- Modify: `tests/integration/tests/indexing/test_repeated_error_state.py`, `tests/integration/tests/indexing/test_initial_permission_sync.py`

- [x] **Step 1: Find every `MockConnectorCheckpoint` caller.** The audit reported one; there are four.
- [x] **Step 2: Delete `test_checkpointing.py`** — entirely about the removed surface.
- [x] **Step 3: Repair the mechanical cases** — substitute the literal `{"has_more": False}` for the mock's `.model_dump(mode="json")` payload; strip the `# noqa: F821,F841`.
- [x] **Step 4: Re-run `ruff`** and handle what the removed `noqa` exposes.
- [x] **Step 5: Delete `test_polling.py`** — it references two further undefined names (`expected_first_start`, `time_before_first_attempt`) whose reconstruction would be unverifiable new coverage.

**Verify:** `ruff check .` clean; all remaining indexing test files parse.

---

### Task 4: Pin the retirements and verify

- [x] **Step 1: Add `tests/unit/test_retired_agent_infrastructure.py`** — both modules fail to import, `ConnectorCheckpoint` absent at all three levels, the live registry still resolves its four loops, live connector models survive.
- [x] `ruff check . --fix && ruff format .`
- [x] `pytest tests/unit` — 4315 passed, 1 skipped.
- [x] Spec and plan committed on the branch.
