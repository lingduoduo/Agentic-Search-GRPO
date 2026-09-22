# Agent Loop Context and Dead Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound `ToolAgentLoop`'s context to its configured budget, stop its crop discarding the system prompt, retire a parser nothing calls, document a stop reason that only looks dead, and make `AgenticRAGLoop`'s fail-open visible to its caller.

**Architecture:** Five independent fixes across three loop modules. No change to what any loop decides — only to how it manages its own token budget, and to what it reports about a check that could not run.

**Tech Stack:** Python 3.10+, pytest, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-22-agent-loop-context-and-dead-paths-design.md`

## Global Constraints

- **Never crop `prompt_ids` inside `ToolAgentLoop.run`.** The teardown recovers the prompt/response split from `len(prompt_ids) - len(response_mask)`. Dropping tokens the mask already covers desyncs the two and corrupts every downstream trainer. Bound the loop by stopping it.
- **Do not delete `StopReason.BUDGET_EXHAUSTED`.** It masks the plateau arm at the budget; removing it makes the final round's evidence skip injection. Mutation-verified — see spec §4.
- **Do not widen `ALLOWED_STATUSES` or `ALLOWED_DETAIL_KEYS`** in `control_flow_trace.py`. Those allowlists are deliberately closed. Use `status="failed"` and the existing `fallback` detail key.
- **Keep `_normalize_query`** when deleting from `planner.py` — `partition_search_requests` depends on it.
- Fixes 1, 2 and 5 are test-first: watch each test fail for the right reason before implementing.
- Fix 4 changes no behaviour, so its tests pass immediately. Verify them by mutation instead of by RED.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: Bound the tool loop's context

**Files:**
- Modify: `src/agents/tool/tool_calling.py`
- Test: `tests/unit/test_tool_loop_context_budget.py` (new)

- [x] **Step 1: Prove the growth.** Test a multi-turn run under a small `prompt_length` with a bulky tool; assert no prompt handed to the backend exceeds it. Watch it fail listing the growth (`618 … 4314`).
- [x] **Step 2: Prove the crop drops the prefix.** Test `_build_prompt_ids_with_tools_sync` over budget with a system message; assert the system text survives. Watch it fail.
- [x] **Step 3: Guard the budget.** Add the `len(prompt_ids) + len(tool_response_ids) > self.prompt_length` break beside the existing response-budget break, with a comment saying why stopping beats cropping.
- [x] **Step 4: Crop like the base loop.** Import `_crop_prompt_ids`; pass `self._encode_system_prefix(messages)` as the preserved prefix.
- [x] **Step 5: Confirm the run still terminates normally** — under the turn caps, `final_answer` set, tool messages still in `trajectory_messages`.

**Verify:** `pytest tests/unit/test_tool_loop_context_budget.py` green; the 55 pre-existing tool-loop tests still green.

---

### Task 2: Retire `Planner.decide`

**Files:**
- Modify: `src/agents/components/planner.py`, `tests/unit/test_components.py`

- [x] **Step 1: Confirm it is unreachable.** Grep `decide(`, `SearchAction`, `RerankAction`, `AnswerAction` across `src/`, `examples/`, `tests/`. Only test callers may appear.
- [x] **Step 2: Delete** `decide()`, the three action dataclasses, `PlannerDecision`, `_SEARCH_RE`, `_RERANK_RE`, `_ANSWER_RE`, `_RETRIEVER_BY_NAME`, `_FALLBACK_QUERY_MAX_CHARS`. Keep `_normalize_query`.
- [x] **Step 3: Rewrite the module docstring** — it documents a precedence rule only the deleted path implemented.
- [x] **Step 4: Delete the eleven `decide()` tests**, re-pointing the whitespace/case duplicate-detection test at `partition_search_requests` so that coverage is moved, not lost.

**Verify:** `pytest tests/unit/test_components.py` green; no residual references; `ruff check` clean.

---

### Task 3: Document and pin the budget/plateau precedence

**Files:**
- Modify: `src/agents/components/loop_controller.py`
- Test: `tests/unit/test_agent_loop.py`

- [x] **Step 1: Add two characterization tests.** At the budget with a plateau also satisfied, the round's `<information>` still reaches the model; below budget, the same plateau early-stops. Expect both to pass immediately.
- [x] **Step 2: Mutation-verify.** Delete the budget arm; confirm both go red and that the failure shows only Round 1's evidence in the transcript. Restore and confirm a clean `git diff`.
- [x] **Step 3: Document the precedence** in `should_continue_searching`'s docstring: why the ordering is load-bearing, that enforcement lives in `partition_search_requests`, and which test pins it.

**Verify:** `pytest tests/unit/test_agent_loop.py tests/unit/test_loop_controller.py` green.

---

### Task 4: Report degraded sufficiency

**Files:**
- Modify: `src/agents/search/agentic_rag.py`, `tests/unit/test_agentic_rag.py`

- [x] **Step 1: Write three failing tests** — the flag on the result, a real verdict *not* marked degraded, and the degradation visible on the control-flow trace.
- [x] **Step 2: Return `(sufficient, degraded)`** from `_is_sufficient`; `degraded` is True only on the fail-open path.
- [x] **Step 3: Carry it through `run()`** onto `AgenticRAGResult.sufficiency_degraded` (defaulted, so callers need no change).
- [x] **Step 4: Emit it** as `status="failed"` with `fallback=degraded`, within the existing trace vocabulary.
- [x] **Step 5: Update the pre-existing fail-open test** to the new tuple contract, asserting both halves.
- [x] **Step 6: Mutation-verify** — make the fail-open path report `degraded=False`; confirm three tests go red and the non-degraded controls stay green.

**Verify:** `pytest tests/unit/test_agentic_rag.py` green (24 tests).

---

### Task 5: Full verification

- [x] `ruff check . --fix && ruff format .`
- [x] `pytest tests/unit` — 4329 passed, 1 skipped.
- [x] Spec and plan committed on the branch.
