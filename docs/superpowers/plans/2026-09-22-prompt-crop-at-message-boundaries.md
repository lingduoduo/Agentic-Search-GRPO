# Prompt Crop At Message Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The context-budget crop must never hand the model a malformed prompt, and must report what it dropped.

**Architecture:** Drop whole oldest messages until the render fits, with the render step overridable so the tool loop (which must also fit its schemas) shares the same fitting. The old token slice stays as a flagged last resort.

**Tech Stack:** Python 3.10+, pytest, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-22-prompt-crop-at-message-boundaries-design.md`

## Global Constraints

- **Do not report through `AgentLoopOutput.truncated`.** It is set by 1 of 4 loops and read by nothing; a second flag there is another dead channel. Use `metrics`, which `post_training/reward.py` reads.
- **Do not widen the control-flow trace allowlist.** No existing detail key fits token accounting, and #617 established the argument against widening it.
- **Always keep the leading system message and the newest message.** Without the second, a single oversized message would fit nothing and loop forever.
- **Always write both metric keys, including zeros** — otherwise a consumer cannot distinguish "nothing dropped" from "loop does not report".
- **Override the render, not the build,** in `ToolAgentLoop`. Overriding the build is what left it with its own tail slice.
- **Commit before any `git checkout`-based mutation script**, and remember the pre-commit `ruff-format` hook aborts the commit when it reformats.

---

### Task 1: Prove the defect

- [x] **Step 1: Probe `_crop_prompt_ids`** with a ChatML-shaped template, six turns, a 300-token budget. Confirm the surviving tail starts mid-message.
- [x] **Step 2: Record the evidence** — a headless `ng 4</think>` fragment glued to the system block, plus an orphaned `<|im_end|>`.

---

### Task 2: Crop at message boundaries

**Files:**
- Modify: `src/agents/core/base.py`, `src/agents/tool/tool_calling.py`
- Test: `tests/unit/test_prompt_crop.py`

- [x] **Step 1: Write the two discriminating failing tests** — nothing headless after the system block, and balanced role markers. Watch both fail.
- [x] **Step 2: Split `_render_prompt_ids`** out of `_build_prompt_ids_sync`.
- [x] **Step 3: Add `_fit_messages_to_budget`** dropping oldest-first, keeping system + newest.
- [x] **Step 4: Keep `_crop_prompt_ids` as the last resort**, documented as such, for a single oversized message; set `prompt_hard_truncated` and log when it fires.
- [x] **Step 5: Override `_render_prompt_ids` in `ToolAgentLoop`** to inject schemas, and reduce `_build_prompt_ids_with_tools_sync` to a call through the base.

**Verify:** the probe now yields a well-formed prompt; `pytest tests/unit/test_prompt_crop.py` green.

---

### Task 3: Make the loss visible

- [x] **Step 1: Write failing tests** that both multi-turn loops surface the counts in `output.metrics`, and that the keys are present as zeros when nothing was dropped.
- [x] **Step 2: Add `record_prompt_budget_metrics`** on the base.
- [x] **Step 3: Call it** from `SearchAgentLoop.run` (before `_finalize_run_metrics`) and `ToolAgentLoop.run`, and seed the keys in the search loop's initial metrics.
- [x] **Step 4: Add `logger.warning`** for both the dropped-message and hard-truncation cases (`base.py` had no logger; add one).

---

### Task 4: Keep fitting cheap

**Files:**
- Modify: `src/agents/core/base.py`
- Add: `examples/benchmark_prompt_fitting.py`
- Test: `tests/unit/test_prompt_crop.py`

- [x] **Step 1: Measure the first implementation.** Drop-and-rerender cost 77 renders and 1.1s for one build at 40 turns / 1024 budget, against 28ms for the raw slice.
- [x] **Step 2: Write a failing test bounding the render count** to <= 4 regardless of messages dropped. Watch it fail at 78.
- [x] **Step 3: Add `_estimate_message_tokens`** — one content `encode` plus a cached per-message overhead; deliberately conservative so the corrective render rarely fires.
- [x] **Step 4: Choose the kept suffix arithmetically** walking back from the newest message, then render once, correcting downward only if the estimate was optimistic.
- [x] **Step 5: Add the benchmark** so the numbers are reproducible; verify both `python -m examples.…` and direct invocation.

**Verify:** renders constant at 3; 42ms at 40 turns against 1077ms before.

---

### Task 5: Mutation-check and verify

- [x] **Step 1: Revert to the token slice** → 6 tests fail across both files. Restore; confirm no `MUTANT` residue and an empty diff.
- [x] **Step 2: Update the one pre-existing test** whose contract changed (`test_build_prompt_ids_falls_back_to_encode` asserted the budget filled by slicing).
- [x] `ruff check . --fix && ruff format .`
- [x] `pytest tests/unit` green.
- [x] Spec and plan committed on the branch.
