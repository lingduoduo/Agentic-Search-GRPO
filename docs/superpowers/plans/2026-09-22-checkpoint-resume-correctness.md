# Checkpoint Resume Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A trainer that cannot checkpoint must be refused loudly, before the run starts — never silently skipped while the step manifest is written or read anyway.

**Architecture:** One named error plus a single hook-resolver used by both checkpoint helpers, and one up-front check in `train_loop` so an unsupported configuration fails at step 0 rather than at the first checkpoint.

**Tech Stack:** Python 3.10+, pytest, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-22-checkpoint-resume-correctness-design.md`

## Global Constraints

- **Do not add checkpointing to `SFTTrainer` / `DPOTrainer` / `GRPOTrainer`.** Whether they need it is a separate question; guessing at it is how the silent no-op arrived.
- **`ckpt_dir` without `ckpt_every` must keep working for every trainer.** That path only writes `metrics.jsonl` and produces no checkpoint, so it is not gated.
- **Guard the save hook at configuration time**, not at the first checkpoint — the whole value of failing is failing before the compute is spent.
- **Do not change the step-skip-on-exception behaviour.** It is a second, separate defect; folding it in would blur this PR.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: Refuse checkpoints a trainer cannot honour

**Files:**
- Modify: `src/model/post_training/grpo/training.py`
- Test: `tests/unit/test_train_loop.py`

- [x] **Step 1: Add an `UncheckpointableTrainer` double** mirroring the shape of `SFTTrainer`/`DPOTrainer`/`GRPOTrainer` — `step_async` only, no hooks.
- [x] **Step 2: Write four failing tests** — refused resume (no step runs), refused periodic-checkpoint config (refusal precedes step 0), metrics-only run still works, capable trainer still resumes. Watch the first two fail with `DID NOT RAISE`; the other two pass from the start as controls.
- [x] **Step 3: Add `CheckpointNotSupportedError`** with a docstring recording which trainers do and do not implement the hooks, and why skipping is worse than raising.
- [x] **Step 4: Add `_require_checkpoint_hook(trainer, hook)`** and route both `save_checkpoint` and `load_checkpoint` through it, so neither can no-op even when called directly.
- [x] **Step 5: Guard `train_loop` up front** — `if config.ckpt_dir is not None and config.ckpt_every: _require_checkpoint_hook(trainer, "save_checkpoint")`.
- [x] **Step 6: Confirm the in-tree caller is unaffected** — `examples/run_retriever_aware_grpo.py` drives `SearchAgentGRPOTrainer`, which inherits both hooks.

**Verify:** `pytest tests/unit/test_train_loop.py` → 9 passed; `-k 'train_loop or grpo or trainer or checkpoint'` → 287 passed.

---

### Task 2: Full verification

- [x] `ruff check . --fix && ruff format .`
- [x] `pytest tests/unit` green.
- [x] Spec and plan committed on the branch.
