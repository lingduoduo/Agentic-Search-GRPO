# Dockerignore Negation Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the corpus actually reach the image, and stop the contract test from confirming a `.dockerignore` that does not work.

**Architecture:** Exclude `data/`'s contents rather than the directory, and ground the test helper's `.dockerignore` model in observed `docker build` behaviour.

**Tech Stack:** Docker, Python 3.10+, pytest, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-22-dockerignore-negation-unreachable-design.md`

## Global Constraints

- **Never verify `.dockerignore` semantics by reading the docs.** "Last match wins" is documented and insufficient: the builder prunes excluded directories, so a negation inside one never applies. Build it both ways.
- **A helper that models an external tool must be pinned to observed behaviour**, or the test and the code share one assumption and agree for the same wrong reason.
- **Keep the probe cheap.** `python:3.11-slim` plus the real `.dockerignore` and the real editable install exercises both fixes without pulling torch.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: Prove the shipped fix does not work

- [x] **Step 1: Start a daemon and build a targeted probe** asserting the corpus is present in the image.
- [x] **Step 2: Observe `CORPUS: MISSING`** against the `.dockerignore` merged in #622.
- [x] **Step 3: Change to `data/*` and rebuild.** Observe the corpus, the registry, and only those two entries under `data/`.
- [x] **Step 4: Confirm #622's other fix in the same build** — finder MAPPING contains `src`, and `import src` resolves from `/`.

---

### Task 2: Fix the artifact

**Files:**
- Modify: `.dockerignore`

- [x] **Step 1: Exclude `data/*`, not `data`,** with a comment recording why and that it was verified by build.

---

### Task 3: Fix the test that let it through

**Files:**
- Modify: `tests/unit/test_docker_stack_contract.py`

- [x] **Step 1: Implement the pruning rule** in `_excluded_by` — an ancestor excluded as a directory blocks all descendant negations, then last-match-wins for the rest.
- [x] **Step 2: Add a semantics test** pinning both observed outcomes, so the model cannot drift back to a guess.
- [x] **Step 3: Mutation-check** by reverting `data/*` to `data`; the corpus test must fail.

---

### Task 4: Verify

- [x] `ruff check . --fix && ruff format .`
- [x] `pytest tests/unit` — 4348 passed, 1 skipped.
- [x] Full `docker build` attempted and **deliberately aborted** part-way: it reached the torch/CUDA install phase having taken free disk from 24Gi to 11Gi, and the marginal value over the targeted probe did not justify the risk of filling the host. Build cache pruned, colima stopped. Recorded in the PR.
- [x] Spec and plan committed on the branch.
