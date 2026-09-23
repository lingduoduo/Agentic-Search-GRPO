# Serving / Heavy Requirements Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `requirements.txt` becomes a serving baseline; backend and training dependencies move to companion files; two wrongly-declared packages go.

**Architecture:** Split by deployment shape, not lifecycle — `faiss-cpu`/`pyserini` are backend dependencies of `servers/retrieval/server.py`, not training ones.

**Tech Stack:** pip requirements files, Docker, pytest, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-23-split-serving-and-heavy-requirements-design.md`

## Global Constraints

- **Use an import pattern that catches function-local imports.** A line-anchored grep reports `pyserini` as unused while missing its deferred import; this repo defers heavily.
- **Never call a package unused because nothing imports it.** `python-multipart` (FastAPI `UploadFile` routes), `pytest-asyncio` (`asyncio_mode = "auto"`) and `urllib3` (floor) all stay.
- **Verify a package is actually declared before calling it unused.** `peft` and `matplotlib` were named from a local environment; neither is in `requirements.txt`.
- **Keep `pytest-asyncio` in the baseline.** CLAUDE.md documents `pip install -r requirements.txt` then `pytest`; moving it breaks the documented flow for negligible weight.
- **Commit before any `git checkout`-based mutation.** This discarded the `requirements.txt` edit — the fourth time in this session.
- Run `ruff check . --fix && ruff format .` before each commit; the pre-commit `ruff-format` hook aborts the commit when it reformats, so re-stage and commit again.

---

### Task 1: Classify before moving anything

- [x] **Step 1: Read all 31 declared packages** rather than a sampled set.
- [x] **Step 2: Import-scan `src/`, `examples/`, `tests/`** with an indentation-tolerant pattern.
- [x] **Step 3: Trace the ambiguous ones** — faiss and pyserini reach `servers/retrieval/server.py` via `internal/retrieval/service.py`, so they are backend, not training.
- [x] **Step 4: Check the never-imported ones for framework need** — `UploadFile` routes require `python-multipart`.
- [x] **Step 5: Confirm the removals are deferred imports** and that `requirements-unit-test.txt` already omits them.

---

### Task 2: Split the files

**Files:**
- Modify: `requirements.txt`
- Add: `requirements-retrieval-heavy.txt`, `requirements-training.txt`

- [x] **Step 1: Remove the six** from the baseline; record in its header what moved where and why two were dropped.
- [x] **Step 2: Write the companions**, each stating what selects it and what it costs (pyserini's JVM and `torch>=2.9`).
- [x] **Step 3: Document the required-but-unimported three** in the baseline's notes.

---

### Task 3: Guard the split

**Files:**
- Add: `tests/unit/test_requirements_layout.py`

- [x] **Step 1: Pin the split, not the contents** — six tests over where each class of package must live.
- [x] **Step 2: Mutation-check** by re-adding `faiss-cpu` to the baseline.

---

### Task 4: Documentation and verification

- [x] **Step 1: Update CLAUDE.md's setup section** with the additive installs and when each is needed.
- [x] `ruff check . --fix && ruff format .`
- [x] `pytest tests/unit` — 4357 passed, 1 skipped.
- [ ] CI reports the new image size against #627's 2.9GB baseline.
