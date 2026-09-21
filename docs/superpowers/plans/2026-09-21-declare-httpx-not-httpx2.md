# Declare httpx, Not httpx2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `requirements.txt`, `requirements-unit-test.txt` and the pyproject `mcp` extra declare `httpx` — the client `src/` imports — instead of the never-imported `httpx2`.

**Architecture:** A declaration fix in three files, plus a test that ties the declaration to the imports so the two cannot drift apart again. No source changes: every `import httpx` is already correct.

**Tech Stack:** Python 3, pytest, `tomllib`/`tomli`, `ast`.

**Spec:** `docs/superpowers/specs/2026-09-21-declare-httpx-not-httpx2-design.md`

## Global Constraints

- **Do not touch `aiohttp==3.9.3`.** The `openai>=1.0.0,<3` cap in `requirements-unit-test.txt:35-41` is load-bearing on it. Lifting it is a separate, deliberate decision.
- **Do not migrate any call site to `httpx2`.** The distributions are unrelated; a migration means renaming imports and exception classes across seven files and is out of scope.
- Use `httpx>=0.28.1,<1.0`. The floor matches `fastmcp`'s own requirement exactly, so this declaration cannot conflict with the dependency that previously supplied httpx transitively.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: Declare httpx in all three dependency files

**Files:**
- Modify: `requirements.txt`, `requirements-unit-test.txt`, `pyproject.toml`

**Interfaces:**
- Produces: `httpx>=0.28.1,<1.0` in both requirements files and in `[project.optional-dependencies].mcp`. Removes `httpx2>=0.28.0` from all three.

- [x] **Step 1: Replace the requirement lines**

In both requirements files, replace `httpx2>=0.28.0` with a three-line comment and `httpx>=0.28.1,<1.0`. The comment records why the package is declared at all — that `src/` imports it directly and it previously arrived only as a transitive — and that it is not `httpx2`.

In `pyproject.toml`, replace `"httpx2>=0.28.0",` in the `mcp` extra with `"httpx>=0.28.1,<1.0",`. No comment there; the extra is a four-line list and the spec carries the reasoning.

**Verify:** `git diff` shows exactly three changed declarations and no source changes.

---

### Task 2: Guard the declaration against the imports

**Files:**
- Create: `tests/unit/test_http_client_declaration.py`

**Interfaces:**
- Produces: `_top_level_imports(root)` (ast walk over `root.rglob("*.py")`, returning top-level module names) and `_declared(requirements)` (regex over non-comment lines, returning distribution names). Both module-private to this test.

- [x] **Step 1: Assert the premise — `src/` imports httpx, never httpx2**

`test_src_imports_httpx_not_httpx2` parses every `.py` under `src/` and asserts `"httpx" in modules` and `"httpx2" not in modules`. This is the fact the other two tests rest on. If the repo ever genuinely adopts httpx2, this fails first and names the reason.

- [x] **Step 2: Assert both requirements files declare httpx and not httpx2**

`test_requirements_declare_httpx_and_not_httpx2` loops over `REQUIREMENTS_FILES` and asserts membership both ways, with the file name in the assertion message.

- [x] **Step 3: Assert the `mcp` extra declares httpx**

`test_mcp_extra_declares_httpx` loads `pyproject.toml` and checks the `mcp` extra. Match on `"httpx>"` rather than `"httpx"` — a bare prefix match would also accept `httpx2`.

**Verify:** `pytest tests/unit/test_http_client_declaration.py -q` → 3 passed.

---

### Task 3: Mutation-check and full verification

- [x] **Step 1: Mutation-check the guard**

Restore `httpx2>=0.28.0` in all three files, run the new test file, confirm it goes red, then `git checkout --` the three files. Expect two of three failing: the `src/`-facing premise test correctly stays green, because the mutation did not touch `src/`.

Commit before mutating, so the restore is a checkout and not a hand-edit.

**Verify:** 2 failed, 1 passed under mutation; `git status --porcelain` clean after restore.

- [x] **Step 2: Full suite**

**Verify:** `pytest -q` → 4306 passed, 1 skipped.
