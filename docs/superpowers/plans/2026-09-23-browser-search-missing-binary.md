# Browser Search Missing Binary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A missing `playwright-cli` must be reported, not rendered as an empty result set.

**Architecture:** A dedicated error from the subprocess wrapper, re-raised past the degrade-to-empty handler, plus a startup refusal in `main()`.

**Tech Stack:** Python 3.10+, pytest.

**Spec:** `docs/superpowers/specs/2026-09-23-browser-search-missing-binary-design.md`

## Global Constraints

- **Keep the degrade-to-empty behaviour for real failures.** It exists so one bad search target does not fail the request; only absence of the tool is re-raised, and a test pins the difference.
- **Put the startup check in `main()`, not `create_app()`.** Tests build the app; a hard failure in the factory trades one bug for a worse one.
- **Do not install `playwright-cli` in the image.** Node plus a browser runtime in an image just cut to 2.04GB, for a slow low-quality fallback, is a decision for the user rather than a side effect of a bug fix.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: Establish what the absence looks like

- [x] **Step 1: Confirm `_run` does not catch `FileNotFoundError`.**
- [x] **Step 2: Trace where it lands** — `_search_and_process`'s broad handler, logged at warning, returning `[]`.
- [x] **Step 3: Note the cascade** — serpapi → browser means `[]` reads as "no results", not "no provider".

---

### Task 2: Make it legible

**Files:**
- Modify: `src/internal/servers/web_search/browser.py`
- Test: `tests/unit/test_browser_search_availability.py`

- [x] **Step 1: Write five failing tests** before any implementation.
- [x] **Step 2: Add `BrowserSearchUnavailableError`** and `playwright_cli_available()`.
- [x] **Step 3: Convert `FileNotFoundError` in `_run`**, with a message naming the binary and both reasons it is absent.
- [x] **Step 4: Re-raise it in `_search_and_process`** ahead of the broad handler.
- [x] **Step 5: Refuse to serve in `main()`.**
- [x] **Step 6: Fix the one test that was wrong** — `SystemExit` carries its message in the exception, not on stderr.

---

### Task 3: Document and verify

- [x] **Step 1: Mark the provider host-only in CLAUDE.md**, beside the command that starts it.
- [x] **Step 2: Mutation-check** — removing the re-raise restores the empty-result disguise.
- [x] `ruff check . --fix && ruff format .`
- [x] `pytest tests/unit` — 4362 passed, 1 skipped.
- [x] Spec and plan committed on the branch.
