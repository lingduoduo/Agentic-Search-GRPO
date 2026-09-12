# Memory Cursor Merge Guard Implementation Plan

> **For agentic workers:** Single bounded task; executed inline with TDD.

**Goal:** `_curate_after_summary` must not wipe a just-saved summary when its state re-read comes back blank.

**Architecture:** One guard in `src/internal/memory/working.py` after the re-read; one test.

**Spec:** `docs/superpowers/specs/2026-09-11-memory-cursor-merge-guard-design.md`

## Global Constraints

- Only `_curate_after_summary` changes; the healthy merge path is unchanged.
- Never commit to `main`; branch `fix/memory-cursor-merge-guard`.
- `ruff check . --fix && ruff format .` before committing; mutation check before committing.

### Task 1: Guard the merge

**Files:** Modify `src/internal/memory/working.py` (`_curate_after_summary`, ~line 179); Modify `tests/unit/memory/test_working_memory.py`.

- [ ] **Step 1: Failing test** — append `test_curate_cursor_write_skips_merge_when_reread_is_blank` (cache subclass whose `get` raises once `fail_reads` is set; curate fake flips it; assert `summary == "S"`, `summarized_through == records[1].id`, `curated_through is None`).
- [ ] **Step 2: Run** `pytest tests/unit/memory/test_working_memory.py -k blank -v` → FAIL (`summary == ""`).
- [ ] **Step 3: Implement** — after `current = load_state(...)`: `if current.summarized_through is None and not current.summary: logger.warning(...); return`.
- [ ] **Step 4: Run** the file → PASS.
- [ ] **Step 5: Mutation check** — remove the guard, test red, restore, `git diff` clean.
- [ ] **Step 6:** lint, `pytest -q`, commit `fix(memory): skip the curated-cursor merge when the state re-read comes back blank`.
