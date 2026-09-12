# Memory Summary Merge-on-Write Implementation Plan

> **For agentic workers:** Single bounded task; executed inline with TDD.

**Goal:** `compress_session` must not overwrite a summary another process advanced while its own summarizer call was running.

**Architecture:** Re-read the state after the LLM call; skip when the cursor moved, else `replace(current, summary=, summarized_through=)`.

**Spec:** `docs/superpowers/specs/2026-09-11-memory-summary-merge-on-write-design.md`

## Global Constraints

- Only the summary write in `compress_session` changes; `_curate_after_summary` is untouched.
- Never commit to `main`; branch `fix/memory-summary-merge-on-write`.
- `ruff check . --fix && ruff format .` and mutation checks before committing.

### Task 1: Merge the summary write

**Files:** Modify `src/internal/memory/working.py` (`compress_session`, the `save_state` after `_complete`); Modify `tests/unit/memory/test_working_memory.py`.

- [ ] **Step 1: Failing tests** — `test_compress_skips_write_when_cursor_moved_during_llm_call`, `test_compress_write_preserves_curated_through_changed_during_llm_call`, `test_compress_first_summary_writes_despite_failed_reread`.
- [ ] **Step 2: Run** `pytest tests/unit/memory/test_working_memory.py -k "during_llm_call or failed_reread" -v` → the first two FAIL.
- [ ] **Step 3: Implement** — `current = load_state(...)`; `if current.summarized_through != state.summarized_through: log; return False`; `save_state(cache, session_id, replace(current, summary=text, summarized_through=last_id))`.
- [ ] **Step 4: Run** the file → PASS.
- [ ] **Step 5: Mutation checks** as in the spec; restore; `git diff` clean.
- [ ] **Step 6:** lint, `pytest -q`, commit `fix(memory): merge the summary write onto the state re-read after the LLM call`.
