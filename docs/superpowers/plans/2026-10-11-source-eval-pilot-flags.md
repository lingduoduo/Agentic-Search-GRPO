# Source-category eval pilot flags — plan

**Spec:** `docs/superpowers/specs/2026-10-11-source-eval-pilot-flags-design.md`

Bounded change; design approved in chat, 2026-10-11. Implemented inline with TDD.

- [x] Failing tests (`tests/unit/test_source_category_eval.py`):
  - first-N-per-category selection in file order, and 16 queries over the frozen file;
  - `--no-plain` never calls the plain query, and a skipped arm does not exclude a row;
  - call count is 33 without plain and 49 with it;
  - the summary reports the plain comparison as not run;
  - `main --dry-run` prints 33.
- [x] `domain_eval.run_source_query(include_plain=...)`, with `plain` and `delta_vs_plain` optional.
- [x] CLI: `select_labels`, `--per-category`, `--no-plain`, `_pending_calls(include_plain)`, and the report's `sample` field.
- [x] Mutation checks each turned a test red: per-category ignored (4 red), plain always run (1 red), pending always counts plain (2 red).
- [x] Docs: pilot command and caveat in `docs/search-engine.md`.
- [x] Checks: `pytest` (5393 passed, 1 skipped), Ruff, `git diff --check`; `--dry-run` with the pilot flags prints 33.
