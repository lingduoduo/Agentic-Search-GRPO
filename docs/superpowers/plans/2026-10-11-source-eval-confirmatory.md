# Source-category confirmatory run — plan

**Spec:** `docs/superpowers/specs/2026-10-11-source-eval-confirmatory-design.md`. Approved in chat on 2026-10-11.

- [x] RED: tests for skip-then-cap in file order, a held-out set of 48 with no pilot overlap, and rejecting a negative skip.
- [x] GREEN: `select_labels(..., skip=)`, the `--skip-per-category` flag with validation, and `sample.skip_per_category` in the report.
- [x] Mutation check: ignoring the skip turns 2 tests red.
- [x] Dry run: 48 labels plus the control, 96 live calls.
- [x] Confirmatory run: +0.094, p=0.005, n=44, "better". Report in `data/eval/source_category_relevance_confirmatory.json`.
- [x] Docs: search-engine (result and caveats), README, configuration row. The default stays off.
- [x] `pytest`: 5408 passed, 1 skipped.
