# Category search parallel legs — plan

**Spec:** `docs/superpowers/specs/2026-10-11-category-search-parallel-legs-design.md`. Approved in chat on 2026-10-11.

- [x] RED: timed-provider tests for:
  - legs starting together;
  - an early restricted win cancelling a slow suffix leg;
  - the grace cut-off;
  - a restricted leg finishing inside the grace still counting;
  - the time budget bounding both legs;
  - outer cancellation cancelling both legs.
- [x] GREEN: `search_category` rewritten around two tasks, plus `RESTRICT_GRACE_SECONDS` and `CategorySearch.fallback_used`. The eval reads `fallback_used`.
- [x] Updated the two tests that assumed sequential calls (suffix never sent).
- [x] Mutation checks, each caught: no grace (1 red), no early win (1), no cancel in `finally` (6), sequential legs (4), eval ignoring `fallback_used` (2).
- [x] Cache re-score of the confirmatory run with the key unset (0 calls) reproduces +0.094, p=0.005, and the same fallback rate and exclusions.
- [x] Docs: search-engine, configuration, README. The default stays off.
- [x] `pytest`: 5415 passed, 1 skipped.
