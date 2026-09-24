# Memory experiment review fixes — Plan

**Spec:** `docs/superpowers/specs/2026-09-23-memory-strategy-experiment-design.md`
(amended in this PR: sampling, lag casualties, what the results can support).

PR #639 squash-merged with its first commit only; the whole-branch review's
fixes were pushed after and never reached `main`. This re-lands them.

1. `ProbeResult.dropped_this_turn`: a fact that left the window on the probe
   turn itself (the previous turn loaded `before - 2` messages). Excluded from
   `summary_retention` and `recall_dropped_excl_lag`; counted as
   `lag_casualties`. `summary_retention` is `None` for non-summary strategies.
   → tests: `test_lag_casualties_are_flagged_and_the_rest_reach_the_summary`
   (echo summarizer), `test_aggregate_excludes_lag_casualties_from_summary_metrics`;
   mutation: flag forced False → red.
2. `append_rows` + `--rows` (default `<out>.rows.jsonl`): rows written per run.
   → `test_rows_are_written_as_each_run_finishes`.
3. Warn when a summary strategy saved < 90% of attempted compressions.
   → `test_summarizer_stats_count_attempts_and_advances`.
4. Isolation test uses an echo summarizer (a marker summarizer hid a leaked
   prior summary). Mutation: fixed session id + shared cache → red.
5. `--windows` must be even; config records pairs, fact_pairs, sampling.
