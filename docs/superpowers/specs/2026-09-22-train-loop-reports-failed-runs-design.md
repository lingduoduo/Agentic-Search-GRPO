# A run whose steps all fail is not a successful run

## Goal

Stop `train_loop` returning an empty history and no error when every training
step failed. Keep the tolerance that made it skip failures in the first place.

## The problem

`train_loop` catches any exception from a step, logs a warning, and continues:

```python
except Exception as exc:  # noqa: BLE001 - a bad step must not abort the run
    logger.warning("Training step %d failed or timed out (%s); skipping.", step, exc)
    continue
```

The comment is right. A rollout that times out, a batch that OOMs, a retrieval
server that blips — none of those should destroy a long run.

Unbounded, though, the skip erases the difference between a run that trained and
a run that did not. With a broken trainer, `train_loop` logs `max_steps`
warnings, returns `history == []`, and raises nothing. Every caller checks for an
exception; none inspects the length of the history. So a run that trained on
nothing is recorded as complete, and against a `resume_from` checkpoint it also
advances the step counter while doing so.

Measured against a trainer whose every step raises, `max_steps=500`: 500
attempts, 500 warnings, empty history, exit status success.

## Why not just remove the skip

Because the transient case is real and common in this stack — `step_async` drives
live search rollouts against a retrieval server, and `step_timeout_s` exists
precisely because one can hang. Aborting on the first failure would make long
runs fragile in exactly the way the `continue` was added to prevent.

The distinction that matters is **isolated failure vs. sustained failure**. One
bad step is weather; ten in a row is a broken trainer, and every further step is
wasted compute against a result nobody can use.

## The change

Two conditions raise `TrainingStepsFailedError`:

1. **A streak.** `max_consecutive_failures` (default 3) failures in a row ends
   the run. Consecutive, not cumulative — the counter resets on every success,
   so isolated failures never accumulate into a false alarm. This also fails
   fast: a broken 500-step run stops after 3 attempts rather than 500.
2. **An empty history.** If the loop finishes having completed no step at all,
   it raises regardless of the streak. Without this, a `max_steps` *below* the
   threshold would still slip through with an empty history and no error — the
   original bug in miniature.

`max_consecutive_failures = 0` disables both and restores the previous
unbounded-skip behaviour, for a caller that genuinely wants it.

The error message carries the streak length, the step it gave up on, and the
last underlying exception, chained with `from exc` so the original traceback
survives.

## What does not change

Isolated failures are still skipped and the run still completes — both existing
tests that depend on this (`test_hung_step_times_out_and_is_skipped`,
`test_failing_step_is_skipped_without_aborting`) pass unmodified, because each
has a single failure followed by successes.

## Testing

Test-first: the three "must raise" tests were watched failing with
`DID NOT RAISE`. Three further tests are tolerance controls that passed from the
start — an isolated failure is skipped, an alternating failure pattern never
trips the threshold, and `0` disables the guard.

Both halves of the guard were mutation-checked independently, because either one
alone would have left a gap:

| Mutation | Test that caught it |
|---|---|
| streak guard removed | `test_a_broken_run_stops_early_instead_of_burning_the_budget` (burned all 500 steps) |
| empty-history check removed | `test_a_short_run_that_wholly_fails_still_raises` |

## Relationship to #616

Same function, same defect class: #616 stops `train_loop` reporting a checkpoint
it did not write; this stops it reporting a run it did not perform. They were
separated because #616 is the one that can corrupt a training *result*, and
bundling would have made that harder to review.

They touch adjacent lines, so whichever merges second needs a rebase.
