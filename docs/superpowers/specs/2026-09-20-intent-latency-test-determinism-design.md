# Intent routing latency: assert on work, not wall time

Closes #593.

## Goal

Stop a contention-sensitive wall-clock assertion from failing in the default
test run, without losing either the measurement or the regression it guards.

## The problem

`tests/unit/test_intent_evaluation.py::test_routing_one_request_stays_under_the_latency_ceiling`
times 50 `sentence_transformers` forward passes and asserts a p95 SLA:

```python
p95 = sorted(timings)[int(0.95 * (len(timings) - 1))]
assert p95 <= _P95_LATENCY_CEILING_MS, p95   # 25.0
```

Observed on one commit during PR #590: it FAILED in a 514-test subset, passed
alone in 8.14 s, and passed in the full 4272-test run. Same code, three results.

Raising the ceiling is the wrong repair, and the file says why itself. The
provenance comment at `test_intent_evaluation.py:638` records a re-measurement
on 2026-08-14: **p95 12.20 ms measured, ceiling set to 25.0 ms**. The assertion
already carries 2x headroom over the real number and still fails. That is the
diagnosis: the variance is not the encoder getting slower, it is contention for
the machine with whatever else pytest is running. Doubling the headroom bought
nothing, so quadrupling it buys nothing either — it would only relocate the
failure to a busier machine while making the assertion vacuous on a quiet one.
The module under test knew this before the test did: `_similarities` carries a
comment about BLAS workers contending with the next CPU encoder call.

This is the second instance of the shape in this repo. The RSS watchdog test
fails under full-suite load for the same structural reason — a tight wall-clock
budget standing in for resource behavior, losing the machine to a neighbour.

## Architecture

Two moves: the measurement relocates, and the regression it guarded is pinned
by counting instead of timing.

**The measurement moves to where this repo already keeps measurements.**
`tests/load/` exists, is marked `load`, and sits outside `testpaths`
(`["tests/unit", "tests/regression"]`), so it runs neither in the default suite
nor in CI (`pytest tests/unit/ -q`). Its own docstring sets the convention:
*"Run with: `pytest tests/load/ -v -s -m load`. Thresholds are intentionally
generous for CI."* The timing loop moves there intact, with its provenance
comment and `_P95_LATENCY_CEILING_MS`. Nothing about the number changes; only
the context it is evaluated in, which is now a context where a wall-clock
number means something.

**The regression is pinned by four deterministic assertions.** What the SLA
actually guarded is visible in `_model`'s own docstring: *"Loading costs
seconds; encoding costs ms."* The failure that a latency ceiling would catch is
a cache break turning every route decision into a multi-second model load. That
is a counting question, not a timing one, and counting does not care what else
is running.

| Assertion | Mechanism | Regression caught |
|---|---|---|
| The encoder is constructed once | fresh `_MODEL_CACHE`, counting fake `SentenceTransformer`, 10 decisions | a per-call model load |
| One encode per decision | same fake; one call, one text each | redundant or unbatched encoding |
| Similarity work is fixed per decision | count `IntentIndex._similarities` over 10 real decisions | scoring that scales with route or module count |
| The index is not re-read | load, then make `np.load` raise | a `decide` that re-reads `index.npz` per call |

The third number is measured, not assumed: `decide` computes `_similarities`
exactly twice per call — once via `route_scores`, once via `_emit_modules` ->
`module_scores` — so 10 decisions cost 20. The test pins that, which documents
a real redundancy without fixing it.

The fourth is the weakest of the four and is included knowingly. What keeps it
from being vacuous is the ordering: `np.load` is broken *after* the index is
loaded, so a passing decision proves `decide` touched no disk. Asserting that
`IntentIndex.load` "was called once" would instead assert the test's own setup.

The first two use a fake encoder, so they need no model download and run in
milliseconds. They test the caching contract rather than the model — which is
the point, and is why they cannot flake. `monkeypatch.setattr` swaps
`_MODEL_CACHE` for a fresh dict and restores it, so the guard cannot leak into
a neighbouring test.

## Testing

The tests are the deliverable, so verification is mutation: each new assertion
is checked by breaking the invariant in the source it guards and confirming the
test goes red. Bypassing the `_MODEL_CACHE` lookup must fail the construction
count; encoding twice per decision must fail the encode count; computing
similarities a third time must fail the fixed-work count; re-reading the index
inside `decide` must fail the no-disk test. A test that survives its mutation
is not testing anything and gets rewritten.

The full default suite must stay green, at the same count: one test is removed
from `tests/unit/` and four are added, and the relocated timing test leaves the
default run entirely.

## Limits

This changes no production code. Routing behavior, the encoder, the index
format, and the ceiling value are all untouched.

Two things this does not do. It does not fix the double `_similarities`
computation in `decide` — that is an optimization, and this PR pins the current
value rather than changing it. And it does not repair the RSS watchdog test,
the other instance of the pattern; PR #504 is the model for that one and it
deserves its own change.

Out of scope: `addopts` and the global pytest configuration, the accuracy and
AUC bars in the same file, and issues #591 and #592.
