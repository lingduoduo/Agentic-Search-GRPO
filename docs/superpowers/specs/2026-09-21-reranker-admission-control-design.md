# Refuse rerank work that cannot finish in time

## Goal

Stop the reranker pool collapsing under load. Past its capacity it should shed
work cheaply and keep serving at capacity, rather than making every request pay
the full deadline for a result it will not get.

## The problem, measured

`AsyncReranker` holds one persistent `ThreadPoolExecutor`, `RERANKER_MAX_WORKERS`
(4 by default), shared by every request, with a `RERANKER_TIMEOUT_MS` deadline
of 500ms. Benchmarked at an 80ms scorer with those defaults
(`examples/benchmark_retrieval_executors.py --target rerank`):

| conc | rps | p50 | wait p50 | failed |
|---|---|---|---|---|
| 4 | 47.6 | 83 | 0.07 | 0% |
| 8 | 48.1 | 166 | 83 | 0% |
| 16 | 48.0 | 332 | 249 | 0% |
| 32 | 63.6 | 503 | 478 | **92.5%** |
| 64 | 126.9 | 503 | 473 | **96.2%** |

At 32 concurrent, **478ms of the 500ms budget is queueing and 80ms is scoring**.

Two details turn saturation into collapse:

1. `future.result(timeout=...)` counts from **submit**, so queue wait spends the
   same budget the work needs. A request that queues behind four batches has
   already lost before it starts.
2. `future.cancel()` cannot stop a task that has begun. A timed-out request has
   often *already occupied a worker* — the pool keeps scoring documents nobody
   is waiting for, which is what pushes the next request over the edge too.

The result is a pool doing full work for almost no delivered results: measured
goodput at 32 concurrent is about **4.8 rps** against a capacity near 48.

## Why refusing is not a downgrade

`RetrievalService` already degrades on a timeout — it keeps the fused ordering
and drops the `+reranked` label:

```python
except RerankerTimeoutError as exc:
    logger.warning("Reranker timed out, keeping fused order: %s", exc)
```

So for a request that was going to time out, refusing reaches **the identical
outcome**, 500ms sooner, without consuming a worker. There is nothing to trade
off: the slow path produces no better answer, only a later one.

## Architecture

Before submitting, estimate whether the work could finish inside the budget,
and refuse if not.

```
projected = (outstanding // max_workers + 1) * observed_seconds
refuse if projected > timeout
```

`observed_seconds` is an EWMA (α = 0.2) of how long scoring actually takes,
recorded by the wrapper around each run. `outstanding` counts admitted work not
yet finished. Integer division is deliberate: it counts whole batches ahead,
each occupying every worker.

**Nothing is refused before anything is measured.** With no observation yet,
refusing would be a guess, so the first requests are always admitted — which is
also what produces the first measurement.

**`RerankerOverloaded` subclasses `RerankerTimeoutError`.** Every existing
caller keeps degrading exactly as before without being touched, while new code
and metrics can tell a refusal from an expiry. `counters` records
`admitted`/`refused`/`expired` for the same reason.

The slot is released in a `finally`, so a scorer that raises does not leak
capacity — a leak there would eventually refuse everything.

## Result

Same benchmark, after:

| conc | goodput/s | refused | expired | p50 |
|---|---|---|---|---|
| 4 | 48.1 | 0% | 0% | 83 |
| 8 | 47.9 | 0% | 0% | 165 |
| 16 | 47.9 | 0% | 0% | 332 |
| 32 | **47.1** | 45.0% | 2.5% | 326 |
| 64 | **47.1** | 90.0% | 6.2% | **0.00** |

Goodput holds flat at the pool's real capacity instead of collapsing to ~4.8
rps, and past capacity the excess is shed in microseconds rather than half a
second each.

The benchmark now reports `goodput_rps` separately, because total throughput
counts instant refusals and reads as *better* the worse things get — 1257 rps
at 64 concurrent, almost none of it work. A measurement that flatters overload
is worse than none.

## What this does not do

It does not add capacity. At 64 concurrent with an 80ms scorer, 4 workers can
serve about 48 rps and the rest is shed — correctly, but shed. Raising
`RERANKER_MAX_WORKERS`, or making scoring cheaper, is the separate question this
makes visible rather than answers.

It also leaves the submit-relative deadline in place. Starting the clock at task
start instead would remove the arbitrary penalty on queued work, but on its own
it converts timeouts into unbounded waits; admission control is what makes the
deadline meaningful, and is the prerequisite for revisiting it.
