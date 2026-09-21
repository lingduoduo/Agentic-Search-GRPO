# Benchmark the two thread pools on the retrieval path

## Goal

Answer one question with evidence: when retrieval concurrency degrades, is
retrieval itself expensive, or are requests queueing behind a concurrency
limit? And if the latter, which pool.

## Why this came up

A request to swap the async HTTP transport for aiohttp turned out to have no
target: `OpenAIEmbedder` is the only OpenAI network call on the retrieval path,
and it is unreachable from any shipped configuration (no env wiring sets
`DenseRetrieverConfig.openai_embedding_model`; the production constructor is
`for_e5_base_v2`, which selects local E5 over FAISS). The dense leg makes no
HTTP call at all.

That left the real question: the retrieval path *does* have parallelism, in two
thread pools. Neither had ever been measured.

## The two pools are not alike

**`RetrievalService._search_one`** (`src/internal/retrieval/service.py:153`)
builds `ThreadPoolExecutor(max_workers=2)` **inside the call** and submits
exactly two tasks, one per leg. The `with` block exits before either
`.result()` is read, so `shutdown(wait=True)` is what blocks.

Consequences: it is not shared, so there is no cross-request queue and no
global concurrency limit here. With workers sized to the task count, nothing
ever waits for a slot. Its cost is two fresh OS threads per request, unbounded.

**`AsyncReranker`** (`src/internal/retrieval/async_reranker.py:26`) holds one
**persistent** pool, `RERANKER_MAX_WORKERS` (4 by default), shared by every
request, gated on `RERANKER_ASYNC` and composed in `reranker_factory.py:28`.
This one queues. Two details make the queue expensive:

- `future.result(timeout=self._timeout_ms / 1000)` counts from **submit**, so
  queue wait spends the same budget as scoring does.
- `future.cancel()` on timeout cannot stop a task that already started, so a
  timed-out request keeps occupying its worker.

## Method

Drive the real production methods. Stub only the work performed *inside* a pool
task, so task cost is a controlled variable and everything measured -- queueing,
thread spawn, scheduling, fan-in -- is the shipped orchestration.

Correlation rides in the query string, because a pool task runs on a different
thread than its caller and `ThreadPoolExecutor` does not propagate contextvars.
The stub records when each task actually began; the caller records submit time;
the difference is wait.

Two task modes bound real behaviour, which sits between them:

- `sleep` releases the GIL -- models FAISS, pyserini-JVM, network
- `cpu` holds the GIL -- models Python-level scoring

That bound matters, because if the legs held the GIL the 2-worker pool would
buy nothing. Measured separately: `faiss.IndexFlatIP.search` over 200k x 128
vectors gives **2.01x** on two threads with OpenMP pinned to one thread, and
**1.55x** at the production OpenMP default (8 threads, 8 cores) -- real, but
diminished, because FAISS already saturates the cores. The sparse leg was
**not** verified; pyserini needs a Lucene index and none exists locally.

## Findings

Both at an 80ms task, `sleep` mode, one 8-core machine.

**search** -- p50 flat at ~85ms and throughput linear from 1 to 64 concurrent
(11.9 -> 737 rps). Wait stays under 1ms, under 1.2% of the request. Zero
timeouts. Peak 194 threads at 64 concurrent (~3N). At 128 concurrent (measured
at a 10ms task) the knee appears: throughput growth collapses to 1.4x for 2x
concurrency, wait p50 jumps 8x to 4ms, 386 threads. The degradation there is
thread spawn and scheduler pressure, not queueing.

**rerank** -- saturates at 4 concurrent (~48 rps, matching 4 workers / 80ms).
Beyond that, latency grows with concurrency until the deadline caps it:

| conc | rps | p50 | wait p50 | timeouts |
|---|---|---|---|---|
| 4 | 47.6 | 83 | 0.07 | 0% |
| 8 | 48.1 | 166 | 83 | 0% |
| 16 | 48.0 | 332 | 249 | 0% |
| 32 | 63.6 | 503 | 478 | **92.5%** |
| 64 | 126.9 | 503 | 473 | **96.2%** |

At 32 concurrent, 478ms of a 500ms budget is spent queueing and only 80ms
scoring. The throughput *rise* at 32 and 64 is not work completed -- it is
timeouts returning early.

**Conclusion.** The per-call pool is not the limit and is not worth optimizing.
The shared reranker pool is the limit, and with the shipped defaults it does not
degrade gracefully: it collapses between 16 and 32 concurrent requests.

## What this does not establish

Synthetic task latencies on one 8-core macOS machine. The sparse leg's GIL
behaviour is unverified. Percentiles come from small samples. The structural
findings -- per-call pool, no queue, two threads per request, deadline measured
from submit -- are independent of the numbers.

Nothing here changes production behaviour. Whether to raise
`RERANKER_MAX_WORKERS`, start the deadline at task start rather than submit, or
add admission control is a separate decision this evidence informs but does not
make.

## Tests

`tests/unit/test_benchmark_retrieval_executors.py` pins the discrimination the
harness exists to make -- the per-call target never times out, the shared target
reports queue-driven timeouts and reports none when given headroom -- plus the
percentile helper and the stubs' correlation. It asserts no duration; following
`test_benchmark_grpo_optimization.py`, a wall-clock assertion would be flaky on
a shared machine.

Mutation-checked: replacing `timeout_pct` with a constant `0.0` turns the
queue-driven-timeout test red.
