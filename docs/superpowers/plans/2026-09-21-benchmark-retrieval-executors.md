# Retrieval Executor Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure both thread pools on the retrieval path so the question "expensive retrieval or queueing behind a concurrency limit?" can be answered with evidence rather than inspection.

**Architecture:** One `examples/` script with two targets. It drives the real production methods and stubs only the work inside a pool task, so task cost is a controlled variable and the orchestration under measurement is the shipped code.

**Tech Stack:** Python 3, `threading`, `concurrent.futures`, argparse, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-benchmark-retrieval-executors-design.md`

## Global Constraints

- **Diagnostic only.** This lands a measurement tool. It must not change `service.py`, `async_reranker.py`, or any default. Acting on the findings is a separate decision.
- **Never assert a duration in a test.** Follow `tests/unit/test_benchmark_grpo_optimization.py`: pin measurement discipline and reported shape, never wall-clock, or the suite goes flaky on a shared machine.
- **`-m` invocation only.** `src` is not importable from a bare script path and peer `examples/` scripts (`beir_to_corpus.py`, `run_domain_relevance_eval.py`) behave the same. Match the peers; do not add a `sys.path` shim.
- Correlate a task to its caller through the query string. A pool task runs on another thread and `ThreadPoolExecutor` does not propagate contextvars.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: Build the harness

**Files:**
- Create: `examples/benchmark_retrieval_executors.py`

**Interfaces:**
- Produces: `StubBackend`, `StubReranker`, `_spend`, `_pct`, `_drive`, `run_level`, `main`.

- [x] **Step 1: Controlled task cost with both GIL behaviours**

`_spend(seconds, mode)` either `time.sleep`s (releases the GIL — models FAISS, pyserini-JVM, network) or burns a loop (holds it — models Python-level scoring). The two modes bound real behaviour.

- [x] **Step 2: Stubs that record when a task actually began**

`_Timed.mark` stores the first start per call id under a lock. `StubBackend` keys legs as `<call_id>:sparse` / `<call_id>:dense`; `StubReranker` keys by query. Wait is then `start - submit`.

- [x] **Step 3: The driver**

`_drive` runs `call(call_id)` from N threads, `per_thread` times each, samples `threading.active_count()` for a peak, counts `RerankerTimeoutError`, and returns latencies, submit times, wall time, peak threads, timeouts.

- [x] **Step 4: Two targets and the report**

`run_level` wires `RetrievalService(stub)._search_one` for `search` and `AsyncReranker(stub, ...)` for `rerank`, then reports p50/p95/p99, wait p50/p99, throughput, timeout %, peak threads.

**Verify:** `python -m examples.benchmark_retrieval_executors --target search --levels 1,4 --per-thread 5` prints a table; `--help` works under `-m`.

---

### Task 2: Pin the behaviour with tests

**Files:**
- Create: `tests/unit/test_benchmark_retrieval_executors.py`

- [x] **Step 1: Helper and stub contracts**

`_pct` returns order statistics on a known list and NaN on empty. `StubBackend` records both legs for one call id; `StubReranker` truncates to `top_k` and records its start.

- [x] **Step 2: The discrimination the harness exists to make**

The `search` target reports every metric key and `timeout_pct == 0.0` — the per-call pool is sized to its task count, so nothing can queue. The `rerank` target with a task far longer than the deadline reports `timeout_pct == 100.0`, and with generous headroom reports `0.0`. Use wide margins so neither is timing-sensitive.

**Verify:** `pytest tests/unit/test_benchmark_retrieval_executors.py -q` → 7 passed.

---

### Task 3: Measure, and verify the GIL premise

- [x] **Step 1: Sweep both targets**

At an 80ms task: `search` scales linearly to 64 concurrent with wait under 1ms and no timeouts; `rerank` saturates at 4 and reaches 92.5% timeouts at 32, 96.2% at 64.

- [x] **Step 2: Verify the premise the `search` result depends on**

The 2-worker pool only helps if the legs release the GIL. Probe `faiss.IndexFlatIP.search` directly: 2.01x on two threads with OpenMP pinned to 1, 1.55x at the production default. Record that the sparse leg is unverified — pyserini needs a Lucene index that does not exist locally.

- [x] **Step 3: Mutation-check and full suite**

Replace `timeout_pct` with a constant `0.0`; the queue-driven-timeout test must go red. Commit first so the restore is a checkout, and clear `examples/` bytecode after restoring.

**Verify:** 1 failed under mutation; `pytest -q` → 4313 passed, 1 skipped.
