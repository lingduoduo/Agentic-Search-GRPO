# Reranker Admission Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the shared reranker pool collapsing past its capacity — shed excess cheaply instead of making every request pay the full deadline for nothing.

**Architecture:** Estimate, before submitting, whether work could finish inside the budget; refuse if not. The estimate comes from an EWMA of observed scoring cost, so it is self-tuning rather than a guessed constant.

**Tech Stack:** Python 3.10+, `concurrent.futures`, `threading`, pytest, `examples/benchmark_retrieval_executors.py`.

**Spec:** `docs/superpowers/specs/2026-09-21-reranker-admission-control-design.md`

## Global Constraints

- **Refusal must remain a `RerankerTimeoutError`.** `RetrievalService` degrades on that type; a new unrelated exception would turn a graceful degradation into a 500.
- **Never refuse before anything has been measured.** With no observation, refusing is a guess — and the first admissions are what produce the measurement.
- **Release the slot in a `finally`.** A scorer that raises must not leak capacity, or the pool eventually refuses everything.
- **Do not raise `RERANKER_MAX_WORKERS` here.** This makes the capacity limit visible; changing it is a separate decision.
- Tests must not assert wall-clock durations — inject `_observed_seconds` and `_outstanding` to test the arithmetic deterministically.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: Admission control in `AsyncReranker`

**Files:**
- Modify: `src/internal/retrieval/async_reranker.py`
- Test: `tests/unit/retrieval/test_async_reranker.py`

- [x] **Step 1: Measure what scoring costs**

Wrap each run so it records its duration into an EWMA (α = 0.2) and releases its slot in a `finally`.

- [x] **Step 2: Refuse what cannot finish**

`projected = (outstanding // max_workers + 1) * observed_seconds`; refuse when it exceeds the budget. Integer division counts whole batches ahead, each occupying every worker.

- [x] **Step 3: `RerankerOverloaded`, plus counters**

Subclass `RerankerTimeoutError` so existing callers are untouched, and record `admitted`/`refused`/`expired`.

**Verify:** refused requests never reach the scorer; a failed run leaks no slot; the first request is always admitted. Mutation — disable the refusal and the refusal test reddens.

---

### Task 2: Make the benchmark report goodput

**Files:**
- Modify: `examples/benchmark_retrieval_executors.py`
- Test: `tests/unit/test_benchmark_retrieval_executors.py`

- [x] **Step 1: Separate refusals from expiries**

Catch `RerankerOverloaded` before `RerankerTimeoutError` — they are both failures, but only one wasted a worker for the full budget.

- [x] **Step 2: Report `goodput_rps`**

Total throughput counts instant refusals, so it reads as *better* the worse overload gets — 1257 rps at 64 concurrent, almost none of it work. Report completed work separately, or the measurement flatters the failure.

**Verify:** the key-contract test covers the new fields; goodput holds flat at capacity across the sweep.

---

### Task 3: Measure the change

- [x] **Step 1: Same sweep, before and after**

80ms scorer, 4 workers, 500ms budget, concurrency 4→64. Goodput at 32 concurrent moves from about 4.8 rps to 47.1, and p50 at 64 from 503ms to 0.00ms.

**Verify:** `pytest -q` green; the sweep reproduces.
