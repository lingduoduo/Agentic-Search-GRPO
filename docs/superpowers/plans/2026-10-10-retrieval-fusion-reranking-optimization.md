# Retrieval Fusion and Reranking Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Correct RRF duplicate scores, two-stage candidate budgets, and candidate configuration wiring in retrieval tuning.

**Architecture:** Extend the shared fusion core and existing service factory; keep the retrieval and reranking pipelines intact.

**Tech Stack:** Python, asyncio, dataclasses, FastAPI, pytest.

**Spec:** `docs/superpowers/specs/2026-10-10-retrieval-fusion-reranking-optimization-design.md`

## Global Constraints

- Python >=3.10; no new dependencies.
- No process-global environment mutation for candidate evaluation.
- Preserve default retrieval scoring, modes, filters, and HTTP shapes.
- Preserve retrieval-leg and reranker degradation behavior.
- Optimization result caches must not be shared across candidates.

## Review Focus

- Repeated keys within one result source must not inflate its ranking (Task 1).
- Weighted sparse/dense fusion must preserve independent source contributions (Tasks 1 and 3).
- A request exceeding prefilter size must still reach the heavy stage with sufficient candidates (Task 2).
- Both synchronous async-fallback stages must run off-loop (Task 2).
- Cached results from one tuning candidate must not determine another candidate's score (Task 3).

### Task 1: Deduplicate per-source RRF contributions

**Files:** `src/internal/retrieval/fusion.py`, `tests/unit/retrieval/test_fusion.py`.
**Interface:** existing `rrf_rank(result_sets, key_fn, weights=None, rrf_k=60)`.

- [x] Add tests comparing duplicate-containing sets against unique equivalents, including weighted and cross-source cases:

```python
assert rrf_rank([["a", "a", "b"]], lambda x: x) == rrf_rank([["a", "b"]], lambda x: x)
```

- [x] Run `pytest tests/unit/retrieval/test_fusion.py -q`; observe duplicate-score mismatch.
- [x] For each source, create a fresh seen-key set and unique rank counter. Skip repeated keys; increment rank only for unseen keys; add the existing weighted contribution. Preserve stable sorting.
- [x] Run fusion and fusion-learner suites and commit `fix(retrieval): count each document once per RRF source`.

### Task 2: Correct two-stage budgets and async fallback

**Files:** `src/internal/retrieval/two_stage_reranker.py`, `tests/unit/retrieval/test_two_stage_reranker.py`.
**Interface:** existing sync `rerank` and async `arerank`; constructor rejects nonpositive prefilter size.

- [x] Add sync/async tests where prefilter is two and top_k is four; assert four results reach the heavy stage. Add nonpositive top_k/no-call and constructor rejection tests. Add thread-identity tests proving each synchronous stage runs outside the event-loop thread, and native-async fixtures proving native async methods are awaited.
- [x] Run two-stage tests; observe budget, validation, and thread-identity failures.
- [x] Validate positive prefilter size, return early on empty/nonpositive requests, use `max(self._pre_n, top_k)` for the fast stage, and wrap sync async-fallback calls in `await asyncio.to_thread(...)`.
- [x] Run two-stage, async-reranker, and reranker-factory suites and commit `fix(retrieval): preserve rerank budgets without blocking async calls`.

### Task 3: Apply optimization candidates to real services

**Files:** `src/internal/retrieval/service.py`, `src/internal/servers/retrieval/optimize_router.py`, `tests/unit/retrieval/test_service.py`, `tests/unit/servers/retrieval/test_optimize_router.py`.
**Interfaces:** `RetrievalService.from_env(*, bm25_k1=None, bm25_b=None, fusion_weights=None, use_result_cache=True)`; constructor gains optional `fusion_weights`; local/backend builders accept named BM25 overrides.

- [x] Add service tests showing explicit weights change sparse/dense order, explicit BM25 overrides reach backend configuration without changing environment, and disabled caching bypasses Redis construction.
- [x] Add endpoint tests patching only the backend builder: sparse/dense candidates vary by BM25 parameters or weight; real `RetrievalService.search` and real grid search select a non-first improving candidate. Spy on `from_env` for `use_result_cache=False`.
- [x] Run service and optimization-router suites; observe missing keyword and unchanged-candidate failures.
- [x] Apply optional overrides through builders, retaining zero-argument default builder calls for compatibility. Store optional weights and call `weighted_rrf_fuse` only when provided. Gate cache construction on `use_result_cache`. Replace ignored positional factories with explicit BM25/fusion keyword calls; disable result cache in `_make_service`.
- [x] Run service, tuner, fusion-learner, optimization-router, and timeout-policy drift suites. Commit `fix(retrieval): evaluate configured BM25 and fusion tuning candidates`.

### Task 4: Review and remote verification

- [x] Run retrieval/server suites, shared RRF consumers, and timeout-policy drift tests. Run Ruff and diff whitespace checks.
- [ ] Request one independent whole-branch review against this spec and Review Focus; fix material findings with reproducing tests.
- [ ] Push the branch and open a focused PR against main. Watch CI for the current commit and fix any attributable failures before claiming completion.
- [ ] Record test results and PR link in the plan without merging the PR.

## Local verification

440 retrieval/server/hybrid/rerank/policy-guard tests and 12 shared search-query tests passed. Ruff lint and format checks passed on all eight changed Python files. Independent review and full remote CI are required before marking the PR ready.
