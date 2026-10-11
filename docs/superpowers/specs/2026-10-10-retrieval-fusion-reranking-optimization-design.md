# Retrieval fusion, reranking, and optimization

## Intent

Improve the existing dense/sparse/hybrid retrieval stack through correct fusion, bounded two-stage reranking, and optimization endpoints that actually evaluate their candidate configurations. The user approved this scope and the established spec, plan, implementation, and PR workflow.

## Existing flow and approach

`RetrievalService` runs sparse and dense legs, fuses their ranked results, optionally fuses query variants, applies MMR, and reranks. Optimization endpoints wrap BM25 and fusion grid searches, but their service factory discards candidate arguments. Two-stage reranking always prefilters to its configured limit even when the caller requests more, and its async fallback calls blocking scorers directly.

Extend these existing boundaries. A replacement backend or dispatcher would duplicate working retrieval; prompt-only tuning cannot fix the configuration plumbing or duplicate-score bug.

## Contracts

### Optimization plumbing

Add optional keyword overrides `bm25_k1`, `bm25_b`, `fusion_weights`, and `use_result_cache=True` to `RetrievalService.from_env`. The local backend receives explicit BM25 overrides without mutating process environment. Default construction remains unchanged. A service stores optional `FusionWeights` and uses weighted RRF for sparse/dense fusion when supplied; default fusion keeps its current unweighted scores. Preserve dense-only/sparse-only failure behavior, filter propagation, and query-variant fusion.

The optimization router supplies candidate parameters by name, including `FusionWeights(w_sparse, w_dense)` for fusion tuning. Evaluation services disable result caching so earlier candidates cannot contaminate later scores. Existing request/response models and admin authorization remain unchanged. Add an endpoint regression that demonstrably selects a non-first candidate whose real service ranking improves; do not mock away candidate-to-service wiring. Actual corpus/model benchmarks are outside this change.

### RRF duplicate handling

Within each source list, count each document key once at its first occurrence. Rank the unique entries consecutively; duplicates neither add contributions nor displace the ranks of subsequent unique documents. Across independent sources, the same key still accumulates contributions normally. Preserve first-seen payloads, deterministic ties, default weights, and mismatch fallback. This is the shared RRF core used by all retrieval consumers.

### Two-stage reranking

Require a positive `pre_filter_top_n`. Return an empty result without scorer calls for nonpositive `top_k` or empty inputs. For positive requests, the fast stage requests `max(pre_filter_top_n, top_k)` candidates, so its configured prefilter cannot underfill the heavy stage by itself. Preserve fast/heavy ordering and exception propagation.

`arerank` awaits native asynchronous scorers. When a stage exposes only synchronous `rerank`, offload it with `asyncio.to_thread`; neither fallback may block the event loop. Keep both stages sequential and preserve their results and errors.

## Constraints

- Python >=3.10; no new dependencies.
- No process-global environment mutation for candidate evaluation.
- Preserve default retrieval scoring, modes, filters, and HTTP shapes.
- Preserve retrieval-leg and reranker degradation behavior.
- Optimization result caches must not be shared across candidates.

## Verification and delivery

Use deterministic backends and scorer fixtures. Cover per-source duplicates, cross-source contributions, weighted fusion, candidate overrides, cache bypass, actual endpoint candidate selection, zero/negative budgets, large top-k requests, native async scorers, and worker-thread fallback. Run retrieval unit and server suites plus timeout-policy drift guards. Request independent branch review, open one separate PR, and verify its remote checks before claiming completion. No live quality or latency gains are claimed.
