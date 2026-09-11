# Intent scoring: keep the small reduction off BLAS

**Date:** 2026-09-10
**Status:** Implemented on `fix/intent-scoring-einsum`, pending PR review

## Problem

`IntentIndex._similarities` scored a query as `self._vectors @ vector`, a
(≈280 × 384) matrix-vector product. On the local OpenBLAS build that dispatch
leaves BLAS worker threads spinning after the call, and they contend with the
next CPU encoder call. The local-only latency bar
(`test_routing_one_request_stays_under_the_latency_ceiling`, 25 ms p95 for
one encode-and-decide) failed at 37.8 ms in the auth-completion session's
notes, and on the same machine today measured 40.8 / 42.1 / 55.9 ms across
three runs.

A parallel session had started the fix uncommitted on `main` (the einsum
switch, a docs paragraph, and a test for wrong-shaped query vectors) and
stopped with the test red: the explicit shape check it asserts was never
written, so a wrong-shaped vector raised einsum's subscript error instead.

## Change

- `_similarities` validates the query shape against the index dimension and
  raises a `ValueError` that names both shapes, then computes the reduction
  with `np.einsum("ij,j->i", ..., optimize=False)` so it stays on the calling
  thread. `optimize=True` may dispatch back through BLAS and reintroduce the
  contention.
- The docs paragraph records the measurement and the reason.
- No change to the encoder, thread configuration, caching, accuracy bars, or
  the 25 ms ceiling.

## Measurement (same machine, three runs each, p95 ms)

| Code | Runs |
|---|---|
| matmul (main) | 40.8, 42.1, 55.9 |
| einsum (this branch) | 25.1, 26.3, 26.9 |

The bar still sits at the ceiling on this machine; the ceiling itself is not
changed here.

## Testing

`test_scoring_rejects_wrong_query_shape` (three wrong shapes → `ValueError`
mentioning "shape"); the existing 24 intent-model tests; the local latency
bar run by hand as above.
