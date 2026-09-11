# Intent scoring einsum — implementation plan

**Goal:** Land the uncommitted einsum scoring change with its shape guard finished, so the intent latency bar stops paying for BLAS thread contention.

**Spec:** `docs/superpowers/specs/2026-09-10-intent-scoring-einsum-design.md`

- [x] Carry the uncommitted diff (`model.py`, `test_intent_model.py`, docs paragraph) onto `fix/intent-scoring-einsum` from origin/main via a patch; leave the main checkout untouched.
- [x] Add the explicit shape check in `_similarities`; `test_scoring_rejects_wrong_query_shape` goes green for all three shapes.
- [x] Run `tests/unit/test_intent_model.py` and the local latency bar (index symlinked from the main checkout's `data/intent_index`).
- [x] Lint, commit, push, open PR with spec + plan.
