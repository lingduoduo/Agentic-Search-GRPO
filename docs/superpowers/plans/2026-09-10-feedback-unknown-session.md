# Feedback unknown-session refusal — implementation plan

**Goal:** Close the fail-open in `POST /api/feedback` so no feedback row is written for a session that does not exist.

**Spec:** `docs/superpowers/specs/2026-09-10-feedback-unknown-session-design.md`

- [x] Test: unknown session id → 404 and `list_retrieval_feedback()` stays empty (red before the change).
- [x] `feedback_router.py`: refuse `session is None` alongside the foreign-session case.
- [x] Router tests create the anonymous sessions they rate; `docs/api-reference.md` documents the rule.
- [x] Run the feedback router, auth-completion, db feedback and evals summary tests; lint; commit, push, open PR.
