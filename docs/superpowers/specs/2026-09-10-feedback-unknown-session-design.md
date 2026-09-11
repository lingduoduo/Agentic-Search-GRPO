# `POST /api/feedback` refuses unknown sessions

**Date:** 2026-09-10
**Status:** Implemented on `fix/feedback-unknown-session`, pending PR review

## Problem

PR #573 added an ownership check to `POST /api/feedback`, but only for a
session that exists locally:

```python
session = db.get_chat_session(request.session_id)
if session is not None and not caller_may_use_session(session, caller):
    raise HTTPException(404)
db.save_retrieval_feedback(...)
```

A `session_id` that matches nothing passes straight through, so an
unauthenticated caller can write `retrieval_feedback` rows against any
made-up id. Those rows are what `load_feedback_examples` and
`load_sft_examples` read back into training. The loaders skip sessions
without chat messages today, which limits the damage, but the endpoint
should not depend on that.

## Change

`session is None` is refused with the same `404` as a foreign session, and
nothing is written. Known sessions behave as before: an anonymous session's
id remains its capability, a user-owned session requires the owner.

The frontend thumbs bar always posts for a session it created through
`/api/sessions`, so it is unaffected. The router's own tests created no
sessions; they now do.

## Testing

Unknown id → `404`, no row; every existing router test against a created
anonymous session; the ownership tests in
`tests/unit/servers/web/test_auth_completion.py` unchanged.
