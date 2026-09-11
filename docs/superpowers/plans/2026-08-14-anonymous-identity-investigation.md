# Plan: anonymous identity — investigation, and the hole it found

Spec: `docs/superpowers/specs/2026-08-14-anonymous-identity-investigation-design.md`

The spec scoped this as investigation-first, on the grounds that acting on a
week-old unverified summary of #500 would be worse than not acting. That was
right, and phase 1 found something the summary did not describe.

## Phase 1 findings

**The question was: can two signed-out callers see each other's transcripts?**
**The answer is broader and worse: anyone could read any session by id, including
a signed-in user's.**

Demonstrated against a real app instance rather than inferred from code:

```
anonymous GET on an OWNED session -> 200
owner user_id exposed: user_alice
messages returned: 2
    user -> my salary is 120k, SSN ends 4471
```

Two vectors:

1. **`GET /api/sessions/{session_id}`** — no authentication, no ownership check.
   Returns the owner id and every message body.
2. **`POST /api/agent` with another caller's `session_id`** — `_ensure_session`
   accepted any *existing* id, then the handler loaded that session's history
   into the model context and appended the caller's message. A read *and* a
   write.

**Scope is contained.** Of every path-parameter endpoint in `app.py`, exactly one
was open. Session ids are `session_<uuid4hex>`, so this was never
brute-forceable — it is an IDOR, where an id leaked through a URL, log line,
referrer or shared link became someone else's transcript.

**Why it survived review.** `check_router_auth` looks like a guard and is purely
advisory: it logs routes as public or "guarded", where *guarded* means only
"expected to have per-handler auth". It never verifies the handler has any. A
reader scanning startup logs sees `/api/sessions/{session_id}` absent from the
public list and concludes it is protected. A test now pins that contract so the
gap reads as missing enforcement rather than a bypassed check.

**On the original #500 question:** anonymous sessions are stored with
`user_id = NULL`, not a shared owner, so the revert did work. The listing query
filters `WHERE user_id = ?` and therefore cannot return them at all.

## Phase 2, scoped narrowly

`_caller_may_use_session` — owned sessions require the matching caller;
anonymous sessions stay readable by anyone holding the id.

**That second half is a deliberate narrowing, not a gap left open.** Signed-out
use has no identity to compare against, so the id is the only capability there
is, and the CLI and local-research flows depend on it. Giving anonymous callers
a per-caller identity is the work #500's revert left unbuilt; until that exists,
two signed-out callers sharing an id share a session. Closing the
authenticated-user exposure did not require solving that, and coupling them
would have delayed a fix for a live hole behind a larger design.

**404, not 403**, on refusal: a 403 confirms the id exists, which is the one bit
an id-guessing caller does not already have.

→ verify: owned + no creds `404`; anonymous + no creds `200` (unchanged);
missing `404`; agent reuse of owned `404`; agent reuse of anonymous `200`
(unchanged).

## On the tests

They began as `xfail(strict=True)` evidence and became assertions when the guard
landed. Strict xfail is what made that safe: the moment the endpoints were fixed
the tests XPASSed, which strict mode reports as a failure — so the markers could
not be left behind quietly protecting nothing.

## Still open

2026-09-10 update: startup route enforcement is implemented, and the remaining
chat/tool continuation and feedback ownership gaps are closed. The old test
pinning the advisory audit was replaced by rejection and acceptance tests.

Per-caller anonymous identity (`anon_<uuid4>` or equivalent), which is the
remaining half of #500's revert. Two signed-out callers who share a session id
still share the session. That needs its own spec.

Full suite: **3192 passed**, 1 deselected (the hardware-sensitive latency bar).
