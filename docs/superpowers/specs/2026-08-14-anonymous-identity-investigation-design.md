# Establish what anonymous callers actually share, then fix it

## Status

Investigation and owned-session fixes completed; see the corresponding plan.
The 2026-09-10 authentication completion extends ownership checks to chat/tool
continuation and feedback. Per-caller anonymous identity remains a separate
design decision; the current session-ID capability and optional shared-memory
contracts are documented in [Authentication](../../authentication.md).

## Problem

PR #500 reverted a shared-owner anonymous session model. The record of that
revert says the real fix — a per-caller anonymous identity rather than a single
pooled one — was never built.

That claim is now roughly a week old and has not been re-checked against the
code. A grep for the obvious markers finds a `is_anonymous` flag on the user
model and an `ANONYMOUS` enum value, but neither establishes what a signed-out
caller's session is actually keyed by today, which is the only thing that
matters.

Two related facts are known and current, and both point at the same area:

- `SearchExperienceSettings.memory_require_auth` defaults to **False**, with the
  comment "Refuse anonymous `/api/memory` callers instead of pooling them into
  the shared `default_user` bucket." So there *is* a shared bucket, and the
  refusal is opt-in.
- The `/api/memory` router is mounted without authentication.

## Why this is an investigation and not a fix

The failure mode being guarded against — one signed-out caller reading another's
conversation — is serious enough that acting on a stale summary is the wrong
move in both directions. If it is already fixed, a "fix" adds risk for nothing.
If it is not fixed, the shape of the correct fix depends entirely on how
ownership is currently resolved, and there are known to be several different
resolvers in this codebase serving different routes.

Writing acceptance criteria before knowing which resolver serves which path
would produce criteria that pass while the hole stays open — which is precisely
how #500 got reverted in the first place.

## Phase 1: establish the facts

Answer, with code references and a reproduction rather than inference:

1. What is a signed-out caller's session owned by, per route? Enumerate the
   resolvers and which routes use each.
2. Can two signed-out callers, in one process, see each other's transcripts?
   Demonstrate it with a test that fails if they can, rather than reasoning about
   it.
3. Does `/api/memory` pool anonymous callers into `default_user` by default, and
   what does that expose?
4. Is the "first user becomes admin" behaviour that #500's revert disturbed
   currently correct?

**Deliverable: a findings document, and a failing test for anything real.** If
nothing is reproducible, that is the result and this closes.

## Phase 2: fix, scoped by what phase 1 found

Only written once phase 1 lands. The likely shape — a per-caller
`anon_<uuid4>` identity — is a hypothesis, not a plan, and must not be treated
as one before the facts are in.

## Acceptance (phase 1)

- Every resolver enumerated with the routes it serves.
- A test that reproduces cross-caller visibility, or a documented demonstration
  that it cannot happen.
- The `default_user` pooling behaviour confirmed or refuted at the current
  default settings.
- An explicit statement of which parts of the #500 record still hold, since that
  is the input everything else has been assuming.

## Out of scope

Any fix, until phase 1 says what is broken. Also out of scope: changing
`memory_require_auth`'s default, which is a product decision about whether the
CLI keeps working unauthenticated, not a security finding.
