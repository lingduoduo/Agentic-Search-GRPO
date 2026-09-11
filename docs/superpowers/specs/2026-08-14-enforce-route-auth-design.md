# Make the route-auth audit enforce instead of describe

## Status

Implemented on 2026-09-10. Startup now rejects unclassified path/method pairs
before seeding or background work. The route triage also closed session
continuation, feedback ownership, forged attribution and debug access gaps.
See [Authentication](../../authentication.md) for the resulting contract and
the source heuristic's limits. The historical problem and proposed phases below
record why this work was needed.

## Problem

`check_router_auth` looks like a guard and is not one. It walks every registered
route and logs each as public or `"guarded"` — where *guarded* means only
"expected to have per-handler auth". **It never checks whether the handler has
any.** Its only failure mode is a warning for allowlist entries that match no
route, i.e. it catches stale allowlist rows and nothing about actual auth.

Two unauthenticated endpoints shipped behind it in one day:

- **#532** — `GET /api/sessions/{session_id}` returned any user's full transcript
- **#533** — `POST /scim/v2/tokens` minted a working bearer token, which then
  satisfied the guard on the rest of that router: provision and deprovision
  users with no credentials

Both sat behind comments describing the intent (`# Token admin endpoints
(internal)`) without the enforcement. Both were found by hand, by someone who
happened to look.

## What the measurement says

Over every registered route, excluding the current public allowlist and static
assets:

| detection | count |
|---|---|
| auth via a `Depends` in the dependency tree | 84 |
| auth called **inline** in the handler body | 11 |
| neither detected | **47** |

Three findings in that table, and all three shape the design:

**1. Dependency inspection alone is not sufficient.** Eleven handlers call
`resolve_active_user` / `_require_admin` inside the body rather than declaring a
dependency — including `/api/sessions/{session_id}`, whose #532 fix is correct.
A purely dependency-based check would flag a properly guarded endpoint.

**2. The allowlist is badly incomplete.** The 47 include `/auth/login`,
`/auth/register`, `/assist`, `/chat` — routes that are obviously and correctly
public but absent from `PUBLIC_ENDPOINT_SPECS`. The allowlist has never been the
authoritative statement of what is public; it has been a partial one nobody
depended on.

**3. The 47 is the actual work.** The mechanism is easy. Classifying 47 routes
as "public by intent" or "missing a guard" is the part that requires judgment,
and it is where any remaining holes are.

## Approach

**Phase 1 — triage.** Classify all 47. Public-by-intent routes join
`PUBLIC_ENDPOINT_SPECS` with a reason. Anything that should be guarded gets one.
This phase is where new findings, if any, surface — and it must be done before
enforcement, or enforcement simply fails on startup.

**Phase 2 — enforce.** `check_router_auth` raises rather than logs when a route
is neither in the allowlist nor detectably guarded. Detection accepts *both*
forms:

- a recognised auth dependency in the route's dependency tree
- a recognised auth call in the handler source

The source scan is a heuristic and should be stated as one. It is the price of
the eleven inline handlers; the alternative is converting them to dependencies,
which is a larger and riskier change to make while closing a security gap.

**Phase 3 — optional.** Convert inline auth to declared dependencies so the
heuristic can be dropped. Not required for enforcement to work, and better done
separately.

## Acceptance

- `check_router_auth` fails startup on an unclassified route, with the path and
  method in the message.
- Every currently-registered route is either in the allowlist with a stated
  reason, or detectably guarded.
- A test registers a deliberately unguarded route and asserts the check rejects
  it — the mechanism must be shown to fire, not assumed to.
- A test asserts an inline-guarded route (`/api/sessions/{session_id}`) is
  accepted, so the heuristic's purpose is pinned against a future refactor.
- SCIM discovery endpoints stay public: SCIM 2.0 requires
  `ServiceProviderConfig`, `Schemas` and `ResourceTypes` unauthenticated, and a
  blanket sweep would break provisioning clients.

## Out of scope

Changing any route's actual auth. Phase 1 fixes only routes found to be missing
a guard; it does not revisit guards that exist.

## Note on the dev bypass

`AGENTIC_SEARCH_DEV_ADMIN=1` in the repo's `.env` makes every admin guard return
a synthetic admin, and **dotenv overrides `env -u`** — the variable must be set
to `"false"` explicitly to audit anything. A first pass at this measurement
reported 39 exposed endpoints that were entirely that artifact. Any test here
must neutralise the flag the same way.
