# Measure the integration suite's drift properly, before deciding its fate

## Goal

Replace the number I gave you with one worth deciding on. The integration-suite
finding in #630 rested on a regex census that was wrong, and the decision it
supports covers 119 files.

## What was wrong with the first number

I reported "69 of 122 files reference at least one removed endpoint" and flagged
the census as approximate. It was worse than approximate in both directions.

The regex matched `/[A-Za-z0-9_{}/-]+` against sources like
`f"{API_SERVER_URL}/admin/api-key/{api_key}"`. It stopped at the brace and
produced `/admin/api-key/{api_key` — an endpoint that matches no route *because
the extraction mangled it*, not because the route is gone. Those artifacts then
marked their files as touching a removed endpoint.

It also counted 122 test files where `test_*.py` finds 119, and treated
`/admin/chat-session-history?{}` as removed when the path is served and only the
query string differs.

## The replacement

`examples/audit_integration_endpoints.py` walks the AST. Every f-string whose
parts reference `API_SERVER_URL` or `MCP_SERVER_URL` is rebuilt with `{}` for each
interpolation, concatenations of the form `API_SERVER_URL + "/path"` are picked up
too, and app routes are normalised the same way — query stripped, parameter names
collapsed — before comparison.

Corrected totals:

| | regex | AST |
|---|---|---|
| Endpoints referenced | 180 | 178 |
| Served by the app | 26 | **35** |
| Test files | 122 | **119** |

## The split that actually drives the decision

Counting files that touch *any* removed endpoint is the wrong question — a file
with one dead call and nine live ones is a repair, not a retirement. The useful
split is per file:

| | helper-aware | direct-only |
|---|---|---|
| Every endpoint gone — cannot be repaired | **22** | **36** |
| Mixed — judgement per test | 80 | 5 |
| Fully served | 2 | 11 |
| No endpoints attributable | 15 | 67 |

Two modes because neither attribution is exact, and the difference is the
interesting part:

- **Direct-only** counts endpoints a file builds itself. It under-counts: most
  tests call `CCPairManager` rather than constructing a URL, which is why 67
  files come out unclassifiable.
- **Helper-aware** also credits a test with the endpoints of every `common_utils`
  module it imports. It over-counts "mixed", because importing a manager does not
  mean calling every method on it.

Together they bracket the answer: **between 22 and 36 files reference nothing the
application still serves**, and a further large group needs per-test judgement
that no static pass can supply.

## What this does not do

It does not delete anything. #630 retired the indexing directory because a route
probe established that no stack could make it pass; the rest is a 119-file
decision and the point of this change is to make that decision cheap rather than
to pre-empt it.

It also cannot see endpoints reached through indirection it does not model — a
URL assembled from a variable, or a helper that takes a path as an argument. The
15 unattributable files in the helper-aware mode are where to look for those.

## Testing

The script is the measurement, so its own correctness is what matters. Three
checks were run rather than assumed: the three sampled "gone" families
(`/admin/api-key`, `/admin/code-interpreter`, `/admin/image-generation`) return
zero prefix matches against a real `create_web_app()`, not merely zero exact
matches; the query-string false positives disappeared once `_normalise` stripped
them, moving six endpoints from "gone" to "served"; and both `python -m` and
direct invocation work, since a `src.` import has silently broken the direct form
in this repo before.
