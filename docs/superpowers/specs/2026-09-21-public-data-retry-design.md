# Retry a transient upstream, once or twice

## Goal

Stop a flaky upstream from turning a working tool into a mostly-broken one,
without letting a dead upstream stall an agent turn.

## The problem

`_fetch` in `public_data/_http.py` is the shared HTTP layer behind all nine
public data tools. It makes exactly one attempt: any status at or above 400
becomes a `PublicDataError`, which `guarded` turns into `{"error": ...}`.

For a permanent failure that is right. For a transient one it discards a
request that would have succeeded on the next try.

Measured against `web.archive.org`, which `search_wayback` calls:

```
single attempt           ~2 of 6 succeed
up to 3, 0.4s/0.8s back  ~4 of 6 succeed
```

The 503s are nondeterministic and independent of both the User-Agent and the
query parameters — a direct curl using curl's own UA sees the same rate. The
upstream is simply rate-limiting or unstable, and with no retry the tool
inherits that instability in full.

This was nearly diagnosed as a malformed request, on the strength of one curl
success against two tool failures. Repeating the request six times showed the
failure hits every combination. The sample, not the hypothesis, was wrong.

## Architecture

`_fetch` retries, bounded by two independent limits.

**What is retried.** Statuses 429, 502, 503 and 504, and the network
exceptions that already collapse into `PublicDataError`. Nothing else: a 404 is
an answer, and a 400 will be one again.

**Only GET.** Every public data tool is a GET. Retrying a POST needs an
idempotency guarantee this layer cannot make.

**Attempt cap:** three, with 0.4s then 0.8s of backoff.

**Elapsed-time cap:** no retry starts once 15 seconds have passed since the
first attempt began. This matters because the per-call timeout is a parameter,
not a constant — `search_nearby_places` overrides it with a much longer
Overpass budget. Capping attempts alone would let that tool spend three full
timeouts on a dead host. Capping elapsed time bounds the damage regardless of
what any caller passes:

| scenario | attempts | roughly |
|---|---|---|
| fast 503s (0.2s each) | 3 | 1.4s |
| default 10s timeout, all failing | 2 | 20s, not 31s |
| Overpass 60s timeout, failing | 1 | unchanged |

The last row is the point: a slow upstream is never multiplied.

## Testing

No live network; the existing `_FakeSession` harness is extended to replay a
sequence of responses rather than one.

- a 503 followed by a 200 returns the 200, having made two requests;
- three 503s raise `PublicDataError`, having made exactly three;
- a 404 raises after exactly one request — non-transient statuses are answers;
- a POST is not retried, even on a 503;
- a network exception is retried, then gives up;
- when the first attempt exhausts the elapsed budget, no retry is made.

Each is mutation-checked: removing the retry, widening the retryable set to all
errors, and dropping the elapsed cap must each turn the matching test red.

## Limits

Retry raises the success rate against an unstable host; it does not fix one.
The measurement above still fails 2 of 6 trials with retry enabled, because the
upstream has sustained bad periods, not only isolated blips.

Worst-case latency rises for a failing call — bounded, but real, and paid
inside an agent turn. The elapsed cap is what keeps it predictable.

This does not add jitter, a circuit breaker, or per-host policy. Those are
worth having if a host proves persistently bad, and none of them is justified
by one flaky upstream on one afternoon.
