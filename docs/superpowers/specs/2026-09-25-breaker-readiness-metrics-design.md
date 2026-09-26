# Breaker and readiness metrics, their alerts, and promtool in CI: design

## Problem

Two gaps are left from the alert-rules work (#661):

- **Circuit-breaker state and `/ready` are not Prometheus metrics,** so
  neither has an alert. An open breaker is visible only in the admin JSON,
  and a not-ready process only to whoever calls `/ready`.
- **CI does not run the alert tests.** It has no promtool, so it skips the
  promtool rule tests and runs only the three structural checks.

## Decision (approved by the user)

### 1. Circuit-breaker state, collected at scrape time

A custom collector registered on the dedicated `REGISTRY`
(`src/internal/observability/prometheus.py`) reads `breaker_snapshots()` when
Prometheus scrapes. It exports:

- `agentic_search_circuit_breaker_open{breaker}`: 1 when the breaker is `open`
  or `half_open`, else 0.
- `agentic_search_circuit_breaker_consecutive_failures{breaker}`.

**The label is the breaker family.** It is the name before the first `:`:
`serpapi`, `browser_search`, `rerank` or `remote_llm`. Per-server
`remote_llm:<base_url>` breakers are named by URL, and metric labels never
carry URLs (#651). So a family reports its **worst** instance: the max of
`open`, and the max of failures.

Breakers are created lazily, so a family appears once the process has used
it. The collector imports the breaker module inside `collect()`, so the
metrics module stays free of import cycles.

### 2. Readiness, recorded per probe

Every `GET /ready` records its result:

- `agentic_search_ready`: 1 or 0;
- `agentic_search_ready_check{check}`: 1 or 0, where `check` is `store` or
  `retrieval`, a fixed vocabulary;
- `agentic_search_ready_checked_timestamp_seconds`: the time of that probe.

The recording happens through `observe_readiness(ready, checks)`, called from
the `/ready` route after `check_readiness`.

A scrape never runs the checks. That would put network calls inside
Prometheus's collection. So something must call `/ready`: an orchestrator
probe, a compose healthcheck, or a blackbox exporter. The docs say so.

### 3. Alerts

Two alerts join `deploy/prometheus/agentic-search-alerts.yml`, each with
firing and quiet promtool cases:

| Alert | Expression | For | Severity |
|---|---|---|---|
| `CircuitBreakerOpen` | `max by (breaker) (agentic_search_circuit_breaker_open) == 1` | 5m | ticket |
| `NotReady` | `agentic_search_ready == 0 and on() (time() - agentic_search_ready_checked_timestamp_seconds) < 300` | 5m | page |

`NotReady` only fires while the last probe is **fresh** (under 5 min old). A
stale `0` that nothing refreshed never pages. Its quiet cases:

- a ready probe;
- a not-ready result whose timestamp stopped advancing.

### 4. promtool in CI

A new `alert-rules` job in `.github/workflows/ci.yml`:

1. downloads `prometheus-3.13.2.linux-amd64.tar.gz` from the official GitHub
   release;
2. **verifies its SHA-256**,
   `0e8c4d46101bd025ea8265e377d2caabc57f488fc1be1c367f37db69ea41be6f`, from
   the release's `sha256sums.txt`;
3. extracts `promtool`;
4. runs `promtool check rules` and `promtool test rules` in
   `deploy/prometheus`.

A unit test pins that the job exists, that it verifies a checksum, and that it
runs `test rules`.

### 5. Docs corrections

`docs/deploy.md` and the #663 spec said new GHCR packages start private.
**Measured:** the first publish was anonymously pullable, because a package
linked to a public repository inherits public visibility. Both are corrected:
make it private in the package settings.

`docs/observability-metrics.md` documents:
- the new metrics;
- the two alerts;
- the "something must call `/ready`" requirement.

It also drops the "not covered yet" note.

## Out of scope

- Per-instance remote-LLM breaker labels.
- Running readiness at scrape time.
- Changing the compose healthcheck.

## Testing

- **Breaker collector:**
  - no breakers exports no series;
  - an open `serpapi` breaker gives `{breaker="serpapi"} 1` and its failure
    count;
  - two `remote_llm:<url>` breakers, one open and one closed, export a single
    `{breaker="remote_llm"}` with value 1, with no URL in any label;
  - a half-open breaker counts as 1.
- **Readiness:**
  - `/ready` with everything fine gives `ready 1`, both checks 1, and a
    timestamp set;
  - a retrieval failure gives `ready 0` and `retrieval 0`, with `store` still
    1;
  - an unknown check name is rejected by `observe_readiness`.
- **Alerts:** the promtool cases above, and the existing coverage and
  exporter-drift tests extended automatically.
- **CI:** the workflow test.
- **Mutation checks:**
  - Count half-open as closed, and the half-open test goes red.
  - Label by the full breaker name, and the no-URL test goes red.
  - Drop the freshness clause, and the stale `NotReady` quiet case goes red.
  - Skip `observe_readiness` in the route, and the readiness tests go red.
