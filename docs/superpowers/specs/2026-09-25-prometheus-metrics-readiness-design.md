# Prometheus metrics export and a readiness probe: design

## Problem

The web app measures itself but exports nothing, and its health probe cannot
tell "up" from "able to serve":

- **Nothing can scrape the metrics.** `ROUTE_LATENCY`
  (`src/internal/servers/middleware/latency_logging.py`) and `STAGE_LATENCY`
  (`src/internal/observability/stage_metrics.py`) are rolling in-memory
  windows of the last 512 samples. They are readable only as JSON from
  `GET /api/admin/metrics` (behind admin auth) or the dev-only
  `/api/debug/latency`. No Prometheus, OTLP or StatsD export exists, so no
  alerting system can watch them. The windows also have no cumulative
  counters, so a rate or an error ratio cannot be derived from them.
- **The probe cannot see an outage.** `GET /health` returns
  `{"status": "ok"}` unconditionally. A web process whose store is broken, or
  whose retrieval server is gone, still reports healthy, so neither an
  orchestrator nor a human can distinguish "process alive" from "can answer
  a query".

## Decision (agreed with the user)

The user chose three things:
- the `prometheus-client` dependency, not a hand-rolled exposition format;
- `/metrics` opt-in, unauthenticated, and restricted at the network layer;
- a readiness probe separate from `/health`.

### 1. Dependency

Add `prometheus-client>=0.20,<1` to `requirements.txt` (the serving baseline)
and `requirements-unit-test.txt` (CI). It is pure Python and imports no torch.

### 2. `src/internal/observability/prometheus.py` (new)

The module owns a dedicated `CollectorRegistry` (`REGISTRY`), not the global
default one. Tests can then read it, and the process-level default collectors
do not mix with ours.

| Metric | Type | Labels | Recorded from |
|---|---|---|---|
| `agentic_search_http_requests_total` | Counter | `method`, `route`, `status` | latency middleware, every request |
| `agentic_search_http_request_duration_seconds` | Histogram | `method`, `route` | latency middleware |
| `agentic_search_stage_duration_seconds` | Histogram | `stage` (`retrieval`, `generation`, `auxiliary`) | where `STAGE_LATENCY.record(...)` runs (`app.py`, ~:2124), one observation per stage the request used |

- **`route` is the matched route template**, the same `_route_key` the
  latency window uses. That keeps label cardinality bounded: an unmatched
  path is recorded as the literal `"<unmatched>"`, not the raw path. This is
  deliberately stricter than the JSON window, because a Prometheus label set
  never expires.
- **`status` is the status class** (`2xx`, `4xx`, `5xx`), which bounds
  cardinality further while still giving an error ratio.
- **An exception** that escapes `call_next` is recorded as `5xx` and then
  re-raised. Today such a request is recorded nowhere.
- **Buckets:**
  - HTTP: prometheus-client's default buckets (5 ms to 10 s) plus 30 s and
    60 s, because agent requests are long.
  - Stage: `0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120`.

The module exposes:

```python
REGISTRY: CollectorRegistry
def observe_request(method: str, route: str, status_code: int, seconds: float) -> None
def observe_stages(metrics: RequestStageMetrics | None) -> None
def render_latest() -> tuple[bytes, str]      # (body, content type)
```

`observe_request` is called from `add_latency_logging_middleware`, next to
`store.record`, so all three servers that use the middleware later get it for
free. Today that is only the web app.

### 3. `GET /metrics`: opt-in, unauthenticated

- It is mounted in `create_web_app` only when
  `AGENTIC_SEARCH_METRICS_ENABLED` is truthy. That becomes a new
  `metrics_enabled: bool = False` on `SearchExperienceSettings`, read with
  `get_env_bool`.
- When the flag is off the route does not exist, so the request returns 404.
- The route returns `Response(render_latest())` with the Prometheus content
  type.
- It is added to `PUBLIC_ENDPOINT_SPECS` (`auth_check.py`) and to
  `LICENSE_ENFORCEMENT_ALLOWED_PREFIXES`, both for `GET` only, as `/health`
  is. The existing `check_router_auth` guard will fail CI if either is
  missed.
- Recording is always on, flag or not. The flag only controls exposure. The
  cost is one counter increment and one histogram observation per request,
  which is small against a request.
- The env var goes in `docs/configuration.md`'s table (the
  documented-env-vars test enforces this), with the warning to restrict the
  route at the network layer.

### 4. `GET /ready`: readiness

`/health` stays as it is, as the liveness probe. The new `/ready` route runs
two checks:

| Check | Pass condition |
|---|---|
| `store` | `db.ping()`, a new public `AgenticSearchStore` method running `SELECT 1`, returns without raising |
| `retrieval` | `GET <retrieval base>/health` returns 2xx within `readiness.probe_timeout_seconds`. The base is `settings.search_url` with its path replaced by `/health`, e.g. `http://localhost:8001/retrieve` becomes `http://localhost:8001/health`. |

- **Status:** 200 if every check passes, otherwise 503. The body is
  `{"status": "ready" | "not_ready", "checks": {"store": {"ok": bool, "error":
  str | None}, "retrieval": {...}}}`.
- **Error strings** are the exception class name only (`"ConnectError"`), so
  that URLs and credentials cannot leak from an unauthenticated route.
- **The retrieval probe** uses `httpx.AsyncClient`, which is already a
  serving dependency. The timeout lives in `timeouts.toml`:

  ```toml
  [readiness]
  probe_timeout_seconds = 2.0     # /ready's retrieval /health probe
  ```

  This adds a `ReadinessPolicy` dataclass and a `readiness` field on
  `TimeoutPolicies`, documented in `docs/configuration/timeouts.md`.
- **Checks run concurrently** (`asyncio.gather`), so the worst case is about
  the probe timeout.
- **Auth:** public and license-exempt, like `/health`.
- **Nothing is cached.** A readiness probe should reflect now, and it runs
  every few seconds at most.

**Compose healthchecks are unchanged.** Pointing the web healthcheck at
`/ready` would couple the web container's health to retrieval, which compose
`depends_on` already orders. This design leaves that choice to operators and
documents it.

## Out of scope

- Breaker-state gauges, which come after the circuit-breaker PR lands.
- Metrics on the retrieval, rerank and web-search servers. They do not use
  the latency middleware today.
- Multiprocess mode (`PROMETHEUS_MULTIPROC_DIR`). The image runs one uvicorn
  worker, and this is documented.
- Alert rules and dashboards. This PR makes them possible; writing them is
  deployment-specific.

## Testing

In `tests/unit/observability/test_prometheus.py` and
`tests/unit/servers/web/test_metrics_ready.py`:

- **Unit:**
  - `observe_request` increments the counter under the route template and
    status class.
  - An unmatched path is recorded as `<unmatched>`.
  - `observe_stages` observes only the stages the request used.
  - `render_latest` returns text/plain exposition containing the metric
    names.
- **App:**
  - With the flag off, `/metrics` returns 404.
  - With the flag on, `/metrics` returns 200 with no auth, and after a
    `GET /health`, `agentic_search_http_requests_total{...route="/health"...}`
    is present.
- **Readiness:**
  - With the store fine and a stubbed retrieval `/health` returning 200,
    `/ready` returns 200.
  - When retrieval raises `ConnectError`, `/ready` returns 503 with
    `checks.retrieval.error == "ConnectError"` and no URL in the body.
  - When the store's `ping` raises, `/ready` returns 503.
  - `/ready` needs no auth.
- **Mutation checks:**
  - Remove the `observe_request` call and watch the counter test fail.
  - Force the retrieval check to `ok=True` and watch the 503 test fail.
- The timeout-policy drift and site tests, the documented-env-vars test and
  the route-auth test must pass.
