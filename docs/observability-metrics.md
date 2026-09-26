# Operational metrics

The web backend records Prometheus metrics for HTTP traffic, agent runs,
tool calls and stale cache serves. This page defines each metric, what it
covers and what it does not, and the PromQL for the five operator questions:
QPS, HTTP error percentage, agent error percentage, average decision rounds,
and tool timeout percentage.

## Export

- Recording is always on. **Exposure** is opt-in: set
  `AGENTIC_SEARCH_METRICS_ENABLED=1` to mount `GET /metrics`
  ([configuration](configuration.md)). The route is unauthenticated, so
  restrict it at the network layer.
- Metrics are **process-local** and live on a dedicated registry
  (`src/internal/observability/prometheus.py`). There is no multiprocess mode;
  the image runs one uvicorn worker. Run one serving worker per process and let
  Prometheus aggregate across replicas with `sum(...)`.
- Counters and histograms are cumulative from process start. Always query them
  through `rate()` / `increase()`, never as raw values.
- Labels are bounded vocabularies. No label ever carries a user ID, request ID,
  query text, tool name or arguments, URL, or exception message.

## Metrics

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `agentic_search_http_requests_total` | counter | `method`, `route` (template, or `<unmatched>`), `status` (`2xx`/`4xx`/`5xx`) | One per HTTP response, including a handler exception, which is counted as `5xx` |
| `agentic_search_http_request_duration_seconds` | histogram | `method`, `route` | HTTP response latency |
| `agentic_search_stage_duration_seconds` | histogram | `stage` | Per-request retrieval / generation / auxiliary time on `/api/agent` |
| `agentic_search_agent_runs_total` | counter | `agent` (`search`, `tool`), `outcome` (`completed`, `error`, `cancelled`) | One per terminated agent loop run |
| `agentic_search_agent_decision_rounds` | histogram (buckets `0,1,2,3,5,8,13,21,34,55,+Inf`) | `agent`, `outcome` | Attempted model generations per terminated run; one observation per run, same labels as the run counter |
| `agentic_search_tool_attempts_total` | counter | `outcome` (`success`, `timeout`, `error`, `cancelled`) | One per validated tool-registry execution |
| `agentic_search_stale_cache_serves_total` | counter | `source` (`web`, `retrieval`) | One per failed live lookup answered from a stale serving-cache entry |

### Semantics and coverage

**HTTP.**
- The counter measures HTTP responses. It does not see individual WebSocket
  messages (`/api/agent/ws`).
- It does not see how a stream ends. A streaming response (`/api/agent/stream`,
  the SSE surfaces) is `2xx` once its headers are sent, even if the agent
  fails mid-stream. Use the agent metrics for terminal failures.

**Agent runs.**
- **Coverage:** `SearchAgentLoop` (`agent="search"`, the `/api/agent` search
  path with a local model) and `ToolAgentLoop` (`agent="tool"`, the
  `/api/agent` tool path and `/tool/send-tool-message`). Plain chat
  (`PlainGenerationLoop`), `AgenticRAGLoop`, direct retrieval, and requests
  rejected before a loop starts are not agent runs.
- **Outcomes:**
  - `completed` means the loop returned normally, including a turn-limit exit.
    It is not a quality or correctness judgement.
  - `error` means the loop raised.
  - `cancelled` means the task was cancelled, for example by a client
    disconnect. The cancellation still propagates.

**Decision rounds.**
- A round is one *attempted* model generation, counted immediately before
  generation. That includes a forced final answer and a generation that
  raises.
- Parallel tool calls, tool results, retrieval subqueries and approval waits
  add nothing.
- A run that fails before its first generation records `0`.
- This is independent of the `num_turns` / retrieval-round fields in responses.

**Tool attempts.**
- Counted at the shared tool-registry execution boundary (`invoke` and
  `invoke_detailed`), once per `create → execute → release` lifecycle.
- **Excluded:** unknown tools, invalid arguments, denied approvals, and tools
  skipped as unavailable. None of these executes anything.
- **Retries:** each loop retry is a separate attempt, so a timeout followed by
  a successful retry is one `timeout` plus one `success`. Retries a provider
  makes internally, such as the public-data HTTP client's, stay inside one
  registry attempt.
- **Timeout** is decided from the exception type, never from message text:
  - `TimeoutError`, `asyncio.TimeoutError` and aiohttp timeouts;
  - `httpx.TimeoutException`;
  - a typed `ToolFailure(is_timeout=True)`, which adapters set when they catch
    such an exception themselves.
- **The corpus and web search tools** report `timeout` when *any* error page
  they received timed out. An explicit timeout outranks a generic failure,
  including the web cascade's "no browser fallback is configured" advisory
  page.
- **Known blind spot:** a provider timeout whose exception has an empty message
  (`asyncio.TimeoutError()`, from SerpAPI, Google or retrieval) produces an
  error page with empty text. Existing code reads that as an empty successful
  result, so the attempt counts as `success`. Tool recovery depends on this
  today, so changing it is outside this metric's scope.
- **Not tool timeouts:** approval waits and escalation waits.

**Stale cache serves.**
- `source="web"` is the `search_tool` fallback for a web provider;
  `source="retrieval"` is the `SearchClient.retrieve` fallback
  ([serving cache](retrieval.md#serving-cache)).
- Counted once per call that the fallback answers, not once per page or row:
  a three-query retrieval batch served stale is `1`.
- A failure with no usable stale entry is not counted; it is returned or
  raised as before.
- A stale serve hides an outage from every other metric: the tool call that
  received it counts as `success`. Alert on this rate, not on its absence
  from the error ratios.


- A labelled series appears only after its first observation. A fresh process,
  or one that has never had a tool time out, exports no
  `outcome="timeout"` series.
- **Absent data is not a measured zero.** In the percentage queries below:
  - Only the *numerator* is zero-filled (`or vector(0)`), because "no timeouts
    among N attempts" really is 0 %.
  - The *denominator* is never zero-filled. With no traffic the query returns
    no data rather than 0 %, which is correct: there was nothing to measure.

## Queries

All queries use five-minute windows. Numerator and denominator always select
the same population.

**QPS on the query surfaces.** List the routes explicitly. Health checks,
`/ready`, `/metrics` and admin routes are deliberately excluded.

```promql
sum(rate(agentic_search_http_requests_total{method="POST",route=~"/api/agent|/api/agent/stream|/chat/send-chat-message|/search/send-search-message|/tool/send-tool-message"}[5m]))
```

**HTTP 5xx percentage.** The same route set on both sides. Report 4xx
separately (swap in `status="4xx"`).

```promql
100 * (sum(rate(agentic_search_http_requests_total{method="POST",route=~"/api/agent|/api/agent/stream|/chat/send-chat-message|/search/send-search-message|/tool/send-tool-message",status="5xx"}[5m])) or vector(0))
/
sum(rate(agentic_search_http_requests_total{method="POST",route=~"/api/agent|/api/agent/stream|/chat/send-chat-message|/search/send-search-message|/tool/send-tool-message"}[5m]))
```

**Agent error percentage.** Cancelled runs are excluded from the denominator.

```promql
100 * (sum by (agent) (rate(agentic_search_agent_runs_total{outcome="error"}[5m])) or (0 * sum by (agent) (rate(agentic_search_agent_runs_total[5m]))))
/
sum by (agent) (rate(agentic_search_agent_runs_total{outcome=~"completed|error"}[5m]))
```

**Agent cancellation percentage**, reported separately:

```promql
100 * (sum by (agent) (rate(agentic_search_agent_runs_total{outcome="cancelled"}[5m])) or (0 * sum by (agent) (rate(agentic_search_agent_runs_total[5m]))))
/
sum by (agent) (rate(agentic_search_agent_runs_total[5m]))
```

**Average decision rounds.** Divide the rate of the sum by the rate of the
count, over the same filter. Do not average per-instance averages. This
default covers all terminated runs:

```promql
sum by (agent) (rate(agentic_search_agent_decision_rounds_sum[5m]))
/
sum by (agent) (rate(agentic_search_agent_decision_rounds_count[5m]))
```

For the completed-only variant, add `{outcome="completed"}` to both sides:

```promql
sum by (agent) (rate(agentic_search_agent_decision_rounds_sum{outcome="completed"}[5m]))
/
sum by (agent) (rate(agentic_search_agent_decision_rounds_count{outcome="completed"}[5m]))
```

**Tool timeout percentage.** Cancelled attempts are excluded and reported
separately.

```promql
100 * (sum(rate(agentic_search_tool_attempts_total{outcome="timeout"}[5m])) or vector(0))
/
sum(rate(agentic_search_tool_attempts_total{outcome=~"success|timeout|error"}[5m]))
```

**Stale-serve rate by source.** Nonzero means a provider or the retrieval
server is failing and the cache is covering for it.

```promql
sum by (source) (rate(agentic_search_stale_cache_serves_total[5m]))
```

The by-agent numerators zero-fill with `0 * <denominator-shaped series>`
rather than `vector(0)`. A bare `vector(0)` has no `agent` label, so it would
not match the per-agent denominator.

## Alerts

`deploy/prometheus/agentic-search-alerts.yml` turns the queries above into
alert rules. To use it:

- Load it with `rule_files: [deploy/prometheus/agentic-search-alerts.yml]`.
- Scrape the web app as job **`agentic-search`**, with
  `AGENTIC_SEARCH_METRICS_ENABLED=1`. The job name is only used by
  `AgenticSearchDown`, so change it there if yours differs.
- Treat every threshold as a starting point to tune to your traffic.

| Alert | Fires when (5m windows) | For | Severity |
|---|---|---|---|
| `AgenticSearchDown` | `up{job="agentic-search"} == 0` | 2m | page |
| `HighHTTP5xxRate` | Query-route 5xx is above 5 %, **and** query traffic is at least 0.1 req/s | 10m | page |
| `HighAgentErrorRate` | Agent error runs are above 10 % of completed plus error runs, per agent | 10m | ticket |
| `HighToolTimeoutRate` | Tool timeouts are above 20 % of executed attempts | 15m | ticket |
| `StaleCacheServing` | Any stale-cache serve, per `source` (a hidden outage) | 10m | ticket |
| `HighAgentDecisionRounds` | Completed runs average more than 8 rounds, per agent | 30m | ticket |

Percentage alerts never zero-fill their denominator, so zero traffic cannot
fire them. The HTTP traffic floor stops a single failure during an idle hour
from reading as 100 %.

**Tests.** `deploy/prometheus/agentic-search-alerts.test.yml` has a firing
case and at least one quiet case for every alert, including zero traffic. Run
it with:

```bash
cd deploy/prometheus && promtool test rules agentic-search-alerts.test.yml
```

`tests/unit/test_prometheus_alert_rules.py` runs promtool when it is on
`PATH`. Without promtool it still checks three things:

- every alert has a severity, a summary and a description;
- every alert has both a firing and a quiet test;
- every metric a rule uses is one the exporter declares.

**Not covered yet.** `/ready` and circuit-breaker state are not Prometheus
metrics, so neither has an alert.

## Also available

- `GET /api/admin/metrics` (admin-only JSON) has rolling windows of the last
  512 responses per route, with `count`, `errors` (5xx within that same window)
  and p50/p95/max.
- Because those windows are bounded, their numbers are not rates. Use
  Prometheus for rates.
