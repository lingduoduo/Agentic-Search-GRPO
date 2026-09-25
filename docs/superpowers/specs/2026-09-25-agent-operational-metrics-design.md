# Agent operational metrics

## Intent and scope

Provide operators with QPS, error rate, average agent decision rounds, and tool timeout rate through the existing opt-in Prometheus endpoint. The user requested a spec, implementation plan, code, and a PR following Superpowers. This spec defines the proposed metric contract; implementation follows written-spec and plan review.

No new observability dependency, dashboard service, or multiprocess support is required. Existing metric names and response fields remain compatible. Labels must never contain user IDs, request IDs, query text, tool arguments, URLs, or exception messages.

## Approach and alternatives

Extend the dedicated Prometheus registry and instrument lifecycle boundaries. This reuses the deployed export path and permits rates across scrape windows and replicas.

Alternatives considered: derive metrics from admin JSON (insufficient because its request count is a bounded window); build a log aggregation pipeline (adds infrastructure and requires consistent event delivery). Neither is needed for these four metrics.

## Metric contract

### HTTP QPS and error rate

Reuse `agentic_search_http_requests_total{method,route,status}`. Document five-minute PromQL queries using an explicit query-route allowlist; exclude health checks, readiness, metrics, and admin routes. HTTP server error percentage is 5xx divided by all statuses for the identical route set. Report 4xx separately. This counter measures HTTP responses; it does not represent individual WebSocket messages or stream completion.

Fix the admin JSON window so `errors` counts 5xx responses within the same retained samples as `count`. Retain the existing response schema. An evicted failed response must decrement the window's error count. Exceptions escaping middleware should enter this window consistently with Prometheus.

### Agent outcomes

Add `agentic_search_agent_runs_total{agent,outcome}` for the search and tool agent loops. `agent` is `search` or `tool`; `outcome` is `completed`, `error`, or `cancelled`. Count once on each loop's termination, including failures before the first model generation. A normal loop return is `completed`, including a configured turn-limit exit; it is not a quality or correctness assessment. A raised exception is `error`; task cancellation is `cancelled` and must propagate.

This metric is independent of HTTP status, so exceptions inside a streaming request are visible. It covers these two agent loops, not every RAG mode or transport admission failure. Document that boundary explicitly. Error percentage excludes cancelled runs from its denominator; cancellation percentage is reported separately.

### Average decision rounds

Add `agentic_search_agent_decision_rounds{agent,outcome}` as a histogram with buckets `0,1,2,3,5,8,13,21,34,55` plus infinity. Observe exactly once for every terminated run with the same labels as the run counter.

A decision round is one attempted model generation cycle in the agent loop. Increment immediately before generation, including forced-final-answer generations and generation attempts that raise. Parallel tool calls, tool-result messages, retrieval subqueries, and approval waits do not increment it. A run failing before generation records zero. Existing `num_turns` and retrieval-round fields remain unchanged.

Average rounds equals the rate of histogram sum divided by the rate of histogram count, with matching filters. Default to all terminated runs and document a completed-only variant. Do not average per-instance averages.

### Tool timeout rate

Add `agentic_search_tool_attempts_total{outcome}` at the shared tool registry execution boundary. Outcomes are `success`, `timeout`, `error`, and `cancelled`. Only actual tool executions enter the denominator: lookup failures, invalid arguments, denied approvals, and unavailable-tool skips do not. Each retry that re-enters execution counts separately. Adapter-internal provider retries remain one registry attempt and are outside this metric's granularity.

Classify timeouts using typed failures and supported timeout exceptions, never substring matching. Preserve all existing retry, error conversion, and cancellation behavior. Record once per attempted execution, including failures during execution lifecycle cleanup. Explicit timeout failure takes precedence over generic failure when the registry exposes that category.

Timeout percentage equals timeout attempts divided by success, timeout, and error attempts. Cancelled attempts are reported separately. Approval and escalation wait timeouts are excluded. Omit tool-name labels in this first version to keep cardinality bounded even with dynamic MCP/OpenAPI registrations.

## Integration

Metric declarations and recording helpers belong in `src/internal/observability/prometheus.py`. Agent hooks belong in the search and tool loops, with a small lifecycle helper if needed to guarantee exactly-once recording and exception propagation. Instrument the shared registry so agent retries and other registry consumers have the same attempt semantics.

Keep the current `AGENTIC_SEARCH_METRICS_ENABLED` exposure flag. Recording remains always enabled; export remains process-local and intended for one serving worker per process. A scrape with no observations may omit labeled series. Documentation must distinguish absent data and a zero denominator from a measured zero rate.

## Validation and acceptance

- Counter and histogram export tests verify exact names, bounded labels, sum/count, and outcomes.
- Agent hook tests exercise both loops, zero-round failure, generation failure, cancellation, forced final answer, and parallel tool batches; every run records once.
- Tool execution tests cover success, typed timeout, generic error, cancellation, timeout followed by successful retry, invalid arguments, and denied approval. Timeout followed by success records two attempts with one timeout.
- Admin-window tests exceed its capacity, evict errors, and exercise escaping exceptions; errors never exceed count.
- Existing metrics endpoint, agent, tool recovery, and middleware tests continue passing.
- Operator documentation includes QPS, HTTP error percentage, agent error percentage, average rounds, and tool timeout percentage queries and their coverage limitations.
- No live load test is required to validate instrumentation; no production performance claims are made.

## Delivery

After spec approval, create a task-by-task implementation plan with failing tests, implementation steps, and validation commands. Execute the reviewed plan on an isolated feature branch, review the final diff, and create a PR containing the spec, plan, implementation, tests, and operator documentation.
