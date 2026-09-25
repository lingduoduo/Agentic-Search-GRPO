# Agent Operational Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Export trustworthy HTTP rates, agent outcomes and decision rounds, and registry tool timeout rates, with documented denominators.

**Architecture:** Extend the existing dedicated Prometheus registry. Track agent lifecycles with context-local state and instrument model-generation boundaries; record tool attempts after validation at the registry lifecycle boundary. Keep existing response fields and retry policies compatible.

**Tech Stack:** Python 3.10+, prometheus-client, asyncio, pytest, FastAPI.

**Spec:** `docs/superpowers/specs/2026-09-25-agent-operational-metrics-design.md` (approved by user).

## Global Constraints

- No new observability dependency, dashboard service, or multiprocess support is required.
- Existing metric names and response fields remain compatible.
- Labels must never contain user IDs, request IDs, query text, tool arguments, URLs, or exception messages.
- Recording remains always enabled; export remains process-local and intended for one serving worker per process.
- A decision round is one attempted model generation cycle in the agent loop.
- Approval and escalation wait timeouts are excluded.

## Review Focus

1. Concurrent and nested agent runs must not mix their decision counts (Task 1).
2. Failed generation and forced-final-answer paths must contribute the correct rounds (Task 2).
3. Tool create/release exceptions and cancellation must preserve existing behavior and record once (Task 3).
4. Timeout followed by successful retry must remain visible as two attempts (Task 3).
5. Evicted HTTP errors and absent Prometheus series must not produce misleading percentages (Tasks 4–5).

## Preparation

- [ ] Use using-git-worktrees at execution time to create an isolated feature branch, carrying only the approved spec and this plan. Leave `.planning/` investigation files outside the PR.
- [ ] Use a Python 3.12 virtual environment, matching CI, and install `requirements-unit-test.txt` if no suitable test environment exists. The default system Python currently lacks pytest.
- [ ] Read the spec and `.github/workflows/ci.yml`. Run the existing exporter, registry, middleware, and agent tests before changes and record any baseline failures.

### Task 1: Metric primitives and run lifecycle

**Files:** Modify `src/internal/observability/prometheus.py`; create `src/internal/observability/agent_metrics.py`, `tests/unit/observability/test_agent_metrics.py`; extend `tests/unit/observability/test_prometheus.py`.

**Interfaces:**
- `observe_agent_run(agent: str, outcome: str, rounds: int) -> None`
- `observe_tool_attempt(outcome: str) -> None`
- `track_agent_run(agent: str)` asynchronous function decorator preserving the wrapped signature with `functools.wraps`.
- `record_decision_round() -> None`, a no-op outside a tracked run.

- [ ] Add a failing helper test that records before/after deltas on the dedicated registry:

```python
labels = {"agent": "search", "outcome": "completed"}
name = "agentic_search_agent_decision_rounds_sum"
before = REGISTRY.get_sample_value(name, labels) or 0
observe_agent_run("search", "completed", 3)
assert REGISTRY.get_sample_value(name, labels) == before + 3
```

Also assert run-counter and histogram-count deltas are exactly one, the configured buckets exist, and tool outcomes render with no name/argument labels.

- [ ] Run `python -m pytest tests/unit/observability/test_prometheus.py tests/unit/observability/test_agent_metrics.py -q`; confirm the new tests fail for missing helpers.
- [ ] Declare the three metrics with the exact names, labels and buckets from the spec. Implement helper validation against fixed agent/outcome vocabularies; reject negative/noninteger rounds without creating series.
- [ ] Implement lifecycle state using `ContextVar` and immutable round counts. The decorator sets a fresh context, defaults outcome to error, sets completed on normal return and cancelled on `asyncio.CancelledError`, observes in `finally`, then resets the context token. Always re-raise cancellation and exceptions.

```python
@track_agent_run("search")
async def sample():
    record_decision_round()
    return "answer"

assert await sample() == "answer"
```

Use immutable context values so inherited contexts in child tasks cannot mutate another task's counter. Generation hooks execute in the run task.

- [ ] Add tests for zero-round failure, cancellation propagation, two concurrent runs with 1 and 3 rounds, nested runs restoring the parent count, calls outside a run, and exact single observation on all outcomes. Inspect histogram sum/count deltas for each case.
- [ ] Re-run both test files; commit as `feat: add agent and tool metric primitives`.

### Task 2: Instrument both agent loops

**Files:** Modify `src/agents/search/search.py`, `src/agents/tool/tool_calling.py`; extend `tests/unit/test_agent_loop.py`, `tests/unit/test_tool_recovery_loop.py`.

**Consumes:** Task 1's `track_agent_run` and `record_decision_round`.
**Produces:** One run observation for each search/tool loop invocation, with attempted-generation counts independent of existing output counters.

- [ ] Extend the existing dummy-manager agent tests to assert a direct answer records 1 round, a tool batch followed by an answer records 2 rounds, and two tools in that batch still record 2 rounds. Read exporter deltas rather than assuming a fresh global registry.
- [ ] Add a generation-failure test: the dummy manager raises `RuntimeError`; assert the exception propagates and an error run records 1 round. Add pre-generation prompt-build failure with 0 rounds and cancellation during generation with 1 round.
- [ ] Run `python -m pytest tests/unit/test_agent_loop.py tests/unit/test_tool_recovery_loop.py -q` and confirm metric assertions fail before adding hooks.
- [ ] Decorate only the concrete search/tool `run` methods:

```python
@track_agent_run("search")
async def run(...):  # retain the existing full signature and body
    ...
```

The ellipses above denote the existing method, not new stub code. For the tool loop use `"tool"`. Add `record_decision_round()` immediately before each awaited `generate_response_ids` in the search `_generate_turn`, search `_force_final_answer`, and tool `run`. Do not increment before prompt construction or on the force-answer no-evidence early return.
- [ ] Extend the existing forced-final-answer fixture: assert one extra round when it actually generates, none when it returns early. Confirm output `num_turns` values remain identical to existing expectations.
- [ ] Re-run agent tests plus Task 1 tests; commit as `feat: measure agent outcomes and decision rounds`.

### Task 3: Preserve timeout identity and count tool attempts

**Files:** Modify `src/internal/tools/base.py`, `src/internal/tools/registry.py`, and typed timeout producers found by searching `ToolFailure(` under `src/internal/tools`; extend `tests/unit/test_tool_registry.py`, `tests/unit/test_tool_recovery_loop.py`, and existing tests for modified producers.

**Consumes:** `observe_tool_attempt(outcome: str) -> None`.
**Produces:** Backward-compatible `ToolFailure.is_timeout: bool = False`; one metric observation per validated registry lifecycle attempt.

- [ ] Add registry tests using `FunctionTool` functions which return normally, raise `TimeoutError`, raise `ValueError`, or raise `asyncio.CancelledError`. Assert outcome counter deltas; cancelled calls must still raise. Unknown names and invalid arguments must leave all counters unchanged.

```python
@registry.tool()
async def times_out() -> str:
    raise TimeoutError()

before = REGISTRY.get_sample_value(
    "agentic_search_tool_attempts_total", {"outcome": "timeout"}
) or 0
result = await registry.invoke_detailed("times_out", {})
assert result.failure.is_timeout is True
assert REGISTRY.get_sample_value(
    "agentic_search_tool_attempts_total", {"outcome": "timeout"}
) == before + 1
```

- [ ] Run `python -m pytest tests/unit/test_tool_registry.py tests/unit/test_tool_recovery_loop.py -q`; verify new tests fail.
- [ ] Add the defaulted boolean at the end of `ToolFailure`, preserving positional construction. Retain `FailureCategory.TRANSIENT` for timeouts so recovery decisions do not change. Set the flag from `TimeoutError`, `httpx.TimeoutException`, and aiohttp timeout subclasses where exceptions are converted to typed failures. Keep optional imports consistent with existing module patterns. For adapters returning typed failures, propagate timeout identity at their catch sites; never infer it from message text. Test each changed conversion path.
- [ ] In `invoke_detailed`, begin observation after `_resolve` and validation, immediately before `tool.create()`. Use an outer try/except/finally around existing create/execute/release control flow. Default outcome to error, classify execute exceptions before existing conversion, and classify returned typed failures through `is_timeout`; successful response has no failure. Catch cancellation separately and re-raise. Preserve create/release exception propagation. Final cleanup exceptions determine terminal outcome; exactly one counter increment happens in the outer finally.
- [ ] Add lifecycle tests for create failure, release failure, and cancellation in release. Confirm release still runs after an execute failure and does not run when create never returns an instance. Add a typed `ToolErrorText` timeout test with no exception.
- [ ] Extend the existing recovery-loop fixture so its first call raises `TimeoutError` and its second succeeds; assert one timeout, one success, unchanged retry count, and a completed logical result. Assert denied approval produces no registry attempt. Assert ordinary transient connection failures count as error, not timeout.
- [ ] Run registry, producer, recovery-loop and recovery-policy tests; commit as `feat: record registry tool timeout outcomes`.

### Task 4: Repair the admin HTTP error window

**Files:** Modify `src/internal/servers/middleware/latency_logging.py`; extend `tests/unit/servers/test_latency_stats.py`.

**Interfaces:** Existing `RouteLatencyStats.record` and `snapshot` remain unchanged.

- [ ] Add the failing eviction regression:

```python
stats = RouteLatencyStats(max_samples_per_route=2)
_record(stats, [1], status_code=500)
_record(stats, [2, 3], status_code=200)
row = stats.snapshot()[0]
assert row["count"] == 2
assert row["errors"] == 0
```

Also retain tests for mixed statuses and percentiles.
- [ ] Add an endpoint that raises; use `TestClient(..., raise_server_exceptions=False)` and assert both the JSON window and Prometheus record one 5xx observation.
- [ ] Run `python -m pytest tests/unit/servers/test_latency_stats.py -q`; confirm the new tests fail.
- [ ] Store each sample as `(elapsed_ms, status_code >= 500)` in the bounded deque. Compute sorted durations and error sum from that same deque during snapshot; remove the lifetime `_errors` dictionary. In the middleware exception path, also call `store.record` with 500 and the same elapsed measurement used by Prometheus, then re-raise.
- [ ] Run middleware and admin-metrics router tests; commit as `fix: align admin error counts with the latency window`.

### Task 5: Operator documentation, final verification and PR

**Files:** Create `docs/observability-metrics.md`; add a link from `docs/configuration.md`; include approved spec and this checked-off plan in the PR.

- [ ] Document exact semantics, coverage, opt-in export flag, single-worker limitation, retry granularity, absent series, and cancellation exclusions. Include these concrete PromQL examples (five-minute windows):

```promql
# HTTP agent-entry QPS; add other intended query routes explicitly.
sum(rate(agentic_search_http_requests_total{method="POST",route=~"/api/agent|/api/agent/stream"}[5m]))

# Average decision rounds for completed runs, by agent.
sum by (agent) (rate(agentic_search_agent_decision_rounds_sum{outcome="completed"}[5m]))
/
sum by (agent) (rate(agentic_search_agent_decision_rounds_count{outcome="completed"}[5m]))

# Tool timeout percentage, excluding cancelled attempts.
100 * (sum(rate(agentic_search_tool_attempts_total{outcome="timeout"}[5m])) or vector(0))
/
sum(rate(agentic_search_tool_attempts_total{outcome=~"success|timeout|error"}[5m]))
```

Include equivalent HTTP 5xx and agent-error percentages with identical numerator/denominator populations. Only zero-fill a missing numerator; never fabricate denominator traffic or treat no observations as measured zero. State that HTTP stream status cannot describe terminal agent failure, that agent outcomes cover search/tool loops, and that tool metrics cover registry lifecycles only.
- [ ] Run `python -m pytest tests/unit/observability/ tests/unit/test_agent_loop.py tests/unit/test_tool_registry.py tests/unit/test_tool_recovery_loop.py tests/unit/test_tool_recovery_policy.py tests/unit/servers/test_latency_stats.py tests/unit/servers/web/test_metrics_router.py tests/unit/servers/web/test_metrics_ready.py -q` plus all modified-producer tests.
- [ ] Run `python -m pytest tests/unit/ -q --tb=short` once after focused checks pass, matching CI. Run the lint/format checks configured in `.github/workflows/ci.yml`, and `git diff --check`. Resolve introduced failures; disclose baseline/environment failures explicitly.
- [ ] Use verification-before-completion and requesting-code-review. Review final branch against the spec, especially exactly-once observation, cancellation, no changes to retries, and metric cardinality. Address findings and rerun affected tests.
- [ ] Commit documentation and plan progress. Push the feature branch and create a PR with `gh pr create --body-file <temporary-description-file>`, describing the behavioral changes, exact validation results, and process-local coverage limitations. Return the PR link.

## Execution handoff

Recommended: Native execution. The five tasks share instrumentation contracts, and implementing in one session keeps those contracts consistent while retaining a final independent branch review. Await user plan review and execution-method selection before product implementation.
