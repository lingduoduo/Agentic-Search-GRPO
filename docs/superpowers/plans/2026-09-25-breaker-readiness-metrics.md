# Breaker And Readiness Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export circuit-breaker and readiness state to Prometheus, alert on
them, and run the promtool alert tests in CI.

**Architecture:** A scrape-time collector for breakers, gauges written by
`/ready`, two rules plus promtool cases, and a checksum-pinned promtool job in
CI.

**Tech Stack:** prometheus-client (custom collector, Gauge), FastAPI,
Prometheus rules and promtool, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-25-breaker-readiness-metrics-design.md`

## Global Constraints

- No URLs in labels: breakers are labelled by family, worst instance.
- A scrape never performs network I/O.
- The promtool download is checksum-verified.
- The torch-free CI job still imports the metrics module.

## Review Focus

- The collector must survive the process-wide breaker registry being empty,
  or being reset between tests.
- `NotReady` must stay quiet on a stale not-ready result.
- The exporter-drift test must see the collector's metric names, which are
  string literals in `prometheus.py`.

---

### Task 1: Breaker collector (TDD)

- [ ] Write tests in `tests/unit/observability/test_prometheus.py`, using the
  real breaker registry (`get_breaker(...)`, with `record_failure` up to the
  threshold from policy). Run them and expect red, then implement
  `_BreakerStateCollector`, register it on `REGISTRY`, and expect green.
  Commit.

### Task 2: Readiness gauges (TDD)

- [ ] Unit-test `observe_readiness`. Add app tests in
  `tests/unit/servers/web/test_metrics_ready.py`: after `/ready` the gauges
  reflect the result. Run them and expect red, then implement the helper and
  the route call, and expect green. Commit.

### Task 3: Alerts and CI (TDD)

- [ ] Add promtool cases (firing and quiet, including the stale case) for the
  two alerts. Run them and expect red, because the rules are missing. Add
  the rules, and expect green.
- [ ] Add the CI job, plus `tests/unit/test_ci_runs_promtool.py`. Run
  actionlint.
- [ ] Commit.

### Task 4: Docs and verification

- [ ] Update `observability-metrics.md` and `deploy.md`, and correct the #663
  spec's visibility claim.
- [ ] Run the mutation checks (from the spec), then the full suite, then ruff,
  then `git diff --check`. Open the PR.
