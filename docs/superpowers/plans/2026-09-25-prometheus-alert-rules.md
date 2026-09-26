# Prometheus Alert Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship tested Prometheus alert rules for the metrics the web app
already exports.

**Architecture:** A rules file and a promtool test file in
`deploy/prometheus/`. A pytest wrapper runs promtool when available, and
checks coverage and metric names without it.

**Tech Stack:** Prometheus rule YAML, `promtool` 3.x, pytest, PyYAML.

**Spec:** `docs/superpowers/specs/2026-09-25-prometheus-alert-rules-design.md`

## Global Constraints

- Every expression reuses the documented PromQL. Denominators are never
  zero-filled.
- The job label is `job="agentic-search"`.
- Every alert has `severity`, `summary` and `description`.

## Review Focus

- Zero traffic must never fire a percentage alert.
- The HTTP alert must not fire below 0.1 req/s.
- The per-agent error alert must fire for the failing agent only.

---

### Task 1: Tests first (red)

- [ ] **Write the tests.** Write `deploy/prometheus/agentic-search-alerts.test.yml`
  with firing and quiet cases for all six alerts. Write
  `tests/unit/test_prometheus_alert_rules.py`:
  - `promtool check rules` / `test rules`, skipped without promtool;
  - a coverage test that every alert in the rules file is named in the test
    file's `alertname` entries;
  - a metric-name drift test, which parses `agentic_search_[a-z_]+` from the
    rules and compares it with the names declared in `prometheus.py`,
    allowing the `_sum`, `_count` and `_bucket` suffixes of histograms.
- [ ] **Run them.** Expect them to fail, because the rules file is missing.

### Task 2: The rules (green)

- [ ] **Write the rules.** Write `deploy/prometheus/agentic-search-alerts.yml`
  per the spec table.
- [ ] **Run them.** Run `promtool check rules`, `promtool test rules`, and
  the pytest wrapper. Expect them to pass.
- [ ] **Mutation checks.** These are in the spec. Restore after each.

### Task 3: Docs and verification

- [ ] **Docs.** Add the Alerts section to `docs/observability-metrics.md`.
- [ ] **Verify.** Run the full unit suite, then ruff, then `git diff --check`.
  Get a review, then open the PR.
