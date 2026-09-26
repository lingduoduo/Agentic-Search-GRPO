# Prometheus alert rules: design

## Problem

The web app exports operational metrics on the opt-in `/metrics` (#648, #651,
#659), and `docs/observability-metrics.md` documents tested PromQL for them.
But no alert rule exists anywhere in the repo. Nobody is notified when error
rates spike, tools time out, or the stale cache starts hiding an outage.

## Decision (approved by the user)

### The rules file

`deploy/prometheus/agentic-search-alerts.yml` is one rule group,
`agentic-search`. Every alert has these labels and annotations:

- `labels.severity`: `page` or `ticket`;
- `annotations.summary` and `annotations.description`.

Every expression reuses the documented PromQL. Each threshold is a commented
starting point for tuning.

| Alert | Expression (5m windows) | For | Severity |
|---|---|---|---|
| `AgenticSearchDown` | `up{job="agentic-search"} == 0` | 2m | page |
| `HighHTTP5xxRate` | 5xx percentage over the query routes > 5, **and** total query-route rate ≥ 0.1 req/s | 10m | page |
| `HighAgentErrorRate` | per-agent error percentage (numerator zero-filled per agent) > 10 | 10m | ticket |
| `HighToolTimeoutRate` | tool timeout percentage over `success\|timeout\|error` > 20 | 15m | ticket |
| `StaleCacheServing` | `sum by (source) (rate(agentic_search_stale_cache_serves_total[5m])) > 0` | 10m | ticket |
| `HighAgentDecisionRounds` | average rounds of completed runs > 8, per agent | 30m | ticket |

- **The job label.** The rules assume the scrape job is named
  `agentic-search`. That is documented, along with how to change it.
- **Idle traffic.** The traffic floor on the HTTP alert keeps it quiet during
  idle periods, where one failed request would read as 100 %.
- **Missing denominators.** The other percentage alerts have no denominator
  when there is no traffic, so they cannot fire then. Their denominators are
  never zero-filled, which follows the doc's absent-data rule.

### Tests

- **The promtool tests.** `deploy/prometheus/agentic-search-alerts.test.yml`
  is a `promtool test rules` file. For **every** alert it has:
  - a firing case, with the expected labels and severity;
  - at least one quiet case.

  The quiet cases include:
  - a healthy ratio;
  - zero traffic;
  - for the HTTP alert, a high percentage below the traffic floor.
- **The pytest wrapper.** `tests/unit/test_prometheus_alert_rules.py`:
  - runs `promtool check rules` and `promtool test rules` when `promtool` is
    on `PATH`, and skips otherwise;
  - also asserts, without promtool, that every alert in the rules file
    appears in the test file, so a new alert cannot ship untested;
  - checks that every metric name the rules reference is one
    `src/internal/observability/prometheus.py` declares, or `up`, so the
    rules cannot drift from the exporter.

### Documentation

`docs/observability-metrics.md` gets an **Alerts** section. It covers:

- the file location;
- how to load it (`rule_files:`);
- the job-label assumption;
- the threshold table;
- how to run the tests locally.

It also states the gap: `/ready` and circuit-breaker state are not Prometheus
metrics yet, so they have no alert.

## Out of scope

- Alertmanager routing.
- Dashboards.
- New metrics, for example readiness or breaker gauges.
- A CI step that installs promtool, noted as a follow-up.

## Mutation checks

- Change a threshold (for example 5xx > 50) and watch its firing test fail.
- Delete an alert's quiet case and watch the coverage assertion fail.
- Reference a made-up metric name and watch the exporter-drift assertion
  fail.
