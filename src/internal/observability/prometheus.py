"""Prometheus export of the web app's request and stage latencies.

``ROUTE_LATENCY`` and ``STAGE_LATENCY`` are rolling windows for humans; they
have no cumulative counters, so an alerting system can derive neither a rate
nor an error ratio from them. This module records the same events into
cumulative Prometheus metrics on a dedicated registry, rendered by the opt-in
``GET /metrics``.

Label values are bounded on purpose: a Prometheus label set never expires, so
``route`` is the matched template (or ``<unmatched>``), ``status`` the status
class, and ``method`` a standard method or ``OTHER``.
"""

from __future__ import annotations

import math

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily

from src.internal.observability.stage_metrics import STAGES, RequestStageMetrics

REGISTRY = CollectorRegistry()
UNMATCHED_ROUTE = "<unmatched>"

_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})
# Agent requests are long, so the default 5 ms..10 s buckets get 30 s and 60 s.
_HTTP_BUCKETS = (*Histogram.DEFAULT_BUCKETS[:-1], 30.0, 60.0)
_STAGE_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)

_REQUESTS = Counter(
    "agentic_search_http_requests_total",
    "HTTP requests by method, route template and status class.",
    ("method", "route", "status"),
    registry=REGISTRY,
)
_REQUEST_SECONDS = Histogram(
    "agentic_search_http_request_duration_seconds",
    "HTTP request duration by method and route template.",
    ("method", "route"),
    buckets=_HTTP_BUCKETS,
    registry=REGISTRY,
)
_STAGE_SECONDS = Histogram(
    "agentic_search_stage_duration_seconds",
    "Per-request time spent in each stage the request used.",
    ("stage",),
    buckets=_STAGE_BUCKETS,
    registry=REGISTRY,
)


def observe_request(method: str, route: str, status_code: int, seconds: float) -> None:
    if not math.isfinite(seconds):
        return
    method = method if method in _METHODS else "OTHER"
    _REQUESTS.labels(method, route, f"{status_code // 100}xx").inc()
    _REQUEST_SECONDS.labels(method, route).observe(seconds)


def observe_stages(metrics: RequestStageMetrics | None) -> None:
    """One observation per stage the finished request used."""
    if metrics is None:
        return
    for stage in STAGES:
        if getattr(metrics, f"{stage}_calls"):
            seconds = getattr(metrics, f"{stage}_ms") / 1000.0
            _STAGE_SECONDS.labels(stage).observe(seconds)


class _BreakerStateCollector:
    """Circuit-breaker state, read from the breaker registry at scrape time.

    Labelled by breaker *family* (the name before the first ``:``) because
    per-server ``remote_llm:<base_url>`` breakers are named by URL and labels
    never carry URLs; a family reports its worst instance.
    """

    def collect(self):
        # Imported here: the breaker module reads timeout policy, and this
        # module is imported by low-level retrieval code.
        from src.internal.resilience.circuit_breaker import breaker_snapshots

        worst: dict[str, tuple[int, int]] = {}
        for snap in breaker_snapshots():
            family = snap.name.split(":", 1)[0]
            is_open = 1 if snap.state in ("open", "half_open") else 0
            prev_open, prev_failures = worst.get(family, (0, 0))
            worst[family] = (
                max(prev_open, is_open),
                max(prev_failures, snap.consecutive_failures),
            )
        open_family = GaugeMetricFamily(
            "agentic_search_circuit_breaker_open",
            "1 while a breaker of this family is open or half-open (worst instance).",
            labels=["breaker"],
        )
        failures_family = GaugeMetricFamily(
            "agentic_search_circuit_breaker_consecutive_failures",
            "Consecutive failures of this breaker family (worst instance).",
            labels=["breaker"],
        )
        for family, (is_open, failures) in sorted(worst.items()):
            open_family.add_metric([family], is_open)
            failures_family.add_metric([family], failures)
        yield open_family
        yield failures_family


REGISTRY.register(_BreakerStateCollector())


def render_latest() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


_AGENT_RUNS = Counter(
    "agentic_search_agent_runs_total",
    "Terminated search and tool agent runs by outcome.",
    ("agent", "outcome"),
    registry=REGISTRY,
)
_AGENT_ROUNDS = Histogram(
    "agentic_search_agent_decision_rounds",
    "Attempted model generations per terminated agent run.",
    ("agent", "outcome"),
    buckets=(0, 1, 2, 3, 5, 8, 13, 21, 34, 55),
    registry=REGISTRY,
)
_TOOL_ATTEMPTS = Counter(
    "agentic_search_tool_attempts_total",
    "Validated registry execution lifecycles by terminal outcome.",
    ("outcome",),
    registry=REGISTRY,
)


def observe_agent_run(agent: str, outcome: str, rounds: int) -> None:
    """Record once when a run terminates, including errors and cancellation."""
    if agent not in {"search", "tool"} or outcome not in {
        "completed",
        "error",
        "cancelled",
    }:
        raise ValueError("Unknown agent metric label")
    if type(rounds) is not int or rounds < 0:
        raise ValueError("Decision rounds must be a nonnegative integer")
    _AGENT_RUNS.labels(agent, outcome).inc()
    _AGENT_ROUNDS.labels(agent, outcome).observe(rounds)


def observe_tool_attempt(outcome: str) -> None:
    """Count an execution lifecycle, excluding lookup/validation failures."""
    if outcome not in {"success", "timeout", "error", "cancelled"}:
        raise ValueError("Unknown tool outcome")
    _TOOL_ATTEMPTS.labels(outcome).inc()


_STALE_SERVES = Counter(
    "agentic_search_stale_cache_serves_total",
    "Failed live lookups answered from a stale serving-cache entry, by source.",
    ("source",),
    registry=REGISTRY,
)


def observe_stale_serve(source: str) -> None:
    """Count one fallback that served stale rows (once per call, not per row)."""
    if source not in {"web", "retrieval"}:
        raise ValueError("Unknown stale-serve source")
    _STALE_SERVES.labels(source).inc()
