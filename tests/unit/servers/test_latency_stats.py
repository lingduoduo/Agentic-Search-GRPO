"""Per-route latency stats and the middleware that feeds them."""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from src.internal.servers.middleware.latency_logging import (
    RouteLatencyStats,
    add_latency_logging_middleware,
)


def _record(stats: RouteLatencyStats, samples, *, route="/x", status_code=200):
    for value in samples:
        stats.record(
            method="GET", route=route, status_code=status_code, elapsed_ms=value
        )


def test_percentiles_over_a_known_sample_set():
    stats = RouteLatencyStats()
    _record(stats, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    row = stats.snapshot()[0]

    assert row["count"] == 10
    assert row["p50_ms"] == 50.0
    assert row["p95_ms"] == 100.0
    assert row["max_ms"] == 100.0


def test_single_sample_reports_itself_at_every_percentile():
    stats = RouteLatencyStats()
    _record(stats, [42.0])

    row = stats.snapshot()[0]

    assert row["p50_ms"] == row["p95_ms"] == row["max_ms"] == 42.0


def test_window_is_bounded_and_keeps_the_newest_samples():
    stats = RouteLatencyStats(max_samples_per_route=3)
    _record(stats, [1000, 1000, 1000, 5, 6, 7])

    row = stats.snapshot()[0]

    assert row["count"] == 3
    assert row["max_ms"] == 7.0


def test_server_errors_are_counted_separately():
    stats = RouteLatencyStats()
    _record(stats, [1, 2], status_code=200)
    _record(stats, [3], status_code=500)
    _record(stats, [4], status_code=404)

    rows = {(r["method"], r["route"], r["count"]): r for r in stats.snapshot()}
    row = next(iter(rows.values()))

    assert row["count"] == 4
    assert row["errors"] == 1


def test_routes_are_sorted_slowest_first():
    stats = RouteLatencyStats()
    _record(stats, [1.0], route="/fast")
    _record(stats, [900.0], route="/slow")

    assert [r["route"] for r in stats.snapshot()] == ["/slow", "/fast"]


def test_snapshot_of_an_empty_store_is_empty():
    assert RouteLatencyStats().snapshot() == []


def test_every_reported_number_is_finite():
    import math

    stats = RouteLatencyStats()
    _record(stats, [1.5, 2.5, 3.5])

    for row in stats.snapshot():
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                assert math.isfinite(value), key


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


@pytest.fixture
def app_with_stats():
    stats = RouteLatencyStats()
    app = FastAPI()

    @app.get("/echo/{item}")
    def echo(item: str) -> dict:
        return {"item": item}

    @app.get("/boom")
    def boom() -> JSONResponse:
        return JSONResponse({"detail": "nope"}, status_code=500)

    add_latency_logging_middleware(
        app, logging.LoggerAdapter(logging.getLogger("test"), {}), stats=stats
    )
    return app, stats


def test_path_parameters_collapse_into_one_route_row(app_with_stats):
    """The whole feature depends on this.

    Keying on the raw path makes every session id its own bucket: thousands of
    one-sample rows whose percentiles all equal that sample, in a panel that
    still looks populated.
    """
    app, stats = app_with_stats
    client = TestClient(app)

    client.get("/echo/alpha")
    client.get("/echo/beta")
    client.get("/echo/gamma")

    rows = stats.snapshot()

    assert len(rows) == 1
    assert rows[0]["route"] == "/echo/{item}"
    assert rows[0]["count"] == 3


def test_unmatched_path_falls_back_to_the_raw_path(app_with_stats):
    app, stats = app_with_stats

    TestClient(app).get("/no-such-route")

    assert [r["route"] for r in stats.snapshot()] == ["/no-such-route"]


def test_middleware_counts_a_server_error(app_with_stats):
    app, stats = app_with_stats

    TestClient(app, raise_server_exceptions=False).get("/boom")

    row = next(r for r in stats.snapshot() if r["route"] == "/boom")
    assert row["errors"] == 1


def test_middleware_returns_the_handlers_response_unchanged(app_with_stats):
    app, _ = app_with_stats

    response = TestClient(app).get("/echo/alpha")

    assert response.status_code == 200
    assert response.json() == {"item": "alpha"}


# ---------------------------------------------------------------------------
# Registration in the web app
# ---------------------------------------------------------------------------


def _middleware_dispatch_names(app) -> set[str]:
    names = set()
    for middleware in app.user_middleware:
        dispatch = middleware.kwargs.get("dispatch") if middleware.kwargs else None
        if dispatch is not None:
            names.add(getattr(dispatch, "__name__", ""))
    return names


def test_web_app_registers_the_latency_middleware(tmp_path):
    """The bug this feature fixes: the middleware existed and nothing called it.

    The other four modules in servers/middleware are registered in
    create_web_app; this one had zero call sites, so the backend measured no
    request latency at all.
    """
    from src.internal.servers.web.app import SearchExperienceSettings, create_web_app

    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "latency.sqlite3"))

    assert "log_latency" in _middleware_dispatch_names(app)


# ---------------------------------------------------------------------------
# /api/debug/latency
# ---------------------------------------------------------------------------


def test_debug_endpoint_reports_the_recorded_routes(monkeypatch):
    import math

    from src.internal.servers.web import debug_router as debug_module

    stats = RouteLatencyStats()
    _record(stats, [10.0, 20.0, 30.0], route="/api/agent")
    monkeypatch.setattr(debug_module, "ROUTE_LATENCY", stats)

    app = FastAPI()
    app.include_router(debug_module.create_debug_router(search_url="http://x/retrieve"))

    payload = TestClient(app).get("/api/debug/latency").json()

    assert [row["route"] for row in payload["routes"]] == ["/api/agent"]
    row = payload["routes"][0]
    assert row["count"] == 3
    for key, value in row.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            assert math.isfinite(value), key


def test_debug_endpoint_is_empty_before_any_request(monkeypatch):
    from src.internal.servers.web import debug_router as debug_module

    from src.internal.observability.stage_metrics import StageLatencyStats

    monkeypatch.setattr(debug_module, "ROUTE_LATENCY", RouteLatencyStats())
    monkeypatch.setattr(debug_module, "STAGE_LATENCY", StageLatencyStats())

    app = FastAPI()
    app.include_router(debug_module.create_debug_router(search_url="http://x/retrieve"))

    assert TestClient(app).get("/api/debug/latency").json() == {
        "routes": [],
        "stages": {"retrieval": {"count": 0}, "generation": {"count": 0}},
    }
