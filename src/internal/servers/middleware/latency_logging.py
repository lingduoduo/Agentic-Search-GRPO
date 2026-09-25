"""Per-request latency: a debug log line, and a bounded per-route sample store.

`GenerationTimings` and `/api/debug/requests` both answer "where did *this*
request spend its time". This answers the question you ask first: which route is
slow, and how often.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Awaitable
from collections.abc import Callable

from fastapi import FastAPI
from fastapi import Request
from fastapi import Response

from src.internal.observability.prometheus import UNMATCHED_ROUTE
from src.internal.observability.prometheus import observe_request

_DEFAULT_MAX_SAMPLES = 512


def _percentile(ordered: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted, non-empty list."""
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


class RouteLatencyStats:
    """Recent request durations, bucketed by (method, route template).

    Each bucket keeps its most recent ``max_samples_per_route`` durations, so
    the percentiles describe recent behaviour and memory stays bounded by
    routes x samples. That rolling window is why there is no reset: it is what a
    reset would be for. Each sample carries its own 5xx flag, so ``errors`` is
    counted over the same retained samples as ``count`` -- an evicted failure
    leaves both.
    """

    def __init__(self, max_samples_per_route: int = _DEFAULT_MAX_SAMPLES) -> None:
        self._max_samples = max_samples_per_route
        self._samples: dict[tuple[str, str], deque[tuple[float, bool]]] = {}

    def record(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        elapsed_ms: float,
    ) -> None:
        if not math.isfinite(elapsed_ms):
            return
        key = (method, route)
        bucket = self._samples.get(key)
        if bucket is None:
            bucket = deque(maxlen=self._max_samples)
            self._samples[key] = bucket
        bucket.append((float(elapsed_ms), status_code >= 500))

    def snapshot(self) -> list[dict]:
        """One row per route, slowest p95 first. Empty buckets are omitted."""
        rows: list[dict] = []
        for (method, route), bucket in self._samples.items():
            if not bucket:
                continue
            ordered = sorted(elapsed for elapsed, _ in bucket)
            rows.append(
                {
                    "method": method,
                    "route": route,
                    "count": len(ordered),
                    "errors": sum(is_error for _, is_error in bucket),
                    "p50_ms": round(_percentile(ordered, 0.50), 3),
                    "p95_ms": round(_percentile(ordered, 0.95), 3),
                    "max_ms": round(ordered[-1], 3),
                }
            )
        rows.sort(key=lambda row: row["p95_ms"], reverse=True)
        return rows


#: Process-wide store the web app records into, mirroring how `request_capture`
#: holds its state. Tests inject their own instance instead.
ROUTE_LATENCY = RouteLatencyStats()


def _route_template(request: Request) -> str | None:
    """The matched route template, or None when nothing matched.

    Recording ``request.url.path`` would give every session id, request id and
    document id its own bucket -- thousands of single-sample rows whose
    percentiles all equal that one sample, in a panel that still looks
    populated. The JSON window falls back to the raw path when nothing
    matched; the Prometheus label uses ``UNMATCHED_ROUTE`` instead, because a
    Prometheus label set never expires.
    """
    route = request.scope.get("route")
    return getattr(route, "path_format", None) or getattr(route, "path", None)


def add_latency_logging_middleware(
    app: FastAPI,
    logger: logging.LoggerAdapter,
    *,
    stats: RouteLatencyStats | None = None,
) -> None:
    """Log each request's duration and record it under its route template."""

    store = ROUTE_LATENCY if stats is None else stats

    @app.middleware("http")
    async def log_latency(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start_time = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            # The response never reaches this middleware; the server turns the
            # exception into a 500 outside it. Record that 500 here, once, in
            # both the JSON window and Prometheus.
            elapsed = time.monotonic() - start_time
            template = _route_template(request)
            store.record(
                method=request.method,
                route=template or request.url.path,
                status_code=500,
                elapsed_ms=elapsed * 1000.0,
            )
            observe_request(request.method, template or UNMATCHED_ROUTE, 500, elapsed)
            raise
        process_time = time.monotonic() - start_time
        # Read the route *after* the handler: routing populates scope["route"].
        template = _route_template(request)
        store.record(
            method=request.method,
            route=template or request.url.path,
            status_code=response.status_code,
            elapsed_ms=process_time * 1000.0,
        )
        observe_request(
            request.method,
            template or UNMATCHED_ROUTE,
            response.status_code,
            process_time,
        )
        logger.debug(
            "Path: %s - Method: %s - Status Code: %s - Time: %s secs",
            request.url.path,
            request.method,
            response.status_code,
            format(process_time, ".4f"),
        )
        return response
