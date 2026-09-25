# Prometheus Metrics Export and Readiness Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the web app's request and stage latencies in Prometheus format on an opt-in `GET /metrics`, and add a `GET /ready` probe that fails when the store or the retrieval server is unusable.

**Architecture:** A new `src/internal/observability/prometheus.py` owns a dedicated `CollectorRegistry` with one counter and two histograms. The existing latency middleware and the `/api/agent` stage-metrics teardown feed it; `/metrics` renders it only when `AGENTIC_SEARCH_METRICS_ENABLED` is on. A new `src/internal/servers/web/readiness.py` runs a store ping and a retrieval `/health` probe concurrently for `/ready`.

**Tech Stack:** FastAPI, prometheus-client, httpx (`AsyncClient` + `MockTransport` in tests), sqlite3, TOML timeout policies.

**Spec:** `docs/superpowers/specs/2026-09-25-prometheus-metrics-readiness-design.md`

## Global Constraints

- Dependency: `prometheus-client>=0.20,<1` in `requirements.txt` and `requirements-unit-test.txt`.
- Metric names: `agentic_search_http_requests_total` (Counter; `method`, `route`, `status`), `agentic_search_http_request_duration_seconds` (Histogram; `method`, `route`), `agentic_search_stage_duration_seconds` (Histogram; `stage`).
- `route` = matched route template; unmatched = literal `"<unmatched>"`. `status` = status class `2xx`/`4xx`/`5xx`.
- HTTP buckets: prometheus-client default buckets plus `30`, `60`. Stage buckets: `0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120`.
- Dedicated `REGISTRY`, not the global default registry.
- Recording is always on; `AGENTIC_SEARCH_METRICS_ENABLED` (default off) only controls whether `/metrics` is mounted.
- `/metrics` and `/ready` are `GET`-only, in `PUBLIC_ENDPOINT_SPECS` and `LICENSE_ENFORCEMENT_ALLOWED_PREFIXES`.
- `/ready` body: `{"status": "ready"|"not_ready", "checks": {"store": {"ok", "error"}, "retrieval": {"ok", "error"}}}`; 200 or 503; error strings are exception class names only.
- `[readiness] probe_timeout_seconds = 2.0` in `timeouts.toml`.
- `/health` and compose healthchecks unchanged.

## Deviations from the spec (verified against the real code)

- `SearchExperienceSettings.from_app_settings` reads its flags with a local `_flag()` helper, not `get_env_bool`. `get_env_bool` exists in `src/internal/configs/app_configs.py` but raises on unrecognized values; to match the neighbouring flags and never fail startup on a typo'd value, `metrics_enabled` uses the same `_flag()` helper.
- A non-2xx retrieval `/health` response raises no exception, so its error string is `"HTTP <code>"` (no URL, no body) rather than a class name.
- The `method` label is bounded to the standard HTTP methods; anything else is recorded as `"OTHER"` (see Review Focus 1). The spec does not mention it but its stated rationale (Prometheus label sets never expire) requires it: an unauthenticated caller can send `FOO /health`, which matches the route and returns 405.
- The JSON window (`ROUTE_LATENCY`) keeps its current raw-path fallback for unmatched paths; only the Prometheus label uses `"<unmatched>"`, as the spec states.

## Review Focus

1. **Arbitrary HTTP method on a matched route** (`FOOBAR /health` -> 405): must not mint a new label value per method; recorded as `method="OTHER"`. Test in Task 1 (`test_unknown_method_is_bounded`).
2. **Unhandled exception in a handler**: recorded once as `5xx` under the route template and still re-raised (client gets 500). Test in Task 2 (`test_exception_is_counted_as_5xx_and_reraised`).
3. **Retrieval error text containing the URL/credentials** (`ConnectError("... http://user:pw@host ...")`): body must contain only the class name. Test in Task 4 (`test_ready_503_when_retrieval_unreachable_without_leaking_url`).
4. **Store closed / broken** (a real closed sqlite connection, not only a mock): `/ready` 503 with `store.error == "ProgrammingError"`. Test in Task 4 (`test_ready_503_when_store_is_closed`).
5. **Retrieval URL with a query string or trailing path** (`http://h:8001/retrieve?x=1`): probe goes to `http://h:8001/health`, not `/retrieve/health` or `/health?x=1`. Test in Task 4 (`test_retrieval_health_url_replaces_path_and_drops_query`).

---

### Task 1: Dependency and `prometheus.py` module

**Files:**
- Modify: `requirements.txt`, `requirements-unit-test.txt`
- Create: `src/internal/observability/prometheus.py`
- Test: `tests/unit/observability/test_prometheus.py`

**Interfaces:**
- Produces: `REGISTRY: CollectorRegistry`, `UNMATCHED_ROUTE = "<unmatched>"`, `observe_request(method: str, route: str, status_code: int, seconds: float) -> None`, `observe_stages(metrics: RequestStageMetrics | None) -> None`, `render_latest() -> tuple[bytes, str]`.

- [ ] **Step 1: Write the failing tests**

```python
"""Prometheus export of request and stage latencies."""

from __future__ import annotations

from src.internal.observability.prometheus import (
    REGISTRY,
    UNMATCHED_ROUTE,
    observe_request,
    observe_stages,
    render_latest,
)
from src.internal.observability.stage_metrics import RequestStageMetrics


def _value(name: str, labels: dict[str, str]) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def _requests(method: str, route: str, status: str) -> float:
    return _value(
        "agentic_search_http_requests_total",
        {"method": method, "route": route, "status": status},
    )


def _stage_count(stage: str) -> float:
    return _value("agentic_search_stage_duration_seconds_count", {"stage": stage})


def test_observe_request_counts_under_route_template_and_status_class():
    before = _requests("GET", "/api/sessions/{session_id}", "4xx")
    observe_request("GET", "/api/sessions/{session_id}", 404, 0.01)
    assert _requests("GET", "/api/sessions/{session_id}", "4xx") == before + 1


def test_observe_request_records_duration_histogram():
    labels = {"method": "POST", "route": "/t/dur"}
    before = _value("agentic_search_http_request_duration_seconds_count", labels)
    observe_request("POST", "/t/dur", 200, 45.0)
    assert (
        _value("agentic_search_http_request_duration_seconds_count", labels)
        == before + 1
    )
    # 45 s lands in the extra 60 s bucket, not only +Inf.
    assert (
        _value(
            "agentic_search_http_request_duration_seconds_bucket",
            {**labels, "le": "60.0"},
        )
        == before + 1
    )


def test_unmatched_route_label_is_the_literal():
    before = _requests("GET", UNMATCHED_ROUTE, "4xx")
    observe_request("GET", UNMATCHED_ROUTE, 404, 0.001)
    assert UNMATCHED_ROUTE == "<unmatched>"
    assert _requests("GET", "<unmatched>", "4xx") == before + 1


def test_unknown_method_is_bounded():
    before = _requests("OTHER", "/t/method", "4xx")
    observe_request("FOOBAR", "/t/method", 405, 0.001)
    assert _requests("OTHER", "/t/method", "4xx") == before + 1
    assert _requests("FOOBAR", "/t/method", "4xx") == 0.0


def test_non_finite_duration_is_ignored():
    before = _requests("GET", "/t/nan", "2xx")
    observe_request("GET", "/t/nan", 200, float("nan"))
    assert _requests("GET", "/t/nan", "2xx") == before


def test_observe_stages_observes_only_stages_the_request_used():
    before = {s: _stage_count(s) for s in ("retrieval", "generation", "auxiliary")}
    observe_stages(
        RequestStageMetrics(
            retrieval_calls=1, retrieval_ms=250.0, generation_calls=0, auxiliary_calls=2,
            auxiliary_ms=40.0,
        )
    )
    assert _stage_count("retrieval") == before["retrieval"] + 1
    assert _stage_count("generation") == before["generation"]
    assert _stage_count("auxiliary") == before["auxiliary"] + 1
    assert (
        _value("agentic_search_stage_duration_seconds_bucket",
               {"stage": "retrieval", "le": "0.25"})
        >= 1
    )


def test_observe_stages_accepts_none():
    observe_stages(None)


def test_render_latest_is_text_exposition_with_metric_names():
    observe_request("GET", "/t/render", 200, 0.002)
    body, content_type = render_latest()
    assert content_type.startswith("text/plain")
    text = body.decode()
    for name in (
        "agentic_search_http_requests_total",
        "agentic_search_http_request_duration_seconds",
        "agentic_search_stage_duration_seconds",
    ):
        assert name in text
    # The dedicated registry carries no default process/python collectors.
    assert "python_gc_objects_collected_total" not in text
```

(ruff format reflows the `RequestStageMetrics(...)` call; that is fine.)

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. python -m pytest tests/unit/observability/test_prometheus.py -q -p no:cacheprovider`
Expected: collection error, `ModuleNotFoundError: No module named 'src.internal.observability.prometheus'`.

- [ ] **Step 3: Implement**

`requirements.txt` — after the `httpx` line:
```
# /metrics exposition (src/internal/observability/prometheus.py). Pure Python.
prometheus-client>=0.20,<1
```
`requirements-unit-test.txt` — after the `httpx` line:
```
prometheus-client>=0.20,<1
```

`src/internal/observability/prometheus.py`:
```python
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
            _STAGE_SECONDS.labels(stage).observe(getattr(metrics, f"{stage}_ms") / 1000.0)


def render_latest() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
```

Note: `render_latest` must name all three families even before any observation. A labelled metric with no children emits only `# HELP`/`# TYPE` lines, which contain the names, so the test holds.

- [ ] **Step 4: Run to verify pass** — same command, expected all PASS.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt requirements-unit-test.txt src/internal/observability/prometheus.py tests/unit/observability/test_prometheus.py
git commit -m "Add Prometheus request and stage metrics on a dedicated registry"
```

---

### Task 2: Feed the metrics from the latency middleware and the stage teardown

**Files:**
- Modify: `src/internal/servers/middleware/latency_logging.py` (`_route_key`, `log_latency`)
- Modify: `src/internal/servers/web/app.py:~2124` (`STAGE_LATENCY.record(...)`)
- Test: `tests/unit/servers/test_latency_stats.py` (append), `tests/unit/servers/web/test_web_experience_app.py` (extend `test_agent_endpoint_persists_stage_metrics_apart_from_pipeline_stages`)

**Interfaces:**
- Consumes: `observe_request`, `observe_stages`, `UNMATCHED_ROUTE`, `REGISTRY` from Task 1.
- Produces: `_route_template(request) -> str | None` in `latency_logging.py`.

- [ ] **Step 1: Write the failing tests** — append to `tests/unit/servers/test_latency_stats.py`:

```python
from src.internal.observability.prometheus import REGISTRY as PROM_REGISTRY


def _prom_requests(method: str, route: str, status: str) -> float:
    return (
        PROM_REGISTRY.get_sample_value(
            "agentic_search_http_requests_total",
            {"method": method, "route": route, "status": status},
        )
        or 0.0
    )


def _prom_app() -> FastAPI:
    app = FastAPI()
    add_latency_logging_middleware(
        app, logging.LoggerAdapter(logging.getLogger("t"), {}), stats=RouteLatencyStats()
    )

    @app.get("/prom/items/{item_id}")
    def item(item_id: str):
        return {"id": item_id}

    @app.get("/prom/boom")
    def boom():
        raise RuntimeError("kaboom")

    return app


def test_middleware_counts_requests_under_the_route_template():
    before = _prom_requests("GET", "/prom/items/{item_id}", "2xx")
    client = TestClient(_prom_app())
    client.get("/prom/items/a")
    client.get("/prom/items/b")
    assert _prom_requests("GET", "/prom/items/{item_id}", "2xx") == before + 2


def test_middleware_records_unmatched_paths_as_one_label():
    before = _prom_requests("GET", "<unmatched>", "4xx")
    client = TestClient(_prom_app())
    client.get("/prom/nope/1")
    client.get("/prom/nope/2")
    assert _prom_requests("GET", "<unmatched>", "4xx") == before + 2
    assert _prom_requests("GET", "/prom/nope/1", "4xx") == 0.0


def test_exception_is_counted_as_5xx_and_reraised():
    before = _prom_requests("GET", "/prom/boom", "5xx")
    client = TestClient(_prom_app(), raise_server_exceptions=False)
    assert client.get("/prom/boom").status_code == 500
    assert _prom_requests("GET", "/prom/boom", "5xx") == before + 1
    with pytest.raises(RuntimeError, match="kaboom"):
        TestClient(_prom_app()).get("/prom/boom")
```

Extend `test_agent_endpoint_persists_stage_metrics_apart_from_pipeline_stages` in `tests/unit/servers/web/test_web_experience_app.py`: before `response = TestClient(app).post(...)` add

```python
    from src.internal.observability.prometheus import REGISTRY as PROM

    def _stage_count(stage):
        return (
            PROM.get_sample_value(
                "agentic_search_stage_duration_seconds_count", {"stage": stage}
            )
            or 0.0
        )

    stage_before = {s: _stage_count(s) for s in ("retrieval", "generation", "auxiliary")}
```
and after the existing `snap` assertions:
```python
    for stage in ("retrieval", "generation", "auxiliary"):
        assert _stage_count(stage) == stage_before[stage] + 1
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. python -m pytest tests/unit/servers/test_latency_stats.py tests/unit/servers/web/test_web_experience_app.py -q -p no:cacheprovider -k "prom or middleware or exception or stage_metrics_apart"`
Expected: the new tests FAIL with `assert 0.0 == 2` style mismatches (nothing records yet); `test_exception_is_counted_as_5xx_and_reraised` fails on the counter assertion.

- [ ] **Step 3: Implement**

In `latency_logging.py`, import:
```python
from src.internal.observability.prometheus import UNMATCHED_ROUTE, observe_request
```
Split the route lookup:
```python
def _route_template(request: Request) -> str | None:
    """The matched route template, or None when nothing matched."""
    route = request.scope.get("route")
    return getattr(route, "path_format", None) or getattr(route, "path", None)


def _route_key(request: Request) -> str:
    """The matched route template, or the raw path when nothing matched.
    ...(existing docstring body kept)...
    """
    return _route_template(request) or request.url.path
```
Replace the middleware body:
```python
        start_time = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            # Recorded nowhere else: the JSON window only sees responses.
            observe_request(
                request.method,
                _route_template(request) or UNMATCHED_ROUTE,
                500,
                time.monotonic() - start_time,
            )
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
        # Stricter than the window: a Prometheus label set never expires.
        observe_request(
            request.method,
            template or UNMATCHED_ROUTE,
            response.status_code,
            process_time,
        )
```
(`_route_key` stays for its docstring and any callers; if it has no remaining callers after the split, delete it and keep its docstring rationale on `_route_template`'s caller.)

In `app.py` add import `from src.internal.observability.prometheus import observe_stages` beside the `STAGE_LATENCY` import, and replace
```python
            STAGE_LATENCY.record(_stage_metrics.finish_request(stage_token))
```
with
```python
            finished_stages = _stage_metrics.finish_request(stage_token)
            STAGE_LATENCY.record(finished_stages)
            observe_stages(finished_stages)
```

- [ ] **Step 4: Run to verify pass** — same command, plus the whole `tests/unit/servers/test_latency_stats.py`. Expected PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/servers/middleware/latency_logging.py src/internal/servers/web/app.py tests/unit/servers/test_latency_stats.py tests/unit/servers/web/test_web_experience_app.py
git commit -m "Record every request and agent stage into the Prometheus registry"
```

---

### Task 3: Opt-in, unauthenticated `GET /metrics`

**Files:**
- Modify: `src/internal/servers/web/app.py` (`SearchExperienceSettings`, `create_web_app` after `/health`)
- Modify: `src/internal/servers/web/auth_check.py` (`PUBLIC_ENDPOINT_SPECS`)
- Modify: `src/internal/configs/license_enforcement_config.py` (`LICENSE_ENFORCEMENT_ALLOWED_PREFIXES`)
- Modify: `docs/configuration.md` (Application and authentication table)
- Test: `tests/unit/servers/web/test_metrics_ready.py`

**Interfaces:**
- Consumes: `render_latest()` from Task 1.
- Produces: `SearchExperienceSettings.metrics_enabled: bool = False`.

- [ ] **Step 1: Write the failing tests** — create `tests/unit/servers/web/test_metrics_ready.py`:

```python
"""GET /metrics (opt-in Prometheus exposition) and GET /ready (readiness)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.internal.configs import AppSettings, AuthSettings
from src.internal.configs.license_enforcement_config import (
    is_license_enforcement_exempt,
)
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


def _app(tmp_path, **settings):
    return create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3", **settings),
        app_settings=AppSettings(auth=AuthSettings()),
    )


def test_metrics_is_404_when_the_flag_is_off(tmp_path):
    assert TestClient(_app(tmp_path)).get("/metrics").status_code == 404


def test_metrics_flag_defaults_off_and_reads_the_env(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_METRICS_ENABLED", raising=False)
    assert SearchExperienceSettings.from_app_settings().metrics_enabled is False
    monkeypatch.setenv("AGENTIC_SEARCH_METRICS_ENABLED", "true")
    assert SearchExperienceSettings.from_app_settings().metrics_enabled is True


def test_metrics_is_public_and_counts_health(tmp_path):
    app = _app(tmp_path, metrics_enabled=True)
    # `with` runs the lifespan, so check_router_auth audits /metrics too.
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert (
        'agentic_search_http_requests_total{method="GET",route="/health",status="2xx"}'
        in response.text
    )


def test_metrics_and_ready_are_license_exempt():
    assert is_license_enforcement_exempt("/metrics")
    assert is_license_enforcement_exempt("/ready")
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. python -m pytest tests/unit/servers/web/test_metrics_ready.py -q -p no:cacheprovider`
Expected: `TypeError: ... unexpected keyword argument 'metrics_enabled'` / `AttributeError`; the flag-off 404 test passes already (acceptable: it pins the default). License test fails.

- [ ] **Step 3: Implement**

`SearchExperienceSettings` — after `search_cache_ttl`:
```python
    # Mount the unauthenticated Prometheus GET /metrics. Recording is always on;
    # this only controls exposure. Restrict the route at the network layer.
    metrics_enabled: bool = False
```
and in `from_app_settings`: `metrics_enabled=_flag("AGENTIC_SEARCH_METRICS_ENABLED"),`.

In `create_web_app`, import `render_latest` next to `observe_stages`, and after the `/health` route:
```python
    if settings.metrics_enabled:

        @app.get("/metrics")
        def metrics() -> Response:
            body, content_type = render_latest()
            return Response(body, media_type=content_type)
```

`auth_check.py`: change the `/health` group to
```python
    # Health, readiness and (opt-in, network-restricted) metrics probes, and the
    # signed-out application shell/assets.
    ("/health", {"GET"}),
    ("/ready", {"GET"}),
    ("/metrics", {"GET"}),
```
`license_enforcement_config.py`: add `"/ready",` and `"/metrics",` after `"/health",`.

`docs/configuration.md` Application table, after the `AGENTIC_SEARCH_WEB_DB_PATH` row:
```
| `AGENTIC_SEARCH_METRICS_ENABLED` | `false` | Mount `GET /metrics`, the Prometheus exposition of `agentic_search_http_requests_total`, `agentic_search_http_request_duration_seconds` and `agentic_search_stage_duration_seconds`. The route is **unauthenticated**: restrict it at the network layer (ingress rule, internal-only port). Recording is always on; the flag only controls exposure. Single-process only (no `PROMETHEUS_MULTIPROC_DIR`); the image runs one uvicorn worker. `GET /ready` (always mounted, also public) returns 503 when the store or the retrieval server's `/health` fails; compose healthchecks keep using the liveness `GET /health` |
```

- [ ] **Step 4: Run to verify pass** — the new file plus `tests/unit/servers/web/test_route_auth_enforcement.py tests/unit/test_documented_env_vars.py`. Expected PASS (the license test for `/ready` passes here too since the prefix is added in this task).

- [ ] **Step 5: Commit**

```bash
git add src/internal/servers/web/app.py src/internal/servers/web/auth_check.py src/internal/configs/license_enforcement_config.py docs/configuration.md tests/unit/servers/web/test_metrics_ready.py
git commit -m "Mount an opt-in unauthenticated GET /metrics behind AGENTIC_SEARCH_METRICS_ENABLED"
```

---

### Task 4: `GET /ready` readiness probe

**Files:**
- Modify: `src/internal/configs/timeouts.toml`, `src/internal/configs/timeouts.py`, `docs/configuration/timeouts.md`
- Modify: `src/internal/db/store.py` (add `ping()`)
- Create: `src/internal/servers/web/readiness.py`
- Modify: `src/internal/servers/web/app.py` (`/ready` route)
- Test: `tests/unit/test_timeout_policies.py` (`TODAY`), `tests/unit/test_timeout_policy_drift.py` (`MIGRATED`), `tests/unit/servers/web/test_metrics_ready.py` (append)

**Interfaces:**
- Consumes: `get_timeout_policies().readiness.probe_timeout_seconds`, `AgenticSearchStore.ping()`.
- Produces: `readiness.retrieval_health_url(search_url: str) -> str`, `readiness.check_readiness(db, search_url: str) -> tuple[bool, dict[str, dict[str, object]]]`, `readiness._retrieval_client(timeout: float) -> httpx.AsyncClient` (test seam).

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_timeout_policies.py` — add to `TODAY`: `"readiness": {"probe_timeout_seconds": 2.0},`.
`tests/unit/test_timeout_policy_drift.py` — add `"src/internal/servers/web/readiness.py",` to `MIGRATED`.

Append to `tests/unit/servers/web/test_metrics_ready.py`:
```python
import httpx
import pytest

from src.internal.configs.timeouts import load_timeout_policies, use_timeout_policies
from src.internal.db import AgenticSearchStore
from src.internal.servers.web import readiness


def _stub_retrieval(monkeypatch, handler, seen_timeouts=None):
    def factory(timeout):
        if seen_timeouts is not None:
            seen_timeouts.append(timeout)
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=timeout)

    monkeypatch.setattr(readiness, "_retrieval_client", factory)


def _ok(request):
    assert request.url.path == "/health"
    return httpx.Response(200, json={"status": "ok"})


def _ready_app(tmp_path, store=None):
    return create_web_app(
        SearchExperienceSettings(
            db_path=tmp_path / "state.sqlite3",
            search_url="http://retrieval.internal:8001/retrieve",
        ),
        app_settings=AppSettings(auth=AuthSettings()),
        store=store,
    )


@pytest.mark.parametrize(
    ("search_url", "expected"),
    [
        ("http://localhost:8001/retrieve", "http://localhost:8001/health"),
        ("http://h:8001/retrieve?x=1#frag", "http://h:8001/health"),
        ("https://h/api/v1/retrieve", "https://h/health"),
    ],
)
def test_retrieval_health_url_replaces_path_and_drops_query(search_url, expected):
    assert readiness.retrieval_health_url(search_url) == expected


def test_ready_200_when_store_and_retrieval_are_fine(tmp_path, monkeypatch):
    _stub_retrieval(monkeypatch, _ok)
    response = TestClient(_ready_app(tmp_path)).get("/ready")  # no auth header
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {
            "store": {"ok": True, "error": None},
            "retrieval": {"ok": True, "error": None},
        },
    }


def test_ready_503_when_retrieval_unreachable_without_leaking_url(
    tmp_path, monkeypatch
):
    def refuse(request):
        raise httpx.ConnectError(f"refused {request.url} user:secret", request=request)

    _stub_retrieval(monkeypatch, refuse)
    response = TestClient(_ready_app(tmp_path)).get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["retrieval"] == {"ok": False, "error": "ConnectError"}
    assert body["checks"]["store"]["ok"] is True
    assert "retrieval.internal" not in response.text
    assert "secret" not in response.text


def test_ready_503_when_retrieval_health_is_not_2xx(tmp_path, monkeypatch):
    _stub_retrieval(monkeypatch, lambda request: httpx.Response(503, text="down"))
    response = TestClient(_ready_app(tmp_path)).get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["retrieval"] == {"ok": False, "error": "HTTP 503"}


def test_ready_503_when_store_ping_raises(tmp_path, monkeypatch):
    _stub_retrieval(monkeypatch, _ok)
    store = AgenticSearchStore(tmp_path / "s.sqlite3")

    def broken():
        raise RuntimeError("disk gone")

    monkeypatch.setattr(store, "ping", broken)
    response = TestClient(_ready_app(tmp_path, store=store)).get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["store"] == {"ok": False, "error": "RuntimeError"}
    store.close()


def test_ready_503_when_store_is_closed(tmp_path, monkeypatch):
    _stub_retrieval(monkeypatch, _ok)
    store = AgenticSearchStore(tmp_path / "s.sqlite3")
    app = _ready_app(tmp_path, store=store)
    store.close()
    response = TestClient(app).get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["store"] == {
        "ok": False,
        "error": "ProgrammingError",
    }


def test_store_ping_succeeds_on_an_open_store(tmp_path):
    store = AgenticSearchStore(tmp_path / "s.sqlite3")
    assert store.ping() is None
    store.close()


def test_retrieval_probe_timeout_follows_the_policy(tmp_path, monkeypatch):
    seen: list[float] = []
    _stub_retrieval(monkeypatch, _ok, seen)
    policies = load_timeout_policies({}, overrides={"readiness": {"probe_timeout_seconds": 0.7}})
    with use_timeout_policies(policies):
        assert TestClient(_ready_app(tmp_path)).get("/ready").status_code == 200
    assert seen == [0.7]


def test_ready_passes_the_route_auth_audit(tmp_path, monkeypatch):
    _stub_retrieval(monkeypatch, _ok)
    with TestClient(_ready_app(tmp_path)) as client:  # lifespan runs check_router_auth
        assert client.get("/ready").status_code == 200
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. python -m pytest tests/unit/servers/web/test_metrics_ready.py tests/unit/test_timeout_policies.py tests/unit/test_timeout_policy_drift.py -q -p no:cacheprovider`
Expected: `ImportError: cannot import name 'readiness'` for the web file; `test_bundled_defaults_are_todays_values` fails (no `readiness` key); drift test for `readiness.py` fails with `FileNotFoundError`.

- [ ] **Step 3: Implement**

`timeouts.toml` — append:
```toml

[readiness]
probe_timeout_seconds = 2.0        # /ready's retrieval /health probe
```

`timeouts.py` — add before `TimeoutPolicies`:
```python
@dataclass(frozen=True)
class ReadinessPolicy:
    probe_timeout_seconds: float
```
and field `readiness: ReadinessPolicy` at the end of `TimeoutPolicies`.

`docs/configuration/timeouts.md` — append:
```markdown

### `[readiness]`

| key | default | bounds |
|---|---|---|
| `probe_timeout_seconds` | 2.0 | `GET /ready`'s probe of the retrieval server's `/health` |
```

`store.py` — after `__exit__`:
```python
    def ping(self) -> None:
        """Raise unless the connection can run a query (readiness probe)."""
        self._conn.execute("SELECT 1").fetchone()
```

`src/internal/servers/web/readiness.py`:
```python
"""``GET /ready``: can this process answer a query right now?

``/health`` is liveness and always says ok. Readiness checks the two things a
query cannot do without -- the store and the retrieval server -- concurrently,
uncached. Errors are reported as exception class names only, because the route
is unauthenticated and an exception message can carry a URL or credentials.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlsplit, urlunsplit

import httpx

from src.internal.configs import get_timeout_policies
from src.internal.db import AgenticSearchStore

Check = dict[str, object]


def retrieval_health_url(search_url: str) -> str:
    """``http://h:8001/retrieve?x=1`` -> ``http://h:8001/health``."""
    parts = urlsplit(search_url)
    return urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))


def _retrieval_client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout)


def _passed() -> Check:
    return {"ok": True, "error": None}


def _failed(error: str) -> Check:
    return {"ok": False, "error": error}


async def _check_store(db: AgenticSearchStore) -> Check:
    try:
        # A worker thread: the store lock may be held by a request thread.
        await asyncio.to_thread(db.ping)
    except Exception as exc:
        return _failed(type(exc).__name__)
    return _passed()


async def _check_retrieval(search_url: str) -> Check:
    timeout = get_timeout_policies().readiness.probe_timeout_seconds
    try:
        async with _retrieval_client(timeout) as client:
            response = await client.get(retrieval_health_url(search_url))
    except Exception as exc:
        return _failed(type(exc).__name__)
    if not response.is_success:
        return _failed(f"HTTP {response.status_code}")
    return _passed()


async def check_readiness(
    db: AgenticSearchStore, search_url: str
) -> tuple[bool, dict[str, Check]]:
    store, retrieval = await asyncio.gather(
        _check_store(db), _check_retrieval(search_url)
    )
    checks = {"store": store, "retrieval": retrieval}
    return all(check["ok"] for check in checks.values()), checks
```

`app.py` — import `from src.internal.servers.web.readiness import check_readiness` and add `JSONResponse` to the `fastapi.responses` import; after `/health`:
```python
    @app.get("/ready")
    async def readiness_probe() -> JSONResponse:
        ready, checks = await check_readiness(db, settings.search_url)
        return JSONResponse(
            {"status": "ready" if ready else "not_ready", "checks": checks},
            status_code=200 if ready else 503,
        )
```

- [ ] **Step 4: Run to verify pass** — same command plus `tests/unit/test_timeout_policy_sites.py tests/unit/servers/web/test_route_auth_enforcement.py`. Expected PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/configs/timeouts.toml src/internal/configs/timeouts.py docs/configuration/timeouts.md src/internal/db/store.py src/internal/servers/web/readiness.py src/internal/servers/web/app.py tests/unit/test_timeout_policies.py tests/unit/test_timeout_policy_drift.py tests/unit/servers/web/test_metrics_ready.py
git commit -m "Add GET /ready: concurrent store ping and retrieval /health probe"
```

---

### Task 5: Mutation checks and final verification

- [ ] **Mutation 1:** delete the success-path `observe_request(...)` call in `latency_logging.py`; run `tests/unit/servers/test_latency_stats.py tests/unit/servers/web/test_metrics_ready.py`; expect `test_middleware_counts_requests_under_the_route_template`, `test_middleware_records_unmatched_paths_as_one_label`, `test_metrics_is_public_and_counts_health` RED. Restore with `git checkout -- <file>`, then `find src tests -name __pycache__ -type d -exec rm -rf {} +`, `git diff --stat` must be empty for that file.
- [ ] **Mutation 2:** make `_check_retrieval` `return _passed()` first; expect `test_ready_503_when_retrieval_unreachable_without_leaking_url` and `test_ready_503_when_retrieval_health_is_not_2xx` RED. Restore, clear `__pycache__`.
- [ ] **Full suite:** `PYTHONPATH=. python -m pytest -q -p no:cacheprovider` — no failures.
- [ ] **Lint:** `ruff check . && ruff format --check .` — clean. Commit any formatting fix.
