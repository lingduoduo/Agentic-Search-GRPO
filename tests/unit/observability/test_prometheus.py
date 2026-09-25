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
    bucket = {"stage": "retrieval", "le": "0.25"}
    bucket_before = _value("agentic_search_stage_duration_seconds_bucket", bucket)
    observe_stages(
        RequestStageMetrics(
            retrieval_calls=1,
            retrieval_ms=250.0,
            generation_calls=0,
            auxiliary_calls=2,
            auxiliary_ms=40.0,
        )
    )
    assert _stage_count("retrieval") == before["retrieval"] + 1
    assert _stage_count("generation") == before["generation"]
    assert _stage_count("auxiliary") == before["auxiliary"] + 1
    # 250 ms is observed in seconds, so it falls in the 0.25 s bucket.
    assert (
        _value("agentic_search_stage_duration_seconds_bucket", bucket)
        == bucket_before + 1
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
