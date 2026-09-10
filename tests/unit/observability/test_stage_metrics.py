"""Per-request retrieval-vs-generation telemetry: a ContextVar accumulator the
choke points write into, and a rolling per-stage window an admin can read."""

from __future__ import annotations

import asyncio

import pytest

from src.internal.observability import stage_metrics as sm


@pytest.fixture(autouse=True)
def _no_leak():
    yield
    assert sm.current() is None, "a test left a request open"


def test_notes_outside_a_request_are_no_ops():
    sm.note_retrieval(elapsed_ms=5.0, docs=3)
    sm.note_generation(elapsed_ms=50.0, prompt_tokens=10, completion_tokens=5)
    assert sm.current() is None
    assert sm.finish_request(None) is None


def test_request_accumulates_into_separate_buckets():
    token = sm.start_request()
    sm.note_retrieval(elapsed_ms=4.0, docs=5)
    sm.note_retrieval(elapsed_ms=0.1, docs=5, cache_hit=True)
    sm.note_generation(elapsed_ms=120.0, prompt_tokens=300, completion_tokens=40)
    sm.note_generation(elapsed_ms=30.0)  # a call whose backend reports no usage
    metrics = sm.finish_request(token)
    assert metrics is not None
    assert metrics.snapshot() == {
        "retrieval": {"calls": 2, "cache_hits": 1, "ms": 4.1, "docs": 10},
        "generation": {
            "calls": 2,
            "ms": 150.0,
            "prompt_tokens": 300,
            "completion_tokens": 40,
        },
    }
    assert sm.current() is None


def test_request_scope_follows_the_task_context():
    async def _inner():
        token = sm.start_request()
        sm.note_retrieval(elapsed_ms=1.0, docs=1)

        async def _child():
            sm.note_retrieval(elapsed_ms=2.0, docs=2)

        await asyncio.create_task(_child())
        return sm.finish_request(token)

    metrics = asyncio.run(_inner())
    assert metrics.snapshot()["retrieval"] == {
        "calls": 2,
        "cache_hits": 0,
        "ms": 3.0,
        "docs": 3,
    }


def _request(retrieval_ms, docs, generation_ms, prompt, completion, *, hit=False):
    token = sm.start_request()
    sm.note_retrieval(elapsed_ms=retrieval_ms, docs=docs, cache_hit=hit)
    sm.note_generation(
        elapsed_ms=generation_ms, prompt_tokens=prompt, completion_tokens=completion
    )
    return sm.finish_request(token)


def test_stage_latency_stats_percentiles_and_averages():
    stats = sm.StageLatencyStats()
    stats.record(_request(10.0, 4, 100.0, 200, 20))
    stats.record(_request(20.0, 6, 300.0, 400, 60, hit=True))
    stats.record(_request(30.0, 8, 200.0, 300, 40))
    snap = stats.snapshot()
    assert snap["retrieval"] == {
        "count": 3,
        "p50_ms": 20.0,
        "p95_ms": 30.0,
        "max_ms": 30.0,
        "avg_docs": 6.0,
        "cache_hit_rate": pytest.approx(1 / 3),
    }
    assert snap["generation"] == {
        "count": 3,
        "p50_ms": 200.0,
        "p95_ms": 300.0,
        "max_ms": 300.0,
        "avg_prompt_tokens": 300.0,
        "avg_completion_tokens": 40.0,
    }


def test_stage_latency_stats_skips_stages_the_request_never_used():
    stats = sm.StageLatencyStats()
    token = sm.start_request()
    sm.note_retrieval(elapsed_ms=5.0, docs=2)
    stats.record(sm.finish_request(token))
    snap = stats.snapshot()
    assert snap["retrieval"]["count"] == 1
    assert snap["generation"] == {"count": 0}


def test_stage_latency_stats_window_is_bounded():
    stats = sm.StageLatencyStats(max_samples=2)
    for ms in (1.0, 2.0, 3.0):
        stats.record(_request(ms, 1, ms, 1, 1))
    snap = stats.snapshot()["retrieval"]
    assert snap["count"] == 2
    assert snap["p50_ms"] == 2.0  # the 1.0 sample fell out of the window
    assert snap["max_ms"] == 3.0


def test_empty_stats_snapshot():
    assert sm.StageLatencyStats().snapshot() == {
        "retrieval": {"count": 0},
        "generation": {"count": 0},
    }
