"""Contracts for the retrieval executor benchmark.

The harness is diagnostic: it exists to tell a per-call pool that cannot queue
apart from a shared pool that can. These tests pin that discrimination and the
shape of what it reports, never an absolute duration -- a wall-clock assertion
would be flaky on any shared machine.
"""

from __future__ import annotations

import argparse

from examples.benchmark_retrieval_executors import (
    StubBackend,
    StubReranker,
    _pct,
    run_level,
)

EXPECTED_KEYS = {
    "conc",
    "reqs",
    "rps",
    "p50",
    "p95",
    "p99",
    "wait50",
    "wait99",
    "timeout_pct",
    "threads",
}


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        target="search",
        per_thread=2,
        task_ms=1.0,
        mode="sleep",
        workers=4,
        timeout_ms=500,
        docs=3,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_pct_picks_order_statistics():
    values = [0.005, 0.001, 0.003, 0.002, 0.004]

    assert _pct(values, 0) == 0.001
    assert _pct(values, 50) == 0.003
    assert _pct(values, 100) == 0.005


def test_pct_of_nothing_is_nan():
    assert _pct([], 50) != _pct([], 50)  # NaN is not equal to itself


def test_stub_backend_records_both_legs_per_call():
    backend = StubBackend(task_s=0.0, mode="sleep")

    backend.search_sparse("call-1", top_k=1)
    backend.search_dense("call-1", top_k=1)

    assert set(backend.starts) == {"call-1:sparse", "call-1:dense"}


def test_stub_reranker_truncates_to_top_k():
    reranker = StubReranker(task_s=0.0, mode="sleep")
    docs = ["a", "b", "c"]

    assert reranker.rerank("q", docs, 2) == ["a", "b"]
    assert "q" in reranker.starts


def test_search_target_reports_every_metric_and_never_times_out():
    """The per-call pool is sized to its task count, so nothing can queue."""
    result = run_level(_args(target="search"), concurrency=2)

    assert set(result) == EXPECTED_KEYS
    assert result["conc"] == 2
    assert result["reqs"] == 4  # concurrency * per_thread
    assert result["timeout_pct"] == 0.0


def test_rerank_target_surfaces_queue_driven_timeouts():
    """A task that cannot finish inside the deadline is reported, not swallowed."""
    result = run_level(
        _args(target="rerank", task_ms=200.0, timeout_ms=5, workers=1),
        concurrency=2,
    )

    assert result["timeout_pct"] == 100.0
    assert result["reqs"] == 4


def test_rerank_target_does_not_time_out_with_headroom():
    """The same shape passes cleanly when the deadline is not the binding limit."""
    result = run_level(
        _args(target="rerank", task_ms=1.0, timeout_ms=5000, workers=4),
        concurrency=2,
    )

    assert result["timeout_pct"] == 0.0
