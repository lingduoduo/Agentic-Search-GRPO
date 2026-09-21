from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import time

import pytest

from src.internal.retrieval.async_reranker import (
    AsyncReranker,
    RerankerOverloaded,
    RerankerTimeoutError,
)
from src.internal.retrieval.backends.base import RetrievalResult


def _result(doc_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="t", text="c", url=None, score=score)


def _make_base(return_val=None):
    base = MagicMock()
    base.rerank.return_value = return_val or [_result("d1")]
    return base


def test_sync_rerank_returns_results():
    base = _make_base([_result("d1"), _result("d2")])
    ar = AsyncReranker(base, timeout_ms=1000)
    results = ar.rerank("query", [_result("d1"), _result("d2")], top_k=2)
    assert [r.doc_id for r in results] == ["d1", "d2"]


def test_sync_rerank_delegates_to_base():
    base = _make_base()
    ar = AsyncReranker(base, timeout_ms=1000)
    ar.rerank("q", [_result("x")], top_k=1)
    base.rerank.assert_called_once_with("q", [_result("x")], 1)


def test_sync_rerank_timeout_raises():
    import time as _time

    base = MagicMock()

    def slow(*_):
        _time.sleep(0.3)
        return [_result("d1")]

    base.rerank.side_effect = slow
    ar = AsyncReranker(base, timeout_ms=50)
    with pytest.raises(RerankerTimeoutError):
        ar.rerank("q", [_result("d1")], top_k=1)


def test_async_rerank_returns_results():
    base = _make_base([_result("d1")])
    ar = AsyncReranker(base, timeout_ms=1000)
    results = asyncio.run(ar.arerank("q", [_result("d1")], top_k=1))
    assert results[0].doc_id == "d1"


def test_async_rerank_timeout_raises():
    import time as _time

    base = MagicMock()

    def slow(*_):
        _time.sleep(0.3)
        return [_result("d1")]

    base.rerank.side_effect = slow
    ar = AsyncReranker(base, timeout_ms=50)
    with pytest.raises(RerankerTimeoutError):
        asyncio.run(ar.arerank("q", [_result("d1")], top_k=1))


def test_from_env_reads_timeout(monkeypatch):
    monkeypatch.setenv("RERANKER_TIMEOUT_MS", "250")
    base = _make_base()
    ar = AsyncReranker.from_env(base)
    assert ar._timeout_ms == 250


# -- admission control -------------------------------------------------------


class _Scorer:
    """A reranker whose cost the test controls."""

    def __init__(self, seconds: float = 0.0) -> None:
        self.seconds = seconds
        self.calls = 0

    def rerank(self, query, results, top_k):
        self.calls += 1
        if self.seconds:
            time.sleep(self.seconds)
        return results[:top_k]


def test_the_first_request_is_admitted_before_anything_is_known():
    """With no measurement yet, refusing would be a guess."""
    reranker = AsyncReranker(_Scorer(), timeout_ms=500, max_workers=4)

    assert reranker.rerank("q", [], 5) == []
    assert reranker.counters["refused"] == 0


def test_work_that_cannot_finish_in_the_budget_is_refused_immediately():
    scorer = _Scorer()
    reranker = AsyncReranker(scorer, timeout_ms=100, max_workers=2)
    # One observed run of 80ms, and both workers already busy with a second
    # batch queued: a newcomer would start at ~160ms, past the 100ms budget.
    reranker._observed_seconds = 0.08
    reranker._outstanding = 4

    with pytest.raises(RerankerOverloaded):
        reranker.rerank("q", [], 5)

    assert scorer.calls == 0, "a refused request must not reach the scorer"
    assert reranker.counters["refused"] == 1


def test_a_refusal_is_still_a_timeout_for_existing_callers():
    """RetrievalService degrades on RerankerTimeoutError; refusals must too."""
    assert issubclass(RerankerOverloaded, RerankerTimeoutError)


def test_headroom_is_admitted():
    scorer = _Scorer()
    reranker = AsyncReranker(scorer, timeout_ms=500, max_workers=4)
    reranker._observed_seconds = 0.08
    reranker._outstanding = 4  # one batch ahead: ~160ms, inside 500ms

    assert reranker.rerank("q", [], 5) == []
    assert reranker.counters["refused"] == 0


def test_a_slot_is_released_even_when_scoring_raises():
    class _Broken:
        def rerank(self, query, results, top_k):
            raise RuntimeError("scorer exploded")

    reranker = AsyncReranker(_Broken(), timeout_ms=500, max_workers=2)

    with pytest.raises(RuntimeError):
        reranker.rerank("q", [], 5)

    assert reranker._outstanding == 0, "a failed run must not leak its slot"


def test_observed_cost_is_learned_from_real_runs():
    scorer = _Scorer(seconds=0.02)
    reranker = AsyncReranker(scorer, timeout_ms=5000, max_workers=2)

    reranker.rerank("q", [], 5)

    assert reranker._observed_seconds is not None
    assert reranker._observed_seconds >= 0.02
