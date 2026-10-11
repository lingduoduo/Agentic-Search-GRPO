from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.two_stage_reranker import TwoStageReranker


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["rerank", "arerank"])
async def test_requested_top_k_exceeds_prefilter_without_underfilling(method):
    inputs = [_result(f"d{i}") for i in range(5)]
    stage = SimpleNamespace(rerank=lambda query, results, top_k: results[:top_k])
    reranker = TwoStageReranker(stage, stage, pre_filter_top_n=2)
    output = (
        reranker.rerank("q", inputs, 4)
        if method == "rerank"
        else await reranker.arerank("q", inputs, 4)
    )
    assert output == inputs[:4]


@pytest.mark.parametrize("budget", [0, -1])
def test_prefilter_budget_must_be_positive(budget):
    with pytest.raises(ValueError, match="pre_filter_top_n"):
        TwoStageReranker(MagicMock(), MagicMock(), pre_filter_top_n=budget)


@pytest.mark.asyncio
@pytest.mark.parametrize("top_k", [0, -1])
async def test_nonpositive_top_k_skips_both_stages(top_k):
    fast, heavy = MagicMock(), MagicMock()
    reranker = TwoStageReranker(fast, heavy)
    assert reranker.rerank("q", [_result("d1")], top_k) == []
    assert await reranker.arerank("q", [_result("d1")], top_k) == []
    fast.rerank.assert_not_called()
    heavy.rerank.assert_not_called()
    fast.arerank.assert_not_called()
    heavy.arerank.assert_not_called()


@pytest.mark.asyncio
async def test_async_sync_fallback_runs_both_stages_off_loop():
    event_thread = threading.get_ident()
    called_on = []

    def score(query, results, top_k):
        called_on.append(threading.get_ident())
        return results[:top_k]

    stage = SimpleNamespace(rerank=score)
    reranker = TwoStageReranker(stage, stage)
    assert await reranker.arerank("q", [_result("d1")], 1) == [_result("d1")]
    assert len(called_on) == 2
    assert all(thread != event_thread for thread in called_on)


@pytest.mark.asyncio
async def test_native_async_stages_are_awaited():
    inputs = [_result("d1")]
    fast = SimpleNamespace(arerank=AsyncMock(return_value=inputs))
    heavy = SimpleNamespace(arerank=AsyncMock(return_value=inputs))
    reranker = TwoStageReranker(fast, heavy, pre_filter_top_n=2)
    assert await reranker.arerank("q", inputs, 1) == inputs
    fast.arerank.assert_awaited_once_with("q", inputs, 2)
    heavy.arerank.assert_awaited_once_with("q", inputs, 1)


def _result(doc_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="t", text="c", url=None, score=score)


def _make_fast(return_val):
    m = MagicMock()
    m.rerank.return_value = return_val
    return m


def _make_heavy(return_val):
    m = MagicMock()
    m.rerank.return_value = return_val
    return m


def test_fast_gets_all_candidates():
    """Fast reranker receives all input results."""
    fast = _make_fast([_result("d1"), _result("d2")])
    heavy = _make_heavy([_result("d1")])
    tsr = TwoStageReranker(fast, heavy, pre_filter_top_n=2)
    inputs = [_result("d1"), _result("d2"), _result("d3")]
    tsr.rerank("q", inputs, top_k=1)
    fast.rerank.assert_called_once_with("q", inputs, 2)


def test_heavy_gets_fast_output():
    """Heavy reranker receives fast reranker's output."""
    fast_out = [_result("d2"), _result("d1")]
    fast = _make_fast(fast_out)
    heavy = _make_heavy([_result("d2")])
    tsr = TwoStageReranker(fast, heavy, pre_filter_top_n=2)
    tsr.rerank("q", [_result("d1"), _result("d2")], top_k=1)
    heavy.rerank.assert_called_once_with("q", fast_out, 1)


def test_returns_heavy_output():
    fast = _make_fast([_result("d1"), _result("d2")])
    heavy = _make_heavy([_result("d1")])
    tsr = TwoStageReranker(fast, heavy, pre_filter_top_n=2)
    result = tsr.rerank("q", [_result("d1"), _result("d2")], top_k=1)
    assert [r.doc_id for r in result] == ["d1"]


def test_empty_inputs_returns_empty():
    fast = _make_fast([])
    heavy = _make_heavy([])
    tsr = TwoStageReranker(fast, heavy, pre_filter_top_n=5)
    assert tsr.rerank("q", [], top_k=3) == []


def test_async_rerank_falls_back_to_sync_when_no_arerank():
    """When arerank is not available, fallback to sync rerank."""
    fast = _make_fast([_result("d1")])
    heavy = _make_heavy([_result("d1")])
    # Explicitly mark that arerank is not available on these mocks
    del fast.arerank
    del heavy.arerank
    tsr = TwoStageReranker(fast, heavy, pre_filter_top_n=2)
    result = asyncio.run(tsr.arerank("q", [_result("d1")], top_k=1))
    assert result[0].doc_id == "d1"
    # Verify sync rerank was called, not arerank
    fast.rerank.assert_called_once_with("q", [_result("d1")], 2)
    heavy.rerank.assert_called_once_with("q", [_result("d1")], 1)


def test_from_env_reads_pre_filter_top_n(monkeypatch):
    monkeypatch.setenv("RERANKER_PRE_FILTER_TOP_N", "25")
    fast = MagicMock()
    heavy = MagicMock()
    tsr = TwoStageReranker.from_env(fast, heavy)
    assert tsr._pre_n == 25
