"""Tests for eval_runner.run_eval."""

from __future__ import annotations

import json
import tempfile
from unittest.mock import MagicMock

import pytest

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.eval_runner import run_eval
from src.internal.retrieval.reranker import Reranker


def _make_service(doc_ids_per_query: list[list[str]]) -> MagicMock:
    svc = MagicMock()
    svc.search.side_effect = [
        (
            [
                RetrievalResult(
                    doc_id=d, title="", text="", url=None, score=0.9 - i * 0.1
                )
                for i, d in enumerate(ids)
            ],
            "sparse",
        )
        for ids in doc_ids_per_query
    ]
    return svc


def _write_qa(qa_pairs: list[dict]) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair) + "\n")
        return f.name


def test_cli_output_flag_writes_the_printed_json(tmp_path, capsys):
    # The CI eval gate invokes `--output`; until now the parser had no such
    # flag and the step would have died on argparse the day a baseline landed.
    from src.internal.retrieval.eval_runner import main

    labels = tmp_path / "routing_labels.jsonl"
    labels.write_text(
        json.dumps({"query": "latest pricing sheet", "retriever": "web"}) + "\n"
    )
    out = tmp_path / "metrics.json"

    with pytest.raises(SystemExit) as excinfo:
        main(["--dataset", str(labels), "--routing-eval", "--output", str(out)])

    assert excinfo.value.code == 0
    written = json.loads(out.read_text())
    assert set(written) == {"routing_accuracy", "num_queries"}
    assert written["num_queries"] == 1
    assert json.loads(capsys.readouterr().out) == written


def test_run_eval_perfect_recall():
    qa = [{"query": "q1", "relevant_doc_ids": ["d1"]}]
    path = _write_qa(qa)
    svc = _make_service([["d1", "d2"]])

    metrics = run_eval(path, service=svc, top_k=5)

    assert metrics["recall@5"] == pytest.approx(1.0)
    assert metrics["ndcg@5"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["num_queries"] == 1


def test_run_eval_zero_recall():
    qa = [{"query": "q1", "relevant_doc_ids": ["d1"]}]
    path = _write_qa(qa)
    svc = _make_service([["x", "y"]])

    metrics = run_eval(path, service=svc, top_k=5)

    assert metrics["recall@5"] == 0.0
    assert metrics["mrr"] == 0.0


def test_run_eval_averages_over_queries():
    qa = [
        {"query": "q1", "relevant_doc_ids": ["d1"]},
        {"query": "q2", "relevant_doc_ids": ["d2"]},
    ]
    path = _write_qa(qa)
    svc = _make_service([["d1"], ["x"]])

    metrics = run_eval(path, service=svc, top_k=5)

    assert metrics["recall@5"] == pytest.approx(0.5)
    assert metrics["num_queries"] == 2


def test_run_eval_with_reranker_returns_reranked_section():
    """run_eval with a reranker must return 'retrieval', 'reranked', and 'latency_ms'."""
    qa = [{"query": "q1", "relevant_doc_ids": ["d1"]}]
    path = _write_qa(qa)

    mock_reranker = MagicMock(spec=Reranker)
    mock_reranker.rerank.return_value = [
        RetrievalResult(doc_id="d1", title="", text="", url=None, score=0.99)
    ]

    svc = _make_service([["d1"]])
    result = run_eval(path, service=svc, top_k=5, reranker=mock_reranker)

    assert "retrieval" in result
    assert "reranked" in result
    assert "latency_ms" in result
    assert "ndcg@5" in result["retrieval"]
    assert "ndcg@5" in result["reranked"]
    assert "p99" in result["latency_ms"]


def test_run_eval_without_reranker_returns_flat_dict():
    """run_eval without reranker returns the existing flat dict format."""
    qa = [{"query": "q1", "relevant_doc_ids": ["d1"]}]
    path = _write_qa(qa)
    svc = _make_service([["d1"]])

    result = run_eval(path, service=svc, top_k=5)

    assert "retrieval" not in result
    assert "ndcg@5" in result


def test_run_eval_reranker_called_once_per_query():
    """reranker.rerank() must be called exactly once per QA pair."""
    qa = [
        {"query": "q1", "relevant_doc_ids": ["d1"]},
        {"query": "q2", "relevant_doc_ids": ["d2"]},
    ]
    path = _write_qa(qa)

    mock_reranker = MagicMock(spec=Reranker)
    mock_reranker.rerank.return_value = [
        RetrievalResult(doc_id="d1", title="", text="", url=None, score=0.9)
    ]

    svc = _make_service([["d1"], ["d2"]])
    run_eval(path, service=svc, top_k=5, reranker=mock_reranker)

    assert mock_reranker.rerank.call_count == 2


def test_run_eval_latency_includes_mean():
    """latency_ms dict now includes 'mean' field."""
    qa_path = _write_qa([{"query": "q", "relevant_doc_ids": ["d1"]}])
    svc = _make_service([["d1"]])
    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [
        RetrievalResult(doc_id="d1", title="", text="", url=None, score=0.9)
    ]
    metrics = run_eval(qa_path, service=svc, top_k=1, reranker=fake_reranker)
    assert "mean" in metrics["latency_ms"]


def test_run_eval_slo_passes_when_fast(monkeypatch):
    """No error when latency is within slo_ms."""
    qa_path = _write_qa([{"query": "q", "relevant_doc_ids": ["d1"]}])
    svc = _make_service([["d1"]])
    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [
        RetrievalResult(doc_id="d1", title="", text="", url=None, score=0.9)
    ]
    # slo_ms=10000 should never be exceeded by a mock
    metrics = run_eval(
        qa_path, service=svc, top_k=1, reranker=fake_reranker, slo_ms=10000
    )
    assert "latency_ms" in metrics


def test_run_eval_slo_raises_when_exceeded():
    """SLOViolationError raised when p99 latency exceeds slo_ms."""
    import time as _time

    qa_path = _write_qa([{"query": "q", "relevant_doc_ids": ["d1"]}])
    svc = _make_service([["d1"]])
    fake_reranker = MagicMock()

    def slow_rerank(*_):
        _time.sleep(0.1)
        return [RetrievalResult(doc_id="d1", title="", text="", url=None, score=0.9)]

    fake_reranker.rerank.side_effect = slow_rerank

    from src.internal.retrieval.eval_runner import SLOViolationError

    with pytest.raises(SLOViolationError):
        run_eval(qa_path, service=svc, top_k=1, reranker=fake_reranker, slo_ms=1)


def test_run_eval_compare_baseline_includes_improvement_ratio():
    qa_path = _write_qa([{"query": "q", "relevant_doc_ids": ["d1"]}])
    svc = _make_service([["d1"]])
    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [
        RetrievalResult(doc_id="d1", title="", text="", url=None, score=0.9)
    ]
    metrics = run_eval(
        qa_path, service=svc, top_k=1, reranker=fake_reranker, compare_baseline=True
    )
    assert "reranker_improvement_ratio" in metrics
