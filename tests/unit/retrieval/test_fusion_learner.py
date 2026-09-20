"""Tests for FusionLearner, weighted RRF fusion, and adaptive MMR lambda."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.fusion import weighted_rrf_fuse
from src.internal.retrieval.fusion_learner import (
    FusionLearner,
    FusionWeights,
)


def _result(doc_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="", text="", url=None, score=score)


def test_fusion_weights_dataclass():
    w = FusionWeights(w_sparse=0.4, w_dense=0.6)
    assert w.w_sparse + w.w_dense == pytest.approx(1.0)


def test_weighted_rrf_fuse_returns_results():
    sparse = [_result("a"), _result("b")]
    dense = [_result("b"), _result("c")]
    weights = FusionWeights(w_sparse=0.5, w_dense=0.5)
    fused = weighted_rrf_fuse([sparse, dense], weights)
    ids = [r.doc_id for r in fused]
    assert set(ids) == {"a", "b", "c"}


def test_weighted_rrf_fuse_higher_dense_weight_promotes_dense_only_doc():
    sparse = [_result("a"), _result("b")]
    dense = [_result("c"), _result("b")]
    weights = FusionWeights(w_sparse=0.1, w_dense=0.9)
    fused = weighted_rrf_fuse([sparse, dense], weights)
    ids = [r.doc_id for r in fused]
    assert ids.index("c") < ids.index("a")


def test_weighted_rrf_fuse_none_weights_is_standard_rrf():
    sparse = [_result("a"), _result("b")]
    dense = [_result("b"), _result("c")]
    fused = weighted_rrf_fuse([sparse, dense], None)
    assert len(fused) == 3


def test_fusion_learner_returns_fusion_weights(tmp_path):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(json.dumps({"query": "q", "relevant_doc_ids": ["d1"]}) + "\n")

    def factory(w_sparse, w_dense):
        svc = MagicMock()
        svc.search.return_value = ([MagicMock(doc_id="d1")], "hybrid")
        return svc

    learner = FusionLearner(factory)
    weights = learner.fit(str(qa), w_sparse_range=[0.3, 0.5, 0.7])
    assert isinstance(weights, FusionWeights)
    assert 0.0 < weights.w_sparse < 1.0
    assert abs(weights.w_sparse + weights.w_dense - 1.0) < 1e-6
