"""Tests for RRF fusion and MMR re-ranking over RetrievalResult objects."""

from __future__ import annotations

import pytest

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.fusion import (
    dedup_variants,
    mmr_rerank,
    rrf_fuse,
    rrf_rank,
    variant_weighted_rrf_fuse,
)


@pytest.mark.parametrize("weights", [None, [0.2, 0.8]])
def test_rrf_duplicate_entries_do_not_change_ranks_or_scores(weights):
    assert rrf_rank(
        [["a", "a", "b", "b"], ["b", "b", "c"]],
        lambda item: item,
        weights=weights,
    ) == rrf_rank([["a", "b"], ["b", "c"]], lambda item: item, weights=weights)


def test_rrf_counts_independent_sources_and_preserves_first_payload():
    first = _r("a", 1)
    first.title = "first"
    duplicate = _r("a", 2)
    results = rrf_fuse([[first, duplicate], [duplicate]])
    assert len(results) == 1
    assert results[0].score == pytest.approx(2 / 61)
    assert results[0].title == "first"


def _r(doc_id: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="T", text="body", url=None, score=score)


def test_rrf_fuse_single_set():
    results = rrf_fuse([[_r("a"), _r("b"), _r("c")]])
    ids = [r.doc_id for r in results]
    assert ids == ["a", "b", "c"]


def test_rrf_fuse_two_sets_accumulates_scores():
    # "a" appears in both sets — should outscore "b" (sparse-only) and "c" (dense-only)
    sparse = [_r("a"), _r("b")]
    dense = [_r("a"), _r("c")]
    results = rrf_fuse([sparse, dense])
    assert results[0].doc_id == "a"


def test_rrf_fuse_deduplicates():
    sparse = [_r("a"), _r("b")]
    dense = [_r("a"), _r("b")]
    results = rrf_fuse([sparse, dense])
    assert len(results) == 2


def test_rrf_fuse_empty_sets():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[]]) == []


def test_rrf_fuse_score_is_rrf():
    results = rrf_fuse([[_r("a")]], rrf_k=60)
    assert results[0].score == pytest.approx(1.0 / (60 + 1))


def test_mmr_rerank_top_k_respected():
    results = [_r(f"d{i}", score=1.0 - i * 0.1) for i in range(5)]
    reranked = mmr_rerank(results, top_k=3)
    assert len(reranked) == 3


def test_mmr_rerank_lambda_1_preserves_order():
    results = [_r("a", 0.9), _r("b", 0.7), _r("c", 0.5)]
    reranked = mmr_rerank(results, top_k=3, mmr_lambda=1.0)
    assert [r.doc_id for r in reranked] == ["a", "b", "c"]


def test_mmr_rerank_penalises_same_source():
    # "chunk-1" and "chunk-2" share prefix "chunk"; "other-1" does not
    results = [
        _r("chunk-1", 0.9),
        _r("chunk-2", 0.8),
        _r("other-1", 0.7),
    ]
    # lambda=0.0 → maximum diversity, so "other-1" beats "chunk-2"
    reranked = mmr_rerank(results, top_k=2, mmr_lambda=0.0)
    ids = [r.doc_id for r in reranked]
    assert "chunk-1" in ids
    assert "other-1" in ids


def test_mmr_rerank_empty():
    assert mmr_rerank([], top_k=5) == []


def test_variant_weight_favours_heavier_set():
    # doc A only in the heavy (original) set, doc B only in a light set, same rank.
    original = [_r("A")]
    expansion = [_r("B")]
    fused = variant_weighted_rrf_fuse([original, expansion], weights=[1.0, 0.1])
    assert fused[0].doc_id == "A"


def test_uniform_weights_match_rank_order():
    fused = variant_weighted_rrf_fuse([[_r("A"), _r("B")]], weights=[1.0])
    assert [r.doc_id for r in fused] == ["A", "B"]


def test_dedup_drops_near_duplicate():
    # Identical embeddings for the first two → second dropped; original (last) kept.
    embs = {"a": [1.0, 0.0], "a2": [1.0, 0.0], "orig": [0.0, 1.0]}
    out = dedup_variants(
        ["a", "a2", "orig"], lambda xs: [embs[x] for x in xs], threshold=0.99
    )
    assert out == ["a", "orig"]


def test_dedup_keeps_distinct():
    embs = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    out = dedup_variants(["a", "b"], lambda xs: [embs[x] for x in xs], threshold=0.99)
    assert out == ["a", "b"]
