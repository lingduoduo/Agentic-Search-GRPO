"""RRF fusion and MMR re-ranking over RetrievalResult objects."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Hashable, TypeVar

from .backends.base import RetrievalResult

_T = TypeVar("_T")
_K = TypeVar("_K", bound=Hashable)


@dataclass
class FusionWeights:
    w_sparse: float = 0.5
    w_dense: float = 0.5


_RRF_K = 60


def _source_prefix(doc_id: str) -> str:
    """Source-level prefix of doc_id used as a cheap similarity proxy for MMR."""
    sep = doc_id.rfind("-")
    return doc_id[:sep] if sep > 0 else doc_id


def rrf_rank(
    result_sets: Sequence[Sequence[_T]],
    key_fn: Callable[[_T], _K],
    *,
    weights: Sequence[float] | None = None,
    rrf_k: int = _RRF_K,
) -> list[tuple[_K, float]]:
    """Reciprocal Rank Fusion scoring — the shared core for every RRF site.

    ``score(key) = Σ_i wᵢ · 1/(rrf_k + rank)`` where ``rank`` is 1-based among
    unique keys within each set. Repeated keys contribute only once per source.
    Returns ``(key, score)`` pairs sorted by score descending; ties
    preserve first-seen order (stable sort over insertion order). ``weights``
    defaults to all-1.0 (standard RRF) and falls back to uniform when its length
    does not match ``result_sets``. Scale-invariant — no score normalisation
    needed. Pure scoring: callers map their own key and rebuild their own type.
    """
    if weights is None or len(weights) != len(result_sets):
        weights = [1.0] * len(result_sets)

    rrf_scores: dict[_K, float] = defaultdict(float)
    for weight, result_set in zip(weights, result_sets):
        seen: set[_K] = set()
        rank = 0
        for item in result_set:
            key = key_fn(item)
            if key in seen:
                continue
            seen.add(key)
            rank += 1
            rrf_scores[key] += weight * (1.0 / (rrf_k + rank))

    return sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)


def _first_seen(
    result_sets: Sequence[Sequence[_T]], key_fn: Callable[[_T], _K]
) -> dict[_K, _T]:
    """First item encountered per key, scanning sets in order."""
    seen: dict[_K, _T] = {}
    for result_set in result_sets:
        for item in result_set:
            seen.setdefault(key_fn(item), item)
    return seen


def _rebuild_results(
    ranked: list[tuple[str, float]], source: dict[str, RetrievalResult]
) -> list[RetrievalResult]:
    """Rebuild RetrievalResults in fused order, carrying the fused score."""
    return [
        RetrievalResult(
            doc_id=doc_id,
            title=source[doc_id].title,
            text=source[doc_id].text,
            url=source[doc_id].url,
            score=score,
            metadata=source[doc_id].metadata,
        )
        for doc_id, score in ranked
    ]


def rrf_fuse(
    result_sets: list[list[RetrievalResult]],
    *,
    rrf_k: int = _RRF_K,
) -> list[RetrievalResult]:
    """Merge ranked result sets via Reciprocal Rank Fusion.

    Score formula: score(doc) = Σ 1 / (k + rank)  for each set the doc appears in.
    Scale-invariant — no normalisation of raw BM25 or cosine scores required.
    """
    ranked = rrf_rank(result_sets, lambda r: r.doc_id, rrf_k=rrf_k)
    return _rebuild_results(ranked, _first_seen(result_sets, lambda r: r.doc_id))


def weighted_rrf_fuse(
    result_sets: list[list[RetrievalResult]],
    weights: FusionWeights | None = None,
    *,
    rrf_k: int = _RRF_K,
) -> list[RetrievalResult]:
    """RRF fusion with per-source weights (defaults to uniform = standard RRF).

    Assumes result_sets[0] = sparse, result_sets[1] = dense.
    Falls back to rrf_fuse() when weights is None or len(result_sets) != 2.
    """
    if weights is None or len(result_sets) != 2:
        return rrf_fuse(result_sets, rrf_k=rrf_k)

    ranked = rrf_rank(
        result_sets,
        lambda r: r.doc_id,
        weights=[weights.w_sparse, weights.w_dense],
        rrf_k=rrf_k,
    )
    return _rebuild_results(ranked, _first_seen(result_sets, lambda r: r.doc_id))


def variant_weighted_rrf_fuse(
    result_sets: list[list[RetrievalResult]],
    weights: list[float],
    *,
    rrf_k: int = _RRF_K,
) -> list[RetrievalResult]:
    """RRF across N variant result sets, each contributing weight / (k + rank).

    weights[i] applies to result_sets[i]. Falls back to uniform when lengths
    mismatch.

    Note: distinct from weighted_rrf_fuse(), which is the 2-set sparse/dense
    variant that takes a FusionWeights dataclass and only supports exactly 2 sets.
    """
    ranked = rrf_rank(result_sets, lambda r: r.doc_id, weights=weights, rrf_k=rrf_k)
    return _rebuild_results(ranked, _first_seen(result_sets, lambda r: r.doc_id))


def mmr_rerank(
    results: list[RetrievalResult],
    *,
    top_k: int,
    mmr_lambda: float = 0.5,
) -> list[RetrievalResult]:
    """Re-rank with Maximal Marginal Relevance.

    Uses source-prefix matching as a cheap inter-document similarity proxy —
    no embeddings required at re-rank time.

    mmr_lambda=1.0 → pure relevance order (no diversity penalty).
    mmr_lambda=0.0 → maximum diversity.
    """
    if not results:
        return []
    if mmr_lambda == 1.0:
        return results[:top_k]

    max_score = max(r.score for r in results) or 1.0
    normalized = [(r, r.score / max_score) for r in results]

    selected: list[RetrievalResult] = []
    selected_prefixes: list[str] = []
    remaining = list(normalized)

    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda x: x[1])
        else:

            def _mmr(
                item: tuple[RetrievalResult, float],
                _prefixes: list[str] = selected_prefixes,
            ) -> float:
                r, rel = item
                sim = 1.0 if _source_prefix(r.doc_id) in _prefixes else 0.0
                return mmr_lambda * rel - (1.0 - mmr_lambda) * sim

            best = max(remaining, key=_mmr)

        result, _ = best
        selected.append(result)
        selected_prefixes.append(_source_prefix(result.doc_id))
        remaining.remove(best)

    return selected


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def dedup_variants(
    variants: list[str],
    embed_fn: Callable[[list[str]], list[list[float]]],
    *,
    threshold: float = 0.95,
) -> list[str]:
    """Drop variants whose embedding cosine ≥ threshold to an earlier kept variant.

    Order-preserving; the last variant (the original) is always kept.
    """
    if len(variants) <= 1:
        return variants
    embs = embed_fn(variants)
    kept: list[str] = []
    kept_embs: list[list[float]] = []
    for i, (v, e) in enumerate(zip(variants, embs)):
        is_last = i == len(variants) - 1
        if is_last or all(_cosine(e, ke) < threshold for ke in kept_embs):
            kept.append(v)
            kept_embs.append(e)
    return kept
