"""Tests for RetrievalService."""

from __future__ import annotations

import logging
import threading

import pytest
from unittest.mock import MagicMock, patch

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.service import RetrievalService


def test_explicit_fusion_weights_change_hybrid_ranking():
    from src.internal.retrieval.fusion import FusionWeights

    backend = MagicMock()
    backend.search_sparse.return_value = [_make_result("sparse")]
    backend.search_dense.return_value = [_make_result("dense")]
    service = RetrievalService(backend, fusion_weights=FusionWeights(0.1, 0.9))
    results, mode = service.search("q", top_k=2, filters={"tenant": "t"})
    assert [result.doc_id for result in results] == ["dense", "sparse"]
    assert mode == "hybrid"
    assert backend.search_sparse.call_args.kwargs["filters"] == {"tenant": "t"}
    assert backend.search_dense.call_args.kwargs["filters"] == {"tenant": "t"}


def test_explicit_bm25_overrides_reach_local_config_without_env_mutation(monkeypatch):
    import os

    monkeypatch.setenv("BM25_INDEX_PATH", "/fake/index")
    monkeypatch.setenv("BM25_K1", "0.9")
    monkeypatch.setenv("BM25_B", "0.5")
    monkeypatch.setenv("RETRIEVAL_BACKEND", "local")
    monkeypatch.delenv("DENSE_MODEL_PATH", raising=False)
    with patch("src.internal.retrieval.backends.local.LocalBackend") as backend:
        RetrievalService.from_env(bm25_k1=2.0, bm25_b=0.8)
    config = backend.call_args.args[0]
    assert config.k1 == 2.0 and config.b == 0.8
    assert os.environ["BM25_K1"] == "0.9"
    assert os.environ["BM25_B"] == "0.5"


def test_disabled_result_cache_skips_redis_construction(monkeypatch):
    monkeypatch.setenv("RESULT_CACHE_REDIS_URL", "redis://example.invalid")
    with (
        patch(
            "src.internal.retrieval.service._build_backend", return_value=MagicMock()
        ),
        patch("redis.from_url") as redis,
    ):
        service = RetrievalService.from_env(use_result_cache=False)
    redis.assert_not_called()
    assert service._result_cache is None


def _make_result(doc_id: str, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, title="T", text="body", url=None, score=score)


def _sparse_only_backend(results):
    """Backend where dense raises NotImplementedError (no dense configured)."""
    backend = MagicMock()
    backend.search_sparse.return_value = results
    backend.search_dense.side_effect = NotImplementedError
    return backend


def test_search_delegates_to_backend_sparse():
    backend = _sparse_only_backend([_make_result("d1")])
    service = RetrievalService(backend)

    results, mode = service.search("procurement", top_k=5)

    # over_fetch multiplier=2 → top_k * 2 = 10
    backend.search_sparse.assert_called_once_with("procurement", top_k=10, filters=None)
    assert mode == "sparse_only"
    assert len(results) == 1
    assert results[0].doc_id == "d1"


def test_search_returns_empty_list_on_no_results():
    backend = _sparse_only_backend([])
    service = RetrievalService(backend)

    results, mode = service.search("nothing", top_k=5)

    assert results == []
    assert mode == "sparse_only"


def test_from_env_raises_on_unknown_backend(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_BACKEND", "pinecone")
    with pytest.raises(ValueError, match="Unknown RETRIEVAL_BACKEND"):
        RetrievalService.from_env()


def test_from_env_local_requires_index_path(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_BACKEND", "local")
    monkeypatch.delenv("BM25_INDEX_PATH", raising=False)
    with pytest.raises(KeyError):
        RetrievalService.from_env()


def test_search_hybrid_when_both_legs_succeed():
    backend = MagicMock()
    backend.search_sparse.return_value = [
        _make_result("s1", 0.9),
        _make_result("s2", 0.7),
    ]
    backend.search_dense.return_value = [
        _make_result("d1", 0.8),
        _make_result("s1", 0.6),
    ]
    service = RetrievalService(backend)

    results, mode = service.search("q", top_k=3)

    assert mode == "hybrid"
    # s1 appears in both sets — highest RRF score
    assert results[0].doc_id == "s1"


def test_search_falls_back_to_sparse_when_dense_raises_not_implemented():
    backend = _sparse_only_backend([_make_result("s1")])
    service = RetrievalService(backend)

    results, mode = service.search("q", top_k=5)

    assert mode == "sparse_only"
    assert results[0].doc_id == "s1"


def test_search_falls_back_to_dense_when_sparse_raises(caplog):
    backend = MagicMock()
    backend.search_sparse.side_effect = RuntimeError("BM25 down")
    backend.search_dense.return_value = [_make_result("d1")]
    service = RetrievalService(backend)

    with caplog.at_level(logging.WARNING):
        results, mode = service.search("q", top_k=5)

    assert mode == "dense_only"
    assert results[0].doc_id == "d1"


def test_search_raises_when_both_legs_fail():
    backend = MagicMock()
    backend.search_sparse.side_effect = RuntimeError("sparse down")
    backend.search_dense.side_effect = RuntimeError("dense down")
    service = RetrievalService(backend)

    with pytest.raises(RuntimeError, match="Both retrieval legs failed"):
        service.search("q", top_k=5)


def test_graph_search_returns_results():
    backend = MagicMock()
    backend.search_sparse.return_value = [
        _make_result("g1"),
    ]
    backend.search_dense.side_effect = NotImplementedError("no dense")
    service = RetrievalService(backend)

    results = service.graph_search(
        "FAISS dense retrieval", top_k=5, max_entity_queries=0
    )

    assert len(results) >= 1
    assert results[0].doc_id == "g1"


def test_graph_search_delegates_to_graph_rag_search():
    backend = MagicMock()
    backend.search_sparse.return_value = [_make_result("g2")]
    backend.search_dense.side_effect = NotImplementedError
    service = RetrievalService(backend)

    results = service.graph_search("q", top_k=3, initial_k=2, max_entity_queries=0)
    assert isinstance(results, list)


def test_search_passes_filters_to_sparse_backend():
    backend = _sparse_only_backend([_make_result("d1")])
    service = RetrievalService(backend)

    service.search("q", top_k=5, filters={"source": "confluence"})

    backend.search_sparse.assert_called_once_with(
        "q", top_k=10, filters={"source": "confluence"}
    )


def test_search_passes_filters_to_both_legs():
    backend = MagicMock()
    backend.search_sparse.return_value = [_make_result("s1")]
    backend.search_dense.return_value = [_make_result("d1")]
    service = RetrievalService(backend)

    service.search("q", top_k=3, filters={"source": "sharepoint"})

    backend.search_sparse.assert_called_once_with(
        "q", top_k=6, filters={"source": "sharepoint"}
    )
    backend.search_dense.assert_called_once_with(
        "q", top_k=6, filters={"source": "sharepoint"}
    )


def test_search_runs_legs_concurrently():
    """Both legs must overlap — proves ThreadPoolExecutor, not sequential."""
    sparse_started = threading.Event()
    dense_started = threading.Event()

    def slow_sparse(query, *, top_k, filters):
        sparse_started.set()
        assert dense_started.wait(timeout=2.0), "dense leg never started"
        return [_make_result("s1")]

    def slow_dense(query, *, top_k, filters):
        dense_started.set()
        assert sparse_started.wait(timeout=2.0), "sparse leg never started"
        return [_make_result("d1")]

    backend = MagicMock()
    backend.search_sparse.side_effect = slow_sparse
    backend.search_dense.side_effect = slow_dense
    service = RetrievalService(backend)

    results, mode = service.search("q", top_k=3)

    assert mode == "hybrid"
    assert sparse_started.is_set() and dense_started.is_set()


def test_sparse_retriever_config_bm25_defaults():
    from src.internal.document_index.retrieval import SparseRetrieverConfig

    config = SparseRetrieverConfig(index_path="/idx", corpus_path="/c.jsonl")
    assert config.k1 == pytest.approx(1.2)
    assert config.b == pytest.approx(0.75)


def test_build_local_backend_passes_bm25_k1_b(monkeypatch):
    monkeypatch.setenv("BM25_INDEX_PATH", "/fake/index")
    monkeypatch.setenv("BM25_K1", "0.9")
    monkeypatch.setenv("BM25_B", "0.5")

    import src.internal.retrieval.service as svc_mod

    captured: dict = {}

    def fake_local_backend():
        from src.internal.document_index.retrieval import SparseRetrieverConfig
        import os

        config = SparseRetrieverConfig(
            index_path=os.environ["BM25_INDEX_PATH"],
            corpus_path=os.environ.get("BM25_CORPUS_PATH", "data/corpus.jsonl"),
            topk=int(os.environ.get("BM25_TOP_K", "20")),
            k1=float(os.environ.get("BM25_K1", "1.2")),
            b=float(os.environ.get("BM25_B", "0.75")),
        )
        captured["k1"] = config.k1
        captured["b"] = config.b
        return MagicMock()

    monkeypatch.setattr(svc_mod, "_build_local_backend", fake_local_backend)
    monkeypatch.setenv("RETRIEVAL_BACKEND", "local")
    svc_mod.RetrievalService.from_env()

    assert captured["k1"] == pytest.approx(0.9)
    assert captured["b"] == pytest.approx(0.5)


def test_reranker_called_when_injected():
    """Reranker.rerank() must be called with the query and fused results."""
    backend = _sparse_only_backend([_make_result("d1"), _make_result("d2")])
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [_make_result("d2"), _make_result("d1")]

    service = RetrievalService(backend, reranker=mock_reranker)
    results, mode = service.search("q", top_k=2)

    mock_reranker.rerank.assert_called_once()
    call_args = mock_reranker.rerank.call_args
    assert call_args[0][0] == "q"  # query
    assert call_args[0][2] == 2  # top_k
    assert results[0].doc_id == "d2"


def test_mode_has_reranked_suffix():
    """retrieval_mode must end with '+reranked' when a reranker is present."""
    backend = _sparse_only_backend([_make_result("d1")])
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [_make_result("d1")]

    service = RetrievalService(backend, reranker=mock_reranker)
    _, mode = service.search("q", top_k=1)

    assert mode.endswith("+reranked")


def test_no_reranker_mode_unchanged():
    """Without a reranker, mode must not contain '+reranked'."""
    backend = _sparse_only_backend([_make_result("d1")])
    service = RetrievalService(backend)
    _, mode = service.search("q", top_k=1)

    assert "+reranked" not in mode


def test_reranker_receives_filters():
    """filters kwarg must be passed through to backend legs even when reranker is set."""
    backend = _sparse_only_backend([_make_result("d1")])
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [_make_result("d1")]

    service = RetrievalService(backend, reranker=mock_reranker)
    service.search("q", top_k=1, filters={"source": "wiki"})

    # With reranker active: over_fetch = ceil(1 * OVER_FETCH_MULTIPLIER=2 * RERANKER_OVER_FETCH_MULTIPLIER=2.0) = 4
    backend.search_sparse.assert_called_once_with(
        "q", top_k=4, filters={"source": "wiki"}
    )


def _pipeline_mock(
    variants: list[str], merged_filters: dict | None = None
) -> MagicMock:
    """Helper: mock QueryTransformPipeline returning given variants."""
    pipeline = MagicMock()
    bundle = MagicMock()
    bundle.retrieval_variants.return_value = variants
    bundle.merged_filters = merged_filters or {}
    pipeline.transform.return_value = bundle
    pipeline.max_variants = 5
    return pipeline


def test_pipeline_transform_called_with_query_and_filters():
    backend = _sparse_only_backend([_make_result("d1")])
    pipeline = _pipeline_mock(["variant q"])

    service = RetrievalService(backend, pipeline=pipeline)
    service.search("original query", top_k=1, filters={"source": "wiki"})

    pipeline.transform.assert_called_once_with("original query", {"source": "wiki"})


def test_rag_fusion_retrieves_once_per_variant():
    backend = _sparse_only_backend([_make_result("d1")])
    pipeline = _pipeline_mock(["q1", "q2", "q3"])

    service = RetrievalService(backend, pipeline=pipeline)
    service.search("q", top_k=1)

    # _search_one runs once per variant; each calls search_sparse once
    assert backend.search_sparse.call_count == 3


def test_mode_has_rag_fusion_suffix_when_pipeline_set():
    backend = _sparse_only_backend([_make_result("d1")])
    pipeline = _pipeline_mock(["q1", "q2"])

    service = RetrievalService(backend, pipeline=pipeline)
    _, mode = service.search("q", top_k=1)

    assert "+rag_fusion" in mode


def test_no_pipeline_path_unchanged():
    """Without pipeline, mode must not contain +rag_fusion and only one retrieval fires."""
    backend = _sparse_only_backend([_make_result("d1")])
    service = RetrievalService(backend)

    _, mode = service.search("q", top_k=1)

    assert "+rag_fusion" not in mode
    backend.search_sparse.assert_called_once()


def test_pipeline_merged_filters_passed_to_backend():
    """Bundle's merged_filters must be forwarded to the retrieval backend."""
    backend = _sparse_only_backend([_make_result("d1")])
    pipeline = _pipeline_mock(["q1"], merged_filters={"source": "arxiv"})

    service = RetrievalService(backend, pipeline=pipeline)
    service.search("q", top_k=1)

    # The backend must receive the merged_filters from the pipeline bundle
    call_args = backend.search_sparse.call_args
    assert call_args[1]["filters"] == {"source": "arxiv"}


def test_pipeline_empty_variants_falls_back_to_original_query():
    """If retrieval_variants() returns [], fall back to original query — no crash."""
    backend = _sparse_only_backend([_make_result("d1")])
    pipeline = _pipeline_mock([])  # empty variants!

    service = RetrievalService(backend, pipeline=pipeline)
    results, mode = service.search("original query", top_k=1)

    # Must not crash; backend must be called with the original query
    assert backend.search_sparse.call_count == 1
    call_args = backend.search_sparse.call_args
    assert call_args[0][0] == "original query"


def test_rag_fusion_degrades_gracefully_when_variant_fails():
    """If one variant's _search_one raises, it should be skipped (not crash)."""
    call_count = 0

    def flaky_sparse(query, *, top_k, filters):
        nonlocal call_count
        call_count += 1
        if query == "q2":
            raise RuntimeError("backend hiccup")
        return [_make_result(f"r_{query}")]

    backend = MagicMock()
    backend.search_sparse.side_effect = flaky_sparse
    backend.search_dense.side_effect = NotImplementedError

    pipeline = _pipeline_mock(["q1", "q2", "q3"])
    service = RetrievalService(backend, pipeline=pipeline)

    results, mode = service.search("q", top_k=2)

    # Should not raise; should return results from q1 and q3
    assert len(results) >= 1
    assert "+rag_fusion" in mode


def test_from_env_builds_async_reranker_chain(monkeypatch):
    """RERANKER_ASYNC=true wraps base reranker in AsyncReranker."""
    monkeypatch.setenv("RERANKER_ASYNC", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "local")
    monkeypatch.setenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    from src.internal.retrieval.async_reranker import AsyncReranker
    from src.internal.retrieval.cached_reranker import CachedReranker

    with (
        patch(
            "src.internal.retrieval.service._build_backend", return_value=MagicMock()
        ),
        patch(
            "src.internal.retrieval.reranker.SentenceTransformerReranker.load",
            return_value=MagicMock(),
        ),
    ):
        svc = RetrievalService.from_env()

    assert isinstance(svc._reranker, (AsyncReranker, CachedReranker))


def test_from_env_no_async_flag_leaves_base_reranker(monkeypatch):
    """Without RERANKER_ASYNC, reranker is the bare Reranker instance."""
    monkeypatch.delenv("RERANKER_ASYNC", raising=False)
    monkeypatch.delenv("RERANKER_PROVIDER", raising=False)

    with patch(
        "src.internal.retrieval.service._build_backend", return_value=MagicMock()
    ):
        svc = RetrievalService.from_env()

    assert svc._reranker is None


def test_search_passes_over_fetch_candidates_to_reranker():
    """When reranker active, fused results are NOT pre-trimmed to top_k before reranking."""
    from src.internal.retrieval.backends.base import RetrievalResult

    results_6 = [
        RetrievalResult(doc_id=f"d{i}", title="", text="", url=None, score=float(i))
        for i in range(6)
    ]

    backend = MagicMock()
    backend.search_sparse.return_value = results_6
    backend.search_dense.side_effect = NotImplementedError

    reranker = MagicMock()
    reranker.rerank.return_value = results_6[:3]

    svc = RetrievalService(backend, reranker=reranker)

    import os

    with patch.dict(
        os.environ,
        {"OVER_FETCH_MULTIPLIER": "2", "RERANKER_OVER_FETCH_MULTIPLIER": "3"},
    ):
        svc.search("q", top_k=2)

    # reranker.rerank should have received more than top_k=2 candidates
    called_results = reranker.rerank.call_args[0][1]
    assert len(called_results) > 2


def test_from_env_builds_two_stage_reranker(monkeypatch):
    monkeypatch.setenv("RERANKER_TWO_STAGE", "true")
    monkeypatch.setenv("RERANKER_PROVIDER", "local")
    monkeypatch.setenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    monkeypatch.setenv("RERANKER_FAST_MODEL", "BAAI/bge-reranker-base")

    from src.internal.retrieval.two_stage_reranker import TwoStageReranker

    with (
        patch(
            "src.internal.retrieval.service._build_backend", return_value=MagicMock()
        ),
        patch(
            "src.internal.retrieval.reranker.SentenceTransformerReranker.load",
            return_value=MagicMock(),
        ),
    ):
        svc = RetrievalService.from_env()

    assert isinstance(svc._reranker, TwoStageReranker)


# --- Bug R1: reranker timeout must degrade, not crash the request ---


def test_search_degrades_when_reranker_times_out(caplog):
    """A RerankerTimeoutError must not fail the request; keep pre-rerank order."""
    from src.internal.retrieval.async_reranker import RerankerTimeoutError

    pre_rerank = [_make_result("d1"), _make_result("d2")]
    backend = _sparse_only_backend(pre_rerank)

    class _TimingOutReranker:
        def rerank(self, query, results, top_k):
            raise RerankerTimeoutError("too slow")

    service = RetrievalService(backend, reranker=_TimingOutReranker())

    with caplog.at_level(logging.WARNING):
        results, mode = service.search("q", top_k=2)

    # Pre-rerank ordering is preserved and mode must NOT claim +reranked.
    assert [r.doc_id for r in results] == ["d1", "d2"]
    assert "+reranked" not in mode


def test_search_degrades_when_reranker_raises_generic(caplog):
    """Any reranker exception (not just timeout) degrades gracefully."""
    pre_rerank = [_make_result("d1"), _make_result("d2")]
    backend = _sparse_only_backend(pre_rerank)

    class _BrokenReranker:
        def rerank(self, query, results, top_k):
            raise RuntimeError("boom")

    service = RetrievalService(backend, reranker=_BrokenReranker())

    with caplog.at_level(logging.WARNING):
        results, mode = service.search("q", top_k=2)

    assert [r.doc_id for r in results] == ["d1", "d2"]
    assert "+reranked" not in mode


# --- Bug R2: QT_REWRITE must open the pipeline gate ---


def test_from_env_qt_rewrite_alone_builds_pipeline(monkeypatch):
    """Enabling only QT_REWRITE must build a non-None pipeline (gate honored)."""
    for flag in (
        "QT_DECOMPOSE",
        "QT_HYDE",
        "QT_STEP_BACK",
        "QT_KEYWORDS",
        "QT_CONSTRUCT_FILTERS",
        "QT_MULTI_QUERY",
        "QT_ROUTER",
    ):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("QT_REWRITE", "1")

    sentinel_pipeline = object()

    with (
        patch(
            "src.internal.retrieval.service._build_backend", return_value=MagicMock()
        ),
        patch("src.internal.retrieval.service._build_llm", return_value=MagicMock()),
        patch(
            "src.internal.retrieval.query_transform_factory."
            "build_query_transform_pipeline_from_env",
            return_value=sentinel_pipeline,
        ),
    ):
        svc = RetrievalService.from_env()

    assert svc._pipeline is sentinel_pipeline


# --- Bug R3: weighted-RRF must not misassign 1.0 when the original is dropped ---


def test_weighted_fusion_falls_back_when_original_variant_fails():
    """If the LAST (original) variant fails, weighted fusion must NOT run with a
    1.0 weight landing on a surviving paraphrase — fall back to unweighted fuse."""

    def flaky_sparse(query, *, top_k, filters):
        if query == "orig":  # the original query (variants[-1]) fails
            raise RuntimeError("original leg down")
        return [_make_result(f"r_{query}")]

    backend = MagicMock()
    backend.search_sparse.side_effect = flaky_sparse
    backend.search_dense.side_effect = NotImplementedError

    pipeline = _pipeline_mock(["p1", "p2", "orig"])
    service = RetrievalService(backend, pipeline=pipeline)

    import os

    with patch.dict(os.environ, {"QT_FUSION_WEIGHTED": "1"}):
        with patch(
            "src.internal.retrieval.service.variant_weighted_rrf_fuse"
        ) as weighted:
            results, mode = service.search("q", top_k=2)

    # Original didn't survive → weighted fuse must not be used (no faked 1.0).
    weighted.assert_not_called()
    assert "+rag_fusion" in mode
    assert len(results) >= 1


def test_weighted_fusion_weights_original_by_identity():
    """When the original survives but an EARLIER variant is dropped, the 1.0
    weight must track the original result set by identity, not by position."""

    def flaky_sparse(query, *, top_k, filters):
        if query == "p1":  # an earlier paraphrase fails and is dropped
            raise RuntimeError("p1 leg down")
        return [_make_result(f"r_{query}")]

    backend = MagicMock()
    backend.search_sparse.side_effect = flaky_sparse
    backend.search_dense.side_effect = NotImplementedError

    pipeline = _pipeline_mock(["p1", "p2", "orig"])
    service = RetrievalService(backend, pipeline=pipeline)

    import os

    with patch.dict(os.environ, {"QT_FUSION_WEIGHTED": "1"}):
        with patch(
            "src.internal.retrieval.service.variant_weighted_rrf_fuse",
            side_effect=lambda sets, weights: sets[0],
        ) as weighted:
            service.search("q", top_k=2)

    weighted.assert_called_once()
    passed_sets, passed_weights = weighted.call_args[0]
    # Surviving sets are [p2, orig]; the 1.0 weight must land on the orig set.
    assert passed_weights.count(1.0) == 1
    orig_index = passed_weights.index(1.0)
    assert passed_sets[orig_index][0].doc_id == "r_orig"
