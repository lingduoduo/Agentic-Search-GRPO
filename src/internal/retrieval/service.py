"""RetrievalService: selects backend from env and exposes search()."""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from .backends.base import RetrievalBackend, RetrievalResult
from .fusion import mmr_rerank, rrf_fuse, variant_weighted_rrf_fuse
from .query_optimizer import QueryOptimizer
from .result_cache import ResultCache

if TYPE_CHECKING:
    from src.internal.retrieval.reranker import Reranker
    from src.context.query_transform import QueryTransformPipeline

logger = logging.getLogger(__name__)


def _build_local_backend() -> RetrievalBackend:
    from src.internal.document_index.retrieval import (
        DenseRetrieverConfig,
        SparseRetrieverConfig,
    )

    from .backends.local import LocalBackend

    sparse_config = SparseRetrieverConfig(
        index_path=os.environ["BM25_INDEX_PATH"],
        corpus_path=os.environ.get("BM25_CORPUS_PATH", "data/corpus.jsonl"),
        topk=int(os.environ.get("BM25_TOP_K", "20")),
        k1=float(os.environ.get("BM25_K1", "1.2")),
        b=float(os.environ.get("BM25_B", "0.75")),
    )
    dense_config: DenseRetrieverConfig | None = None
    if os.environ.get("DENSE_MODEL_PATH"):
        dense_config = DenseRetrieverConfig.for_e5_base_v2(
            model_path=os.environ["DENSE_MODEL_PATH"],
            index_path=os.environ["DENSE_INDEX_PATH"],
            corpus_path=os.environ.get("DENSE_CORPUS_PATH", "data/corpus.jsonl"),
            topk=int(os.environ.get("BM25_TOP_K", "20")),
            device=os.environ.get("DENSE_DEVICE", "cpu"),
            redis_url=os.environ.get("DENSE_REDIS_URL"),
        )
    return LocalBackend(sparse_config, dense_config=dense_config)


def _build_llm() -> object:
    """Build an LLM client from GEN_AI_* environment variables."""
    from src.internal.llm.interfaces import LLMConfig
    from src.internal.llm.providers import OpenAICompatibleLLM

    return OpenAICompatibleLLM(
        LLMConfig(
            model_provider=os.environ.get("GEN_AI_MODEL_PROVIDER", "openai"),
            model_name=os.environ.get("GEN_AI_MODEL_VERSION", "gpt-4o-mini"),
            api_key=os.environ.get("GEN_AI_API_KEY")
            or os.environ.get("OPENAI_API_KEY"),
            api_base=os.environ.get("GEN_AI_API_BASE"),
            max_input_tokens=int(os.environ.get("GEN_AI_MAX_INPUT_TOKENS", "8192")),
        )
    )


def _build_backend() -> RetrievalBackend:
    name = os.environ.get("RETRIEVAL_BACKEND", "local").lower()
    if name == "local":
        return _build_local_backend()
    raise ValueError(f"Unknown RETRIEVAL_BACKEND: {name!r}. Supported values: local")


class RetrievalService:
    def __init__(
        self,
        backend: RetrievalBackend,
        reranker: "Reranker | None" = None,
        pipeline: "QueryTransformPipeline | None" = None,
        optimizer: "QueryOptimizer | None" = None,
        result_cache: "ResultCache | None" = None,
        router: object | None = None,
    ) -> None:
        self._backend = backend
        self._reranker = reranker
        self._pipeline = pipeline
        self._optimizer = optimizer
        self._result_cache = result_cache
        self._router = router

    @classmethod
    def from_env(cls) -> "RetrievalService":
        """Construct service from environment variables."""
        pipeline = None
        _qt_flags = (
            "QT_DECOMPOSE",
            "QT_HYDE",
            "QT_STEP_BACK",
            "QT_KEYWORDS",
            "QT_CONSTRUCT_FILTERS",
            "QT_MULTI_QUERY",
            "QT_ROUTER",
            "QT_REWRITE",
        )
        if any(
            os.environ.get(v, "").lower() in ("1", "true", "yes") for v in _qt_flags
        ):
            from src.internal.retrieval.query_transform_factory import (
                build_query_transform_pipeline_from_env,
            )

            pipeline = build_query_transform_pipeline_from_env(_build_llm())

        optimizer = QueryOptimizer.from_env()
        result_cache = None
        redis_url = os.environ.get("RESULT_CACHE_REDIS_URL")
        if redis_url:
            try:
                import redis as _redis

                rc = _redis.from_url(redis_url)
                result_cache = ResultCache(
                    rc, ttl_seconds=int(os.environ.get("RESULT_CACHE_TTL", "300"))
                )
            except Exception as exc:
                logger.warning("Result cache disabled: %s", exc)
        from src.internal.retrieval.reranker_factory import build_reranker_from_env
        from src.internal.routing.routing_factory import build_router_from_env

        router = build_router_from_env()
        return cls(
            _build_backend(),
            reranker=build_reranker_from_env(),
            pipeline=pipeline,
            optimizer=optimizer,
            result_cache=result_cache,
            router=router,
        )

    def _search_one(
        self,
        query: str,
        over_fetch: int,
        filters: dict | None,
    ) -> tuple[list[RetrievalResult], str]:
        """Run sparse+dense retrieval for one query variant. Returns (results, base_mode)."""
        sparse_results: list[RetrievalResult] = []
        dense_results: list[RetrievalResult] = []
        sparse_ok = dense_ok = False

        bm25_query = self._optimizer.expand(query) if self._optimizer else query
        with ThreadPoolExecutor(max_workers=2) as executor:
            sparse_future = executor.submit(
                self._backend.search_sparse,
                bm25_query,
                top_k=over_fetch,
                filters=filters,
            )
            dense_future = executor.submit(
                self._backend.search_dense, query, top_k=over_fetch, filters=filters
            )

        try:
            sparse_results = sparse_future.result()
            sparse_ok = True
        except Exception as exc:
            logger.warning("Sparse retrieval leg failed: %s", exc)

        try:
            dense_results = dense_future.result()
            dense_ok = True
        except NotImplementedError:
            pass
        except Exception as exc:
            logger.warning("Dense retrieval leg failed: %s", exc)

        if not sparse_ok and not dense_ok:
            raise RuntimeError("Both retrieval legs failed")

        if not dense_ok:
            return sparse_results, "sparse_only"
        elif not sparse_ok:
            return dense_results, "dense_only"
        else:
            return rrf_fuse([sparse_results, dense_results]), "hybrid"

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> tuple[list[RetrievalResult], str]:
        """Run retrieval, fuse with RRF+MMR, fall back gracefully.

        When pipeline is set: generates query variants, retrieves per variant in
        parallel, fuses all result sets via RRF → mode gains '+rag_fusion' suffix.
        Without pipeline: single query, identical behaviour to previous releases.
        Result cache (if enabled) is checked before retrieval and populated after.
        """
        routed = ""
        if self._router is not None:
            from src.internal.routing.route import RetrieverTarget

            decision = self._router.route(query)
            if decision.retriever in (
                RetrieverTarget.SQL,
                RetrieverTarget.GRAPH,
                RetrieverTarget.API,
            ):
                # No execution backend for these targets — construct-only.
                # Run ordinary retrieval anyway, since an unbacked route must
                # not cost the caller its results, and record the decision.
                routed = f"+routed:{decision.retriever.value}"

        if self._result_cache:
            cached = self._result_cache.get(query, filters, top_k)
            if cached is not None:
                return cached, "cached" + routed

        reranker_multiplier = (
            float(os.environ.get("RERANKER_OVER_FETCH_MULTIPLIER", "2.0"))
            if self._reranker
            else 1.0
        )
        over_fetch = math.ceil(
            top_k
            * int(os.environ.get("OVER_FETCH_MULTIPLIER", "2"))
            * reranker_multiplier
        )

        if self._pipeline:
            bundle = self._pipeline.transform(query, filters)
            variants = bundle.retrieval_variants(self._pipeline.max_variants)
            active_filters: dict | None = bundle.merged_filters or None
        else:
            variants = [query]
            active_filters = filters

        if (
            not variants
        ):  # guard: retrieval_variants() returned [] (shouldn't happen but be safe)
            variants = [query]

        embed_fn = getattr(self._backend, "embed", None)
        if (
            os.environ.get("QT_SEMANTIC_DEDUP", "").lower() in ("1", "true", "yes")
            and embed_fn is not None
            and len(variants) > 1
        ):
            from .fusion import dedup_variants

            threshold = float(os.environ.get("QT_SEMANTIC_DEDUP_THRESHOLD", "0.95"))
            variants = dedup_variants(variants, embed_fn, threshold=threshold)

        # retrieval_variants() places the original query LAST; track it by
        # identity so weighted fusion keeps its heaviest weight even if some
        # earlier variants are dropped on failure.
        last_idx = len(variants) - 1
        max_workers = min(len(variants), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                (
                    i == last_idx,
                    executor.submit(self._search_one, v, over_fetch, active_filters),
                )
                for i, v in enumerate(variants)
            ]
        result_sets_with_modes = []
        original_result_set: list[RetrievalResult] | None = None
        for is_original, f in futures:
            try:
                result = f.result()
            except Exception as exc:
                logger.warning("Variant retrieval failed, skipping: %s", exc)
                continue
            result_sets_with_modes.append(result)
            if is_original:
                original_result_set = result[0]
        if not result_sets_with_modes:
            result_sets_with_modes = [
                self._search_one(query, over_fetch, active_filters)
            ]
        all_result_sets = [rs for rs, _ in result_sets_with_modes]
        base_mode = (
            result_sets_with_modes[0][1] if result_sets_with_modes else "sparse_only"
        )

        reranker_fetch = math.ceil(top_k * reranker_multiplier)

        if len(all_result_sets) > 1:
            if (
                os.environ.get("QT_FUSION_WEIGHTED", "").lower() in ("1", "true", "yes")
                and original_result_set is not None
            ):
                # Give the original query the heaviest weight by identity, not
                # by position — earlier variants may have been dropped on failure.
                weights = [
                    1.0 if rs is original_result_set else 0.3 for rs in all_result_sets
                ]
                fused = variant_weighted_rrf_fuse(all_result_sets, weights)
            else:
                fused = rrf_fuse(all_result_sets)
            fused = mmr_rerank(fused, top_k=reranker_fetch)
            mode = f"{base_mode}+rag_fusion"
        else:
            raw = all_result_sets[0] if all_result_sets else []
            if base_mode == "hybrid":
                fused = mmr_rerank(raw, top_k=reranker_fetch)
            else:
                fused = raw[:reranker_fetch]
            mode = base_mode

        if self._reranker:
            from src.internal.retrieval.async_reranker import RerankerTimeoutError

            try:
                fused = self._reranker.rerank(query, fused, top_k)
                mode = f"{mode}+reranked"
            except RerankerTimeoutError as exc:
                # Async reranker timed out — degrade to the pre-rerank ordering
                # instead of failing the whole request.
                logger.warning("Reranker timed out, keeping fused order: %s", exc)
            except Exception as exc:
                # Any other reranker failure also degrades gracefully; do not
                # claim '+reranked' when the rerank fell back.
                logger.warning("Reranker failed, keeping fused order: %s", exc)

        if self._result_cache:
            self._result_cache.set(query, filters, top_k, fused)

        return fused, mode + routed

    def graph_search(
        self,
        query: str,
        top_k: int = 10,
        initial_k: int = 5,
        max_entity_queries: int = 3,
    ) -> list[RetrievalResult]:
        """Graph-augmented retrieval: seed search → entity expansion → RRF fusion."""
        from .graph_rag import graph_rag_search

        return graph_rag_search(
            query,
            service=self,
            top_k=top_k,
            initial_k=initial_k,
            max_entity_queries=max_entity_queries,
        )
