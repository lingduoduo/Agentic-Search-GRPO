from __future__ import annotations

import asyncio
import os

from src.internal.retrieval.backends.base import RetrievalResult


class TwoStageReranker:
    """Chains a fast pre-filter on all candidates, then a heavy scorer on top-N."""

    def __init__(
        self,
        fast_reranker,
        heavy_reranker,
        *,
        pre_filter_top_n: int = 50,
    ) -> None:
        if pre_filter_top_n <= 0:
            raise ValueError("pre_filter_top_n must be positive")
        self._fast = fast_reranker
        self._heavy = heavy_reranker
        self._pre_n = pre_filter_top_n

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        if not results or top_k <= 0:
            return []
        candidates = self._fast.rerank(query, results, max(self._pre_n, top_k))
        return self._heavy.rerank(query, candidates, top_k)

    async def arerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        if not results or top_k <= 0:
            return []
        candidate_count = max(self._pre_n, top_k)
        if hasattr(self._fast, "arerank"):
            candidates = await self._fast.arerank(query, results, candidate_count)
        else:
            candidates = await asyncio.to_thread(
                self._fast.rerank, query, results, candidate_count
            )
        if hasattr(self._heavy, "arerank"):
            return await self._heavy.arerank(query, candidates, top_k)
        return await asyncio.to_thread(self._heavy.rerank, query, candidates, top_k)

    @classmethod
    def from_env(cls, fast_reranker, heavy_reranker) -> "TwoStageReranker":
        return cls(
            fast_reranker,
            heavy_reranker,
            pre_filter_top_n=int(os.environ.get("RERANKER_PRE_FILTER_TOP_N", "50")),
        )
