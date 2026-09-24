"""Session-aware composition of retrieval, ranking, and grounded inference."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from src.context import ChatMessage

from .models import CandidateSet, GeneratedAnswer, RankedEvidence

from .context import build_retrieval_context

logger = logging.getLogger(__name__)


class RetrievalStage(Protocol):
    async def retrieve(self, query, history, filters, top_k) -> CandidateSet: ...


class RankingStage(Protocol):
    async def rank(self, query, candidates, top_k) -> RankedEvidence: ...


class InferenceStage(Protocol):
    async def generate(self, query, history, evidence) -> GeneratedAnswer: ...


class SearchPipeline:
    """Compose normalized stages behind the existing web result contract."""

    def __init__(
        self,
        retrieval: RetrievalStage,
        ranking: RankingStage,
        inference: InferenceStage,
    ) -> None:
        self._retrieval = retrieval
        self._ranking = ranking
        self._inference = inference

    async def run(
        self,
        query: str,
        history: list[ChatMessage],
        filters: dict[str, Any] | None,
        top_k: int,
        source_provider: str,
        *,
        retrieval_query: str | None = None,
    ) -> tuple:
        context = build_retrieval_context(query, history)
        search_query = retrieval_query or context.retrieval_query
        extra: dict[str, Any] = {
            "source_provider": source_provider,
            "retrieval_query": search_query,
        }
        try:
            candidates = await self._retrieval.retrieve(
                search_query,
                context.history,
                filters,
                top_k,
            )
        except Exception as exc:
            logger.warning("Retrieval failed for %r: %s", query, exc)
            extra["search_fallback"] = "retrieval_unavailable"
            return (
                "No sources are reachable right now. Please try again shortly.",
                [],
                [],
                "search",
                extra,
            )

        extra.update(candidates.metadata)
        if candidates.metadata.get("status") == "unreachable":
            executed = candidates.metadata.get("executed_queries", [])
            query_lines = "\n".join(f"- {item}" for item in executed)
            suffix = f"\n\nExecuted queries:\n{query_lines}" if query_lines else ""
            return (
                "No sources are reachable right now. Please try again shortly."
                + suffix,
                [],
                [],
                "search",
                extra,
            )

        evidence = await self._ranking.rank(search_query, candidates, top_k)
        extra["ranking"] = evidence.metadata
        if not evidence.evidence:
            return f"No results found for: {query}", [], [], "search", extra

        try:
            generated = await self._inference.generate(query, context.history, evidence)
            extra["inference"] = generated.metadata
            return (
                generated.answer,
                generated.citations,
                evidence.evidence,
                "search",
                extra,
            )
        except Exception as exc:
            logger.warning("Inference failed for %r: %s", query, exc)
            extra["inference_fallback"] = "synthesis_failed"
            answer = "Search (synthesis failed) results:\n\n" + "\n".join(
                f"{doc.citation} {doc.title}" for doc in evidence.evidence
            )
            return (
                answer,
                [doc.citation for doc in evidence.evidence],
                evidence.evidence,
                "search",
                extra,
            )
