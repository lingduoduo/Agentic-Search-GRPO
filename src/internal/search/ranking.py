"""Default candidate ranking policy shared by web search entry points."""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

from src.context import ContextDocument
from src.context.utils import mmr_rerank
from src.internal.resilience.circuit_breaker import CircuitOpenError

from .models import CandidateSet, RankedEvidence

logger = logging.getLogger(__name__)


class _Reranker(Protocol):
    async def rank(
        self, query: str, candidates: CandidateSet, top_k: int
    ) -> RankedEvidence: ...


class DefaultRankingStage:
    """Deduplicate, optionally rerank, diversify, and label candidates."""

    def __init__(self, reranker: _Reranker | None = None) -> None:
        self._reranker = reranker

    @staticmethod
    def documents(candidates: CandidateSet) -> list[ContextDocument]:
        documents = []
        for index, result in enumerate(candidates.candidates, 1):
            metadata = dict(result.metadata)
            metadata.setdefault("source_provider", candidates.provider)
            documents.append(
                ContextDocument.from_search_result(
                    result, index=index, metadata=metadata
                )
            )
        return documents

    async def rank(
        self, query: str, candidates: CandidateSet, top_k: int
    ) -> RankedEvidence:
        deduplicated = self._deduplicate(candidates)
        documents = self.documents(deduplicated)
        operations = ["deduplicate"]
        rerank_status = "disabled"
        degraded = False

        if self._reranker is not None and documents:
            try:
                result = await self._reranker.rank(query, deduplicated, len(documents))
                if result.evidence:
                    documents = result.evidence
                operations.append("external_rerank")
                rerank_status = "applied"
            except CircuitOpenError as exc:
                logger.warning("Rerank skipped, using original order: %s", exc)
                rerank_status = "circuit_open"
                degraded = True
            except httpx.TimeoutException as exc:
                logger.warning(
                    "Rerank request timed out, using original order: %s", exc
                )
                rerank_status = "timeout"
                degraded = True
            except Exception as exc:
                logger.warning("Rerank request failed, using original order: %s", exc)
                rerank_status = "error"
                degraded = True

        if degraded:
            diversified = documents[:top_k]
            operations.append("truncate")
        else:
            diversified = mmr_rerank(documents, topk=top_k)
            operations.append("mmr")
        evidence = [
            ContextDocument(
                id=f"D{index}",
                title=document.title,
                content=document.content,
                url=document.url,
                score=document.score,
                metadata={**document.metadata, "mmr_rank": index},
            )
            for index, document in enumerate(diversified, 1)
        ]
        return RankedEvidence(
            query=query,
            evidence=evidence,
            metadata={
                "operations": operations,
                "candidate_count": len(candidates.candidates),
                "deduplicated_count": len(deduplicated.candidates),
                "returned_count": len(evidence),
                "rerank_status": rerank_status,
                "degraded": degraded,
            },
        )

    @staticmethod
    def _deduplicate(candidates: CandidateSet) -> CandidateSet:
        unique = []
        seen: set[tuple[str | None, str]] = set()
        for candidate in candidates.candidates:
            key = (candidate.url, candidate.contents[:160])
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return CandidateSet(
            query=candidates.query,
            candidates=unique,
            provider=candidates.provider,
            filters=candidates.filters,
            metadata=candidates.metadata,
        )
