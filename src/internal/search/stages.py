"""Stage protocols and compatibility adapters for the existing services."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

import httpx

from src.context import ChatMessage, ContextDocument
from src.context.retrieval.client import SearchClient
from src.context.search import SearchResult
from src.internal.cache.serving import serving_cache
from src.internal.retrieval.acl import acl_allows
from src.internal.search.process_search_query import weighted_reciprocal_rank_fusion

if TYPE_CHECKING:
    from src.model.serving import ServerManager

from .models import CandidateSet, GeneratedAnswer, RankedEvidence


@runtime_checkable
class RetrievalStage(Protocol):
    async def retrieve(
        self,
        query: str,
        history: list[ChatMessage],
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> CandidateSet: ...


@runtime_checkable
class RankingStage(Protocol):
    async def rank(
        self, query: str, candidates: CandidateSet, top_k: int
    ) -> RankedEvidence: ...


@runtime_checkable
class InferenceStage(Protocol):
    async def generate(
        self,
        query: str,
        history: list[ChatMessage],
        evidence: RankedEvidence,
    ) -> GeneratedAnswer: ...


class SearchClientRetrievalStage:
    """Translate the normalized retrieval call to the existing SearchClient."""

    def __init__(self, client: SearchClient, *, provider: str = "retrieval") -> None:
        self._client = client
        self._provider = provider

    async def retrieve(
        self,
        query: str,
        history: list[ChatMessage],
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> CandidateSet:
        results = await self._client.retrieve_one(query, topk=top_k, filters=filters)
        # Enforce, don't just forward: the bundled servers honour access_acl but
        # a third-party backend need not, and these candidates go on to the
        # model's context.
        results = [r for r in results if acl_allows(r.metadata, filters)]
        return CandidateSet(
            query=query,
            candidates=results,
            provider=self._provider,
            filters=filters,
            metadata={"history_messages": len(history)},
        )


def _as_document(result: SearchResult, index: int, provider: str) -> ContextDocument:
    metadata = dict(result.metadata)
    metadata.setdefault("source_provider", provider)
    return ContextDocument.from_search_result(result, index=index, metadata=metadata)


class FusionRankingStage:
    """Adapt existing weighted RRF fusion to the normalized ranking contract."""

    def __init__(
        self,
        additional_sets: Sequence[CandidateSet] = (),
        *,
        weights: Sequence[float] | None = None,
    ) -> None:
        self._additional_sets = list(additional_sets)
        self._weights = list(weights) if weights is not None else None

    async def rank(
        self, query: str, candidates: CandidateSet, top_k: int
    ) -> RankedEvidence:
        sets = [candidates, *self._additional_sets]
        weights = self._weights or [1.0] * len(sets)
        if len(weights) != len(sets):
            raise ValueError("weights and candidate sets must have the same length")
        fused = weighted_reciprocal_rank_fusion(
            [item.candidates for item in sets], weights
        )
        provider_by_key = {
            (result.url, result.contents[:100]): candidate_set.provider
            for candidate_set in sets
            for result in candidate_set.candidates
        }
        documents = [
            _as_document(
                result,
                index,
                provider_by_key[(result.url, result.contents[:100])],
            )
            for index, result in enumerate(fused[:top_k], 1)
        ]
        return RankedEvidence(
            query=query,
            evidence=documents,
            metadata={
                "operations": ["weighted_rrf"],
                "providers": [s.provider for s in sets],
            },
        )


class RerankHTTPRankingStage:
    """Translate candidates to the unchanged standalone ``/rerank`` payload."""

    def __init__(
        self,
        rerank_url: str,
        *,
        timeout: float = 10.0,
        document_contents: Callable[[SearchResult], str] | None = None,
        send_top_k: bool = True,
    ) -> None:
        self._url = f"{rerank_url.rstrip('/')}/rerank"
        self._timeout = timeout
        self._document_contents = document_contents or (
            lambda candidate: candidate.contents
        )
        self._send_top_k = send_top_k

    async def rank(
        self, query: str, candidates: CandidateSet, top_k: int
    ) -> RankedEvidence:
        payloads = [
            {
                "document": {
                    "contents": self._document_contents(candidate),
                    "_idx": str(index),
                }
            }
            for index, candidate in enumerate(candidates.candidates)
        ]
        cache = serving_cache()
        cache_key = (
            "rerank",
            self._url,
            query,
            top_k if self._send_top_k else None,
            tuple(p["document"]["contents"] for p in payloads),
        )
        ranked = cache.get(cache_key) if cache is not None else None
        if ranked is None:
            async with httpx.AsyncClient() as client:
                body = {
                    "queries": [query],
                    "documents": [payloads],
                    "return_scores": True,
                }
                if self._send_top_k:
                    body["rerank_topk"] = top_k
                response = await client.post(
                    self._url, json=body, timeout=self._timeout
                )
                response.raise_for_status()
            ranked = response.json()["result"][0]
            if cache is not None and ranked:
                cache.set(cache_key, ranked)
        evidence = []
        for position, item in enumerate(ranked, 1):
            try:
                index = int(item.get("document", {}).get("_idx", -1))
            except (TypeError, ValueError):
                continue
            if not 0 <= index < len(candidates.candidates):
                continue
            original = candidates.candidates[index]
            scored = SearchResult(
                contents=original.contents,
                title=original.title,
                url=original.url,
                score=float(item.get("score", original.score)),
                metadata=original.metadata,
            )
            evidence.append(_as_document(scored, position, candidates.provider))
        return RankedEvidence(
            query=query,
            evidence=evidence,
            metadata={"operations": ["external_rerank"], "rerank_url": self._url},
        )


_CITATION = re.compile(r"\[([^\[\]]+)\]")


class ServingInferenceStage:
    """Adapt text/evidence inputs to the token-based ServerManager boundary."""

    def __init__(
        self,
        manager: ServerManager,
        *,
        encode: Callable[[str], list[int]],
        decode: Callable[[list[int]], str],
        sampling_params: dict[str, Any] | None = None,
        request_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._manager = manager
        self._encode = encode
        self._decode = decode
        self._sampling_params = dict(sampling_params or {})
        self._request_id_factory = request_id_factory or (lambda: str(uuid4()))

    async def generate(
        self,
        query: str,
        history: list[ChatMessage],
        evidence: RankedEvidence,
    ) -> GeneratedAnswer:
        prompt = self._build_prompt(query, history, evidence.evidence)
        output_ids = await self._manager.generate(
            self._request_id_factory(), self._encode(prompt), self._sampling_params
        )
        answer = self._decode(output_ids)
        valid = {document.id for document in evidence.evidence}
        citations = list(
            dict.fromkeys(c for c in _CITATION.findall(answer) if c in valid)
        )
        return GeneratedAnswer(
            answer=answer,
            citations=citations,
            metadata={"evidence_count": len(evidence.evidence)},
        )

    @staticmethod
    def _build_prompt(
        query: str, history: list[ChatMessage], evidence: list[ContextDocument]
    ) -> str:
        history_text = "\n".join(f"{m.role}: {m.content}" for m in history)
        evidence_text = "\n\n".join(
            f"{doc.citation} {doc.title}\n{doc.content}" for doc in evidence
        )
        return (
            "Answer the question using only the cited evidence.\n"
            f"Conversation:\n{history_text}\n\nEvidence:\n{evidence_text}\n\n"
            f"Question: {query}"
        )


# Concise adapter aliases for callers that describe the dependency, not its role.
SearchClientAdapter = SearchClientRetrievalStage
FusionAdapter = FusionRankingStage
RerankHTTPAdapter = RerankHTTPRankingStage
ServingAdapter = ServingInferenceStage
