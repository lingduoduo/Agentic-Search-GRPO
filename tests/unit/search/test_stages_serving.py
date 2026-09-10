"""Serving-side behaviour of the pipeline stages: the retrieval adapter enforces
the ACL it forwarded, and the rerank adapter serves repeats from the serving
cache."""

from __future__ import annotations

import asyncio

import pytest

from src.context import ChatMessage
from src.context.search import SearchResult
from src.internal.cache import serving
from src.internal.search.models import CandidateSet
from src.internal.search.stages import (
    RerankHTTPRankingStage,
    SearchClientRetrievalStage,
)


class _LeakyClient:
    """Returns a document the filters forbid, as a third-party backend might."""

    async def retrieve_one(self, query, topk=None, filters=None):
        return [
            SearchResult(contents="open", title="Open", score=0.9, metadata={}),
            SearchResult(
                contents="secret",
                title="Secret",
                score=0.8,
                metadata={"acl": ["user:b"]},
            ),
            SearchResult(
                contents="mine",
                title="Mine",
                score=0.7,
                metadata={"acl": ["user:a"]},
            ),
        ]


def test_retrieval_stage_enforces_the_acl_it_forwarded():
    stage = SearchClientRetrievalStage(_LeakyClient())
    result = asyncio.run(
        stage.retrieve(
            "q", [ChatMessage(role="user", content="q")], {"access_acl": ["user:a"]}, 5
        )
    )
    assert [c.title for c in result.candidates] == ["Open", "Mine"]
    assert result.filters == {"access_acl": ["user:a"]}


def test_retrieval_stage_without_filters_keeps_everything():
    stage = SearchClientRetrievalStage(_LeakyClient())
    result = asyncio.run(
        stage.retrieve("q", [ChatMessage(role="user", content="q")], None, 5)
    )
    assert len(result.candidates) == 3


class _HTTPResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "result": [
                [
                    {"document": {"_idx": "1"}, "score": 0.95},
                    {"document": {"_idx": "0"}, "score": 0.4},
                ]
            ]
        }


class _HTTPClient:
    calls: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, *, json, timeout):
        self.calls.append((url, json, timeout))
        return _HTTPResponse()


@pytest.fixture
def http(monkeypatch):
    _HTTPClient.calls = []
    monkeypatch.setattr("src.internal.search.stages.httpx.AsyncClient", _HTTPClient)
    return _HTTPClient


@pytest.fixture
def cache():
    serving.configure_serving_cache(60)
    yield serving.serving_cache()
    serving.reset_serving_cache()


def _candidates(second: str = "two") -> CandidateSet:
    return CandidateSet(
        query="q",
        candidates=[
            SearchResult(contents="one", title="One", score=0.2),
            SearchResult(contents=second, title="Two", score=0.3),
        ],
        provider="retrieval",
    )


def test_rerank_repeat_is_served_from_cache(http, cache):
    stage = RerankHTTPRankingStage("http://reranker/")
    first = asyncio.run(stage.rank("q", _candidates(), 2))
    second = asyncio.run(stage.rank("q", _candidates(), 2))
    assert len(http.calls) == 1
    assert [d.title for d in second.evidence] == [d.title for d in first.evidence]
    assert second.evidence[0].score == 0.95


def test_rerank_different_contents_miss(http, cache):
    stage = RerankHTTPRankingStage("http://reranker/")
    asyncio.run(stage.rank("q", _candidates("two"), 2))
    asyncio.run(stage.rank("q", _candidates("deux"), 2))
    assert len(http.calls) == 2


def test_rerank_without_cache_posts_every_time(http):
    serving.reset_serving_cache()
    stage = RerankHTTPRankingStage("http://reranker/")
    asyncio.run(stage.rank("q", _candidates(), 2))
    asyncio.run(stage.rank("q", _candidates(), 2))
    assert len(http.calls) == 2
