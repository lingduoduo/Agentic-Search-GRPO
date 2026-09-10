"""`SearchClient.retrieve` serves repeats from the serving cache, per query,
with the serialised filters in the key so two callers with different ACLs never
share an entry."""

from __future__ import annotations

import asyncio

import pytest

from src.context.retrieval.client import SearchClient, SearchClientConfig
from src.internal.cache import serving


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return self._payload


class _FakeSession:
    """Answers each query with one document titled after the query."""

    def __init__(self, posts: list, *, empty_for: set[str] = frozenset()) -> None:
        self._posts = posts
        self._empty_for = empty_for
        self.closed = False

    def post(self, url, json):
        self._posts.append(json)
        rows = [
            []
            if q in self._empty_for
            else [{"document": {"title": q, "contents": f"body of {q}"}, "score": 1.0}]
            for q in json["queries"]
        ]
        return _FakeResponse({"results": rows})

    async def close(self):
        self.closed = True


@pytest.fixture
def posts(monkeypatch) -> list:
    posts: list = []
    monkeypatch.setattr(
        "src.context.retrieval.client.aiohttp.ClientSession",
        lambda *, timeout: _FakeSession(posts),
    )
    return posts


@pytest.fixture
def cache():
    serving.configure_serving_cache(60)
    yield serving.serving_cache()
    serving.reset_serving_cache()


def _client() -> SearchClient:
    return SearchClient(SearchClientConfig(url="http://localhost:8001/retrieve"))


def _run(coro):
    return asyncio.run(coro)


def test_repeat_is_served_without_a_post(posts, cache):
    client = _client()
    first = _run(client.retrieve(["a"], topk=3, filters={"access_acl": ["public"]}))
    second = _run(client.retrieve(["a"], topk=3, filters={"access_acl": ["public"]}))
    assert len(posts) == 1
    assert [r.title for r in second[0]] == [r.title for r in first[0]] == ["a"]
    # A hit is a fresh object; mutating it cannot poison later hits.
    assert second[0][0] is not first[0][0]


def test_different_filters_miss(posts, cache):
    client = _client()
    _run(client.retrieve(["a"], filters={"access_acl": ["public"]}))
    _run(client.retrieve(["a"], filters={"access_acl": ["public", "user:x"]}))
    assert len(posts) == 2


def test_different_topk_and_url_miss(posts, cache):
    _run(_client().retrieve(["a"], topk=3))
    _run(_client().retrieve(["a"], topk=5))
    other = SearchClient(SearchClientConfig(url="http://localhost:9999/retrieve"))
    _run(other.retrieve(["a"], topk=3))
    assert len(posts) == 3


def test_batch_posts_only_the_misses_and_keeps_order(posts, cache):
    client = _client()
    _run(client.retrieve(["b"]))
    rows = _run(client.retrieve(["a", "b", "c"]))
    assert posts[-1]["queries"] == ["a", "c"]
    assert [row[0].title for row in rows] == ["a", "b", "c"]


def test_fully_cached_batch_makes_no_post(posts, cache):
    client = _client()
    _run(client.retrieve(["a", "b"]))
    _run(client.retrieve(["b", "a"]))
    assert len(posts) == 1


def test_empty_rows_are_not_cached(monkeypatch, cache):
    posts: list = []
    monkeypatch.setattr(
        "src.context.retrieval.client.aiohttp.ClientSession",
        lambda *, timeout: _FakeSession(posts, empty_for={"nothing"}),
    )
    client = _client()
    assert _run(client.retrieve(["nothing"])) == [[]]
    assert _run(client.retrieve(["nothing"])) == [[]]
    assert len(posts) == 2


def test_unconfigured_cache_posts_every_time(posts):
    serving.reset_serving_cache()
    client = _client()
    _run(client.retrieve(["a"]))
    _run(client.retrieve(["a"]))
    assert len(posts) == 2
