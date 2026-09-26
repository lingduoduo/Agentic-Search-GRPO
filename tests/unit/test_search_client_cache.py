"""`SearchClient.retrieve` serves repeats from the serving cache, per query,
with the serialised filters in the key so two callers with different ACLs never
share an entry."""

from __future__ import annotations

import asyncio

import aiohttp
import pytest
from src.internal.observability.prometheus import REGISTRY

from src.context.retrieval.client import SearchClient, SearchClientConfig
from src.internal.cache import serving
from src.internal.cache.ttl_cache import TTLCache


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
            else [
                {
                    "document": {
                        "title": q,
                        "contents": f"body of {q}",
                        "metadata": {"acl": ["public"]},
                    },
                    "score": 1.0,
                }
            ]
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


def test_mutating_a_result_cannot_poison_later_hits(posts, cache):
    client = _client()
    first = _run(client.retrieve(["a"]))
    first[0][0].metadata["acl"].append("user:mallory")
    second = _run(client.retrieve(["a"]))
    second[0][0].metadata["acl"].append("user:eve")
    third = _run(client.retrieve(["a"]))
    assert len(posts) == 1
    assert second[0][0].metadata["acl"] == ["public", "user:eve"]
    assert third[0][0].metadata["acl"] == ["public"]


def test_short_server_response_is_padded_and_logged(monkeypatch, cache, caplog):
    class _ShortSession(_FakeSession):
        def post(self, url, json):
            self._posts.append(json)
            return _FakeResponse({"results": [[{"document": {"title": "only"}}]]})

    posts: list = []
    monkeypatch.setattr(
        "src.context.retrieval.client.aiohttp.ClientSession",
        lambda *, timeout: _ShortSession(posts),
    )
    with caplog.at_level("WARNING", logger="src.context.retrieval.client"):
        rows = _run(_client().retrieve(["a", "b"]))
    assert [len(row) for row in rows] == [1, 0]
    assert "returned 1 rows for 2 queries" in caplog.text


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


class _FailingSession(_FakeSession):
    def __init__(self, posts, exc):
        super().__init__(posts)
        self._exc = exc

    def post(self, url, json):
        self._posts.append(json)
        raise self._exc


class _NoBackoff:
    backoff_base_seconds = 0.0
    timeout_seconds = 1.0
    max_retries = 1


@pytest.fixture
def stale_cache(monkeypatch):
    now = [100.0]
    cache = TTLCache(60, stale_seconds=3600, clock=lambda: now[0])
    monkeypatch.setattr(serving, "_cache", cache)
    return cache, now


def _fail_with(monkeypatch, exc):
    monkeypatch.setattr(
        "src.context.retrieval.client.aiohttp.ClientSession",
        lambda *, timeout: _FailingSession([], exc),
    )
    monkeypatch.setattr(
        "src.context.retrieval.client._client_policy", lambda: _NoBackoff()
    )


def test_failed_post_serves_stale_rows_labelled(monkeypatch, posts, stale_cache):
    cache, now = stale_cache
    _run(_client().retrieve(["a", "b"]))
    now[0] += 61
    _fail_with(monkeypatch, ConnectionError("down"))
    rows = _run(_client().retrieve(["a", "b"]))
    assert [row[0].title for row in rows] == ["a", "b"]
    assert rows[0][0].metadata == {"acl": ["public"], "stale": True}
    key = ("retrieve", "http://localhost:8001/retrieve", "a", 5, "")  # default topk
    assert cache.get_stale(key)[0]["document"]["metadata"] == {"acl": ["public"]}


def test_one_query_without_a_stale_row_raises(monkeypatch, posts, stale_cache):
    _, now = stale_cache
    _run(_client().retrieve(["a"]))
    now[0] += 61
    _fail_with(monkeypatch, ConnectionError("down"))
    with pytest.raises(RuntimeError):
        _run(_client().retrieve(["a", "b"]))


def test_fresh_hits_plus_stale_misses_label_only_the_stale(
    monkeypatch, posts, stale_cache
):
    _, now = stale_cache
    _run(_client().retrieve(["a"]))
    now[0] += 61
    _run(_client().retrieve(["b"]))  # b is fresh, a is stale
    _fail_with(monkeypatch, ConnectionError("down"))
    rows = _run(_client().retrieve(["a", "b"]))
    assert rows[0][0].metadata.get("stale") is True
    assert "stale" not in rows[1][0].metadata


def test_cancellation_never_serves_stale(monkeypatch, posts, stale_cache):
    _, now = stale_cache
    _run(_client().retrieve(["a"]))
    now[0] += 61
    _fail_with(monkeypatch, asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        _run(_client().retrieve(["a"]))


def _http_error(status: int) -> aiohttp.ClientResponseError:
    return aiohttp.ClientResponseError(None, (), status=status)


@pytest.mark.parametrize("status", [400, 403])
def test_client_error_reraises_despite_a_stale_row(
    monkeypatch, posts, stale_cache, status
):
    _, now = stale_cache
    _run(_client().retrieve(["a"]))
    now[0] += 61
    _fail_with(monkeypatch, _http_error(status))
    with pytest.raises(aiohttp.ClientResponseError) as exc_info:
        _run(_client().retrieve(["a"]))
    assert exc_info.value.status == status


@pytest.mark.parametrize("status", [429, 503, 500])
def test_rate_limit_and_server_errors_serve_stale(
    monkeypatch, posts, stale_cache, status
):
    # 429 re-raises directly from _post_json; 5xx exhaust the single retry and
    # arrive as RuntimeError whose __cause__ is the ClientResponseError.
    _, now = stale_cache
    _run(_client().retrieve(["a"]))
    now[0] += 61
    _fail_with(monkeypatch, _http_error(status))
    rows = _run(_client().retrieve(["a"]))
    assert rows[0][0].metadata == {"acl": ["public"], "stale": True}


def test_is_client_error_reads_the_cause():
    from src.context.retrieval.client import _is_client_error

    wrapped = RuntimeError("retries exhausted")
    wrapped.__cause__ = _http_error(404)
    assert _is_client_error(wrapped)
    wrapped.__cause__ = _http_error(500)
    assert not _is_client_error(wrapped)
    assert not _is_client_error(_http_error(429))


def _stale_serves() -> float:
    return (
        REGISTRY.get_sample_value(
            "agentic_search_stale_cache_serves_total", {"source": "retrieval"}
        )
        or 0.0
    )


def test_stale_serve_counts_once_per_call(monkeypatch, posts, stale_cache):
    _, now = stale_cache
    _run(_client().retrieve(["a", "b", "c"]))
    now[0] += 61
    _fail_with(monkeypatch, ConnectionError("down"))
    before = _stale_serves()
    _run(_client().retrieve(["a", "b", "c"]))
    assert _stale_serves() == before + 1


def test_no_stale_serve_counts_nothing(monkeypatch, posts, stale_cache):
    _, now = stale_cache
    _run(_client().retrieve(["a"]))
    now[0] += 61
    before = _stale_serves()
    _fail_with(monkeypatch, ConnectionError("down"))
    with pytest.raises(RuntimeError):
        _run(_client().retrieve(["a", "b"]))  # b has no stale row
    _fail_with(monkeypatch, _http_error(400))
    with pytest.raises(aiohttp.ClientResponseError):
        _run(_client().retrieve(["a"]))
    assert _stale_serves() == before


def test_mutating_a_stale_result_cannot_poison_the_next_stale_serve(
    monkeypatch, posts, stale_cache
):
    _, now = stale_cache
    _run(_client().retrieve(["a"]))
    now[0] += 61
    _fail_with(monkeypatch, ConnectionError("down"))
    first = _run(_client().retrieve(["a"]))
    first[0][0].metadata["acl"].append("user:mallory")
    second = _run(_client().retrieve(["a"]))
    assert second[0][0].metadata == {"acl": ["public"], "stale": True}
