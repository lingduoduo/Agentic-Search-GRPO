"""`search_tool` serves repeated web-provider lookups from the serving cache.
The ``retrieval`` provider is not cached here — that lives in ``SearchClient``."""

from __future__ import annotations

import asyncio
import copy

import pytest

from src.internal.cache import serving
from src.internal.cache.ttl_cache import TTLCache
from src.internal.tools.search import (
    SERPAPI_CIRCUIT_OPEN_ERROR,
    SearchPage,
    search_tool,
)


@pytest.fixture
def cache():
    serving.configure_serving_cache(60)
    yield serving.serving_cache()
    serving.reset_serving_cache()


@pytest.fixture
def serp_calls(monkeypatch) -> list:
    calls: list = []

    async def _fake_serpapi(query, *, page, page_size, timeout_seconds):
        calls.append((query, page, page_size))
        if query == "broken":
            return [SearchPage(error="rate limited")]
        if query == "nothing":
            return []
        if query == "timeout":  # empty-message provider timeout
            return [SearchPage(error="", timed_out=True)]
        if query == "blank":
            return [SearchPage()]
        return [
            SearchPage(
                title=query,
                summary="s",
                url=f"https://{query}",
                metadata={"tags": ["fresh"]},
            )
        ]

    monkeypatch.setattr("src.internal.tools.search.serpapi_search", _fake_serpapi)
    return calls


def _search(query: str, **kwargs) -> list[SearchPage]:
    return asyncio.run(search_tool(query, provider="serpapi", **kwargs))


def test_repeat_web_lookup_is_served_from_cache(serp_calls, cache):
    first = _search("faiss", page_size=3)
    second = _search("faiss", page_size=3)
    assert serp_calls == [("faiss", 1, 3)]
    assert [p.url for p in second] == [p.url for p in first] == ["https://faiss"]


def test_mutating_a_page_cannot_poison_later_hits(serp_calls, cache):
    first = _search("faiss")
    first[0].metadata["tags"].append("stale")  # SearchPage is frozen; dict is not
    second = _search("faiss")
    second[0].metadata["tags"].append("staler")  # a hit must be a copy too
    third = _search("faiss")
    assert len(serp_calls) == 1
    assert second[0].metadata == {"tags": ["fresh", "staler"]}
    assert third[0].metadata == {"tags": ["fresh"]}


def test_page_and_page_size_are_part_of_the_key(serp_calls, cache):
    _search("faiss", page_size=3)
    _search("faiss", page_size=5)
    _search("faiss", page_size=3, page=2)
    assert len(serp_calls) == 3


def test_error_pages_and_empty_results_are_not_cached(serp_calls, cache):
    _search("broken")
    _search("broken")
    _search("nothing")
    _search("nothing")
    assert len(serp_calls) == 4


def test_retrieval_provider_bypasses_the_web_cache(monkeypatch, cache):
    calls: list = []

    async def _fake_retrieval(query, **kwargs):
        calls.append(query)
        return [SearchPage(title=query, summary="s", url="https://r")]

    monkeypatch.setattr("src.internal.tools.search.retrieval_search", _fake_retrieval)
    asyncio.run(search_tool("q", provider="retrieval"))
    asyncio.run(search_tool("q", provider="retrieval"))
    assert calls == ["q", "q"]


def test_unconfigured_cache_calls_the_provider_every_time(serp_calls):
    serving.reset_serving_cache()
    _search("faiss")
    _search("faiss")
    assert len(serp_calls) == 2


@pytest.mark.parametrize("query", ["timeout", "blank"])
def test_blank_page_is_not_cached(serp_calls, cache, query):
    _search(query)
    _search(query)
    assert [c[0] for c in serp_calls] == [query, query]


def test_is_blank_only_for_an_empty_page():
    assert SearchPage().is_blank
    assert SearchPage(timed_out=True).is_blank
    assert not SearchPage(url="https://x").is_blank
    assert not SearchPage(error="boom").is_blank
    assert not SearchPage(title="t").is_blank


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return _Clock()


@pytest.fixture
def stale_cache(monkeypatch, clock):
    cache = TTLCache(60, stale_seconds=3600, clock=clock)
    monkeypatch.setattr(serving, "_cache", cache)
    return cache


@pytest.fixture
def serp(monkeypatch):
    state = {
        "reply": [
            SearchPage(
                title="t", summary="s", url="https://x", metadata={"acl": ["public"]}
            )
        ],
        "calls": 0,
    }

    async def _fake(query, *, page, page_size, timeout_seconds):
        state["calls"] += 1
        return copy.deepcopy(state["reply"])

    monkeypatch.setattr("src.internal.tools.search.serpapi_search", _fake)
    return state


FAILURES = {
    "timeout": [SearchPage(timed_out=True)],
    "blank": [SearchPage()],
    "error": [SearchPage(error="rate limited")],
    "circuit_open": [SearchPage(error=SERPAPI_CIRCUIT_OPEN_ERROR)],
}


@pytest.mark.parametrize("failure", FAILURES)
def test_failure_after_expiry_serves_stale_pages(serp, stale_cache, clock, failure):
    _search("q")
    clock.now += 61
    serp["reply"] = FAILURES[failure]
    pages = _search("q")
    assert serp["calls"] == 2
    assert [p.url for p in pages] == ["https://x"]
    assert pages[0].metadata == {"acl": ["public"], "stale": True}


def test_empty_result_never_falls_back(serp, stale_cache, clock):
    _search("q")
    clock.now += 61
    serp["reply"] = []
    assert _search("q") == []


@pytest.mark.parametrize("failure", FAILURES)
def test_no_stale_entry_returns_the_failure_unchanged(serp, stale_cache, failure):
    serp["reply"] = FAILURES[failure]
    assert _search("q") == FAILURES[failure]


def test_beyond_the_grace_window_returns_the_failure(serp, stale_cache, clock):
    _search("q")
    clock.now += 60 + 3600 + 1
    serp["reply"] = FAILURES["timeout"]
    assert _search("q") == FAILURES["timeout"]


def test_mixed_list_is_returned_as_is(serp, stale_cache, clock):
    _search("q")
    clock.now += 61
    mixed = [SearchPage(title="new", url="https://new"), SearchPage(error="partial")]
    serp["reply"] = mixed
    assert _search("q") == mixed


def test_stale_hit_is_a_copy(serp, stale_cache, clock):
    _search("q")
    clock.now += 61
    serp["reply"] = FAILURES["error"]
    _search("q")[0].metadata["acl"].append("user:mallory")
    again = _search("q")
    assert again[0].metadata == {"acl": ["public"], "stale": True}
    cached = stale_cache.get_stale(("web", "serpapi", "q", 1, 5))
    assert cached[0].metadata == {"acl": ["public"]}


def test_recovery_after_a_stale_fallback_is_fresh_again(serp, stale_cache, clock):
    _search("q")
    clock.now += 61
    serp["reply"] = FAILURES["error"]
    _search("q")
    serp["reply"] = [SearchPage(title="t2", url="https://y")]
    assert [p.url for p in _search("q")] == ["https://y"]
    fresh = _search("q")
    assert serp["calls"] == 3
    assert fresh[0].metadata == {}
