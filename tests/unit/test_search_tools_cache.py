"""`search_tool` serves repeated web-provider lookups from the serving cache.
The ``retrieval`` provider is not cached here — that lives in ``SearchClient``."""

from __future__ import annotations

import asyncio

import pytest

from src.internal.cache import serving
from src.internal.tools.search import SearchPage, search_tool


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
    assert len(serp_calls) == 1
    assert second[0].metadata == {"tags": ["fresh"]}


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
