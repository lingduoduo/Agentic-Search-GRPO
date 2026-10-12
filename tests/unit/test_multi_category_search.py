"""search_categories: up to three categories in parallel, fused with RRF."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.internal.tools.base import InvalidToolInput
from src.internal.tools.search import (
    SearchPage,
    normalize_search_domains,
    prepare_domain_query,
    restricted_query,
    search_categories,
    search_category,
)


def _pages(*urls):
    return [SearchPage(title=u, summary="s", url=u) for u in urls]


class _Provider:
    def __init__(self, answers, delays=None):
        self.answers, self.delays = answers, delays or {}
        self.calls: list[str] = []
        self.started: dict[str, float] = {}

    async def __call__(self, query, page_size):
        self.calls.append(query)
        self.started[query] = time.perf_counter()
        await asyncio.sleep(self.delays.get(query, 0.0))
        answer = self.answers.get(query, [])
        if isinstance(answer, BaseException):
            raise answer
        return answer


def _scoped(q, d):
    return restricted_query(q, d)


def _suffix(q, d):
    return prepare_domain_query(q, d)


# --- normalize_search_domains ----------------------------------------------


def test_normalize_dedupes_after_aliasing_then_caps():
    assert normalize_search_domains(["finance", "financial", "news", "code"]) == [
        "financial",
        "news",
        "code",
    ]
    assert normalize_search_domains(["web", "news"]) == ["web", "news"]


@pytest.mark.parametrize(
    "values",
    [[], ["financial", "news", "code", "legal"], ["nope"], "financial", None],
)
def test_normalize_rejects(values):
    with pytest.raises(InvalidToolInput):
        normalize_search_domains(values)


# --- search_categories -------------------------------------------------------


@pytest.mark.asyncio
async def test_one_category_is_exactly_search_category():
    answers = {
        _scoped("q", "financial"): _pages(
            "https://sec.gov/a", "https://sec.gov/b", "https://sec.gov/c"
        )
    }
    one = await search_categories(
        "q",
        domains=["financial"],
        page_size=5,
        search_fn=_Provider(answers),
        restricted_fn=_Provider(answers),
    )
    ref = await search_category(
        "q",
        domain="financial",
        page_size=5,
        search_fn=_Provider(answers),
        restricted_fn=_Provider(answers),
    )
    assert one == ref
    assert all("categories" not in p.metadata for p in one.pages)


@pytest.mark.asyncio
async def test_categories_run_in_parallel():
    # Barrier: each call waits until both categories have started, so a serial
    # implementation deadlocks into a TimeoutError instead of a timing flake.
    started: list[str] = []
    both_started = asyncio.Event()

    async def provider(query, page_size):
        started.append(query)
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        return []

    await asyncio.wait_for(
        search_categories(
            "q", domains=["news", "code"], page_size=5, search_fn=provider
        ),
        timeout=2.0,
    )
    assert started == [_suffix("q", "news"), _suffix("q", "code")]


@pytest.mark.asyncio
async def test_rrf_ranks_shared_pages_first_and_records_categories():
    p = _Provider(
        {
            _suffix("q", "news"): _pages(
                "https://a.example/1", "https://shared.example/x"
            ),
            _suffix("q", "code"): _pages(
                "https://b.example/2", "https://shared.example/x"
            ),
        }
    )
    out = await search_categories(
        "q", domains=["news", "code"], page_size=5, search_fn=p
    )
    urls = [pg.url for pg in out.pages]
    assert urls[0] == "https://shared.example/x"
    # Tie between the two rank-1 singles: first category (news) wins.
    assert urls[1:] == ["https://a.example/1", "https://b.example/2"]
    shared = out.pages[0]
    assert shared.metadata["categories"] == ["news", "code"]
    assert shared.metadata["category"] == "news"
    assert out.pages[1].metadata["categories"] == ["news"]
    assert out.executed_queries == [_suffix("q", "news"), _suffix("q", "code")]


@pytest.mark.asyncio
async def test_fused_list_is_truncated_to_page_size():
    p = _Provider(
        {
            _suffix("q", "news"): _pages(*[f"https://n.example/{i}" for i in range(5)]),
            _suffix("q", "code"): _pages(*[f"https://c.example/{i}" for i in range(5)]),
        }
    )
    out = await search_categories(
        "q", domains=["news", "code"], page_size=4, search_fn=p
    )
    assert len(out.pages) == 4


@pytest.mark.asyncio
async def test_one_category_raising_keeps_the_others():
    p = _Provider(
        {
            _suffix("q", "news"): RuntimeError("news down"),
            _suffix("q", "code"): _pages("https://c.example/1"),
        }
    )
    out = await search_categories(
        "q", domains=["news", "code"], page_size=5, search_fn=p
    )
    assert [pg.url for pg in out.pages] == ["https://c.example/1"]


@pytest.mark.asyncio
async def test_every_category_raising_propagates_the_first():
    p = _Provider(
        {
            _suffix("q", "news"): RuntimeError("first"),
            _suffix("q", "code"): RuntimeError("second"),
        }
    )
    with pytest.raises(RuntimeError, match="first"):
        await search_categories("q", domains=["news", "code"], page_size=5, search_fn=p)


@pytest.mark.asyncio
async def test_no_usable_pages_returns_the_first_outcome_unchanged():
    err = SearchPage(error="quota", timed_out=False)
    p = _Provider({_suffix("q", "news"): [err], _suffix("q", "code"): []})
    out = await search_categories(
        "q", domains=["news", "code"], page_size=5, search_fn=p
    )
    assert len(out.pages) == 1 and out.pages[0].error == "quota"


@pytest.mark.asyncio
async def test_fallback_used_is_any():
    scoped = _scoped("q", "financial")
    p = _Provider(
        {
            scoped: [],
            _suffix("q", "financial"): _pages("https://x.example/1"),
            _suffix("q", "news"): _pages("https://n.example/1"),
        }
    )
    out = await search_categories(
        "q", domains=["financial", "news"], page_size=5, search_fn=p, restricted_fn=p
    )
    assert out.fallback_used is True


@pytest.mark.asyncio
async def test_outer_cancellation_cancels_every_category():
    started: list[str] = []
    cancelled: list[str] = []
    both_started = asyncio.Event()

    async def provider(query, page_size):
        started.append(query)
        if len(started) == 2:
            both_started.set()
        try:
            await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            cancelled.append(query)
            raise
        return []

    task = asyncio.create_task(
        search_categories(
            "q", domains=["news", "code"], page_size=5, search_fn=provider
        )
    )
    await asyncio.wait_for(both_started.wait(), timeout=2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert sorted(cancelled) == sorted([_suffix("q", "news"), _suffix("q", "code")])
