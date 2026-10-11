"""search_category: restricted leg first, suffix fallback when thin."""

from __future__ import annotations

import asyncio

import pytest

from src.internal.tools.search import (
    CategorySearch,
    SearchPage,
    restricted_query,
    search_category,
)

SCOPED = "etf fees (site:sec.gov OR site:nasdaq.com OR site:finance.yahoo.com OR site:marketwatch.com)"
SUFFIX = "etf fees financial market data filings"


def _pages(*urls):
    return [SearchPage(title=u, summary="s", url=u) for u in urls]


class _Provider:
    """Answers per query; records every call in order."""

    def __init__(self, answers):
        self.answers = answers
        self.calls: list[tuple[str, int]] = []

    async def __call__(self, query, page_size):
        self.calls.append((query, page_size))
        answer = self.answers.get(query, [])
        if isinstance(answer, BaseException):
            raise answer
        return answer


def test_restricted_query_shape():
    assert restricted_query("etf fees", "finance") == SCOPED
    assert restricted_query("etf fees", "web") is None
    assert restricted_query("   ", "financial") is None


@pytest.mark.asyncio
async def test_enough_restricted_results_skip_the_fallback():
    p = _Provider(
        {
            SCOPED: _pages(
                "https://sec.gov/a", "https://nasdaq.com/b", "https://sec.gov/c"
            )
        }
    )
    out = await search_category(
        "etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p
    )
    assert p.calls == [(SCOPED, 5)]
    assert out.executed_queries == [SCOPED]
    assert [pg.metadata["category_leg"] for pg in out.pages] == ["restricted"] * 3
    assert out.pages[0].metadata["category"] == "financial"
    assert out.pages[0].metadata["category_query"] == SCOPED


@pytest.mark.asyncio
async def test_thin_restricted_results_are_topped_up_by_the_fallback():
    p = _Provider(
        {
            SCOPED: _pages("https://sec.gov/a", "https://nasdaq.com/b"),
            SUFFIX: _pages(
                "https://sec.gov/a", "https://blog.example/x", "https://news.example/y"
            ),
        }
    )
    out = await search_category(
        "etf fees", domain="financial", page_size=4, search_fn=p, restricted_fn=p
    )
    assert [q for q, _ in p.calls] == [SCOPED, SUFFIX]
    assert [pg.url for pg in out.pages] == [
        "https://sec.gov/a",
        "https://nasdaq.com/b",
        "https://blog.example/x",
        "https://news.example/y",
    ]
    assert [pg.metadata["category_leg"] for pg in out.pages] == [
        "restricted",
        "restricted",
        "fallback",
        "fallback",
    ]
    assert out.executed_queries == [SCOPED, SUFFIX]


@pytest.mark.asyncio
async def test_threshold_is_capped_by_page_size():
    p = _Provider({SCOPED: _pages("https://sec.gov/a", "https://sec.gov/b")})
    out = await search_category(
        "etf fees", domain="financial", page_size=2, search_fn=p, restricted_fn=p
    )
    assert p.calls == [(SCOPED, 2)]
    assert len(out.pages) == 2


@pytest.mark.asyncio
async def test_blank_and_error_pages_are_not_usable():
    p = _Provider(
        {
            SCOPED: [
                SearchPage(),
                SearchPage(error="boom"),
                *_pages("https://sec.gov/a"),
            ],
            SUFFIX: _pages("https://x.example/1"),
        }
    )
    out = await search_category(
        "etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p
    )
    assert [pg.url for pg in out.pages] == ["https://sec.gov/a", "https://x.example/1"]


@pytest.mark.asyncio
async def test_a_raising_restricted_leg_falls_back():
    p = _Provider({SCOPED: TimeoutError("slow"), SUFFIX: _pages("https://x.example/1")})
    out = await search_category(
        "etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p
    )
    assert [pg.url for pg in out.pages] == ["https://x.example/1"]
    assert out.pages[0].metadata["category_leg"] == "fallback"


@pytest.mark.asyncio
async def test_both_legs_failing_returns_the_fallback_outcome_unchanged():
    timed_out = SearchPage(error="timeout", timed_out=True)
    p = _Provider({SCOPED: RuntimeError("x"), SUFFIX: [timed_out]})
    out = await search_category(
        "etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p
    )
    assert (
        len(out.pages) == 1
        and out.pages[0].timed_out
        and out.pages[0].error == "timeout"
    )

    p = _Provider({SCOPED: [], SUFFIX: RuntimeError("provider down")})
    with pytest.raises(RuntimeError, match="provider down"):
        await search_category(
            "etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p
        )


@pytest.mark.asyncio
async def test_a_failed_fallback_never_hides_restricted_results():
    p = _Provider({SCOPED: _pages("https://sec.gov/a"), SUFFIX: RuntimeError("down")})
    out = await search_category(
        "etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p
    )
    assert [pg.url for pg in out.pages] == ["https://sec.gov/a"]

    p = _Provider(
        {SCOPED: _pages("https://sec.gov/a"), SUFFIX: [SearchPage(error="quota")]}
    )
    out = await search_category(
        "etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p
    )
    assert [pg.url for pg in out.pages] == ["https://sec.gov/a"]
    assert not any(pg.error for pg in out.pages)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "domain,expected",
    [
        ("web", "etf fees"),
        ("product", "etf fees product specifications prices reviews"),
    ],
)
async def test_categories_without_sites_make_one_hint_call(domain, expected):
    p = _Provider({})
    out = await search_category(
        "etf fees", domain=domain, page_size=5, search_fn=p, restricted_fn=p
    )
    assert p.calls == [(expected, 5)]
    assert out.executed_queries == [expected]


@pytest.mark.asyncio
async def test_no_restricted_fn_means_one_hint_call():
    p = _Provider({})
    out = await search_category(
        "etf fees", domain="financial", page_size=5, search_fn=p
    )
    assert p.calls == [(SUFFIX, 5)]
    assert isinstance(out, CategorySearch)


@pytest.mark.asyncio
async def test_empty_query_is_never_scoped():
    p = _Provider({})
    await search_category(
        "  ", domain="financial", page_size=5, search_fn=p, restricted_fn=p
    )
    assert p.calls == [("  ", 5)]


class _SlowFallback(_Provider):
    """Restricted answers at once; the suffix query sleeps past any budget."""

    async def __call__(self, query, page_size):
        if query == SUFFIX:
            self.calls.append((query, page_size))
            await asyncio.sleep(5)
        return await super().__call__(query, page_size)


@pytest.mark.asyncio
async def test_a_slow_fallback_keeps_restricted_results_within_the_budget():
    p = _SlowFallback({SCOPED: _pages("https://sec.gov/a")})
    out = await asyncio.wait_for(
        search_category(
            "etf fees",
            domain="financial",
            page_size=5,
            search_fn=p,
            restricted_fn=p,
            time_budget=0.05,
        ),
        timeout=1.0,
    )
    assert [pg.url for pg in out.pages] == ["https://sec.gov/a"]
    assert out.executed_queries == [SCOPED, SUFFIX]


@pytest.mark.asyncio
async def test_a_slow_fallback_with_nothing_restricted_times_out():
    p = _SlowFallback({})
    loop = asyncio.get_running_loop()
    start = loop.time()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            search_category(
                "etf fees",
                domain="financial",
                page_size=5,
                search_fn=p,
                restricted_fn=p,
                time_budget=0.05,
            ),
            timeout=1.0,
        )
    # The budget raised it, not the outer one-second guard.
    assert loop.time() - start < 0.5
