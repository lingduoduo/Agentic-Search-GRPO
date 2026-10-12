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
async def test_enough_restricted_results_use_no_fallback_pages():
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
    # Both legs are sent in parallel; only restricted pages are used.
    assert sorted(p.calls) == sorted([(SCOPED, 5), (SUFFIX, 5)])
    assert out.executed_queries == [SCOPED, SUFFIX]
    assert out.fallback_used is False
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
    assert (SCOPED, 2) in p.calls
    assert len(out.pages) == 2
    assert out.fallback_used is False


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


# --- restriction is enforced on the results, not only requested -------------

from src.internal.tools.search import site_matches  # noqa: E402


@pytest.mark.parametrize(
    "url,sites,expected",
    [
        ("https://www.sec.gov/x", ("sec.gov",), True),
        ("https://sec.gov", ("sec.gov",), True),
        ("https://notsec.gov/x", ("sec.gov",), False),
        ("https://sec.gov.evil.com/x", ("sec.gov",), False),
        ("https://www.linkedin.com/jobs/view/1", ("linkedin.com/jobs",), True),
        ("https://www.linkedin.com/jobsearch", ("linkedin.com/jobs",), False),
        ("https://www.linkedin.com/in/someone", ("linkedin.com/jobs",), False),
        ("", ("sec.gov",), False),
        ("not a url", ("sec.gov",), False),
    ],
)
def test_site_matches_hosts_subdomains_and_paths(url, sites, expected):
    assert site_matches(url, sites) is expected


@pytest.mark.asyncio
async def test_off_list_restricted_results_do_not_count():
    # Google relaxes site: — the restricted leg returns mostly other hosts.
    p = _Provider(
        {
            SCOPED: _pages(
                "https://www.sec.gov/a",
                "https://studocu.com/b",
                "https://scribd.com/c",
                "https://stocktitan.net/d",
            ),
            SUFFIX: _pages("https://blog.example/x", "https://news.example/y"),
        }
    )
    out = await search_category(
        "etf fees", domain="financial", page_size=4, search_fn=p, restricted_fn=p
    )
    # One on-list page is below the threshold, so the fallback tops it up.
    assert [q for q, _ in p.calls] == [SCOPED, SUFFIX]
    assert [pg.url for pg in out.pages] == [
        "https://www.sec.gov/a",
        "https://blog.example/x",
        "https://news.example/y",
    ]
    assert [pg.metadata["category_leg"] for pg in out.pages] == [
        "restricted",
        "fallback",
        "fallback",
    ]


@pytest.mark.asyncio
async def test_no_on_list_results_returns_the_fallback_as_today():
    p = _Provider(
        {
            SCOPED: _pages(
                "https://studocu.com/b", "https://scribd.com/c", "https://x.example/d"
            ),
            SUFFIX: _pages("https://blog.example/x"),
        }
    )
    out = await search_category(
        "etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p
    )
    assert [pg.url for pg in out.pages] == ["https://blog.example/x"]
    assert out.pages[0].metadata["category_leg"] == "fallback"


# --- parallel legs with a grace window -----------------------------------------

import time  # noqa: E402

from src.internal.tools import search as search_module  # noqa: E402


class _TimedProvider:
    """Answers per query after a per-query delay; records start times."""

    def __init__(self, answers, delays):
        self.answers, self.delays = answers, delays
        self.started: dict[str, float] = {}
        self.cancelled: list[str] = []

    async def __call__(self, query, page_size):
        self.started[query] = time.perf_counter()
        try:
            await asyncio.sleep(self.delays.get(query, 0.0))
        except asyncio.CancelledError:
            self.cancelled.append(query)
            raise
        answer = self.answers.get(query, [])
        if isinstance(answer, BaseException):
            raise answer
        return answer


ON_LIST = _pages("https://sec.gov/a", "https://nasdaq.com/b", "https://sec.gov/c")


@pytest.mark.asyncio
async def test_both_legs_start_together():
    p = _TimedProvider(
        {SCOPED: ON_LIST, SUFFIX: _pages("https://x.example/1")},
        {SCOPED: 0.2, SUFFIX: 0.2},
    )
    t0 = time.perf_counter()
    out = await search_category(
        "etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p
    )
    assert time.perf_counter() - t0 < 0.35  # parallel, not 0.4 sequential
    assert abs(p.started[SCOPED] - p.started[SUFFIX]) < 0.05
    assert [pg.metadata["category_leg"] for pg in out.pages] == ["restricted"] * 3
    assert out.fallback_used is False


@pytest.mark.asyncio
async def test_enough_restricted_results_cancel_a_slow_suffix_leg():
    p = _TimedProvider(
        {SCOPED: ON_LIST, SUFFIX: _pages("https://x.example/1")},
        {SCOPED: 0.0, SUFFIX: 5.0},
    )
    t0 = time.perf_counter()
    out = await search_category(
        "etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p
    )
    assert time.perf_counter() - t0 < 0.5
    await asyncio.sleep(0)  # a cancelled leg sees CancelledError on its next turn
    assert SUFFIX in p.cancelled
    assert out.fallback_used is False and len(out.pages) == 3


@pytest.mark.asyncio
async def test_a_slow_restricted_leg_is_cut_off_after_the_grace(monkeypatch):
    monkeypatch.setattr(search_module, "RESTRICT_GRACE_SECONDS", 0.2)
    p = _TimedProvider(
        {SCOPED: ON_LIST, SUFFIX: _pages("https://x.example/1")},
        {SCOPED: 5.0, SUFFIX: 0.0},
    )
    t0 = time.perf_counter()
    out = await search_category(
        "etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p
    )
    assert time.perf_counter() - t0 < 0.6  # suffix + grace, not 5 s
    await asyncio.sleep(0)
    assert SCOPED in p.cancelled
    assert [pg.url for pg in out.pages] == ["https://x.example/1"]
    assert out.fallback_used is True


@pytest.mark.asyncio
async def test_a_restricted_leg_inside_the_grace_still_counts(monkeypatch):
    monkeypatch.setattr(search_module, "RESTRICT_GRACE_SECONDS", 1.0)
    p = _TimedProvider(
        {SCOPED: ON_LIST, SUFFIX: _pages("https://x.example/1")},
        {SCOPED: 0.2, SUFFIX: 0.0},
    )
    out = await search_category(
        "etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p
    )
    assert [pg.metadata["category_leg"] for pg in out.pages] == ["restricted"] * 3
    assert out.fallback_used is False


@pytest.mark.asyncio
async def test_parallel_failure_rules_match_the_sequential_ones():
    # Suffix fails, restricted thin but present: restricted pages are kept.
    p = _TimedProvider(
        {SCOPED: _pages("https://sec.gov/a"), SUFFIX: RuntimeError("down")}, {}
    )
    out = await search_category(
        "etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p
    )
    assert [pg.url for pg in out.pages] == ["https://sec.gov/a"]
    # Suffix fails and restricted is empty: the suffix error propagates.
    p = _TimedProvider({SCOPED: [], SUFFIX: RuntimeError("provider down")}, {})
    with pytest.raises(RuntimeError, match="provider down"):
        await search_category(
            "etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p
        )


@pytest.mark.asyncio
async def test_time_budget_bounds_both_legs():
    p = _TimedProvider(
        {SCOPED: ON_LIST, SUFFIX: _pages("https://x.example/1")},
        {SCOPED: 5.0, SUFFIX: 5.0},
    )
    t0 = time.perf_counter()
    with pytest.raises(asyncio.TimeoutError):
        await search_category(
            "etf fees",
            domain="financial",
            page_size=5,
            search_fn=p,
            restricted_fn=p,
            time_budget=0.2,
        )
    assert time.perf_counter() - t0 < 0.5
    await asyncio.sleep(0)
    assert set(p.cancelled) == {SCOPED, SUFFIX}


@pytest.mark.asyncio
async def test_outer_cancellation_cancels_both_legs():
    p = _TimedProvider({}, {SCOPED: 5.0, SUFFIX: 5.0})
    task = asyncio.create_task(
        search_category(
            "etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert set(p.cancelled) == {SCOPED, SUFFIX}
