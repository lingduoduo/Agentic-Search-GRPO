# Category Source Sites Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** A search-source category with a site list first searches only those sites, falls back to today's suffix query when that comes back thin, and a pre-registered three-arm evaluation measures whether it helps.

**Architecture:** `DOMAIN_REGISTRY` gains `source_sites`. One pure function, `search_category`, owns restricted-then-fallback for one query and one category; every call site that today builds `prepare_domain_query` and calls a provider calls it instead. Callers decide whether restriction may run by passing a browser-free `restricted_fn` (or `None`); a typed setting turns it off globally. A new label file and CLI measure plain vs suffix vs category arms.

**Tech Stack:** Python 3.10+, pytest/pytest-asyncio, FastAPI, TypeScript type only. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-10-category-source-sites-design.md`

## Global Constraints

- Site lists exactly as the spec table; at most 5 entries; bare host or host/path, no scheme, no `site:`.
- `web`, `official`, `product`, `local` have no `source_sites` and keep one suffix call.
- Restricted query: `f"{query} ({' OR '.join('site:' + s for s in sites)})"`, no suffix words.
- Usable page: has `url`, no `error`, not `is_blank`. Threshold `min(3, page_size)`.
- Corpus (`retrieval`) legs always get the raw query; browser legs always get the suffix query, one call.
- Setting `AGENTIC_SEARCH_CATEGORY_SITES`, default on; off ⇒ every path issues exactly today's queries.
- Each leg is an ordinary provider call: its own cache key, existing timeout/retry policies.
- Page metadata gains `category`, `category_leg` (`restricted`/`fallback`/`hint`), `category_query`.
- Eval: 64 labels (4 × 16 site-listed categories), drafted blind to `source_sites`, user spot-checked before any live run, never edited after it. Decision rule pre-registered (Task 6).
- Every new test is mutation-checked: remove the feature line, see red, restore.
- Separate PR against main; verify CI on the final commit; do not merge.

## Plan rulings (deviations from the spec's literal text, with reasons)

1. **Interface.** The spec sketched `search_category(query, *, domain, provider, page_size, **provider_kwargs)`. The plan uses `search_category(query, *, domain, page_size, search_fn, restricted_fn=None)` where both functions are `async (query: str, page_size: int) -> list[SearchPage]`. Reason: two of the eight call sites (`web_search` tool, `DomainSearch`) call `make_web_cascade_search`, which falls back to the browser internally. Passing it as the restricted leg would send a restricted query to the browser and could double a 30–50 s call — exactly what the spec forbids. A separate `restricted_fn` lets each caller supply a browser-free function.
2. **Setting read by callers, not by `search_category`.** Callers pass `restricted_fn=None` when `category_sites_enabled()` is false. Keeps `search_category` pure and lets the eval force the category arm on regardless of the operator's environment.
3. **`retrieval` handling stays in callers.** `app.py` already routes the corpus leg with the raw query; it keeps doing so and never calls `search_category` for it. The multi-query `web_search` tool keeps today's behavior for non-web providers (one suffix call, `restricted_fn=None`), so "setting off ⇒ identical" holds there too.
4. **Fallback latency in the eval** is measured as the suffix arm's live call time on queries where the fallback fired: the fallback query equals the suffix arm's query, so inside the eval it is always a cache hit and cannot be timed directly.

## Review Focus

1. SerpAPI unusable (no key/quota) with the browser configured: the `web_search` tool and `DomainSearch` must not issue two browser calls for one query. Pinned in Task 4 (restricted leg built with `browser_search_url=None`).
2. Restricted leg returns 1–2 usable pages and the fallback raises or returns only error pages: the user gets those pages, not an exception or an error card. Pinned in Task 2.
3. `DomainSearch` `.web` raises on any error page today; a merged result must never contain a fallback error page next to usable restricted pages. Pinned in Task 2 + Task 4.
4. A category query that is empty/whitespace: no `site:` clause on an empty query, one call as today. Pinned in Task 2.
5. Existing suffix-pinning tests silently pass because fakes return `[]` and the fallback query is the last one seen: after wiring, tests that assert the suffix query must assert the full call sequence. Pinned in Task 3/4 step "audit existing hint tests".

---

### Task 1: Registry `source_sites`, setting, discovery

**Files:**
- Modify: `src/internal/tools/search.py` (`SearchDomain`, `DOMAIN_REGISTRY`, `get_sub_domains` directory dict)
- Modify: `src/internal/configs/app_configs.py` (`ServiceSettings`, `load_app_settings`)
- Modify: `src/internal/servers/web/app.py` (`list_search_domains`)
- Modify: `web/src/types.ts` (`SearchDomainOption`)
- Modify: `docs/configuration.md` (Web request routing table)
- Test: `tests/unit/test_search_domains.py`, `tests/unit/servers/web/test_search_domain_http.py`, `tests/unit/test_configs.py`

**Interfaces:**
- Produces: `SearchDomain.source_sites: tuple[str, ...] = ()` (5th field, after `typical_sources`); `ServiceSettings.category_sites: bool = True`; `category_sites_enabled() -> bool` in `src/internal/tools/search.py`; discovery dicts carry `"source_sites": list[str]`.

- [x] **Step 1: Write the failing tests**

Append to `tests/unit/test_search_domains.py`:

```python
import re

from src.internal.tools.search import DOMAIN_REGISTRY, category_sites_enabled

EXPECTED_SITES = {
    "academic": ("arxiv.org", "semanticscholar.org", "pubmed.ncbi.nlm.nih.gov", "openreview.net", "aclanthology.org"),
    "code": ("github.com", "gitlab.com", "stackoverflow.com", "pypi.org", "npmjs.com"),
    "social_media": ("reddit.com", "news.ycombinator.com", "x.com"),
    "resource": ("huggingface.co", "kaggle.com", "data.gov", "zenodo.org"),
    "news": ("reuters.com", "apnews.com", "bbc.com", "npr.org"),
    "legal": ("law.cornell.edu", "courtlistener.com", "justia.com", "govinfo.gov", "eur-lex.europa.eu"),
    "financial": ("sec.gov", "nasdaq.com", "finance.yahoo.com", "marketwatch.com"),
    "patent": ("patents.google.com", "uspto.gov", "wipo.int", "epo.org"),
    "business": ("opencorporates.com", "crunchbase.com"),
    "travel": ("tripadvisor.com", "booking.com", "lonelyplanet.com"),
    "media": ("youtube.com", "commons.wikimedia.org", "vimeo.com", "podcasts.apple.com"),
    "books": ("openlibrary.org", "books.google.com", "goodreads.com", "worldcat.org"),
    "education": ("coursera.org", "edx.org", "khanacademy.org", "ocw.mit.edu"),
    "events": ("eventbrite.com", "meetup.com"),
    "jobs": ("indeed.com", "linkedin.com/jobs", "glassdoor.com"),
    "statistics": ("census.gov", "data.worldbank.org", "data.oecd.org", "ourworldindata.org"),
}


def test_source_sites_match_the_spec():
    actual = {n: e.source_sites for n, e in DOMAIN_REGISTRY.items() if e.source_sites}
    assert actual == EXPECTED_SITES
    for name in ("web", "official", "product", "local"):
        assert DOMAIN_REGISTRY[name].source_sites == ()


def test_source_sites_are_bare_hosts_capped_at_five():
    shape = re.compile(r"^[a-z0-9.-]+\.[a-z]{2,}(/[a-z0-9_-]+)*$")
    for name, entry in DOMAIN_REGISTRY.items():
        assert len(entry.source_sites) <= 5, name
        for site in entry.source_sites:
            assert shape.match(site), (name, site)


def test_category_sites_setting_defaults_on(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_CATEGORY_SITES", raising=False)
    assert category_sites_enabled() is True
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    assert category_sites_enabled() is False


def test_discovery_directory_exposes_source_sites():
    from src.internal.tools.search import DomainSearch

    directory = DomainSearch(web_search_fn=lambda *a, **k: [], tools=[]).get_sub_domains(
        ["financial", "web"]
    )["domains"]
    assert directory[0]["source_sites"] == list(EXPECTED_SITES["financial"])
    assert directory[1]["source_sites"] == []
```

Append to `tests/unit/servers/web/test_search_domain_http.py`:

```python
def test_registry_endpoint_exposes_source_sites(client):
    domains = {d["name"]: d for d in client.get("/api/search-domains").json()["domains"]}
    assert domains["patent"]["source_sites"] == ["patents.google.com", "uspto.gov", "wipo.int", "epo.org"]
    assert domains["web"]["source_sites"] == []
```

- [x] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/test_search_domains.py tests/unit/servers/web/test_search_domain_http.py -q`
Expected: ImportError on `category_sites_enabled`, then (after a stub) failures on missing `source_sites`.

- [x] **Step 3: Implement**

In `src/internal/tools/search.py` add the field and populate the sixteen entries as the 5th positional argument (after `typical_sources`), exactly per `EXPECTED_SITES`:

```python
@dataclass(frozen=True)
class SearchDomain:
    description: str
    query_hint: str
    capabilities: tuple["Capability", ...] = ()
    typical_sources: tuple[str, ...] = ()
    # Hosts (or host/path) the restricted leg searches. Distinct from the
    # descriptive typical_sources: these are what a query actually runs against.
    source_sites: tuple[str, ...] = ()
```

Example entry:

```python
    "financial": SearchDomain(
        "Market data, filings, earnings, financial statements",
        "financial market data filings",
        (...unchanged capabilities...),
        ("SEC EDGAR", "exchange data", "financial providers"),
        ("sec.gov", "nasdaq.com", "finance.yahoo.com", "marketwatch.com"),
    ),
```

Below `search_domain_parameter`:

```python
def category_sites_enabled() -> bool:
    """Whether category searches may restrict to their source sites."""
    from ..configs.app_configs import load_app_settings

    return load_app_settings().services.category_sites
```

In `get_sub_domains`, the directory dict gains `"source_sites": list(DOMAIN_REGISTRY[domain].source_sites),` after `typical_sources`.

In `app_configs.py`: `ServiceSettings` gains `category_sites: bool = True` (comment: "Restrict category web searches to their source sites first; see docs/search-engine.md."), and `load_app_settings` passes `category_sites=get_env_bool(source, "AGENTIC_SEARCH_CATEGORY_SITES", True),`.

In `app.py` `list_search_domains`, add `"source_sites": list(entry.source_sites),` to each dict.

In `web/src/types.ts` `SearchDomainOption`, add:

```ts
  // Hosts a category's restricted web search runs against; empty for suffix-only categories.
  source_sites?: string[];
```

In `docs/configuration.md` Web request routing table, add after the browser row:

```markdown
| `AGENTIC_SEARCH_CATEGORY_SITES` | Search-source categories with a site list (see [search engine](search-engine.md#category-source-sites)) first search only those sites and fall back to the category hint when fewer than three usable results come back; `true` by default. `false` issues the hint-only query, as before |
```

(The anchor is created in Task 6; `tests/unit/test_documented_env_vars.py` needs the row and the `src` reference.)

- [x] **Step 4: Run to verify they pass**

Run: `pytest tests/unit/test_search_domains.py tests/unit/servers/web/test_search_domain_http.py tests/unit/test_configs.py tests/unit/test_documented_env_vars.py -q && (cd web && npm run typecheck)`
Expected: PASS.

- [x] **Step 5: Mutation check** — delete one site from `financial`, re-run `test_source_sites_match_the_spec`: FAIL; restore.

- [x] **Step 6: Commit**

```bash
git add src/internal/tools/search.py src/internal/configs/app_configs.py src/internal/servers/web/app.py web/src/types.ts docs/configuration.md tests/unit/test_search_domains.py tests/unit/servers/web/test_search_domain_http.py
git commit -m "feat: declare category source sites and the setting that enables them"
```

### Task 2: `search_category` core

**Files:**
- Modify: `src/internal/tools/search.py` (new code directly below `prepare_domain_query`)
- Test: `tests/unit/test_category_search.py` (create)

**Interfaces:**
- Consumes: `DOMAIN_REGISTRY[...].source_sites`, `prepare_domain_query`, `normalize_search_domain`, `SearchPage.is_blank` (Task 1 / existing).
- Produces:

```python
PageSearch = Callable[[str, int], Awaitable[list[SearchPage]]]
RESTRICT_MIN_USABLE = 3
RESTRICTABLE_PROVIDERS = frozenset({"google", "serpapi", "serper"})

def restricted_query(query: str, domain: str) -> str | None
@dataclass(frozen=True)
class CategorySearch:
    pages: list[SearchPage]
    executed_queries: list[str]
async def search_category(query: str, *, domain: str, page_size: int,
                          search_fn: PageSearch,
                          restricted_fn: PageSearch | None = None) -> CategorySearch
```

- [x] **Step 1: Write the failing tests** — create `tests/unit/test_category_search.py`:

```python
"""search_category: restricted leg first, suffix fallback when thin."""

from __future__ import annotations

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
    p = _Provider({SCOPED: _pages("https://sec.gov/a", "https://nasdaq.com/b", "https://sec.gov/c")})
    out = await search_category("etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p)
    assert p.calls == [(SCOPED, 5)]
    assert out.executed_queries == [SCOPED]
    assert [pg.metadata["category_leg"] for pg in out.pages] == ["restricted"] * 3
    assert out.pages[0].metadata["category"] == "financial"
    assert out.pages[0].metadata["category_query"] == SCOPED


@pytest.mark.asyncio
async def test_thin_restricted_results_are_topped_up_by_the_fallback():
    p = _Provider({
        SCOPED: _pages("https://sec.gov/a", "https://nasdaq.com/b"),
        SUFFIX: _pages("https://sec.gov/a", "https://blog.example/x", "https://news.example/y"),
    })
    out = await search_category("etf fees", domain="financial", page_size=4, search_fn=p, restricted_fn=p)
    assert [q for q, _ in p.calls] == [SCOPED, SUFFIX]
    assert [pg.url for pg in out.pages] == [
        "https://sec.gov/a", "https://nasdaq.com/b", "https://blog.example/x", "https://news.example/y",
    ]
    assert [pg.metadata["category_leg"] for pg in out.pages] == ["restricted", "restricted", "fallback", "fallback"]
    assert out.executed_queries == [SCOPED, SUFFIX]


@pytest.mark.asyncio
async def test_threshold_is_capped_by_page_size():
    p = _Provider({SCOPED: _pages("https://sec.gov/a", "https://sec.gov/b")})
    out = await search_category("etf fees", domain="financial", page_size=2, search_fn=p, restricted_fn=p)
    assert p.calls == [(SCOPED, 2)]
    assert len(out.pages) == 2


@pytest.mark.asyncio
async def test_blank_and_error_pages_are_not_usable():
    p = _Provider({
        SCOPED: [SearchPage(), SearchPage(error="boom"), *_pages("https://sec.gov/a")],
        SUFFIX: _pages("https://x.example/1"),
    })
    out = await search_category("etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p)
    assert [pg.url for pg in out.pages] == ["https://sec.gov/a", "https://x.example/1"]


@pytest.mark.asyncio
async def test_a_raising_restricted_leg_falls_back():
    p = _Provider({SCOPED: TimeoutError("slow"), SUFFIX: _pages("https://x.example/1")})
    out = await search_category("etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p)
    assert [pg.url for pg in out.pages] == ["https://x.example/1"]
    assert out.pages[0].metadata["category_leg"] == "fallback"


@pytest.mark.asyncio
async def test_both_legs_failing_returns_the_fallback_outcome_unchanged():
    timed_out = SearchPage(error="timeout", timed_out=True)
    p = _Provider({SCOPED: RuntimeError("x"), SUFFIX: [timed_out]})
    out = await search_category("etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p)
    assert len(out.pages) == 1 and out.pages[0].timed_out and out.pages[0].error == "timeout"

    p = _Provider({SCOPED: [], SUFFIX: RuntimeError("provider down")})
    with pytest.raises(RuntimeError, match="provider down"):
        await search_category("etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p)


@pytest.mark.asyncio
async def test_a_failed_fallback_never_hides_restricted_results():
    p = _Provider({SCOPED: _pages("https://sec.gov/a"), SUFFIX: RuntimeError("down")})
    out = await search_category("etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p)
    assert [pg.url for pg in out.pages] == ["https://sec.gov/a"]

    p = _Provider({SCOPED: _pages("https://sec.gov/a"), SUFFIX: [SearchPage(error="quota")]})
    out = await search_category("etf fees", domain="financial", page_size=5, search_fn=p, restricted_fn=p)
    assert [pg.url for pg in out.pages] == ["https://sec.gov/a"]
    assert not any(pg.error for pg in out.pages)


@pytest.mark.asyncio
@pytest.mark.parametrize("domain,expected", [("web", "etf fees"), ("product", "etf fees product specifications prices reviews")])
async def test_categories_without_sites_make_one_hint_call(domain, expected):
    p = _Provider({})
    out = await search_category("etf fees", domain=domain, page_size=5, search_fn=p, restricted_fn=p)
    assert p.calls == [(expected, 5)]
    assert out.executed_queries == [expected]


@pytest.mark.asyncio
async def test_no_restricted_fn_means_one_hint_call():
    p = _Provider({})
    out = await search_category("etf fees", domain="financial", page_size=5, search_fn=p)
    assert p.calls == [(SUFFIX, 5)]
    assert isinstance(out, CategorySearch)


@pytest.mark.asyncio
async def test_empty_query_is_never_scoped():
    p = _Provider({})
    await search_category("  ", domain="financial", page_size=5, search_fn=p, restricted_fn=p)
    assert p.calls == [("  ", 5)]
```

- [x] **Step 2: Run to verify it fails**

Run: `pytest tests/unit/test_category_search.py -q`
Expected: ImportError (`CategorySearch` not defined).

- [x] **Step 3: Implement** — in `src/internal/tools/search.py` below `prepare_domain_query` (add `Awaitable` to the `collections.abc` import; `replace` is already imported from `dataclasses`):

```python
PageSearch = Callable[[str, int], Awaitable[list["SearchPage"]]]

# Below this many usable restricted pages the suffix query tops the list up.
RESTRICT_MIN_USABLE = 3
# Providers whose query string accepts ``site:`` operators and that never fall
# back to the browser internally.
RESTRICTABLE_PROVIDERS = frozenset({"google", "serpapi", "serper"})


def restricted_query(query: str, domain: str) -> str | None:
    """The site-scoped query for a category, or None when it has no sites."""
    sites = DOMAIN_REGISTRY[normalize_search_domain(domain)].source_sites
    if not sites or not query.strip():
        return None
    return f"{query} ({' OR '.join('site:' + site for site in sites)})"


@dataclass(frozen=True)
class CategorySearch:
    pages: list["SearchPage"]
    executed_queries: list[str]


def _usable_page(page: "SearchPage") -> bool:
    return bool(page.url) and not page.error and not page.is_blank


def _tag_pages(pages, domain: str, leg: str, query: str) -> list["SearchPage"]:
    return [
        replace(
            page,
            metadata={
                **(page.metadata or {}),
                "category": domain,
                "category_leg": leg,
                "category_query": query,
            },
        )
        for page in pages
    ]


async def search_category(
    query: str,
    *,
    domain: str,
    page_size: int,
    search_fn: PageSearch,
    restricted_fn: PageSearch | None = None,
) -> CategorySearch:
    """Search one category: its source sites first, the suffix query if thin.

    ``restricted_fn`` must not fall back to the browser; pass None (the
    setting is off, or the provider cannot take ``site:``) for one hint call.
    """
    canonical = normalize_search_domain(domain)
    suffix = prepare_domain_query(query, canonical)
    scoped = restricted_query(query, canonical) if restricted_fn is not None else None
    if scoped is None:
        pages = await search_fn(suffix, page_size)
        return CategorySearch(_tag_pages(pages, canonical, "hint", suffix), [suffix])

    try:
        restricted = [p for p in await restricted_fn(scoped, page_size) if _usable_page(p)]
    except Exception as exc:  # timeout or provider error: the fallback decides
        logger.info("restricted %s search failed: %s", canonical, exc)
        restricted = []
    restricted = _tag_pages(restricted[:page_size], canonical, "restricted", scoped)
    if len(restricted) >= min(RESTRICT_MIN_USABLE, page_size):
        return CategorySearch(restricted, [scoped])

    executed = [scoped, suffix]
    try:
        fallback = await search_fn(suffix, page_size)
    except Exception:
        if restricted:
            return CategorySearch(restricted, executed)
        raise
    if not restricted:
        # Exactly what a single suffix call returns today, error pages included.
        return CategorySearch(_tag_pages(fallback, canonical, "fallback", suffix), executed)
    seen = {page.url for page in restricted}
    extra = [p for p in fallback if _usable_page(p) and p.url not in seen]
    merged = restricted + _tag_pages(extra, canonical, "fallback", suffix)
    return CategorySearch(merged[:page_size], executed)
```

(`logger` is defined further down the module; that is fine — it is looked up when `search_category` runs, not at import.)

- [x] **Step 4: Run to verify it passes**

Run: `pytest tests/unit/test_category_search.py -q`
Expected: all PASS.

- [x] **Step 5: Mutation checks** — (a) change `>=` to `>` in the threshold: `test_enough_restricted_results_skip_the_fallback` FAILS; (b) delete the `if restricted: return ...` in the fallback `except`: `test_a_failed_fallback_never_hides_restricted_results` FAILS; (c) drop `_usable_page(p) and` from `extra`: the error-page assertion FAILS. Restore each.

- [x] **Step 6: Commit**

```bash
git add src/internal/tools/search.py tests/unit/test_category_search.py
git commit -m "feat: search_category restricts to source sites with a suffix fallback"
```

### Task 3: Wire the HTTP search paths

**Files:**
- Modify: `src/internal/servers/web/app.py` — imports; `_run_direct_search` provider loop (currently `provider_query = (query if provider == "retrieval" else prepare_domain_query(query, domain))` then `search_tool(...)`); `_run_hybrid_search._fetch_provider` (`dispatched = [...]` + `asyncio.gather(search_tool(...))`). Browser calls stay as they are.
- Test: `tests/unit/servers/web/test_search_domain_http.py`

**Interfaces:**
- Consumes: `search_category`, `CategorySearch`, `RESTRICTABLE_PROVIDERS`, `category_sites_enabled`, `restricted_query` (Tasks 1–2).

- [x] **Step 1: Write the failing tests** — append to `tests/unit/servers/web/test_search_domain_http.py`:

```python
from src.internal.tools.search import restricted_query


def _recording_search_tool(calls, answers=None):
    async def fake(query, *, provider, **_kwargs):
        calls.append((provider, query))
        return list((answers or {}).get(query, []))

    return fake


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["serpapi", "google", "serper"])
async def test_direct_search_scopes_web_providers_then_falls_back(monkeypatch, provider):
    calls: list = []
    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _recording_search_tool(calls))
    monkeypatch.setattr("src.internal.servers.web.app.fetch_pages_concurrently", _passthrough_fetch)
    await _run_direct_search("etf fees", source_provider=provider, search_url="http://x/retrieve", top_k=2, domain="financial")
    assert calls == [
        (provider, restricted_query("etf fees", "financial")),
        (provider, "etf fees financial market data filings"),
    ]


@pytest.mark.asyncio
async def test_direct_search_corpus_leg_stays_raw(monkeypatch):
    calls: list = []
    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _recording_search_tool(calls))
    await _run_direct_search("etf fees", source_provider="retrieval", search_url="http://x/retrieve", top_k=2, domain="financial")
    assert calls == [("retrieval", "etf fees")]


@pytest.mark.asyncio
async def test_direct_search_setting_off_issues_todays_query(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    calls: list = []
    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _recording_search_tool(calls))
    monkeypatch.setattr("src.internal.servers.web.app.fetch_pages_concurrently", _passthrough_fetch)
    await _run_direct_search("etf fees", source_provider="serpapi", search_url="http://x/retrieve", top_k=2, domain="financial")
    assert calls == [("serpapi", "etf fees financial market data filings")]


@pytest.mark.asyncio
async def test_direct_search_tags_documents_with_their_leg(monkeypatch):
    scoped = restricted_query("etf fees", "financial")
    answers = {scoped: [SearchPage(title=f"t{i}", url=f"https://sec.gov/{i}", summary="c") for i in range(3)]}
    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _recording_search_tool([], answers))
    monkeypatch.setattr("src.internal.servers.web.app.fetch_pages_concurrently", _passthrough_fetch)
    docs = await _run_direct_search("etf fees", source_provider="serpapi", search_url="http://x/retrieve", top_k=3, domain="financial")
    assert docs and all(d.metadata["category_leg"] == "restricted" for d in docs)


@pytest.mark.asyncio
async def test_hybrid_scopes_each_expanded_query_once(monkeypatch):
    calls: list = []
    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _recording_search_tool(calls))
    monkeypatch.setattr("src.internal.servers.web.app.fetch_pages_concurrently", _passthrough_fetch)
    monkeypatch.setattr("src.internal.servers.web.app._expanded_queries", lambda query, llm: [query, f"{query} alt"])
    await _run_hybrid_search("etf fees", llm=None, search_url="http://x/retrieve", top_k=2, filters=None, source_provider="all", domain="financial")
    web = [q for p, q in calls if p == "serpapi"]
    assert sorted(web) == sorted([
        restricted_query("etf fees", "financial"), "etf fees financial market data filings",
        restricted_query("etf fees alt", "financial"), "etf fees alt financial market data filings",
    ])
    assert sorted(q for p, q in calls if p == "retrieval") == ["etf fees", "etf fees alt"]
```

- [x] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/servers/web/test_search_domain_http.py -q`
Expected: the five new tests FAIL (only suffix queries are issued).

- [x] **Step 3: Implement** — in `app.py` import `RESTRICTABLE_PROVIDERS, category_sites_enabled, search_category` from `...tools.search` alongside `prepare_domain_query`. Add one module helper next to `_is_web_provider`:

```python
def _category_search_fns(provider: str, **search_kwargs):
    """(search_fn, restricted_fn) for one web provider; restricted is None
    when the setting is off or the provider cannot take site: operators."""

    async def provider_search(query: str, page_size: int) -> list[SearchPage]:
        return await search_tool(query, provider=provider, page_size=page_size, **search_kwargs)

    restrictable = provider in RESTRICTABLE_PROVIDERS and category_sites_enabled()
    return provider_search, (provider_search if restrictable else None)
```

`_run_direct_search` loop body (non-browser branch) becomes:

```python
        if provider == "retrieval":
            # The corpus is never hinted: a topic word there is just another
            # scored term.
            provider_query = query
            pages = await search_tool(
                query,
                provider=provider,
                search_url=search_url,
                page_size=fetch_k,
                # Access filters apply to the internal corpus only.
                filters=filters,
            )
        else:
            search_fn, restricted_fn = _category_search_fns(provider, search_url=search_url)
            category = await search_category(
                query, domain=domain, page_size=fetch_k,
                search_fn=search_fn, restricted_fn=restricted_fn,
            )
            pages = category.pages
            provider_query = category.executed_queries[0]
```

(The rest of the loop — `fetch_pages_concurrently` and `_documents_from_search_pages(... query=provider_query ...)` — is unchanged. Keep the existing comment explaining why `_is_web_provider` is not the right test.)

In `_run_hybrid_search._fetch_provider`, replace the `dispatched = [...]` list and the `asyncio.gather(search_tool(...))` block with:

```python
        if provider == "retrieval":
            dispatched = list(executed_queries)
            page_lists: list[list[SearchPage]] = list(
                await asyncio.gather(
                    *[
                        search_tool(
                            eq,
                            provider=provider,
                            search_url=search_url,
                            page_size=top_k,
                            timeout_seconds=hybrid.provider_timeout_seconds,
                            max_retries=hybrid.provider_max_retries,
                            filters=_filters_payload(filters),
                        )
                        for eq in executed_queries
                    ]
                )
            )
        else:
            # One category search per expanded query, applied after expansion:
            # hinting the base query would feed the hint to the LLM expander.
            search_fn, restricted_fn = _category_search_fns(
                provider,
                search_url=search_url,
                timeout_seconds=hybrid.provider_timeout_seconds,
                max_retries=hybrid.provider_max_retries,
            )
            results = await asyncio.gather(
                *[
                    search_category(
                        eq, domain=domain, page_size=top_k,
                        search_fn=search_fn, restricted_fn=restricted_fn,
                    )
                    for eq in executed_queries
                ]
            )
            dispatched = [r.executed_queries[0] for r in results]
            page_lists = [r.pages for r in results]
```

Everything after (`_is_web_provider` page fetch, `zip(dispatched, page_lists)`, `_enforce_access`) is unchanged.

- [x] **Step 4: Audit existing hint tests (Review Focus 5)** — tests in this file that assert a single suffix query for a site-listed category (`test_direct_search_hints_web_providers`, `test_direct_search_hints_providers_outside_web_provider_set`, `test_hybrid_hints_each_expanded_query_exactly_once`) now pass only because their fake returns `[]` and the suffix is the last call. Add `monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")` to each and keep their assertions: they now pin hint-only behavior explicitly, while the new tests pin the scoped behavior.

- [x] **Step 5: Run**

Run: `pytest tests/unit/servers/web/ tests/unit/test_category_search.py -q`
Expected: PASS.

- [x] **Step 6: Mutation check** — in `_category_search_fns` return `(provider_search, None)` unconditionally: the scoped tests FAIL; restore.

- [x] **Step 7: Commit**

```bash
git add src/internal/servers/web/app.py tests/unit/servers/web/test_search_domain_http.py
git commit -m "feat: HTTP direct and hybrid search reach category source sites"
```

### Task 4: Wire the `web_search` tool, `DomainSearch`, and MCP

**Files:**
- Modify: `src/internal/tools/search.py` — `MultiQueryWebSearchTool.__init__` and `execute`; `DomainSearch.__init__` and the `.web` branch of `search`.
- Modify: `src/internal/tools/knowledge_base.py` — pass a browser-free restricted function.
- Modify: `src/internal/mcp_server/tools/search.py` — `search_web`, `_domain_service`.
- Test: `tests/unit/test_domain_search_tools.py`, `tests/unit/test_domain_search.py`, `tests/unit/test_domain_search_mcp.py`

**Interfaces:**
- Consumes: Task 2 exports.
- Produces: `MultiQueryWebSearchTool(..., restricted_search_fn=None)`; `DomainSearch(..., restricted_search_fn=None)`; both forward to `search_category`.

Restricted-function rules (Plan ruling 1):
- `MultiQueryWebSearchTool`: `restricted_search_fn` if given; else, when `search_fn` was not injected and `provider in RESTRICTABLE_PROVIDERS`, the default `search_tool` bound to that provider; else `None`.
- `DomainSearch`: `restricted_search_fn` if given; else, when `web_search_fn` was not injected, `make_web_cascade_search(browser_search_url=None)` (SerpAPI only); else `None`.
- `knowledge_base.tool_knowledge_base`: build `restricted_fn = make_web_cascade_search(browser_search_url=None)` once and pass it to both.
- MCP `_domain_service` passes its browser-free `web_search` as both functions; `search_web` uses its provider function for both.
- In every case the setting gates use: `restricted_fn if category_sites_enabled() else None`, evaluated per call.

- [x] **Step 1: Write the failing tests**

Append to `tests/unit/test_domain_search.py`:

```python
from src.internal.tools.search import restricted_query


@pytest.mark.asyncio
async def test_web_capability_scopes_with_a_browser_free_function():
    seen = []

    async def cascade(query, *, page_size):
        seen.append(("cascade", query))
        return [SearchPage(title="f", summary="s", url="https://blog.example/1")]

    async def serp_only(query, *, page_size):
        seen.append(("serp", query))
        return []

    client = DomainSearch(web_search_fn=cascade, restricted_search_fn=serp_only, tools=[])
    result = await client.search("battery", domain="academic")
    assert seen == [
        ("serp", restricted_query("battery", "academic")),
        ("cascade", "battery academic research"),
    ]
    assert result["executed_query"] == restricted_query("battery", "academic")
    assert result["executed_queries"] == [restricted_query("battery", "academic"), "battery academic research"]


@pytest.mark.asyncio
async def test_web_capability_keeps_restricted_results_when_fallback_errors():
    async def cascade(query, *, page_size):
        return [SearchPage(error="quota exhausted")]

    async def serp_only(query, *, page_size):
        return [SearchPage(title="a", summary="s", url="https://arxiv.org/abs/1")]

    client = DomainSearch(web_search_fn=cascade, restricted_search_fn=serp_only, tools=[])
    result = await client.search("battery", domain="academic")
    assert [r["url"] for r in result["results"]] == ["https://arxiv.org/abs/1"]


def test_default_service_never_scopes_through_the_browser(monkeypatch):
    built = []
    import src.internal.tools.search as module

    real = module.make_web_cascade_search
    monkeypatch.setattr(module, "make_web_cascade_search", lambda **kw: built.append(kw) or real(**kw))
    monkeypatch.setenv("AGENTIC_SEARCH_BROWSER_SEARCH_URL", "http://browser/retrieve")
    DomainSearch(tools=[])
    assert {"browser_search_url": None} in built
```

Append to `tests/unit/test_domain_search_tools.py`:

```python
@pytest.mark.asyncio
async def test_web_search_tool_scopes_then_falls_back():
    from src.internal.tools.search import MultiQueryWebSearchTool, SearchPage, restricted_query

    calls = []

    async def cascade(query, **kwargs):
        calls.append(("cascade", query))
        return [SearchPage(title="t", summary="s", url="https://x.example/1")]

    async def serp_only(query, **kwargs):
        calls.append(("serp", query))
        return []

    tool = MultiQueryWebSearchTool(search_fn=cascade, restricted_search_fn=serp_only, page_size=3)
    _, _, metadata = await tool.execute("i", {"queries": ["etf fees"], "domain": "financial"})
    assert calls == [
        ("serp", restricted_query("etf fees", "financial")),
        ("cascade", "etf fees financial market data filings"),
    ]
    assert metadata["executed_queries"] == [
        restricted_query("etf fees", "financial"), "etf fees financial market data filings",
    ]


@pytest.mark.asyncio
async def test_web_search_tool_injected_without_restricted_fn_issues_todays_query():
    from src.internal.tools.search import MultiQueryWebSearchTool

    calls = []

    async def fn(query, **kwargs):
        calls.append(query)
        return []

    tool = MultiQueryWebSearchTool(search_fn=fn, page_size=3)
    await tool.execute("i", {"queries": ["etf fees"], "domain": "financial"})
    assert calls == ["etf fees financial market data filings"]


def test_seeded_tools_share_one_browser_free_restricted_function(monkeypatch):
    import src.internal.tools.search as module
    from src.internal.tools.knowledge_base import tool_knowledge_base

    built = []
    real = module.make_web_cascade_search
    monkeypatch.setattr(
        "src.internal.tools.knowledge_base.make_web_cascade_search",
        lambda **kw: built.append(kw) or real(**kw),
    )
    monkeypatch.setenv("AGENTIC_SEARCH_BROWSER_SEARCH_URL", "http://browser/retrieve")
    tool_knowledge_base()
    assert {"browser_search_url": None} in built
```

(Check `tool_knowledge_base`'s required arguments before running; pass `search_url="http://x/retrieve"` if it requires one.)

Append to `tests/unit/test_domain_search_mcp.py`:

```python
@pytest.mark.asyncio
async def test_mcp_search_web_scopes_then_falls_back(monkeypatch):
    from src.internal.mcp_server.tools import search as module
    from src.internal.tools.search import restricted_query

    calls = []

    async def serper(query, *, page_size):
        calls.append(query)
        return []

    monkeypatch.setattr(module, "serper_dev_search", serper)
    monkeypatch.setenv("MCP_WEB_SEARCH_PROVIDER", "serper")
    result = await module.search_web("battery", domain="academic")
    assert calls == [restricted_query("battery", "academic"), "battery academic research"]
    assert result["executed_query"] == restricted_query("battery", "academic")
    assert result["executed_queries"] == calls
```

and update `test_mcp_web_capability_uses_existing_provider`'s expected `calls` to the two-call sequence:

```python
    assert calls == [
        (restricted_query("battery", "academic"), {"provider": "serper", "page_size": 5}),
        ("battery academic research", {"provider": "serper", "page_size": 5}),
    ]
```

- [x] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/test_domain_search.py tests/unit/test_domain_search_tools.py tests/unit/test_domain_search_mcp.py -q`
Expected: new tests FAIL (`restricted_search_fn` unexpected keyword; single suffix call).

- [x] **Step 3: Implement**

`MultiQueryWebSearchTool.__init__` gains `restricted_search_fn: Any = None` and stores:

```python
        if restricted_search_fn is not None:
            self._restricted_search_fn = restricted_search_fn
        elif search_fn is None and provider in RESTRICTABLE_PROVIDERS:
            self._restricted_search_fn = search_tool
        else:
            self._restricted_search_fn = None
```

In `execute`, replace `executed_queries = [prepare_domain_query(...)]` and the `asyncio.gather(self._search_fn(...))` with:

```python
        def _bind(fn):
            async def run(query: str, page_size: int) -> list[SearchPage]:
                return await fn(
                    query,
                    provider=self._provider,
                    search_url=self._search_url,
                    page_size=page_size,
                    timeout_seconds=self._timeout_seconds,
                )

            return run

        restricted = (
            _bind(self._restricted_search_fn)
            if self._restricted_search_fn is not None and category_sites_enabled()
            else None
        )
        searches = await asyncio.gather(
            *[
                search_category(
                    q, domain=domain, page_size=self._page_size,
                    search_fn=_bind(self._search_fn), restricted_fn=restricted,
                )
                for q in queries
            ]
        )
        results_per_query = [s.pages for s in searches]
        executed_queries = [q for s in searches for q in s.executed_queries]
```

The merge loop and `metadata.update(domain=domain, executed_queries=executed_queries)` stay as they are.

`DomainSearch.__init__` gains `restricted_search_fn=None`:

```python
        if restricted_search_fn is not None:
            self.restricted_search_fn = restricted_search_fn
        elif web_search_fn is None:
            # SerpAPI only: a restricted query must never reach the browser.
            self.restricted_search_fn = make_web_cascade_search(browser_search_url=None)
        else:
            self.restricted_search_fn = None
```

The `.web` branch of `DomainSearch.search` becomes:

```python
            restricted = (
                (lambda q, n: self.restricted_search_fn(q, page_size=n))
                if self.restricted_search_fn is not None and category_sites_enabled()
                else None
            )
            category = await search_category(
                query,
                domain=canonical,
                page_size=max_results,
                search_fn=lambda q, n: self.web_search_fn(q, page_size=n),
                restricted_fn=restricted,
            )
            pages = category.pages
            errors = [page.error for page in pages if page.error]
            if errors:
                raise ValueError("; ".join(errors))
            result.update(
                executed_query=category.executed_queries[0],
                executed_queries=category.executed_queries,
                results=[
                    {"title": page.title, "content": page.summary, "url": page.url}
                    for page in pages[:max_results]
                ],
            )
            return result
```

`knowledge_base.tool_knowledge_base`:

```python
    web_search_fn = make_web_cascade_search(
        browser_search_url=os.getenv("AGENTIC_SEARCH_BROWSER_SEARCH_URL")
    )
    # Category searches restrict to source sites through SerpAPI only: a
    # restricted query that fell back to the browser could double a 30-50 s call.
    restricted_search_fn = make_web_cascade_search(browser_search_url=None)
```

and pass `restricted_search_fn=restricted_search_fn` to both `MultiQueryWebSearchTool(...)` and `DomainSearch(...)`.

MCP `_domain_service`: `return DomainSearch(web_search_fn=web_search, restricted_search_fn=web_search)`.

MCP `search_web`: replace the `executed_query = prepare_domain_query(...)` and the provider `if/elif/else` call with:

```python
    domain = normalize_search_domain(domain)
    provider = os.getenv("MCP_WEB_SEARCH_PROVIDER", "google")
    provider_fn = {"serpapi": serpapi_search, "serper": serper_dev_search}.get(
        provider, google_custom_search
    )

    async def run(q: str, page_size: int):
        return await provider_fn(q, page_size=page_size)

    logger.info("MCP Server: web search: query='%s', limit=%s", query, limit)
    executed_queries = [prepare_domain_query(query, domain)]
    try:
        category = await search_category(
            query, domain=domain, page_size=limit, search_fn=run,
            restricted_fn=run if category_sites_enabled() else None,
        )
        pages = category.pages
        executed_queries = category.executed_queries
    except Exception as exc:
        logger.error("MCP Server: Web search error: %s", exc, exc_info=True)
        return {
            "error": f"Web search failed: {str(exc)}",
            "results": [],
            "query": query,
            **_domain_metadata(domain, executed_queries),
        }
```

with a module helper

```python
def _domain_metadata(domain: str, executed_queries: list[str]) -> dict:
    if domain == "web":
        return {}
    return {"domain": domain, "executed_query": executed_queries[0], "executed_queries": executed_queries}
```

used for the success payload too (replacing the old `domain_metadata` dict). Update the `search_web` docstring line "`domain` optionally appends a source-category hint" to: "`domain` optionally scopes the search to the category's source sites (falling back to a category hint when that returns few results). It is not a hard filter."

- [x] **Step 4: Audit existing hint tests (Review Focus 5)** — run `pytest tests/unit/test_domain_search*.py tests/unit/test_search_domain_tools.py tests/unit/test_mcp_server.py -q`. Any test now failing because it pinned a single suffix call for a site-listed category: if it tests hint semantics, add `monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")`; if it tests call plumbing, update the expected sequence to scoped-then-suffix. Never weaken an assertion to `in`/subset.

- [x] **Step 5: Run**

Run: `pytest tests/unit/test_domain_search.py tests/unit/test_domain_search_tools.py tests/unit/test_domain_search_mcp.py tests/unit/test_search_domain_tools.py tests/unit/test_mcp_server.py tests/unit/test_category_search.py -q`
Expected: PASS.

- [x] **Step 6: Mutation check** — in `knowledge_base.py` change `browser_search_url=None` to the env value: `test_seeded_tools_share_one_browser_free_restricted_function` FAILS; restore.

- [x] **Step 7: Commit**

```bash
git add src/internal/tools/search.py src/internal/tools/knowledge_base.py src/internal/mcp_server/tools/search.py tests/unit/test_domain_search.py tests/unit/test_domain_search_tools.py tests/unit/test_domain_search_mcp.py
git commit -m "feat: web_search, DomainSearch and MCP reach category source sites"
```

### Task 5: Three-arm evaluation harness

**Files:**
- Modify: `src/internal/retrieval/domain_eval.py`
- Create: `examples/run_source_category_eval.py`
- Test: `tests/unit/test_source_category_eval.py` (create); `tests/unit/test_domain_eval.py` (one test)

**Interfaces:**
- Consumes: `search_category`, `restricted_query`, `prepare_domain_query`, `DOMAIN_REGISTRY` (Tasks 1–2).
- Produces (in `domain_eval.py`):

```python
SOURCE_TAXONOMY_MARKER = "source_categories"
DEFAULT_SOURCE_LABELS_PATH: Path  # data/eval/source_category_queries.json
ArmResult(urls, empty, error=None, seconds: float | None = None)  # seconds added; None when cached
@dataclass(frozen=True) class SourceLabelledQuery: domain: str; query: str; authority_hosts: set[str]
def load_source_labels(path=DEFAULT_SOURCE_LABELS_PATH) -> list[SourceLabelledQuery]
def on_list_share(domain: str, hosts: set[str]) -> float
@dataclass(frozen=True) class SourceQueryOutcome:
    domain: str; query: str; plain: ArmResult; suffix: ArmResult; category: ArmResult
    fallback_fired: bool; delta_vs_suffix: float; delta_vs_plain: float
    jaccard_vs_suffix: float; excluded: bool; on_list_share: float
async def run_source_query(label, *, search_fn, cache, provider="serpapi") -> SourceQueryOutcome
```

- Produces (in the CLI): `summarise_source(outcomes) -> dict` with keys `interpretable`, `excluded_count`, `paired_vs_suffix`, `paired_vs_plain` (each `{n, mean_delta, p_value, cliffs_delta}`), `empty_rate` (`{plain, suffix, category}`), `fallback_rate`, `fallback_added_seconds`, `on_list_share`, `control` (`{n, max_abs_delta, min_jaccard}`), `per_domain` (descriptive), `decision` (`"worse"|"better"|"null"|None`).

- [x] **Step 1: Write the failing tests** — create `tests/unit/test_source_category_eval.py`:

```python
"""Offline tests for the three-arm source-category evaluation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.internal.retrieval.domain_eval import (
    DiskCache,
    SourceLabelledQuery,
    load_labels,
    load_source_labels,
    on_list_share,
    run_source_query,
)
from src.internal.tools.search import restricted_query

SUFFIX = "etf fees financial market data filings"


class _Page:
    def __init__(self, url, error=None):
        self.url, self.error = url, error


class _Stub:
    def __init__(self, by_query):
        self.by_query, self.calls = by_query, []

    async def __call__(self, query, **_kwargs):
        self.calls.append(query)
        return [_Page(u) for u in self.by_query.get(query, [])]


def _write(tmp_path, payload):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(payload))
    return path


def test_source_loader_requires_the_marker(tmp_path):
    with pytest.raises(ValueError, match="source_categories"):
        load_source_labels(_write(tmp_path, {"domains": {}}))


def test_source_loader_rejects_categories_without_sites(tmp_path):
    payload = {"taxonomy": "source_categories", "domains": {"product": [{"query": "q", "authority_hosts": ["a.com"]}]}}
    with pytest.raises(ValueError, match="source_sites"):
        load_source_labels(_write(tmp_path, payload))


def test_historical_loader_rejects_a_marked_file(tmp_path):
    payload = {"taxonomy": "source_categories", "domains": {}}
    with pytest.raises(ValueError, match="load_source_labels"):
        load_labels(_write(tmp_path, payload))


def test_on_list_share_counts_subdomains_and_paths():
    assert on_list_share("financial", {"www.sec.gov", "blog.example"}) == 0.5
    assert on_list_share("jobs", {"linkedin.com"}) == 1.0


@pytest.mark.asyncio
async def test_three_arms_and_a_cached_fallback(tmp_path):
    scoped = restricted_query("etf fees", "financial")
    stub = _Stub({
        "etf fees": ["https://blog.example/a"],
        SUFFIX: ["https://blog.example/a", "https://blog.example/b"],
        scoped: ["https://sec.gov/x"],
    })
    label = SourceLabelledQuery("financial", "etf fees", {"sec.gov"})
    out = await run_source_query(label, search_fn=stub, cache=DiskCache(tmp_path))
    # plain, suffix, scoped; the fallback (= suffix) is a cache hit.
    assert stub.calls == ["etf fees", SUFFIX, scoped]
    assert out.fallback_fired
    # One restricted hit (< 3) is topped up from the cached suffix response.
    assert out.category.urls == [
        "https://sec.gov/x", "https://blog.example/a", "https://blog.example/b",
    ]
    assert out.delta_vs_plain == pytest.approx(1 / 3)
    assert out.delta_vs_suffix == pytest.approx(1 / 3)
    assert out.suffix.seconds is not None


@pytest.mark.asyncio
async def test_web_control_has_identical_arms(tmp_path):
    stub = _Stub({"q": ["https://a.example/1"]})
    out = await run_source_query(SourceLabelledQuery("web", "q", {"a.example"}), search_fn=stub, cache=DiskCache(tmp_path))
    assert stub.calls == ["q"]
    assert out.delta_vs_suffix == 0.0 and out.jaccard_vs_suffix == 1.0


@pytest.mark.asyncio
async def test_category_arm_ignores_the_operator_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    scoped = restricted_query("etf fees", "financial")
    stub = _Stub({})
    await run_source_query(SourceLabelledQuery("financial", "etf fees", {"sec.gov"}), search_fn=stub, cache=DiskCache(tmp_path))
    assert scoped in stub.calls


def _cli():
    path = Path(__file__).resolve().parents[2] / "examples" / "run_source_category_eval.py"
    spec = importlib.util.spec_from_file_location("run_source_category_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _outcome(domain, d_suffix, *, excluded=False, fallback=False):
    from src.internal.retrieval.domain_eval import ArmResult, SourceQueryOutcome

    arm = ArmResult(urls=["u"], empty=False, seconds=1.0)
    return SourceQueryOutcome(domain, "q", arm, arm, arm, fallback, d_suffix, d_suffix, 1.0 if domain == "web" else 0.5, excluded, 0.25)


def test_summary_applies_the_preregistered_rule():
    cli = _cli()
    worse = cli.summarise_source([_outcome("financial", -0.5) for _ in range(12)])
    assert worse["decision"] == "worse"
    better = cli.summarise_source([_outcome("financial", 0.5) for _ in range(12)])
    assert better["decision"] == "better"
    null = cli.summarise_source([_outcome("financial", d) for d in (0.1, -0.1) * 6])
    assert null["decision"] == "null"


def test_summary_excludes_controls_and_excluded_queries():
    cli = _cli()
    summary = cli.summarise_source([_outcome("web", 0.0)] + [_outcome("financial", 0.5)] * 3 + [_outcome("legal", 0.9, excluded=True)])
    assert summary["paired_vs_suffix"]["n"] == 3
    assert summary["control"]["n"] == 1
    assert summary["excluded_count"] == 1


def test_summary_reports_fallback_rate_and_circularity_share():
    cli = _cli()
    summary = cli.summarise_source([_outcome("financial", 0.1, fallback=True), _outcome("financial", 0.1)])
    assert summary["fallback_rate"] == 0.5
    assert summary["fallback_added_seconds"] == 1.0
    assert summary["on_list_share"] == 0.25
```

- [x] **Step 2: Run to verify it fails**

Run: `pytest tests/unit/test_source_category_eval.py -q`
Expected: ImportError (`SourceLabelledQuery`).

- [x] **Step 3: Implement `domain_eval.py` additions**

Imports: `import time`, `from functools import partial`, `from src.internal.tools.search import DOMAIN_REGISTRY, SearchPage, prepare_domain_query, search_category`.

`ArmResult` gains `seconds: float | None = None`. In `_run_arm`, time the live call:

```python
    started = time.perf_counter()
    try:
        pages = await search_fn(query, provider=provider, page_size=TOP_K)
    except Exception as exc:
        return ArmResult(urls=[], empty=True, error=str(exc))
    ...
    return ArmResult(urls=urls, empty=not urls, seconds=time.perf_counter() - started)
```

`load_labels` gains, right after parsing JSON:

```python
    if raw.get("taxonomy") is not None:
        raise ValueError(
            f"label file {path} is a {raw['taxonomy']!r} file; "
            "load it with load_source_labels"
        )
```

New code:

```python
SOURCE_TAXONOMY_MARKER = "source_categories"
DEFAULT_SOURCE_LABELS_PATH = DEFAULT_LABELS_PATH.with_name("source_category_queries.json")


@dataclass(frozen=True)
class SourceLabelledQuery:
    domain: str
    query: str
    authority_hosts: set[str] = field(default_factory=set)


def load_source_labels(
    path: str | Path = DEFAULT_SOURCE_LABELS_PATH,
) -> list[SourceLabelledQuery]:
    """Read source-category labels; only categories with source_sites."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"label file not found: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"label file is not valid JSON: {path}") from exc
    if raw.get("taxonomy") != SOURCE_TAXONOMY_MARKER:
        raise ValueError(f"label file {path} lacks taxonomy={SOURCE_TAXONOMY_MARKER!r}")
    out: list[SourceLabelledQuery] = []
    for domain, entries in raw.get("domains", {}).items():
        if domain not in DOMAIN_REGISTRY or not DOMAIN_REGISTRY[domain].source_sites:
            raise ValueError(f"{domain!r} has no source_sites; it cannot be labelled")
        for entry in entries:
            out.append(SourceLabelledQuery(domain, entry["query"], set(entry["authority_hosts"])))
    return out


def on_list_share(domain: str, hosts: set[str]) -> float:
    """Share of authority hosts inside the category's own site list.

    A high share means the category arm's precision is inflated by
    construction; the report states it so a reader can discount it.
    """
    sites = {site.split("/", 1)[0] for site in DOMAIN_REGISTRY[domain].source_sites}
    if not hosts:
        return 0.0
    return sum(1 for h in hosts if any(host_matches(h, s) for s in sites)) / len(hosts)


@dataclass(frozen=True)
class SourceQueryOutcome:
    domain: str
    query: str
    plain: ArmResult
    suffix: ArmResult
    category: ArmResult
    fallback_fired: bool
    delta_vs_suffix: float
    delta_vs_plain: float
    jaccard_vs_suffix: float
    excluded: bool
    on_list_share: float


async def _cached_pages(query, page_size, *, search_fn, cache, provider):
    """search_category's search function: the same cache keys as _run_arm, so
    the fallback leg reuses the suffix arm's response."""
    del page_size  # every arm is scored at TOP_K
    cached = cache.get(cache_key(query, provider=provider))
    if cached is None:
        pages = await search_fn(query, provider=provider, page_size=TOP_K)
        cached = [p.url for p in pages if getattr(p, "url", None) and not getattr(p, "error", None)]
        cache.set(cache_key(query, provider=provider), cached)
    return [SearchPage(url=url) for url in cached]


async def run_source_query(
    label: SourceLabelledQuery, *, search_fn, cache: DiskCache, provider: str = "serpapi"
) -> SourceQueryOutcome:
    plain = await _run_arm(label.query, search_fn=search_fn, cache=cache, provider=provider)
    suffix = await _run_arm(
        prepare_domain_query(label.query, label.domain),
        search_fn=search_fn, cache=cache, provider=provider,
    )
    fetch = partial(_cached_pages, search_fn=search_fn, cache=cache, provider=provider)
    try:
        # restricted_fn is passed unconditionally: the eval measures the
        # category arm whatever the operator's AGENTIC_SEARCH_CATEGORY_SITES.
        result = await search_category(
            label.query, domain=label.domain, page_size=TOP_K,
            search_fn=fetch, restricted_fn=fetch,
        )
        urls = [page.url for page in result.pages]
        category = ArmResult(urls=urls, empty=not urls)
        fallback = len(result.executed_queries) > 1
    except Exception as exc:
        category = ArmResult(urls=[], empty=True, error=str(exc))
        fallback = False
    hosts = label.authority_hosts
    excluded = any(arm.error or arm.empty for arm in (plain, suffix, category))
    return SourceQueryOutcome(
        domain=label.domain,
        query=label.query,
        plain=plain,
        suffix=suffix,
        category=category,
        fallback_fired=fallback,
        delta_vs_suffix=authority_precision(category.urls, hosts) - authority_precision(suffix.urls, hosts),
        delta_vs_plain=authority_precision(category.urls, hosts) - authority_precision(plain.urls, hosts),
        jaccard_vs_suffix=jaccard(suffix.urls, category.urls),
        excluded=excluded,
        on_list_share=on_list_share(label.domain, hosts) if label.domain != "web" else 0.0,
    )
```

Note: `on_list_share` for `web` would fail on an empty site list only through division by `len(hosts)`; the guard keeps controls out of the share regardless.

- [x] **Step 4: Implement the CLI** — create `examples/run_source_category_eval.py`:

```python
"""Compare plain, suffix, and category (site-scoped) search arms.

Answers one question: does restricting a category's web search to its source
sites move results toward authoritative hosts, compared with today's suffix
hint? Source alignment, not answer quality. The decision rule is fixed in
docs/superpowers/specs/2026-10-10-category-source-sites-design.md.

Usage:
    python -m examples.run_source_category_eval --dry-run
    python -m examples.run_source_category_eval --control "what is a bond"
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from examples.run_domain_relevance_eval import (
    DEFAULT_CACHE_DIR,
    MIN_USABLE_SHARE,
    _finite,
    _mean,
)
from src.internal.retrieval.domain_eval import (
    DEFAULT_SOURCE_LABELS_PATH,
    DiskCache,
    SourceLabelledQuery,
    SourceQueryOutcome,
    cache_key,
    load_source_labels,
    run_source_query,
)
from src.internal.tools.search import prepare_domain_query, restricted_query, search_tool
from src.model.post_training.eval.stats import cliffs_delta, paired_permutation_p

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = REPO_ROOT / "data" / "eval" / "source_category_relevance.json"
ALPHA = 0.05


def _paired(deltas: list[float], interpretable: bool) -> dict:
    if deltas and interpretable:
        p_value = paired_permutation_p(deltas, alternative="two-sided")
        effect = cliffs_delta(deltas, [0.0] * len(deltas))
    else:
        p_value = effect = None
    return {"n": len(deltas), "mean_delta": _mean(deltas), "p_value": p_value, "cliffs_delta": effect}


def summarise_source(outcomes: list[SourceQueryOutcome]) -> dict:
    """`web` rows are controls; excluded rows leave the paired comparison."""
    control = [o for o in outcomes if o.domain == "web"]
    data = [o for o in outcomes if o.domain != "web"]
    usable = [o for o in data if not o.excluded]
    interpretable = bool(data) and len(usable) >= MIN_USABLE_SHARE * len(data)
    vs_suffix = _paired([o.delta_vs_suffix for o in usable], interpretable)
    vs_plain = _paired([o.delta_vs_plain for o in usable], interpretable)

    decision = None
    if vs_suffix["p_value"] is not None:
        if vs_suffix["p_value"] < ALPHA and vs_suffix["mean_delta"] < 0:
            decision = "worse"
        elif vs_suffix["p_value"] < ALPHA and vs_suffix["mean_delta"] > 0:
            decision = "better"
        else:
            decision = "null"

    fired = [o for o in data if o.fallback_fired]
    live_fallback_seconds = [o.suffix.seconds for o in fired if o.suffix.seconds is not None]
    per_domain: dict[str, dict] = {}
    for o in data:
        b = per_domain.setdefault(o.domain, {"deltas": [], "fallback": 0, "n": 0, "excluded": 0, "share": []})
        b["n"] += 1
        b["fallback"] += int(o.fallback_fired)
        b["share"].append(o.on_list_share)
        if o.excluded:
            b["excluded"] += 1
        else:
            b["deltas"].append(o.delta_vs_suffix)
    return {
        "interpretable": interpretable,
        "excluded_count": len(data) - len(usable),
        "paired_vs_suffix": vs_suffix,
        "paired_vs_plain": vs_plain,
        "empty_rate": {
            arm: _mean([float(getattr(o, arm).empty) for o in data])
            for arm in ("plain", "suffix", "category")
        },
        "fallback_rate": _mean([float(o.fallback_fired) for o in data]),
        "fallback_added_seconds": _mean(live_fallback_seconds) if live_fallback_seconds else None,
        "on_list_share": _mean([o.on_list_share for o in data]),
        "control": {
            "n": len(control),
            "max_abs_delta": max((abs(o.delta_vs_suffix) for o in control), default=0.0),
            "min_jaccard": min((o.jaccard_vs_suffix for o in control), default=1.0),
        },
        # Descriptive only: four queries cannot support a per-category claim.
        "per_domain": {
            name: {
                "mean_delta_vs_suffix": _mean(b["deltas"]),
                "fallback_rate": b["fallback"] / b["n"],
                "on_list_share": _mean(b["share"]),
                "n_usable": len(b["deltas"]),
                "excluded": b["excluded"],
            }
            for name, b in per_domain.items()
        },
        "decision": decision,
    }


def _pending_calls(labels, cache: DiskCache, provider: str) -> int:
    pending: set[str] = set()
    for label in labels:
        queries = {label.query, prepare_domain_query(label.query, label.domain)}
        scoped = restricted_query(label.query, label.domain)
        if scoped is not None:
            queries.add(scoped)
        pending |= {q for q in queries if cache.get(cache_key(q, provider=provider)) is None}
    return len(pending)


async def _run(labels, cache: DiskCache, provider: str) -> dict:
    outcomes = []
    for index, label in enumerate(labels, start=1):
        outcomes.append(await run_source_query(label, search_fn=search_tool, cache=cache, provider=provider))
        print(f"  [{index:>2}/{len(labels)}] {label.domain:<14} {label.query[:48]}")
    return summarise_source(outcomes)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=str(DEFAULT_SOURCE_LABELS_PATH))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--provider", default="serpapi")
    parser.add_argument("--control", action="append", default=[], help="A query run as a `web` control: all three arms are identical.")
    parser.add_argument("--dry-run", action="store_true", help="Report how many live provider calls the run would make, then exit.")
    args = parser.parse_args(argv)

    labels = load_source_labels(args.labels)
    labels += [SourceLabelledQuery("web", q, {"example.com"}) for q in args.control]
    cache = DiskCache(args.cache_dir)
    print(f"{len(labels)} labelled queries; {_pending_calls(labels, cache, args.provider)} live provider calls needed.")
    if args.dry_run:
        return
    report = _finite(asyncio.run(_run(labels, cache, args.provider)))
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("paired_vs_suffix", "control", "decision")}, indent=2))
    print(f"report written to {args.report}")


if __name__ == "__main__":
    main()
```

`DEFAULT_CACHE_DIR` is imported from `examples/run_domain_relevance_eval.py` so both CLIs share one cache (the plain arm of a repeated query is then free).

- [x] **Step 5: Run**

Run: `pytest tests/unit/test_source_category_eval.py tests/unit/test_domain_eval.py -q`
Expected: PASS (the torch-free guard in `test_domain_eval.py` must still pass — `domain_eval.py` imports only `src.internal.tools.search`).

- [x] **Step 6: Mutation checks** — (a) drop the `taxonomy` check in `load_labels`: `test_historical_loader_rejects_a_marked_file` FAILS; (b) in `summarise_source` swap `< 0` for `> 0` in the "worse" branch: `test_summary_applies_the_preregistered_rule` FAILS; (c) pass `restricted_fn=None` in `run_source_query`: `test_three_arms_and_a_cached_fallback` FAILS. Restore.

- [x] **Step 7: Commit**

```bash
git add src/internal/retrieval/domain_eval.py examples/run_source_category_eval.py tests/unit/test_source_category_eval.py
git commit -m "feat: three-arm source-category relevance evaluation"
```

### Task 6: Labels, documentation, review, PR, and the measured run

**Files:**
- Create: `data/eval/source_category_queries.json`
- Modify: `docs/search-engine.md`; `README.md` (only after the run)
- Modify: this plan's checkboxes and Validation record

- [x] **Step 1: Draft labels blind.** Dispatch one fresh subagent with this prompt (it must not read `src/internal/tools/search.py`, this plan, or the spec):

> Write `data/eval/source_category_queries.json` in /Users/linghuang/Git/Agentic-Search. Do not open src/internal/tools/search.py, docs/superpowers/, or any file mentioning "source_sites". Format: `{"taxonomy": "source_categories", "k": 10, "domains": {"<category>": [{"query": "...", "authority_hosts": ["host", ...]}, ...]}}`. Categories and what each covers: academic (papers, preprints, citations, research); code (repositories, APIs, libraries, examples, issues); social_media (discussions, opinions, community experiences); resource (datasets, directories, tools, templates, benchmarks); news (breaking news, reporting, investigations); legal (statutes, regulations, case law); financial (market data, filings, earnings, financial statements); patent (patents, trademarks, inventions); business (company profiles, industry information); travel (flights, hotels, destinations, transportation); media (images, videos, audio, podcasts); books (books, authors, editions); education (courses, tutorials, textbooks); events (conferences, concerts, meetups, schedules); jobs (job openings, careers, salaries); statistics (public statistics, indicators, surveys, census). Exactly four realistic user queries per category, phrased as a person would type them, with no site names or category words in the query. For each query list 3–6 hosts (bare hostnames, no scheme) whose pages would directly and authoritatively answer it — judge each query on its own merits, and include hosts of any kind (official bodies, publishers, primary sources, reputable references). Do not commit. Report the file path when done.

- [x] **Step 2: Validate shape** — `python -c "from src.internal.retrieval.domain_eval import load_source_labels as l; ls=l(); print(len(ls)); assert len(ls)==64"` and `python -m examples.run_source_category_eval --dry-run` (expect `64 labelled queries; N live provider calls needed`, N ≤ 192). Record the pooled `on_list_share` by running `python -c "from src.internal.retrieval.domain_eval import load_source_labels, on_list_share as s; ls=load_source_labels(); print(sum(s(x.domain,x.authority_hosts) for x in ls)/len(ls))"`.

- [x] **Step 3: USER GATE — spot-check.** Show the user the label file path, the on-list share, and five random entries; wait for approval. Apply only the user's corrections. Then commit:

```bash
git add data/eval/source_category_queries.json
git commit -m "data: source-category relevance labels (drafted blind to site lists)"
```

From this commit on, the label file is frozen.

- [x] **Step 4: Document** — in `docs/search-engine.md`, add a section `### Category source sites` directly after `### Legacy identifiers`, containing: the `source_sites` table (copy from the spec); the restricted-then-fallback rule (threshold `min(3, page_size)`, leg tags, browser and corpus exceptions); `AGENTIC_SEARCH_CATEGORY_SITES`; that `/api/search-domains` and `get_sub_domains` expose `source_sites`; and a placeholder-free sentence: "Whether this improves relevance is measured below." Add a subsection `### Does restricting to source sites help?` stating the arms, the circularity guard with the measured on-list share, the pre-registered decision rule, and the run command — with the result paragraph written in Step 8. Update the `typical_sources` paragraph ("no dedicated connector exists … every category searches through the configured web providers") to say site-listed categories scope their web search to those sites first.

- [x] **Step 5: Full verification** — `pytest -q`; `ruff check` and `ruff format --check` on changed Python files; `cd web && npm run typecheck && npm run test:unit`; `git diff --check`; the link/anchor check used for #680 over the edited docs.

- [x] **Step 6: Whole-branch review** — one fresh reviewer against `origin/main`, the spec, this plan, and Review Focus. Fix material findings RED→GREEN; record rulings below.

- [x] **Step 7: PR** — push the branch; open a **draft** PR against main titled "Restrict category web searches to their source sites"; body states the behavior, the rulings, and that the measurement is pending. Watch CI for the final commit; fix attributable failures.

- [ ] **Step 8: USER GATE — measured run.** Ask the user to run (with `SERP_API_KEY` set):

```bash
python -m examples.run_source_category_eval --dry-run
python -m examples.run_source_category_eval --control "what is a bond" --control "how do vaccines work"
```

Then read `data/eval/source_category_relevance.json`. Reject the run if `control.max_abs_delta != 0` or `control.min_jaccard != 1`. Apply the decision rule: `"worse"` ⇒ change `get_env_bool(source, "AGENTIC_SEARCH_CATEGORY_SITES", True)` and the `ServiceSettings` default to `False`, update the configuration row and docs to say why, with a test asserting the default; `"better"` ⇒ document the measured gain; `"null"` or not interpretable ⇒ document "not detectably better" (or "uninterpretable" with the excluded count). Write the result paragraph in `docs/search-engine.md` with: pooled mean delta vs suffix and vs plain, p-values, Cliff's delta, n, fallback rate, fallback added seconds, empty rates, on-list share. Update the README "Search-source agents" bullet only to reflect what was measured. Commit the report JSON and docs.

- [ ] **Step 9: Finish** — update this plan's checkboxes and Validation record, push, verify CI on the final commit, record CI in the PR body, mark ready, report the PR URL. Do not merge.

## Validation record

Spec approved by the user 2026-10-10; plan executed subagent-driven (fresh implementer + reviewer per task).

- Tasks 1–5 complete; Tasks 3, 4, 5 each needed one fix round (fetch dropped leg metadata; setting-off and seeded-wiring tests missing; failed restricted calls scored as 0 and cached, control could not reject).
- Labels: 64 queries drafted blind to `source_sites`, approved by the user 2026-10-10, frozen at `af576471`. On-list share 90/245 hosts (36.7%); per-query mean 36.4%.
- Final whole-branch review: 0 critical, 4 important (hybrid fallback cancellation lost restricted pages; control gate passed on failed controls; stale MCP and search-engine docs) — all fixed in `524c2600`, `66b44ed6`, scoped re-review clean.
- Local: `pytest` 5386 passed, 1 skipped (one wall-clock auth `nbf` flake outside the diff, 3/3 green on rerun); vitest 246 passed; typecheck, Ruff, `git diff --check`, docs link check clean. `--dry-run` with two controls: 66 queries, 194 live calls.
- Draft PR #681. Remaining: Step 8 (measured run with the user's SerpAPI key, decision rule, result docs) and Step 9.
