# Multi-category Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a query use up to three search-source categories, searched in parallel and fused with RRF, through HTTP (direct and hybrid), the agent tools, MCP and the Assist page. A single category must behave exactly as today.

**Architecture:** One validator, `normalize_search_domains`, and one fan-out function, `search_categories`, both in `src/internal/tools/search.py`. With one category, `search_categories` *is* `search_category`. With several, it runs them in parallel and fuses the results by URL with the existing `rrf_rank`. Every call site switches from `search_category` to `search_categories`.

Inside `app.py` the category selection keeps travelling through the existing `domain` parameter. It is typed `str | tuple[str, ...]`: a single category is still a plain `str`, so every existing caller, test and log is unchanged. Only the leaves expand it, with `_domain_list`.

**Tech Stack:** Python 3.10+, pytest/pytest-asyncio, FastAPI/Pydantic, React/TypeScript, Vitest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-12-multi-category-search-design.md`

## Global Constraints

- `MAX_SEARCH_DOMAINS = 3`, counted **after** alias normalization and deduplication (first occurrence kept). `web` is allowed and counts toward the cap.
- An empty list, an unknown value, or more than 3 categories raises `InvalidToolInput` (HTTP 400) **before any provider call**.
- `domain` plus `domains` is rejected, unless `domain` is omitted or `web`. A `tag` plus `domains` is rejected.
- One category ⇒ exactly today's behaviour: the same calls, the same pages and the same metadata. No `categories` key is added.
- Fusion: `rrf_rank` (k = 60) over each category's usable pages (a URL, no error, not blank), keyed by URL, truncated to `page_size`. Ties go first-seen in category order.
- A fused page keeps the `category`, `category_leg` and `category_query` of the first category that found it, plus `categories` (every category that found it, in category order).
- `executed_queries` is concatenated in category order. `fallback_used` is true if any category used its fallback.
- Failures:
  - a category that raises is dropped;
  - if every category raises, the first exception propagates;
  - if no category returns a usable page, the first outcome that did not raise is returned unchanged.
- Hybrid with several categories: categories apply to the original query (`executed_queries[0]`) only. The other variants run through `search_category(eq, domain="web")`, which is one plain call. Hybrid with one category is unchanged.
- Browser legs with several categories get one call with the plain query. With one category they get the suffix query as today. Corpus legs always get the raw query.
- Every new test is mutation-checked. Separate PR against main; verify CI on the final commit; do not merge.

## Review Focus

1. **Four categories, or three that only collapse after aliasing:** must never reach a provider on any entry point (HTTP, `web_search`, `search_domain`, `batch_search`, MCP). Pinned by guard tests in Tasks 2 and 3.
2. **A single `domain` request after the change:** must produce byte-identical provider queries and response shape. Pinned by the existing suites (unchanged) and a `search_categories([x]) is search_category(x)` equality test.
3. **One category raising while another returns pages:** the user gets the surviving pages, not an error. Pinned in Task 1.
4. **Hybrid with several categories plus query expansion:** variants must not carry a hint or a `site:` clause, and the browser is called once. Pinned in Task 2.
5. **Assist page selection:** picking a category while only `web` is selected *replaces* `web`. It does not produce `[web, x]`, which would silently send two categories. Pinned in Task 4.

---

### Task 1: `normalize_search_domains` and `search_categories`

**Files:**
- Modify: `src/internal/tools/search.py` (new code directly below `search_category`)
- Test: `tests/unit/test_multi_category_search.py` (create)

**Interfaces:**
- Consumes: `normalize_search_domain`, `search_category`, `CategorySearch(pages, executed_queries, fallback_used=False)`, `_usable_page`, `PageSearch`, `InvalidToolInput` (existing); `rrf_rank` from `src/internal/retrieval/fusion.py`.
- Produces:

```python
MAX_SEARCH_DOMAINS = 3
def normalize_search_domains(values) -> list[str]
async def search_categories(query: str, *, domains: list[str], page_size: int,
    search_fn: PageSearch, restricted_fn: PageSearch | None = None,
    time_budget: float | None = None) -> CategorySearch
```

- [ ] **Step 1: Write the failing tests.** Create `tests/unit/test_multi_category_search.py`:

```python
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
        "financial", "news", "code",
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
    answers = {_scoped("q", "financial"): _pages("https://sec.gov/a", "https://sec.gov/b", "https://sec.gov/c")}
    one = await search_categories("q", domains=["financial"], page_size=5, search_fn=_Provider(answers), restricted_fn=_Provider(answers))
    ref = await search_category("q", domain="financial", page_size=5, search_fn=_Provider(answers), restricted_fn=_Provider(answers))
    assert one == ref
    assert all("categories" not in p.metadata for p in one.pages)


@pytest.mark.asyncio
async def test_categories_run_in_parallel():
    p = _Provider({}, delays={_suffix("q", "news"): 0.2, _suffix("q", "code"): 0.2})
    t0 = time.perf_counter()
    await search_categories("q", domains=["news", "code"], page_size=5, search_fn=p)
    assert time.perf_counter() - t0 < 0.35


@pytest.mark.asyncio
async def test_rrf_ranks_shared_pages_first_and_records_categories():
    p = _Provider({
        _suffix("q", "news"): _pages("https://a.example/1", "https://shared.example/x"),
        _suffix("q", "code"): _pages("https://b.example/2", "https://shared.example/x"),
    })
    out = await search_categories("q", domains=["news", "code"], page_size=5, search_fn=p)
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
    p = _Provider({
        _suffix("q", "news"): _pages(*[f"https://n.example/{i}" for i in range(5)]),
        _suffix("q", "code"): _pages(*[f"https://c.example/{i}" for i in range(5)]),
    })
    out = await search_categories("q", domains=["news", "code"], page_size=4, search_fn=p)
    assert len(out.pages) == 4


@pytest.mark.asyncio
async def test_one_category_raising_keeps_the_others():
    p = _Provider({
        _suffix("q", "news"): RuntimeError("news down"),
        _suffix("q", "code"): _pages("https://c.example/1"),
    })
    out = await search_categories("q", domains=["news", "code"], page_size=5, search_fn=p)
    assert [pg.url for pg in out.pages] == ["https://c.example/1"]


@pytest.mark.asyncio
async def test_every_category_raising_propagates_the_first():
    p = _Provider({
        _suffix("q", "news"): RuntimeError("first"),
        _suffix("q", "code"): RuntimeError("second"),
    })
    with pytest.raises(RuntimeError, match="first"):
        await search_categories("q", domains=["news", "code"], page_size=5, search_fn=p)


@pytest.mark.asyncio
async def test_no_usable_pages_returns_the_first_outcome_unchanged():
    err = SearchPage(error="quota", timed_out=False)
    p = _Provider({_suffix("q", "news"): [err], _suffix("q", "code"): []})
    out = await search_categories("q", domains=["news", "code"], page_size=5, search_fn=p)
    assert len(out.pages) == 1 and out.pages[0].error == "quota"


@pytest.mark.asyncio
async def test_fallback_used_is_any():
    scoped = _scoped("q", "financial")
    p = _Provider({scoped: [], _suffix("q", "financial"): _pages("https://x.example/1"), _suffix("q", "news"): _pages("https://n.example/1")})
    out = await search_categories("q", domains=["financial", "news"], page_size=5, search_fn=p, restricted_fn=p)
    assert out.fallback_used is True


@pytest.mark.asyncio
async def test_outer_cancellation_cancels_every_category():
    p = _Provider({}, delays={_suffix("q", "news"): 5.0, _suffix("q", "code"): 5.0})
    task = asyncio.create_task(search_categories("q", domains=["news", "code"], page_size=5, search_fn=p))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
```

- [ ] **Step 2: Run to verify it fails.** Run `pytest tests/unit/test_multi_category_search.py -q`. Expect an ImportError on `normalize_search_domains`.

- [ ] **Step 3: Implement.** In `src/internal/tools/search.py`, add `from ..retrieval.fusion import rrf_rank` with the other relative imports. It is light and pulls in no torch. Then add directly below `search_category`:

```python
# Each category is up to two provider calls; three keeps a query at six.
MAX_SEARCH_DOMAINS = 3


def normalize_search_domains(values) -> list[str]:
    """Canonical, de-duplicated categories for one request, or reject it.

    Aliases collapse before the cap is counted, so ["finance", "financial"]
    is one category. Raised before any provider call.
    """
    if not isinstance(values, list) or not values:
        raise InvalidToolInput("domains must be a non-empty list of categories")
    canonical: list[str] = []
    for value in values:
        name = normalize_search_domain(value)
        if name not in canonical:
            canonical.append(name)
    if len(canonical) > MAX_SEARCH_DOMAINS:
        raise InvalidToolInput(
            f"at most {MAX_SEARCH_DOMAINS} categories per search; got {len(canonical)}"
        )
    return canonical


async def search_categories(
    query: str,
    *,
    domains: list[str],
    page_size: int,
    search_fn: PageSearch,
    restricted_fn: PageSearch | None = None,
    time_budget: float | None = None,
) -> CategorySearch:
    """Search up to three categories in parallel and fuse them with RRF.

    One category is exactly ``search_category``. With several, each category
    contributes its usable pages as a ranked list; RRF (k=60) fuses them by
    URL, ties first-seen in category order, truncated to ``page_size``.
    """
    if len(domains) == 1:
        return await search_category(
            query,
            domain=domains[0],
            page_size=page_size,
            search_fn=search_fn,
            restricted_fn=restricted_fn,
            time_budget=time_budget,
        )
    outcomes = await asyncio.gather(
        *[
            search_category(
                query,
                domain=domain,
                page_size=page_size,
                search_fn=search_fn,
                restricted_fn=restricted_fn,
                time_budget=time_budget,
            )
            for domain in domains
        ],
        return_exceptions=True,
    )
    for outcome in outcomes:
        # CancelledError is a BaseException; never treat it as a dropped category.
        if isinstance(outcome, BaseException) and not isinstance(outcome, Exception):
            raise outcome
    succeeded = [
        (domain, o) for domain, o in zip(domains, outcomes) if not isinstance(o, Exception)
    ]
    for domain, o in zip(domains, outcomes):
        if isinstance(o, Exception):
            logger.info("category %s search failed: %s", domain, o)
    if not succeeded:
        raise next(o for o in outcomes if isinstance(o, Exception))

    executed = [q for _, o in succeeded for q in o.executed_queries]
    fallback_used = any(o.fallback_used for _, o in succeeded)
    ranked_lists = [
        (domain, [p for p in o.pages if _usable_page(p)]) for domain, o in succeeded
    ]
    if not any(pages for _, pages in ranked_lists):
        first = succeeded[0][1]
        return CategorySearch(first.pages, executed, fallback_used=fallback_used)

    first_page: dict[str, SearchPage] = {}
    found_by: dict[str, list[str]] = {}
    for domain, pages in ranked_lists:
        for page in pages:
            first_page.setdefault(page.url, page)
            found_by.setdefault(page.url, [])
            if domain not in found_by[page.url]:
                found_by[page.url].append(domain)
    ranked = rrf_rank([pages for _, pages in ranked_lists], lambda p: p.url)
    fused = [
        replace(
            first_page[url],
            metadata={**(first_page[url].metadata or {}), "categories": found_by[url]},
        )
        for url, _score in ranked[:page_size]
    ]
    return CategorySearch(fused, executed, fallback_used=fallback_used)
```

Check `rrf_rank`'s signature and its tie rule in `fusion.py` before relying on them. It returns `(key, score)` pairs sorted by score descending, with ties kept in first-seen order across the sets as given. Also confirm that `logger` and `replace` are in module scope; both already are.

- [ ] **Step 4: Run.** Run `pytest tests/unit/test_multi_category_search.py tests/unit/test_category_search.py -q`. Expect PASS.
- [ ] **Step 5: Mutation-check each of these; each must turn red, then restore it:**
  - (a) drop the dedup in `normalize_search_domains` (the alias test fails);
  - (b) replace `rrf_rank` with plain concatenation (the shared-first test fails);
  - (c) remove the `return_exceptions` handling so the first exception always raises (the one-raising test fails);
  - (d) remove the `len(domains) == 1` short-circuit (the `categories` key appears and the equality test fails).
- [ ] **Step 6: Commit.** Message: `feat: search_categories fans out up to three categories with RRF`.

### Task 2: HTTP `domains`, direct and hybrid

**Files:**
- Modify: `src/internal/servers/web/app.py`
- Test: `tests/unit/servers/web/test_search_domain_http.py`

**Interfaces:**
- Consumes: `normalize_search_domains`, `search_categories`, `MAX_SEARCH_DOMAINS` (Task 1).
- Produces:
  - `AgentExperienceRequest.domains: list[str] | None = None`;
  - `AgentExperienceResponse.domains: list[str] | None = None`, set only for several categories;
  - inside `app.py`, the `domain` parameter of `_run_direct_search`, `_run_hybrid_search`, `_auto_search_pipeline`, `_run_search_direct_or_escalate` and `_WebHybridRetrievalStage` accepts `str | tuple[str, ...]`;
  - the helper `_domain_list(domain) -> list[str]`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/unit/servers/web/test_search_domain_http.py`. Reuse the file's `_recording_search_tool`, `_passthrough_fetch`, `_capture_kwarg` and `client` helpers, and its imports of `restricted_query`/`prepare_domain_query`. Import `prepare_domain_query` if the file doesn't already.

```python
def test_request_domains_reach_the_search_as_a_tuple(client, monkeypatch):
    captured: dict = {}
    _capture_kwarg(monkeypatch, "_run_direct_search", captured)
    r = client.post("/api/agent", json={"query": "acme earnings", "mode": "search_tool", "domains": ["finance", "news"]})
    assert captured["domain"] == ("financial", "news")
    assert r.status_code == 200


def test_single_entry_domains_is_a_plain_domain(client, monkeypatch):
    captured: dict = {}
    _capture_kwarg(monkeypatch, "_run_direct_search", captured)
    client.post("/api/agent", json={"query": "q", "mode": "search_tool", "domains": ["finance"]})
    assert captured["domain"] == "financial"


@pytest.mark.parametrize(
    "body",
    [
        {"domains": ["financial", "news", "code", "legal"]},
        {"domains": []},
        {"domains": ["nope"]},
        {"domain": "news", "domains": ["code"]},
    ],
)
def test_bad_domains_are_rejected_before_any_search(client, monkeypatch, body):
    called = []
    async def boom(*a, **k):
        called.append(1)
        return []
    monkeypatch.setattr("src.internal.servers.web.app.search_tool", boom)
    r = client.post("/api/agent", json={"query": "q", "mode": "search_tool", **body})
    assert r.status_code == 400
    assert called == []


def test_domains_respect_the_mode_rule(client):
    r = client.post("/api/agent", json={"query": "q", "mode": "chat_once", "domains": ["news", "code"]})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_direct_search_fans_out_and_tags_categories(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    answers = {
        prepare_domain_query("q", "news"): [SearchPage(title="n", url="https://shared.example/x", summary="c")],
        prepare_domain_query("q", "code"): [SearchPage(title="c", url="https://shared.example/x", summary="c")],
    }
    calls: list = []
    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _recording_search_tool(calls, answers))
    monkeypatch.setattr("src.internal.servers.web.app.fetch_pages_concurrently", _passthrough_fetch)
    docs = await _run_direct_search("q", source_provider="serpapi", search_url="http://x/retrieve", top_k=2, domain=("news", "code"))
    assert sorted(q for _, q in calls) == sorted([prepare_domain_query("q", "news"), prepare_domain_query("q", "code")])
    assert docs and docs[0].metadata["categories"] == ["news", "code"]


@pytest.mark.asyncio
async def test_hybrid_several_categories_scope_only_the_original_query(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    calls: list = []
    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _recording_search_tool(calls))
    monkeypatch.setattr("src.internal.servers.web.app.fetch_pages_concurrently", _passthrough_fetch)
    monkeypatch.setattr("src.internal.servers.web.app._expanded_queries", lambda query, llm: [query, f"{query} alt"])
    browser: list = []
    async def fake_browser(query, **_k):
        browser.append(query)
        return []
    monkeypatch.setattr("src.internal.servers.web.app._run_browser_search", fake_browser)
    await _run_hybrid_search("q", llm=None, search_url="http://x/retrieve", browser_search_url="http://b/retrieve",
                             top_k=2, filters=None, source_provider="all", domain=("news", "code"))
    web = sorted(q for p, q in calls if p == "serpapi")
    assert web == sorted([prepare_domain_query("q", "news"), prepare_domain_query("q", "code"), "q alt"])
    assert browser == ["q"]


def test_response_echoes_domains(client, monkeypatch):
    _capture_kwarg(monkeypatch, "_run_direct_search", {})
    r = client.post("/api/agent", json={"query": "q", "mode": "search_tool", "domains": ["news", "code"]})
    assert r.json()["domains"] == ["news", "code"]
```

(Set `AGENTIC_SEARCH_CATEGORY_SITES=0` where a test pins the plain fan-out shape. Restricted-leg behaviour is already covered by `search_category`'s tests.)

- [ ] **Step 2: Run to verify it fails.** Run `pytest tests/unit/servers/web/test_search_domain_http.py -q`. The new tests should fail.

- [ ] **Step 3: Implement in `app.py`:**
  - Import `normalize_search_domains, search_categories` from `...tools.search`.
  - `AgentExperienceRequest` gains:

    ```python
    domains: list[str] | None = Field(
        default=None,
        description=(
            "Up to three search-source categories searched in parallel and "
            "fused. Mutually exclusive with a non-'web' domain."
        ),
    )
    ```

  - `AgentExperienceResponse` gains `domains: list[str] | None = None`.
  - Add a helper next to `_category_search_fns`:

    ```python
    def _domain_list(domain: "str | tuple[str, ...]") -> list[str]:
        return [domain] if isinstance(domain, str) else list(domain)
    ```

  - In `_run_agent_impl`, replace the single `normalize_search_domain(request.domain)` block with:

    ```python
    try:
        # Before any provider call, per the taxonomy and multi-category specs.
        if request.domains is not None:
            if normalize_search_domain(request.domain) != "web":
                raise ValueError("send either domain or domains, not both")
            selected = normalize_search_domains(request.domains)
            domain = selected[0] if len(selected) == 1 else tuple(selected)
        else:
            domain = normalize_search_domain(request.domain)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _reject_domain_for_mode(domain, request.mode)
    ```

  - Make `_reject_domain_for_mode` accept the selection: return early when `all(d == "web" for d in _domain_list(domain))` or `mode is None`.
  - Pass `domains=list(domain) if isinstance(domain, tuple) else None` to every `AgentExperienceResponse(...)` built in `_run_agent_impl`. Grep for them; there are two.
  - **`_run_direct_search`:** compute `domains = _domain_list(domain)` once.
    - The web-provider branch calls `search_categories(query, domains=domains, page_size=fetch_k, search_fn=..., restricted_fn=...)` instead of `search_category`.
    - The browser sidecar sends `query if len(domains) > 1 else prepare_domain_query(query, domains[0])`.
  - **`_run_hybrid_search`:** compute `domains = _domain_list(domain)`.
    - Both browser call sites send the same `query if len(domains) > 1 else prepare_domain_query(query, domains[0])`.
    - In the web branch, `len(domains) == 1` keeps today's per-variant `search_category(..., domain=domains[0], ...)` loop unchanged.
    - With several categories:

    ```python
    budget = 0.75 * hybrid.provider_wait_seconds
    results = await asyncio.gather(
        search_categories(
            executed_queries[0], domains=domains, page_size=top_k,
            search_fn=search_fn, restricted_fn=restricted_fn, time_budget=budget,
        ),
        *[
            # Variants run plain: categories apply to the original query only.
            search_category(eq, domain="web", page_size=top_k,
                            search_fn=search_fn, time_budget=budget)
            for eq in executed_queries[1:]
        ],
    )
    ```

    `dispatched` and `page_lists` are built from `results` as today.
  - Update the type hint of `domain` in `_run_direct_search`, `_run_hybrid_search`, `_auto_search_pipeline`, `_run_search_direct_or_escalate` and `_WebHybridRetrievalStage.__init__` to `"str | tuple[str, ...]"`, keeping the default `"web"`. Grep for any other place that reads `domain` as a string, such as logs or hook payloads, and make it use `_domain_list`.

- [ ] **Step 4: Run.** Run `pytest tests/unit/servers/web/ tests/unit/test_multi_category_search.py tests/unit/test_category_search.py -q`, then `pytest -q`. Expect PASS. Existing single-`domain` tests must pass **unmodified**.
- [ ] **Step 5: Mutation-check each of these; each must turn red, then restore it:**
  - send the scoped variant for every expanded query in multi-category hybrid (the hybrid test fails);
  - call `search_category(domain=domains[0])` instead of `search_categories` in direct (the fan-out test fails);
  - drop the `domain` + `domains` rejection (the 400 test fails).
- [ ] **Step 6: Commit.** Message: `feat: HTTP domains fans out direct and hybrid search`.

### Task 3: `web_search`, `search_domain`, `batch_search` and MCP

**Files:**
- Modify: `src/internal/tools/search.py` (`MultiQueryWebSearchTool` schema and `execute`; `DomainSearch.search` and `batch_search`; `_search_properties`)
- Modify: `src/internal/mcp_server/tools/search.py` (`search_web`, `search_domain`, `batch_search`, `_domain_metadata`)
- Test: `tests/unit/test_domain_search.py`, `tests/unit/test_domain_search_tools.py`, `tests/unit/test_domain_search_mcp.py`

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - the `domains` argument (array of category names, 1–3 items) on `web_search`, `search_domain`, each `batch_search` item, and MCP `search_web`/`search_domain`;
  - multi-category results carry `domains: list[str]` in their metadata or result dict.

- [ ] **Step 1: Write the failing tests:**
  - `tests/unit/test_domain_search_tools.py`: a registry `web_search` call with `domains=["finance", "news"]` issues both suffix queries (with `AGENTIC_SEARCH_CATEGORY_SITES=0`) and returns metadata `domains == ["financial", "news"]`. With `domains` of 4 it returns a validation error and makes no provider call. `domain` plus `domains` is an error.
  - `tests/unit/test_domain_search.py`: `DomainSearch.search(query, domains=["news", "code"])` issues both and returns `{"query", "domains": ["news", "code"], "executed_queries", "results"}`. `tag="financial.quote"` plus `domains` raises `InvalidToolInput` with no call. `domains=["news"]` returns exactly the `domain="news"` result. `batch_search` lets a per-item `domains` override the shared `domain`, and a per-item `tag` override the shared `domains`.
  - `tests/unit/test_domain_search_mcp.py`: MCP `search_web(query, domains=["news", "code"])` issues both through `MCP_WEB_SEARCH_PROVIDER` and returns `domains`. `domains` of 4 returns the error payload with no call.

  Write each test with real stub providers, using the patterns already in those files.

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement:**
  - **`MultiQueryWebSearchTool`:**
    - Schema: add `"domains": {"type": "array", "items": search_domain_parameter(), "minItems": 1, "maxItems": MAX_SEARCH_DOMAINS, "description": "Up to three categories searched in parallel and fused; instead of domain."}`. Drop `default` from the item copy.
    - `execute`: if both `domain` and `domains` are present, raise `InvalidToolInput`. Set `domains = normalize_search_domains(arguments["domains"])` if present, else `[normalize_search_domain(arguments.get("domain", "web"))]`, and call `search_categories(q, domains=domains, ...)` per query in place of `search_category`.
    - Metadata: unchanged for a single category. For several, add `domains=domains` and `executed_queries`.
    - `_source_validation_arguments` already normalizes a `domains` list for validation; keep it.
  - **`DomainSearch.search`** gains `domains: list[str] | None = None`:
    - `domains` with `tag` raises `InvalidToolInput`; `domains` with a `domain` other than `None` or `web` raises `InvalidToolInput`.
    - If `domains` normalizes to one category, continue with `domain=that` (the identical path).
    - Otherwise, take the web route through `search_categories` and return `{"query": query, "domains": canonical, "executed_query": executed[0], "executed_queries": executed, "results": [...]}`. Error-page handling is the same as the `.web` branch.
  - **`batch_search`:** per item, an explicit `tag` drops the shared `domain`, `domains` and `tag`; an explicit `domain` or `domains` drops the shared `domain`, `domains` and `tag`.
  - **`_search_properties`:** add the same `domains` array schema. Use `"items": {"type": "string", "enum": list(AVAILABLE_DOMAINS)}`, matching the existing `domain` property.
  - **MCP `search_web(query, limit=5, domain="web", domains: list[str] | None = None)`:**
    - If `domains` is given with `domain` other than `web`, return the error payload.
    - Validation runs inside the existing `try`, so a bad value returns the error payload.
    - Run `search_categories`.
    - `_domain_metadata(domains, executed_queries)` returns `{}` for `["web"]`, the existing single keys for one category, and `{"domains": [...], "executed_query": ..., "executed_queries": ...}` for several.
  - **MCP `search_domain` / `batch_search`:** pass `domains` through to `DomainSearch`.
  - Update the docstrings to mention `domains` (up to three, fused).

- [ ] **Step 4: Run.** Run `pytest tests/unit/test_domain_search*.py tests/unit/test_search_domain_tools.py tests/unit/test_mcp_server.py tests/unit/test_multi_category_search.py -q`, then `pytest -q`. Expect PASS, with existing tests unmodified.
- [ ] **Step 5: Mutation-check:**
  - remove the `tag` + `domains` rejection;
  - make the tool ignore `domains`;
  - make MCP ignore `domains`.

  Each must turn red. Restore.
- [ ] **Step 6: Commit.** Message: `feat: domains on web_search, search_domain, batch_search and MCP`.

### Task 4: Assist page multi-select and source-card categories

**Files:**
- Create: `web/src/lib/domains.ts`, `web/src/lib/__tests__/domains.test.ts`
- Modify: `web/src/types.ts`, `web/src/components/SearchComposer.tsx`, `web/src/pages/AssistPage.tsx`, `web/src/components/SourceGrid.tsx`
- Test: `web/src/components/__tests__/SearchComposer.test.tsx`, `web/src/pages/__tests__/AssistPage.test.tsx`, `web/src/components/__tests__/App.test.tsx`, `web/src/components/__tests__/SourceGrid.test.tsx`

**Interfaces:**
- Produces: `MAX_SEARCH_DOMAINS = 3`, `toggleDomain(current: string[], name: string): string[]`; the `SearchComposer` props `domains?: string[]` and `onDomainsChange(value: string[])`, replacing `domain`/`onDomainChange`; `AgentRequest.domains?: string[]`.

- [ ] **Step 1: Write the failing tests.** Create `web/src/lib/__tests__/domains.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { MAX_SEARCH_DOMAINS, toggleDomain } from "../domains";

describe("toggleDomain", () => {
  it("replaces the web default instead of adding to it", () => {
    expect(toggleDomain(["web"], "financial")).toEqual(["financial"]);
  });
  it("adds up to the cap, then ignores more", () => {
    expect(toggleDomain(["financial"], "news")).toEqual(["financial", "news"]);
    const full = ["financial", "news", "code"];
    expect(full).toHaveLength(MAX_SEARCH_DOMAINS);
    expect(toggleDomain(full, "legal")).toEqual(full);
  });
  it("removes a selected category and falls back to web when empty", () => {
    expect(toggleDomain(["financial", "news"], "news")).toEqual(["financial"]);
    expect(toggleDomain(["financial"], "financial")).toEqual(["web"]);
  });
  it("selecting web explicitly clears the others", () => {
    expect(toggleDomain(["financial", "news"], "web")).toEqual(["web"]);
  });
});
```

Then make these test updates:
- **`SearchComposer.test.tsx`:**
  - The selector is a group: `getByRole("group", { name: "Search source" })`.
  - Each option is a checkbox labelled with its readable name.
  - Clicking "financial" calls `onDomainsChange(["financial"])` from `["web"]`.
  - With 3 selected, unselected checkboxes are disabled and the text "Up to 3 search sources" is shown.
- **`AssistPage.test.tsx`:**
  - Selecting financial and then news sends `domains: ["financial", "news"]`.
  - Selecting only financial sends `domain: "financial"` and no `domains`.
  - The discovery-failure fallback still shows a checked `web`.
- **`App.test.tsx`:** update the domain test to click the "financial" checkbox. The payload is still `domain: "financial"`, and the default is still `domain: "web"`.
- **`SourceGrid.test.tsx`:** a document whose `metadata.categories` is `["financial", "news"]` renders "financial · news". A single-entry or missing `categories` renders nothing extra.

- [ ] **Step 2: Run to verify they fail.** Run `cd web && npm run test:unit`.

- [ ] **Step 3: Implement.** Create `web/src/lib/domains.ts`:

```ts
// Mirrors MAX_SEARCH_DOMAINS in src/internal/tools/search.py.
export const MAX_SEARCH_DOMAINS = 3;

// One click on a "Search source" checkbox. "web" is the empty selection:
// choosing a real category replaces it, and clearing everything returns to it.
export function toggleDomain(current: string[], name: string): string[] {
  if (name === "web") return ["web"];
  const real = current.filter((d) => d !== "web");
  if (real.includes(name)) {
    const next = real.filter((d) => d !== name);
    return next.length ? next : ["web"];
  }
  if (real.length >= MAX_SEARCH_DOMAINS) return current;
  return [...real, name];
}
```

Then:
- **`types.ts`:** add `domains?: string[];` to the agent request type, below `domain`, with the comment "Up to three categories, fused; mutually exclusive with a non-web domain."
- **`SearchComposer.tsx`:**
  - Replace the `<label>Search source<select>` with `<fieldset className="search-sources"><legend>Search source</legend>`, holding one `<label title=…><input type="checkbox" checked={domains.includes(opt.name)} disabled={!checked && atLimit} onChange={() => onDomainsChange(toggleDomain(domains, opt.name))} />{opt.name.replace(/_/g, " ")}</label>` per option.
  - `atLimit = domains.filter((d) => d !== "web").length >= MAX_SEARCH_DOMAINS`.
  - When `atLimit`, render `<small>Up to 3 search sources</small>`.
  - Keep the existing title text (description plus typical sources).
  - Props: `domains = ["web"]` and `onDomainsChange`.
- **`AssistPage.tsx`:**
  - State: `const [domains, setDomains] = useState<string[]>(["web"]);`.
  - Request: `...(domains.length === 1 ? { domain: domains[0] } : { domains })`.
  - Pass `domains` and `onDomainsChange={setDomains}` to the composer.
- **`SourceGrid.tsx`:**
  - After the MMR-rank span, when `Array.isArray(document.metadata.categories) && document.metadata.categories.length > 1`, render `<span className="category-chips" title="Found by these search sources">{(document.metadata.categories as string[]).join(" · ")}</span>`.
  - Add minimal CSS for `.search-sources` (an inline-wrapped checkbox list) and `.category-chips` (small, muted) to the stylesheet the composer already uses. Check with `rg "composer" web/src --glob "*.css"`.

- [ ] **Step 4: Run.** Run `cd web && npm run typecheck && npm run test:unit`. Expect PASS.
- [ ] **Step 5: Mutation-check:**
  - make `toggleDomain` append to `["web"]`, so the replace test fails;
  - send `domains` for a single selection, so the AssistPage single test fails.

  Restore both.
- [ ] **Step 6: Commit.** Message: `feat: Assist page selects up to three search sources`.

### Task 5: Docs, review, PR

**Files:** `docs/search-engine.md`, `docs/api-reference.md`, `docs/mcp.md`, `docs/frontend.md`, `README.md`; this plan's checkboxes and Validation record.

- [ ] **Step 1: Docs:**
  - **`docs/search-engine.md`:** add a `### Multi-category search` section after `### Category source sites`. It covers:
    - fields: the HTTP `domains`, the tool and MCP `domains`, and the Assist page multi-select;
    - the cap of 3, counted after aliasing;
    - fusion: RRF with k = 60, first-category tie-breaks, and `categories` metadata;
    - the failure rules;
    - per-mode cost: at most 6 calls on direct, tools and MCP, and 6 plus the plain variants on hybrid;
    - the hybrid rule: categories on the original query only, and one plain browser call;
    - that a single category is unchanged;
    - **"Multi-category relevance is not measured; a labelled evaluation is a follow-up."**
  - **`api-reference.md` and `mcp.md`:** the new field.
  - **`frontend.md`:** the multi-select and the category chips.
  - **README:** in the "Search-source agents" bullet, replace "Today a request uses one category; searching several at once with fused results is not implemented yet" with "A request may use up to three categories, searched in parallel with results fused (RRF), reranked and cited; multi-category relevance is not yet measured".

  Verify every claim against the code.
- [ ] **Step 2: Full verification:** `pytest -q`; Ruff check and format on changed Python; `cd web && npm run typecheck && npm run test:unit`; `git diff --check`; the docs link and anchor check over `README.md` and `docs/*.md` (excluding `docs/superpowers`).
- [ ] **Step 3: Whole-branch review.** One fresh reviewer, on the most capable model, against `origin/main`, the spec, this plan and the Review Focus. Fix material findings RED→GREEN and record rulings.
- [ ] **Step 4: PR.** Before each commit, check that `git branch --show-current` is `feat/multi-category-search`. Before pushing, check `git log origin/main -3` for direct pushes. Then push, open a PR against main, watch CI for the final commit, mark it ready, and report the URL. Do not merge.

## Validation record

Spec approved by the user on 2026-10-12. Plan awaiting review.
