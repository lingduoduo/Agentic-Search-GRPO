# HTTP and UI Search Domains Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a caller select one of the 17 search domains on `POST /api/agent` and from the Assist page, so the topic hint reaches web search providers.

**Architecture:** `AgentExperienceRequest` gains a `domain` field validated once at the route. The value threads to the two leaf search helpers, which apply `prepare_domain_query` inside their existing per-provider loops, guarded on the provider not being `retrieval`. A registry endpoint feeds the UI selector so the taxonomy has one source.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, pytest, React 19, TypeScript, Vitest.

**Spec:** `docs/superpowers/specs/2026-09-15-http-ui-search-domains-design.md`

## Global Constraints

- Python >=3.10; no new runtime dependencies.
- Preserve the exact 17 identifiers and their supplied order from `DOMAIN_REGISTRY`.
- `domain="general"` must leave behavior byte-identical to today.
- Topic hints never reach the `retrieval` (corpus) provider.
- Apply topic hints exactly once, before provider dispatch and cache lookup.
- Reject invalid explicit domains before any network call.
- Never reuse `_is_web_provider` for the domain decision; its `_WEB_PROVIDERS` set is `{"serpapi"}` and means "needs full-page fetch".
- Topic domains never replace, relax, or infer authorization filters.
- No changes to `src/agents`, the `/search` corpus route, or index schemas.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/internal/servers/web/app.py` | Request field, route validation, registry endpoint, hint application in both leaves, threading through the auto path |
| `tests/unit/servers/web/test_search_domain_http.py` | New — all backend domain behavior |
| `web/src/types.ts` | `SearchDomainOption` type, `domain` on the agent request body type |
| `web/src/api.ts` | `fetchSearchDomains()`, `domain` in the agent request payload |
| `web/src/components/SearchComposer.tsx` | The selector control |
| `web/src/pages/AssistPage.tsx` | Domain state, fetch on mount, send on submit |
| `web/src/components/__tests__/SearchComposer.test.tsx` | Selector rendering and change handling |
| `docs/search-engine.md` | Document the field and the endpoint |

---

### Task 1: Request field, validation, and registry endpoint

**Files:**
- Modify: `src/internal/servers/web/app.py:240-270` (`AgentExperienceRequest`), `src/internal/servers/web/app.py:1573-1590` (`_run_agent_impl`), near `src/internal/servers/web/app.py:1530` (new route)
- Test: `tests/unit/servers/web/test_search_domain_http.py`

**Interfaces:**
- Consumes: `normalize_search_domain`, `DOMAIN_REGISTRY` from `src.internal.tools.search`
- Produces: `AgentExperienceRequest.domain: str`; `GET /api/search-domains` returning `{"domains": [{"name": str, "description": str}, ...]}`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/servers/web/test_search_domain_http.py`:

```python
"""HTTP surface for the 17-domain search taxonomy.

Covers the request field, its validation, and the registry endpoint that
feeds the UI selector. Provider-level hint behavior lives in the tests
added by later tasks in the same file.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.internal.servers.web.app import AgentExperienceRequest, create_web_app
from src.internal.tools.search import AVAILABLE_DOMAINS, DOMAIN_REGISTRY


@pytest.fixture
def client():
    with TestClient(create_web_app()) as c:
        yield c


def test_domain_defaults_to_general():
    assert AgentExperienceRequest(query="q").domain == "general"


def test_registry_endpoint_lists_all_domains_in_order(client):
    body = client.get("/api/search-domains").json()
    assert [d["name"] for d in body["domains"]] == AVAILABLE_DOMAINS
    assert len(body["domains"]) == 17


def test_registry_endpoint_carries_descriptions(client):
    body = client.get("/api/search-domains").json()
    by_name = {d["name"]: d["description"] for d in body["domains"]}
    assert by_name["finance"] == DOMAIN_REGISTRY["finance"].description


def test_invalid_domain_is_rejected(client):
    r = client.post("/api/agent", json={"query": "q", "domain": "nonsense"})
    assert r.status_code == 400
    assert "domain must be one of" in r.json()["detail"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/servers/web/test_search_domain_http.py -v`
Expected: FAIL — `AgentExperienceRequest` has no `domain`; `/api/search-domains` returns 404.

- [ ] **Step 3: Add the field**

In `AgentExperienceRequest`, after the `route` field:

```python
    domain: str = Field(
        default="general",
        description=(
            "Optional topic query hint applied to web search providers only. "
            "One of the 17 identifiers in DOMAIN_REGISTRY, or 'general' "
            "(default) to leave the query unchanged. Not a result filter."
        ),
    )
```

- [ ] **Step 4: Validate it at the route**

Add the import alongside the other `src.internal.tools.search` imports near line 119:

```python
from src.internal.tools.search import (
    AVAILABLE_DOMAINS,
    DOMAIN_REGISTRY,
    normalize_search_domain,
    prepare_domain_query,
)
```

In `_run_agent_impl`, immediately after the empty-query guard:

```python
        try:
            domain = normalize_search_domain(request.domain)
        except ValueError as exc:
            # Before any provider call, per the taxonomy spec.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
```

- [ ] **Step 5: Add the registry endpoint**

Next to the other `@app.get` routes near line 1547:

```python
    @app.get("/api/search-domains")
    def list_search_domains() -> dict[str, list[dict[str, str]]]:
        """Expose the taxonomy so the UI has one source for names and order."""
        return {
            "domains": [
                {"name": name, "description": entry.description}
                for name, entry in DOMAIN_REGISTRY.items()
            ]
        }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/servers/web/test_search_domain_http.py -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add tests/unit/servers/web/test_search_domain_http.py src/internal/servers/web/app.py
git commit -m "feat(web): accept a search domain on /api/agent and expose the registry"
```

---

### Task 2: Apply the hint in `_run_direct_search`

**Files:**
- Modify: `src/internal/servers/web/app.py:2315-2370`
- Test: `tests/unit/servers/web/test_search_domain_http.py`

**Interfaces:**
- Consumes: `prepare_domain_query(query: str, domain: str) -> str`
- Produces: `_run_direct_search(..., domain: str = "general")`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/servers/web/test_search_domain_http.py`:

```python
from src.internal.servers.web.app import _run_direct_search
from src.internal.tools.search import SearchPage


async def _passthrough_fetch(pages, **_kwargs):
    return pages


def _capture_search_tool(seen: dict[str, str]):
    async def fake_search_tool(query, *, provider, **_kwargs):
        seen[provider] = query
        return [SearchPage(title="t", url="http://x", contents="c")]

    return fake_search_tool


@pytest.mark.asyncio
async def test_direct_search_hints_web_providers(monkeypatch):
    seen: dict[str, str] = {}
    monkeypatch.setattr(
        "src.internal.servers.web.app.search_tool", _capture_search_tool(seen)
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", _passthrough_fetch
    )
    await _run_direct_search(
        "etf fees",
        source_provider="serpapi",
        search_url="http://x/retrieve",
        top_k=2,
        domain="finance",
    )
    assert seen["serpapi"] == "etf fees finance"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["google", "serper"])
async def test_direct_search_hints_providers_outside_web_provider_set(
    monkeypatch, provider
):
    # _WEB_PROVIDERS is {"serpapi"} and means "needs full-page fetch".
    # These two must still be hinted.
    seen: dict[str, str] = {}
    monkeypatch.setattr(
        "src.internal.servers.web.app.search_tool", _capture_search_tool(seen)
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", _passthrough_fetch
    )
    await _run_direct_search(
        "etf fees",
        source_provider=provider,
        search_url="http://x/retrieve",
        top_k=2,
        domain="finance",
    )
    assert seen[provider] == "etf fees finance"


@pytest.mark.asyncio
async def test_direct_search_leaves_the_corpus_query_raw(monkeypatch):
    seen: dict[str, str] = {}
    monkeypatch.setattr(
        "src.internal.servers.web.app.search_tool", _capture_search_tool(seen)
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", _passthrough_fetch
    )
    await _run_direct_search(
        "etf fees",
        source_provider="retrieval",
        search_url="http://x/retrieve",
        top_k=2,
        domain="finance",
    )
    assert seen["retrieval"] == "etf fees"


@pytest.mark.asyncio
async def test_direct_search_general_domain_is_a_no_op(monkeypatch):
    seen: dict[str, str] = {}
    monkeypatch.setattr(
        "src.internal.servers.web.app.search_tool", _capture_search_tool(seen)
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", _passthrough_fetch
    )
    await _run_direct_search(
        "etf fees",
        source_provider="serpapi",
        search_url="http://x/retrieve",
        top_k=2,
        domain="general",
    )
    assert seen["serpapi"] == "etf fees"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/servers/web/test_search_domain_http.py -v -k direct_search`
Expected: FAIL — `_run_direct_search() got an unexpected keyword argument 'domain'`.

- [ ] **Step 3: Implement**

Add the parameter to the signature after `filters`:

```python
    filters: dict | None = None,
    domain: str = "general",
```

Inside the `for provider in _source_providers_for(source_provider):` loop, replace the `search_tool(` call's first argument. The loop body becomes:

```python
        # Hint web providers only: appending a topic word to a corpus query
        # upweights documents containing that word rather than focusing the
        # search. Not _is_web_provider — that set excludes google and serper.
        provider_query = (
            query if provider == "retrieval" else prepare_domain_query(query, domain)
        )
        pages = await search_tool(
            provider_query,
            provider=provider,
            search_url=search_url,
            page_size=fetch_k,
            filters=filters,
        )
```

Then use `provider_query` for the document `query=` kwarg in the same iteration so cards report what actually ran:

```python
        documents.extend(
            _documents_from_search_pages(
                pages,
                source_provider=provider,
                query=provider_query,
                start_index=len(documents) + 1,
            )
        )
```

And hint the browser sidecar, which is not the corpus:

```python
    if browser_search_url and source_provider not in {"browser", "all", "auto"}:
        browser_docs = await _run_browser_search(
            prepare_domain_query(query, domain),
            browser_search_url=browser_search_url,
            top_k=fetch_k,
            existing_count=len(documents),
        )
```

Leave the final `_rank_documents(documents, query, rerank_url, top_k)` call on the raw `query`: reranking scores documents against the user's real question, not the hinted string.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/servers/web/test_search_domain_http.py -v -k direct_search`
Expected: 5 passed.

- [ ] **Step 5: Run the neighbouring suites for regressions**

Run: `pytest tests/unit/servers/web/ tests/unit/test_search_filters_plumbing.py tests/unit/test_search_route_access_filters.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/servers/web/test_search_domain_http.py src/internal/servers/web/app.py
git commit -m "feat(web): hint web providers with the selected domain in direct search"
```

---

### Task 3: Apply the hint in `_run_hybrid_search`

**Files:**
- Modify: `src/internal/servers/web/app.py:2488-2606`
- Test: `tests/unit/servers/web/test_search_domain_http.py`

**Interfaces:**
- Consumes: `prepare_domain_query`
- Produces: `_run_hybrid_search(..., domain: str = "general")`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/servers/web/test_search_domain_http.py`:

```python
from src.internal.servers.web.app import _run_hybrid_search


@pytest.mark.asyncio
async def test_hybrid_hints_each_expanded_query_exactly_once(monkeypatch):
    seen: list[str] = []

    async def fake_search_tool(query, *, provider, **_kwargs):
        if provider != "retrieval":
            seen.append(query)
        return [SearchPage(title="t", url="http://x", contents="c")]

    monkeypatch.setattr(
        "src.internal.servers.web.app.search_tool", fake_search_tool
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", _passthrough_fetch
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app._expanded_queries",
        lambda query, llm: [query, f"{query} alt"],
    )
    await _run_hybrid_search(
        "etf fees",
        llm=None,
        search_url="http://x/retrieve",
        top_k=2,
        filters=None,
        source_provider="serpapi",
        domain="finance",
    )
    assert seen == ["etf fees finance", "etf fees alt finance"]
    # Exactly once: no query carries the hint twice.
    assert all(q.count("finance") == 1 for q in seen)


@pytest.mark.asyncio
async def test_hybrid_corpus_branch_stays_raw(monkeypatch):
    seen: list[str] = []

    async def fake_expanded(query, **_kwargs):
        seen.append(query)
        from src.internal.search.process_search_query import SearchQueryResult

        return SearchQueryResult(executed_queries=[query], results=[])

    monkeypatch.setattr(
        "src.internal.servers.web.app.run_expanded_search", fake_expanded
    )
    await _run_hybrid_search(
        "etf fees",
        llm=None,
        search_url="http://x/retrieve",
        top_k=2,
        filters=None,
        source_provider="retrieval",
        domain="finance",
    )
    assert seen == ["etf fees"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/servers/web/test_search_domain_http.py -v -k hybrid`
Expected: FAIL — unexpected keyword argument `domain`.

- [ ] **Step 3: Implement**

Add to the signature after `source_provider`:

```python
    source_provider: str,
    domain: str = "general",
```

The `source_provider == "retrieval"` branch is left exactly as-is — the corpus never gets a hint.

In the `source_provider == "browser"` branch, hint the call (browser is not the corpus):

```python
            browser_docs = await _run_browser_search(
                prepare_domain_query(query, domain),
                browser_search_url=browser_search_url,
                top_k=top_k * 2,
                existing_count=0,
            )
```

In `_fetch_provider`, hint the browser call and each dispatched query. The hint goes here, after `_expanded_queries` has already run on the raw query — hinting before expansion would feed the topic word to the LLM expander and duplicate it into every variant:

```python
    async def _fetch_provider(provider: str) -> list[ContextDocument]:
        if provider == "browser":
            if not browser_search_url:
                return []
            # IDs are globally reassigned by _finalize_hybrid -> _reindex_documents, so starting at 0 here is safe.
            return await _run_browser_search(
                prepare_domain_query(query, domain),
                browser_search_url=browser_search_url,
                top_k=top_k * 2,
                existing_count=0,
            )
        # Corpus stays raw; every other provider gets one hint per dispatched
        # query. Not _is_web_provider — that set excludes google and serper.
        dispatched = [
            eq if provider == "retrieval" else prepare_domain_query(eq, domain)
            for eq in executed_queries
        ]
        page_lists: list[list[SearchPage]] = list(
            await asyncio.gather(
                *[
                    search_tool(
                        dispatched_query,
                        provider=provider,
                        search_url=search_url,
                        page_size=top_k,
                        timeout_seconds=5,
                        max_retries=1,
                        **(
                            {"filters": _filters_payload(filters)}
                            if provider == "retrieval"
                            else {}
                        ),
                    )
                    for dispatched_query in dispatched
                ]
            )
        )
```

Then zip the document construction over `dispatched` rather than `executed_queries`, so each card reports the query that actually ran:

```python
        for dispatched_query, pages in zip(dispatched, page_lists):
            docs.extend(
                _documents_from_search_pages(
                    pages,
                    source_provider=provider,
                    query=dispatched_query,
                    start_index=len(docs) + 1,
                    entry_point="hybrid_search",
                )
            )
```

Leave the returned `executed_queries` as the unhinted list: it is shown to the user as what was asked, and the prior spec's multi-query metadata rule distinguishes requested from executed at the tool layer, not here.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/servers/web/test_search_domain_http.py -v -k hybrid`
Expected: 2 passed.

- [ ] **Step 5: Run the hybrid regression suites**

Run: `pytest tests/unit/servers/web/test_hybrid_web_fallback.py tests/unit/servers/web/test_browser_pipeline.py tests/unit/test_execution_fallbacks.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/servers/web/test_search_domain_http.py src/internal/servers/web/app.py
git commit -m "feat(web): hint web providers with the selected domain in hybrid search"
```

---

### Task 4: Thread the domain through the route and reject non-honoring modes

**Files:**
- Modify: `src/internal/servers/web/app.py:496-525` (`_WebHybridRetrievalStage`), `:565-592` (`_auto_search_pipeline`), `:1161-1182` + `:984` + `:1004` + `:1085` + `:1299` (`_run_auto_routed`), `:1672` + `:1719` + `:1762` (dispatch call sites)
- Test: `tests/unit/servers/web/test_search_domain_http.py`

**Interfaces:**
- Consumes: `_run_direct_search(..., domain=)`, `_run_hybrid_search(..., domain=)` from Tasks 2 and 3
- Produces: `domain` honored on auto, `search_tool`, and `hybrid_search`; HTTP 400 on a non-`general` domain with a non-honoring explicit mode

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/servers/web/test_search_domain_http.py`:

```python
_NON_HONORING_MODES = ["chat_once", "chat_loop", "search_agent", "tool_agent"]


@pytest.mark.parametrize("mode", _NON_HONORING_MODES)
def test_non_honoring_mode_rejects_a_domain(client, mode):
    r = client.post(
        "/api/agent", json={"query": "q", "mode": mode, "domain": "finance"}
    )
    assert r.status_code == 400
    assert "does not support" in r.json()["detail"]


@pytest.mark.parametrize("mode", _NON_HONORING_MODES)
def test_non_honoring_mode_accepts_general(client, mode):
    r = client.post(
        "/api/agent", json={"query": "q", "mode": mode, "domain": "general"}
    )
    assert r.status_code != 400


def test_honoring_modes_accept_a_domain(client):
    for mode in ("search_tool", "hybrid_search"):
        r = client.post(
            "/api/agent", json={"query": "q", "mode": mode, "domain": "finance"}
        )
        assert r.status_code != 400, mode
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/servers/web/test_search_domain_http.py -v -k mode`
Expected: FAIL — non-honoring modes currently return something other than 400.

- [ ] **Step 3: Add the mode guard**

Next to `_VALID_AGENT_MODES` near line 2196:

```python
# Modes where one caller-supplied query reaches a provider. Agent loops build
# their own per-round queries inside src/agents, so a single entry-point hint
# cannot apply exactly once across rounds; chat_once does no retrieval.
_DOMAIN_HONORING_MODES = {"search_tool", "hybrid_search"}
```

In `_run_agent_impl`, right after the `domain` normalization from Task 1:

```python
        requested_mode = _MODE_ALIASES.get(request.mode or "", request.mode)
        if (
            domain != "general"
            and requested_mode is not None
            and requested_mode not in _DOMAIN_HONORING_MODES
        ):
            # Refusing beats silently dropping the field, per the taxonomy spec.
            raise HTTPException(
                status_code=400,
                detail=(
                    f"mode {requested_mode!r} does not support a search domain; "
                    "use 'search_tool', 'hybrid_search', or omit mode for auto"
                ),
            )
```

- [ ] **Step 4: Thread the value through the auto path**

`_WebHybridRetrievalStage.__init__` gains `domain: str = "general"` stored as `self._domain`, and its `retrieve` passes `domain=self._domain` to `_run_hybrid_search`.

`_auto_search_pipeline` gains `domain: str = "general"` and passes it to the `_WebHybridRetrievalStage(...)` constructor.

`_run_auto_routed` gains `domain: str = "general"` after `source_provider`, and passes `domain=domain` to both `_auto_search_pipeline` calls (lines ~984 and ~1299) and to the external `_run_direct_search` call (~1085). The corpus-only `_run_direct_search` call at ~1004 is left alone — it passes `source_provider="retrieval"`, so Task 2's guard already makes a hint impossible there; not passing it states the intent.

- [ ] **Step 5: Thread the value at the three dispatch call sites**

Pass `domain=domain` to `_run_auto_routed` (~1672), `_run_direct_search` (~1719), and `_run_hybrid_search` (~1762).

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/servers/web/test_search_domain_http.py -v`
Expected: all pass.

- [ ] **Step 7: Run the full backend suite**

Run: `pytest -q`
Expected: no new failures against the pre-change baseline.

- [ ] **Step 8: Commit**

```bash
git add tests/unit/servers/web/test_search_domain_http.py src/internal/servers/web/app.py
git commit -m "feat(web): thread the search domain through auto routing and guard other modes"
```

---

### Task 5: Frontend selector

**Files:**
- Modify: `web/src/types.ts`, `web/src/api.ts`, `web/src/components/SearchComposer.tsx:29-47` and `:96-113`, `web/src/pages/AssistPage.tsx:130-140` and `:295-312`
- Test: `web/src/components/__tests__/SearchComposer.test.tsx`

**Interfaces:**
- Consumes: `GET /api/search-domains` from Task 1; `domain` on the agent body from Task 4
- Produces: `SearchDomainOption`, `fetchSearchDomains()`, `domain` / `onDomainChange` props on `SearchComposer`

- [ ] **Step 1: Write the failing test**

Append to `web/src/components/__tests__/SearchComposer.test.tsx`:

```tsx
it("renders the domain selector and reports a change", async () => {
  const onDomainChange = vi.fn();
  render(
    <SearchComposer
      query="q"
      searchUrl=""
      topK={5}
      sourceProvider="auto"
      isLoading={false}
      domain="general"
      domainOptions={[
        { name: "general", description: "Broad or mixed-topic search; default" },
        { name: "finance", description: "Markets, investments, banking" },
      ]}
      onQueryChange={() => {}}
      onSearchUrlChange={() => {}}
      onTopKChange={() => {}}
      onSourceProviderChange={() => {}}
      onDomainChange={onDomainChange}
      onSubmit={() => {}}
    />,
  );
  const select = screen.getByLabelText("Domain");
  expect(select).toHaveValue("general");
  await userEvent.selectOptions(select, "finance");
  expect(onDomainChange).toHaveBeenCalledWith("finance");
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd web && npx vitest run src/components/__tests__/SearchComposer.test.tsx`
Expected: FAIL — no element labelled "Domain".

- [ ] **Step 3: Add the type and the fetcher**

In `web/src/types.ts`:

```ts
// Mirrors the GET /api/search-domains payload in
// src/internal/servers/web/app.py. Order is the registry's order.
export interface SearchDomainOption {
  name: string;
  description: string;
}
```

In `web/src/api.ts`:

```ts
export function fetchSearchDomains(): Promise<{ domains: SearchDomainOption[] }> {
  return requestJson<{ domains: SearchDomainOption[] }>("/search-domains");
}
```

- [ ] **Step 4: Add the control**

In `SearchComposerProps`:

```ts
  domain: string;
  domainOptions: SearchDomainOption[];
  onDomainChange: (value: string) => void;
```

In the `composer-controls` div, before the `Top K` label — always visible, unlike the dev-gated Source and URL fields:

```tsx
        <label>
          Domain
          <select
            value={domain}
            onChange={(e) => onDomainChange(e.currentTarget.value)}
          >
            {domainOptions.map((opt) => (
              <option key={opt.name} value={opt.name} title={opt.description}>
                {opt.name.replace(/_/g, " ")}
              </option>
            ))}
          </select>
        </label>
```

- [ ] **Step 5: Wire the page**

In `AssistPage.tsx`, add state and load the options once on mount, falling back to `general` alone if the request fails so the composer always renders:

```tsx
  const [domain, setDomain] = useState("general");
  const [domainOptions, setDomainOptions] = useState<SearchDomainOption[]>([
    { name: "general", description: "Broad or mixed-topic search; default" },
  ]);

  useEffect(() => {
    fetchSearchDomains()
      .then((r) => setDomainOptions(r.domains))
      .catch(() => {});
  }, []);
```

Send it in the request body next to `top_k`:

```tsx
        top_k: topK,
        domain,
```

Pass the props to `SearchComposer`:

```tsx
        domain={domain}
        domainOptions={domainOptions}
        onDomainChange={setDomain}
```

The selection is held in page state, so it persists across turns within the session and resets on reload.

- [ ] **Step 6: Run the test to verify it passes**

Run: `cd web && npx vitest run src/components/__tests__/SearchComposer.test.tsx`
Expected: PASS.

- [ ] **Step 7: Type-check and run the frontend suite**

Run: `cd web && npm run typecheck && npx vitest run`
Expected: no type errors, no new test failures.

- [ ] **Step 8: Commit**

```bash
git add web/src
git commit -m "feat(web): add a search domain selector to the Assist composer"
```

---

### Task 6: Documentation

**Files:**
- Modify: `docs/search-engine.md`

- [ ] **Step 1: Document the field and the endpoint**

Add a section describing: the `domain` field on `POST /api/agent` and its default; that hints reach web providers only and never the corpus; which three modes honor it and that the others return 400; the `GET /api/search-domains` endpoint; and that no relevance improvement is claimed pending the evaluation the taxonomy spec requires.

- [ ] **Step 2: Verify the referenced paths exist**

Run: `grep -n "search-domains\|domain" docs/search-engine.md`
Expected: the new section is present and names only real routes.

- [ ] **Step 3: Commit**

```bash
git add docs/search-engine.md
git commit -m "docs: describe the search domain field and registry endpoint"
```

---

## Self-Review

**Spec coverage:** request field (Task 1), validation before network calls (Task 1), registry endpoint and single-source taxonomy (Task 1), per-provider hint with the corpus excluded (Tasks 2 and 3), the `_is_web_provider` trap covered by a test (Task 2), exactly-once across expansion (Task 3), honored-mode table and 400 on the rest (Task 4), auto path threading (Task 4), always-visible selector defaulting to general and persisted per session (Task 5), limits documented (Task 6). The spec's cache-separation property is a consequence of hinting before dispatch; it is asserted indirectly by the Task 3 exactly-once test rather than by a separate cache test, because the serving cache is configured in the app lifespan and not reachable from these helper-level tests.

**Placeholder scan:** no TBD, TODO, or "handle edge cases" steps; every code step carries real code.

**Type consistency:** `domain: str` throughout the backend; `SearchDomainOption` used identically in `types.ts`, `api.ts`, and `SearchComposer.tsx`; `prepare_domain_query(query, domain)` called with the same argument order in both leaves.
