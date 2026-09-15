# Search Domains Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking. Subagent execution is an alternative only if selected by the user.

**Goal:** Make the requested 17 search domains discoverable and usable as optional query hints through search tools and MCP web search.

**Architecture:** A dependency-free registry owns identifiers, descriptions, validation, and query preparation. Tool entry points apply hints once and pass ordinary strings through the existing provider, cache, and fallback paths. Default searches retain their behavior.

**Tech Stack:** Python >=3.10, dataclasses, existing ToolSchema/FunctionTool, existing MCP registration, pytest with mocked providers.

**Spec:** [Search domains design](../specs/2026-09-14-search-domains-design.md)

**Status:** Implemented on `feat/search-domains` after user authorization to add code and create a PR. The checklist records completed execution steps.

## Global Constraints

- Python >=3.10; no new runtime dependencies.
- Preserve the exact 17 identifiers and their supplied order.
- Omitted domain and explicit `general` preserve existing search behavior.
- Topic domains never replace, relax, or infer authorization filters.
- Domain selection never changes provider selection or fallback order.
- Apply topic hints exactly once, before provider dispatch and cache lookup.
- Reject invalid explicit domains before any network call.
- No HTTP API, UI, index schema, or search-agent state changes in this increment.

## File map and execution order

| File | Responsibility |
| --- | --- |
| Create `src/internal/tools/search_domains.py` | Registry, validation, preparation, schema descriptions |
| Modify `src/internal/tools/search.py` | Single-query and multi-query entry points only |
| Modify `src/internal/mcp_server/tools/search.py` | MCP web-search argument, dispatch, response metadata |
| Create `tests/unit/test_search_domains.py` | Registry contracts |
| Create `tests/unit/test_search_domain_tools.py` | Tool behavior, injected signatures, cache and fallback |
| Modify `tests/unit/test_mcp_server.py` | MCP web-search coverage using established fixtures |
| Modify `docs/search-engine.md` and `docs/mcp.md` | Supported usage and limits |

Execute Tasks 1–4 in order. Preserve unrelated workspace edits. Establish a clean
baseline for the targeted tests before changing production code; report existing
failures separately. Create an isolated worktree at execution time if needed.
Commit commands below are local checkpoints, not instructions to push or merge.

### Task 1: Shared domain registry

**Files:** Create `src/internal/tools/search_domains.py` and `tests/unit/test_search_domains.py`.

**Interfaces:**

- Consumes: the exact taxonomy table in the spec.
- Produces: `SearchDomain(description: str, query_hint: str)`, `DOMAIN_REGISTRY: dict[str, SearchDomain]`, `AVAILABLE_DOMAINS: list[str]`, `normalize_search_domain(value: str = "general") -> str`, `prepare_domain_query(query: str, domain: str = "general") -> str`, `search_domain_parameter() -> dict[str, object]`.

- [x] **Step 1: Add failing contract tests.**

```python
import pytest

from src.internal.tools.search_domains import (
    AVAILABLE_DOMAINS, DOMAIN_REGISTRY, normalize_search_domain,
    prepare_domain_query, search_domain_parameter,
)

EXPECTED = [
    "general", "resource", "social_media", "finance", "academic", "legal",
    "health", "business", "security", "ip", "code", "energy",
    "environment", "agriculture", "travel", "film", "gaming",
]

def test_registry_contract():
    assert AVAILABLE_DOMAINS == EXPECTED
    assert list(DOMAIN_REGISTRY) == EXPECTED
    assert all(item.description for item in DOMAIN_REGISTRY.values())
    assert DOMAIN_REGISTRY["general"].query_hint == ""
    assert DOMAIN_REGISTRY["ip"].query_hint == "intellectual property"

@pytest.mark.parametrize("value", ["Social Media", "social-media", " social_media "])
def test_normalize_alias(value):
    assert normalize_search_domain(value) == "social_media"

@pytest.mark.parametrize("value", [None, 1, [], "", " ", "unknown"])
def test_invalid_domain(value):
    with pytest.raises(ValueError, match="general"):
        normalize_search_domain(value)

def test_prepare_query():
    assert prepare_domain_query("  battery recycling  ") == "  battery recycling  "
    assert prepare_domain_query("battery recycling", "academic") == "battery recycling academic research"
    assert prepare_domain_query("  ", "academic") == "  "

def test_schema_is_independent():
    first = search_domain_parameter()
    first["enum"].clear()
    second = search_domain_parameter()
    assert second["enum"] == EXPECTED
    assert second["default"] == "general"
    assert "intellectual property" in second["description"].lower()
```

- [x] **Step 2: Run `python -m pytest tests/unit/test_search_domains.py -q`.** Expect import failure for the new module; confirm it is the intended missing implementation.
- [x] **Step 3: Implement the module.** Populate every registry entry from the spec's table, in order, with the complete meaning text as `description` and the exact hint column as `query_hint`. Use this implementation around that literal registry:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SearchDomain:
    description: str
    query_hint: str

DOMAIN_REGISTRY: dict[str, SearchDomain] = {
    'general': SearchDomain('Broad or mixed-topic search; default', ''),
    'resource': SearchDomain('Datasets, reference materials, directories, and reusable tools', 'resources'),
    'social_media': SearchDomain('Public social posts, communities, and discussions', 'social media'),
    'finance': SearchDomain('Markets, investments, banking, and financial analysis', 'finance'),
    'academic': SearchDomain('Scholarly literature, research methods, and publications', 'academic research'),
    'legal': SearchDomain('Law, regulation, case law, and legal procedure', 'law'),
    'health': SearchDomain('Medicine, public health, and clinical information', 'health'),
    'business': SearchDomain('Companies, operations, strategy, and commerce', 'business'),
    'security': SearchDomain('Cybersecurity, vulnerabilities, and defensive practices', 'cybersecurity'),
    'ip': SearchDomain('Intellectual property: patents, trademarks, copyright, and licensing', 'intellectual property'),
    'code': SearchDomain('Source code, programming, APIs, and developer documentation', 'programming'),
    'energy': SearchDomain('Generation, fuels, storage, and energy systems', 'energy'),
    'environment': SearchDomain('Climate, ecosystems, conservation, and pollution', 'environment'),
    'agriculture': SearchDomain('Farming, crops, livestock, and agricultural systems', 'agriculture'),
    'travel': SearchDomain('Destinations, transport, lodging, and trip planning', 'travel'),
    'film': SearchDomain('Cinema, films, filmmaking, and the film industry', 'film'),
    'gaming': SearchDomain('Video games, game development, and gaming communities', 'video games'),
}
AVAILABLE_DOMAINS = list(DOMAIN_REGISTRY)

def normalize_search_domain(value: str = "general") -> str:
    if isinstance(value, str):
        canonical = value.strip().lower().replace("-", "_").replace(" ", "_")
        if canonical in DOMAIN_REGISTRY:
            return canonical
    raise ValueError("domain must be one of: " + ", ".join(DOMAIN_REGISTRY))

def prepare_domain_query(query: str, domain: str = "general") -> str:
    canonical = normalize_search_domain(domain)
    hint = DOMAIN_REGISTRY[canonical].query_hint
    return f"{query} {hint}" if hint and query.strip() else query

def search_domain_parameter() -> dict[str, object]:
    descriptions = "; ".join(
        f"{name}: {entry.description}" for name, entry in DOMAIN_REGISTRY.items()
    )
    return {
        "type": "string",
        "enum": list(DOMAIN_REGISTRY),
        "default": "general",
        "description": (
            "Optional topic query hint, not a guaranteed result filter. "
            "Use general for broad or mixed topics. " + descriptions
        ),
    }
```

- [x] **Step 4: Run `python -m pytest tests/unit/test_search_domains.py -q` and `ruff check src/internal/tools/search_domains.py tests/unit/test_search_domains.py`.** Expect all tests and lint checks to pass.
- [x] **Step 5: Commit the two files** with `git add src/internal/tools/search_domains.py tests/unit/test_search_domains.py` and `git commit -m "feat(search): define search domain taxonomy"`.

### Task 2: Function-calling tool integration

**Files:** Modify `src/internal/tools/search.py`; create `tests/unit/test_search_domain_tools.py`.

**Interfaces:**

- Consumes: Task 1's three helper functions.
- Produces: optional `domain` on the two tool schemas and executable calls; non-general multi-query metadata includes `domain` and `executed_queries`.
- Keeps: `search_tool`, `_search_fn`, and cascade signatures unchanged.

- [x] **Step 1: Add failing tests for both public tool entry points.**

```python
import asyncio
import pytest
from src.internal.tools.search import MultiQueryWebSearchTool, build_search_tool

@pytest.mark.parametrize("domain", ["general", "academic"])
def test_multi_query_preserves_injected_signature(domain):
    seen = []
    async def fake(query, *, provider, search_url, page_size, timeout_seconds):
        seen.append(query)
        return []
    tool = MultiQueryWebSearchTool(search_fn=fake)
    _, _, meta = asyncio.run(tool.execute("i", {"queries": [" battery "], "domain": domain}))
    assert seen == (["battery"] if domain == "general" else ["battery academic research"])
    assert meta["queries"] == ["battery"]
    if domain == "general":
        assert meta == {"queries": ["battery"]}
    else:
        assert meta["domain"] == "academic"
        assert meta["executed_queries"] == seen

@pytest.mark.parametrize("queries", [[], ["battery"]])
def test_invalid_domain_never_dispatches(queries):
    async def fake(*args, **kwargs):
        pytest.fail("invalid domain reached search")
    with pytest.raises(ValueError):
        asyncio.run(MultiQueryWebSearchTool(search_fn=fake).execute(
            "i", {"queries": queries, "domain": "unknown"}
        ))

def test_single_query_domain(monkeypatch):
    seen = []
    async def fake(query, **kwargs):
        seen.append(query)
        return "formatted"
    monkeypatch.setattr("src.internal.tools.search.search_for_tool_string", fake)
    tool = build_search_tool()
    assert tool.schema.parameters["properties"]["domain"]["default"] == "general"
    assert asyncio.run(tool.execute("i", {"query": "patent", "domain": "ip"})) == (
        "formatted", "formatted", {}
    )
    assert seen == ["patent intellectual property"]
```

- [x] **Step 2: Run `python -m pytest tests/unit/test_search_domain_tools.py -q`.** Expect failures because schemas lack domain and calls ignore/reject the argument.
- [x] **Step 3: Add schema properties and prepare queries at entry points.** Import the shared helpers. Add `"domain": search_domain_parameter()` to both property dictionaries without changing required fields. In multi-query execution, use:

```python
domain = normalize_search_domain(arguments.get("domain", "general"))
queries = _normalize_queries_input(arguments.get("queries", []))
if not queries:
    return "No results found.", [], {}
executed_queries = [prepare_domain_query(q, domain) for q in queries]
# Existing asyncio.gather iterates executed_queries; all existing kwargs remain.
# Existing result merging and formatting remain.
metadata = {"queries": queries}
if domain != "general":
    metadata.update(domain=domain, executed_queries=executed_queries)
return format_search_pages(merged), merged, metadata
```

For `build_search_tool`, change only the inner callable and schema:

```python
async def search(query: str, domain: str = "general") -> str:
    return await search_for_tool_string(
        prepare_domain_query(query, domain),
        provider=provider,
        search_url=search_url,
        page_size=page_size,
    )
```

- [x] **Step 4: Add cache and fallback tests before final verification.** These tests exercise prepared queries through existing infrastructure and must fail if the hint is omitted or applied twice.

```python
def test_domain_queries_use_existing_cache(monkeypatch):
    from src.internal.cache import serving
    from src.internal.tools.search import SearchPage
    calls = []
    async def fake(query, **kwargs):
        calls.append(query)
        return [SearchPage(url="https://example.test/result")]
    monkeypatch.setattr("src.internal.tools.search.serpapi_search", fake)
    serving.configure_serving_cache(60)
    try:
        tool = MultiQueryWebSearchTool(provider="serpapi")
        for domain in ("general", "academic", "academic"):
            asyncio.run(tool.execute("i", {"queries": ["battery"], "domain": domain}))
        assert calls == ["battery", "battery academic research"]
    finally:
        serving.reset_serving_cache()

def test_domain_applied_once_through_cascade():
    from src.internal.tools.search import SearchPage, make_web_cascade_search
    seen = []
    async def serp(query, **kwargs):
        seen.append(("serp", query))
        return []
    async def browser(query, **kwargs):
        seen.append(("browser", query))
        return [SearchPage(url="https://example.test/result")]
    cascade = make_web_cascade_search(
        browser_search_url="http://browser/retrieve", serpapi_fn=serp, browser_fn=browser
    )
    tool = MultiQueryWebSearchTool(search_fn=cascade)
    asyncio.run(tool.execute("i", {"queries": ["battery"], "domain": "academic"}))
    assert seen == [
        ("serp", "battery academic research"),
        ("browser", "battery academic research"),
    ]
```

- [x] **Step 5: Run `python -m pytest tests/unit/test_search_domain_tools.py tests/unit/test_search_tools.py tests/unit/test_search_tools_cache.py tests/unit/test_web_cascade_search.py tests/unit/test_tool_categories.py tests/unit/test_tool_search_acl.py -q`.** Expect existing default behavior, ACL, deduplication, and new hint tests to pass. Resolve new failures without weakening existing assertions.
- [x] **Step 6: Commit** with `git add src/internal/tools/search.py tests/unit/test_search_domain_tools.py` and `git commit -m "feat(search): support optional domain hints in search tools"`.

### Task 3: MCP web-search integration

**Files:** Modify `src/internal/mcp_server/tools/search.py` and `tests/unit/test_mcp_server.py`.

**Interfaces:**

- Consumes: Task 1 helpers; existing MCP provider helpers and registration decorator.
- Produces: `search_web(query: str, limit: int = 5, domain: str = "general") -> dict[str, Any]`; optional `domain`/`executed_query` response fields for non-general calls.

- [x] **Step 1: Extend the existing MCP unit module with failing tests.** Follow its current registration/import fixture setup; do not load integration-server fixtures.

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("provider,helper", [
    ("google", "google_custom_search"),
    ("serpapi", "serpapi_search"),
    ("serper", "serper_dev_search"),
])
@pytest.mark.parametrize("broken", [False, True])
async def test_search_web_domain(monkeypatch, provider, helper, broken):
    from src.internal.mcp_server.tools import search as module
    seen = []
    async def fake(query, **kwargs):
        seen.append(query)
        if broken:
            raise RuntimeError("provider unavailable")
        return []
    monkeypatch.setenv("MCP_WEB_SEARCH_PROVIDER", provider)
    monkeypatch.setattr(module, helper, fake)
    result = await module.search_web("battery", domain="Academic")
    assert seen == ["battery academic research"]
    assert result["query"] == "battery"
    assert result["domain"] == "academic"
    assert result["executed_query"] == seen[0]
    assert ("error" in result) == broken

@pytest.mark.asyncio
async def test_search_web_rejects_domain_before_dispatch(monkeypatch):
    from src.internal.mcp_server.tools import search as module
    async def fake(*args, **kwargs):
        pytest.fail("invalid domain reached provider")
    for name in ("google_custom_search", "serpapi_search", "serper_dev_search"):
        monkeypatch.setattr(module, name, fake)
    with pytest.raises(ValueError):
        await module.search_web("battery", domain="unknown")
```

- [x] **Step 2: Run `python -m pytest tests/unit/test_mcp_server.py -k 'search_web' -q`.** Expect the new tests to fail on the unsupported domain argument.
- [x] **Step 3: Add the optional argument, shared normalization, and response metadata.** Immediately before existing logging/provider dispatch:

```python
domain = normalize_search_domain(domain)
executed_query = prepare_domain_query(query, domain)
domain_metadata = (
    {"domain": domain, "executed_query": executed_query}
    if domain != "general" else {}
)
```

Replace the query argument to all three provider calls with `executed_query`.
Add `**domain_metadata` to success and exception response dicts; keep `query`
as the original value. Keep provider selection and page-error filtering unchanged.

Generate the tool's domain documentation before MCP registration using a small
local decorator, beneath `@mcp_server.tool()` so Python applies it first:

```python
def _describe_search_domain(fn):
    fn.__doc__ = (fn.__doc__ or "") + "\n\n" + str(
        search_domain_parameter()["description"]
    )
    return fn

```

Insert `@_describe_search_domain` immediately below the existing
`@mcp_server.tool()` on `search_web`; do not create a second registered tool. The decorator only modifies documentation and preserves function
identity/signature. The MCP parameter is a string with runtime validation; only
the manually constructed FunctionTool schemas expose an enum in this increment.

- [x] **Step 4: Verify generated documentation and default responses.** Add:

```python
@pytest.mark.asyncio
async def test_search_web_general_contract(monkeypatch):
    from src.internal.mcp_server.tools import search as module
    from src.internal.tools.search_domains import AVAILABLE_DOMAINS
    async def fake(query, **kwargs):
        assert query == "battery"
        return []
    monkeypatch.setenv("MCP_WEB_SEARCH_PROVIDER", "google")
    monkeypatch.setattr(module, "google_custom_search", fake)
    for kwargs in ({}, {"domain": "general"}):
        assert await module.search_web("battery", **kwargs) == {"results": [], "query": "battery"}
    assert all(name in module.search_web.__doc__ for name in AVAILABLE_DOMAINS)
```

- [x] **Step 5: Run `python -m pytest tests/unit/test_mcp_server.py -q`.** Expect all new and existing MCP cases to pass, including indexed-document search. If the local environment lacks optional MCP dependencies, record the missing dependency and run this suite in the repository's MCP-enabled test environment before declaring this task verified.
- [x] **Step 6: Commit** with `git add src/internal/mcp_server/tools/search.py tests/unit/test_mcp_server.py` and `git commit -m "feat(mcp): add search domain hints to public web search"`.

### Task 4: Documentation and final regression checks

**Files:** Modify `docs/search-engine.md` and `docs/mcp.md`.

**Interfaces:** Consumes the completed tool signatures and metadata contracts; produces user documentation and a recorded verification result.

- [x] **Step 1: Add a “Search topic domains” section to `docs/search-engine.md`.** Include all 17 identifiers and their meanings from the spec table, the single-domain-per-call rule, normalization behavior, and these executable tool-argument examples:

```json
{"query": "battery recycling", "domain": "academic"}
```

```json
{"queries": ["battery recycling", "lithium recovery"], "domain": "academic"}
```

Include this behavior explanation:

> Domains are optional topic hints on the function-calling search tools and MCP
> public-web search. `general` preserves the original query. `academic` appends
> `academic research`; other domains use the documented fixed hints. Domains
> do not guarantee category membership or select a provider. The HTTP search
> endpoints and UI do not accept a domain selector in this release. `ip` means
> intellectual property; `resource` covers reusable reference material and tools.

Document all overlap boundaries from the spec and that hints may reduce recall.

- [x] **Step 2: Update `docs/mcp.md` with the public-web call and response example.**

```json
{"query": "battery recycling", "limit": 5, "domain": "academic"}
```

```json
{
  "results": [],
  "query": "battery recycling",
  "domain": "academic",
  "executed_query": "battery recycling academic research"
}
```

Explain that omitted/general responses retain their existing shape, invalid
domains produce tool errors, and indexed-document search has no domain argument.
Link to the search-engine taxonomy instead of maintaining a second table.

- [x] **Step 3: Run the complete affected regression set once.**

```bash
python -m pytest tests/unit/test_search_domains.py tests/unit/test_search_domain_tools.py tests/unit/test_search_tools.py tests/unit/test_search_tools_cache.py tests/unit/test_web_cascade_search.py tests/unit/test_tool_categories.py tests/unit/test_tool_search_acl.py tests/unit/test_mcp_server.py -q
ruff check src/internal/tools/search_domains.py src/internal/tools/search.py src/internal/mcp_server/tools/search.py tests/unit/test_search_domains.py tests/unit/test_search_domain_tools.py tests/unit/test_mcp_server.py
git diff --check
```

Expect all checks to pass. If a check fails, fix the introduced issue and rerun
the affected check; report unrelated baseline failures explicitly. No live search
credentials or external network calls belong in these unit tests.

- [x] **Step 4: Review the final diff against acceptance criteria 1–10.** Confirm
no hint reaches the provider twice, no `domain` keyword leaks into injected
callables, no required schema field changes, and no permissions/provider logic
changes. Record executed test results and the limit that mocked tests do not
measure real-world relevance. Do not claim native vertical-search support.
- [x] **Step 5: Commit documentation** with `git add docs/search-engine.md docs/mcp.md` and `git commit -m "docs(search): explain topic domain hints and supported entry points"`.

## Spec coverage and handoff

| Requirement | Task |
| --- | --- |
| Exact taxonomy, overlap definitions, normalization, validation | 1, 4 |
| Query hints and fresh schema descriptions | 1 |
| Single-/multi-query tools, metadata, callable compatibility | 2 |
| Once-only transformation, cache and fallback behavior | 2 |
| MCP providers, invalid input, success/error metadata, discovery text | 3 |
| Default compatibility, authorization and category regressions | 2, 3, 4 |
| Supported surfaces and quality limitations | 4 |

Executed inline with `superpowers:executing-plans`, using an isolated worktree
at `/tmp/agentic-search-domains`. A read-only reviewer checked the production
change separately through `superpowers:requesting-code-review`.

## Execution results

- Existing baseline: 92 tests passed.
- Registry tests failed first on the absent module, then 12 passed.
- Tool tests failed first on absent domain handling, then the affected tool suites
  passed (65 tests).
- MCP domain tests failed first on the absent argument, then all 42 MCP tests passed.
- Combined affected regression suite: 119 passed.
- Ruff checks and `git diff --check` passed.
- Provider calls were mocked; live relevance improvements were not evaluated.
- Documentation includes the full taxonomy and limits for both supported surfaces.

Implementation follows the planned signatures and preserves the general/default
contracts. Full HTTP/UI integration and strict topic filtering remain outside
this increment. The completed spec, plan, code, tests, and user documentation
are included together in the PR.
