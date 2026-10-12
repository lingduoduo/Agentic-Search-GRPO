"""Search features are native tools with the repository's result contracts."""

import json

import pytest

from src.internal.tools.search import DomainSearch
from src.internal.tools.search import build_domain_search_tools
from src.internal.tools.registry import ToolRegistry
from src.internal.tools.search import SearchPage


@pytest.mark.asyncio
async def test_tool_registry_supports_all_four_features():
    async def web(query, **kwargs):
        if query.startswith("boom"):
            raise RuntimeError("provider down")
        return [SearchPage(title="Page", summary=query, url="https://example.test")]

    async def fetch(url, **kwargs):
        return "Full text"

    service = DomainSearch(web_search_fn=web, fetch_fn=fetch, tools=[])
    registry = ToolRegistry()
    for tool in build_domain_search_tools(service=service):
        registry.register(tool)
    result, _, errors = await registry.invoke(
        "search_domain", {"query": "battery", "domain": "academic"}
    )
    assert not errors
    assert json.loads(result)[0]["content"] == "battery academic research"
    result, _, errors = await registry.invoke("get_sub_domains", {"domains": ["film"]})
    assert not errors
    assert (
        json.loads(result)["domains"][0]["sub_domains"][0]["sub_domain"] == "media.web"
    )
    result, _, errors = await registry.invoke(
        "extract_page", {"url": "https://example.test"}
    )
    assert not errors
    assert json.loads(result)[0]["content"] == "Full text"
    # A malformed item is rejected by the schema before anything runs...
    _, _, errors = await registry.invoke(
        "batch_search", {"queries": [{"query": "one"}, {}]}
    )
    assert errors == ["Missing required argument: 'queries.1.query'"]
    # ...while a runtime failure stays a per-item error beside the other results.
    result, _, errors = await registry.invoke(
        "batch_search", {"queries": [{"query": "one"}, {"query": "boom"}]}
    )
    assert not errors
    items = json.loads(result)["queries"]
    assert items[0]["query"] == "one" and "error" in items[1]
    assert not registry.get("search_domain").citeable
    assert registry.get("extract_page").citeable
    assert not registry.get("get_sub_domains").citeable
    assert not registry.get("batch_search").citeable


@pytest.mark.asyncio
async def test_invalid_native_tool_arguments_are_json_errors():
    service = DomainSearch(tools=[])
    tools = {tool.name: tool for tool in build_domain_search_tools(service=service)}
    result, _, _ = await tools["search_domain"].execute(
        "i", {"query": "x", "tag": "legal.missing"}
    )
    assert "unsupported capability" in json.loads(result)["error"]


# The three facade tools route to public-data tools the agent already holds
# directly, so offering both doubles the menu without adding reach. PR #479
# established that a system prompt alone does not fix that; withholding does.
FACADE_TOOLS = frozenset({"search_domain", "get_sub_domains", "batch_search"})


def test_seeded_features_stay_registered_and_invocable():
    """Withholding is an agent-menu decision, not a removal: /admin and MCP keep them."""
    from src.internal.tools.knowledge_base import seed_tools

    registry = ToolRegistry()
    seed_tools(registry)
    all_four = FACADE_TOOLS | {"extract_page"}

    assert all_four <= {summary["name"] for summary in registry.all_summaries()}
    assert all(registry.get(name) is not None for name in all_four)


def test_facade_tools_are_withheld_from_agents_but_extract_page_is_offered():
    from src.internal.tools.knowledge_base import seed_tools

    registry = ToolRegistry()
    seed_tools(registry)
    offered = {tool.name for tool in registry.agent_tools()}

    assert FACADE_TOOLS.isdisjoint(offered)
    # Nothing else seeded fetches a URL, so this one adds reach rather than a
    # second path to it.
    assert "extract_page" in offered


def test_no_capability_target_is_reachable_two_ways_from_the_agent_menu():
    """The invariant that makes the withholding above durable."""
    from src.internal.tools.knowledge_base import seed_tools
    from src.internal.tools.search import iter_capabilities

    registry = ToolRegistry()
    seed_tools(registry)
    offered = {tool.name for tool in registry.agent_tools()}
    direct_targets = {
        cap.tool_name for _tag, cap in iter_capabilities() if cap.tool_name
    }

    assert direct_targets <= offered, "capability targets must stay directly callable"
    assert FACADE_TOOLS.isdisjoint(offered), (
        "a facade over already-offered tools must not also be on the menu"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "fetched", "error"),
    [
        # Passes the schema's scheme pattern but has no host.
        ("https://", "unused", "InvalidToolInput: url must be an HTTP(S) URL"),
        (
            "https://dead.test",
            "[fetch error] timeout",
            "InvalidToolInput: [fetch error] timeout",
        ),
    ],
)
async def test_extract_page_bad_url_or_dead_link_feeds_back(url, fetched, error):
    """One bad link is about this argument; it must not disable the tool."""
    from src.internal.tools import FailureCategory

    async def fetch(url, **kwargs):
        return fetched

    registry = ToolRegistry()
    for tool in build_domain_search_tools(
        service=DomainSearch(fetch_fn=fetch, tools=[])
    ):
        registry.register(tool)
    outcome = await registry.invoke_detailed("extract_page", {"url": url})
    assert outcome.failure.category is FailureCategory.INVALID_INPUT
    assert json.loads(outcome.response) == {"error": error}  # same text as before


@pytest.mark.asyncio
async def test_registry_accepts_legacy_selectors_without_mutating_arguments():
    async def web(query, **kwargs):
        return [SearchPage(title="Page", summary=query, url="https://example.test")]

    registry = ToolRegistry()
    for tool in build_domain_search_tools(
        service=DomainSearch(web_search_fn=web, tools=[])
    ):
        registry.register(tool)
    for arguments in (
        {"query": "earnings", "domain": " Finance "},
        {"queries": [{"query": "earnings", "domain": "finance"}]},
        {"domains": ["general", "film"]},
    ):
        import copy

        before = copy.deepcopy(arguments)
        name = (
            "get_sub_domains"
            if "domains" in arguments
            else "batch_search"
            if "queries" in arguments
            else "search_domain"
        )
        response, _, errors = await registry.invoke(name, arguments)
        assert not errors
        assert "error" not in json.loads(response)
        assert arguments == before
    _, _, errors = await registry.invoke(
        "search_domain", {"query": "q", "domain": "unknown"}
    )
    assert errors


@pytest.mark.asyncio
async def test_web_search_tool_scopes_then_falls_back(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "1")
    from src.internal.tools.search import MultiQueryWebSearchTool, restricted_query

    calls = []

    async def cascade(query, **kwargs):
        calls.append(("cascade", query))
        return [SearchPage(title="t", summary="s", url="https://x.example/1")]

    async def serp_only(query, **kwargs):
        calls.append(("serp", query))
        return []

    tool = MultiQueryWebSearchTool(
        search_fn=cascade, restricted_search_fn=serp_only, page_size=3
    )
    _, _, metadata = await tool.execute(
        "i", {"queries": ["etf fees"], "domain": "financial"}
    )
    assert calls == [
        ("serp", restricted_query("etf fees", "financial")),
        ("cascade", "etf fees financial market data filings"),
    ]
    assert metadata["executed_queries"] == [
        restricted_query("etf fees", "financial"),
        "etf fees financial market data filings",
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

    def build(**kw):
        fn = real(**kw)
        built.append((kw, fn))
        return fn

    monkeypatch.setattr(
        "src.internal.tools.knowledge_base.make_web_cascade_search", build
    )
    monkeypatch.setenv("AGENTIC_SEARCH_BROWSER_SEARCH_URL", "http://browser/retrieve")
    tools = tool_knowledge_base()
    browser_free = [fn for kw, fn in built if kw == {"browser_search_url": None}]
    assert len(browser_free) == 1
    captured = browser_free[0]
    web_tool = next(t for t in tools if t.name == "web_search")
    # extract_page is bound to the DomainSearch the domain tools were built over.
    service = next(
        t for t in tools if t.name == "extract_page"
    )._fn.__wrapped__.__self__
    assert web_tool._restricted_search_fn is captured
    assert service.restricted_search_fn is captured


@pytest.mark.asyncio
async def test_web_search_tool_setting_off_issues_one_suffix_call(monkeypatch):
    from src.internal.tools.search import MultiQueryWebSearchTool

    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    normal, restricted = [], []

    async def fn(query, **kwargs):
        normal.append(query)
        return []

    async def serp_only(query, **kwargs):
        restricted.append(query)
        return []

    tool = MultiQueryWebSearchTool(
        search_fn=fn, restricted_search_fn=serp_only, page_size=3
    )
    await tool.execute("i", {"queries": ["battery"], "domain": "academic"})
    assert normal == ["battery academic research"]
    assert restricted == []


def _web_search_tool(calls):
    from src.internal.tools.search import MultiQueryWebSearchTool

    async def fake(query, **kwargs):
        calls.append(query)
        return [SearchPage(title=query, summary="s", url=f"https://x.test/{query}")]

    return MultiQueryWebSearchTool(search_fn=fake)


@pytest.mark.asyncio
async def test_web_search_domains_issues_each_suffix_query(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    calls = []
    tool = _web_search_tool(calls)
    registry = ToolRegistry()
    registry.register(tool)
    arguments = {"queries": ["rates"], "domains": ["finance", "news"]}
    _, _, errors = await registry.invoke("web_search", arguments)
    assert not errors
    assert len(calls) == 2
    calls.clear()
    _, _, meta = await tool.execute("i", arguments)
    assert meta["domains"] == ["financial", "news"]
    assert meta["executed_queries"] == calls and len(calls) == 2


@pytest.mark.asyncio
async def test_web_search_rejects_four_domains_and_mixed_selectors():
    from src.internal.tools.base import InvalidToolInput

    calls = []
    tool = _web_search_tool(calls)
    registry = ToolRegistry()
    registry.register(tool)
    _, _, errors = await registry.invoke(
        "web_search",
        {"queries": ["q"], "domains": ["finance", "news", "code", "film"]},
    )
    assert errors
    with pytest.raises(InvalidToolInput):
        await tool.execute(
            "i", {"queries": ["q"], "domain": "news", "domains": ["finance"]}
        )
    assert calls == []


def test_web_search_domains_items_carry_no_category_descriptions():
    from src.internal.tools.search import DOMAIN_REGISTRY

    properties = _web_search_tool([]).schema.parameters["properties"]
    assert properties["domains"]["items"] == {
        "type": "string",
        "enum": list(DOMAIN_REGISTRY) + ["auto"],
    }
    assert "financial:" in properties["domain"]["description"]


@pytest.mark.asyncio
async def test_web_search_domain_aliases_pass_schema_validation(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    calls = []
    registry = ToolRegistry()
    registry.register(_web_search_tool(calls))
    _, _, errors = await registry.invoke(
        "web_search", {"queries": ["q"], "domains": ["Finance", "financial", "news"]}
    )
    assert not errors
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_search_domain_tool_accepts_domains(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    calls = []

    async def web(query, **kwargs):
        calls.append(query)
        return [SearchPage(title=query, summary="s", url=f"https://x.test/{query}")]

    registry = ToolRegistry()
    for tool in build_domain_search_tools(
        service=DomainSearch(web_search_fn=web, tools=[])
    ):
        registry.register(tool)
    _, _, errors = await registry.invoke(
        "search_domain", {"query": "q", "domains": ["news", "code"]}
    )
    assert not errors and len(calls) == 2
    _, _, errors = await registry.invoke(
        "search_domain", {"query": "q", "domains": ["news", "code", "film", "academic"]}
    )
    assert errors
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_web_search_domain_web_is_allowed_with_domains(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    calls = []
    registry = ToolRegistry()
    registry.register(_web_search_tool(calls))
    _, _, errors = await registry.invoke(
        "web_search",
        {"queries": ["q"], "domain": "web", "domains": ["news", "code"]},
    )
    assert not errors and len(calls) == 2
    calls.clear()
    with pytest.raises(ValueError):
        await registry.invoke(
            "web_search",
            {"queries": ["q"], "domain": "news", "domains": ["news", "code"]},
        )
    assert calls == []


@pytest.mark.asyncio
async def test_web_search_one_domains_entry_equals_domain(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    calls = []
    tool = _web_search_tool(calls)
    single = await tool.execute("i", {"queries": ["q"], "domain": "news"})
    single_calls, calls[:] = list(calls), []
    listed = await tool.execute("i", {"queries": ["q"], "domains": ["news"]})
    assert calls == single_calls
    assert listed[2] == single[2] and listed[0] == single[0]


def _patch_search_resolver(monkeypatch, selected, status="ok"):
    seen = []

    async def fake_resolve(query, domains):
        seen.append((query, domains))
        if domains == ["auto"]:
            return list(selected), {"selected": list(selected), "status": status}
        return domains, None

    monkeypatch.setattr("src.internal.tools.search.resolve_auto_domains", fake_resolve)
    return seen


@pytest.mark.asyncio
async def test_web_search_auto_passes_validation_and_searches_the_picks(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    seen = _patch_search_resolver(monkeypatch, ["news", "code"])
    calls = []
    tool = _web_search_tool(calls)
    registry = ToolRegistry()
    registry.register(tool)
    _, _, errors = await registry.invoke(
        "web_search", {"queries": ["rates"], "domains": ["auto"]}
    )
    assert not errors
    assert len(calls) == 2
    assert seen[0] == ("rates", ["auto"])
    calls.clear()
    _, _, meta = await tool.execute("i", {"queries": ["rates"], "domains": ["auto"]})
    assert meta["domains"] == ["news", "code"]
    assert meta["auto_domains"] == {"selected": ["news", "code"], "status": "ok"}
    assert meta["executed_queries"] == calls


@pytest.mark.asyncio
async def test_web_search_single_category_auto_reports_domains(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    _patch_search_resolver(monkeypatch, ["news"])
    tool = _web_search_tool([])
    _, _, meta = await tool.execute("i", {"queries": ["q"], "domains": ["auto"]})
    assert meta["domains"] == ["news"]
    assert meta["auto_domains"]["selected"] == ["news"]


@pytest.mark.asyncio
async def test_web_search_without_auto_has_no_auto_domains(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    tool = _web_search_tool([])
    _, _, meta = await tool.execute(
        "i", {"queries": ["q"], "domains": ["news", "code"]}
    )
    assert "auto_domains" not in meta


@pytest.mark.asyncio
async def test_auto_mixed_with_names_makes_no_provider_call():
    from src.internal.tools.base import InvalidToolInput

    calls = []
    with pytest.raises(InvalidToolInput):
        await _web_search_tool(calls).execute(
            "i", {"queries": ["q"], "domains": ["auto", "news"]}
        )

    async def web(query, **kwargs):
        calls.append(query)
        return []

    registry = ToolRegistry()
    for tool in build_domain_search_tools(
        service=DomainSearch(web_search_fn=web, tools=[])
    ):
        registry.register(tool)
    result, _, errors = await registry.invoke(
        "search_domain", {"query": "q", "domains": ["auto", "news"]}
    )
    assert errors or "auto" in str(result)
    assert calls == []


@pytest.mark.asyncio
async def test_search_domain_tool_schema_accepts_auto(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    _patch_search_resolver(monkeypatch, ["news", "code"])
    calls = []

    async def web(query, **kwargs):
        calls.append(query)
        return [SearchPage(title=query, summary="s", url=f"https://x.test/{query}")]

    registry = ToolRegistry()
    for tool in build_domain_search_tools(
        service=DomainSearch(web_search_fn=web, tools=[])
    ):
        registry.register(tool)
    _, _, errors = await registry.invoke(
        "search_domain", {"query": "q", "domains": ["auto"]}
    )
    assert not errors and len(calls) == 2
    _, _, errors = await registry.invoke(
        "batch_search", {"queries": [{"query": "q", "domains": ["auto"]}]}
    )
    assert not errors and len(calls) == 4
