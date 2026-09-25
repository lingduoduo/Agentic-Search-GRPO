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
        json.loads(result)["domains"][0]["sub_domains"][0]["sub_domain"] == "film.web"
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
