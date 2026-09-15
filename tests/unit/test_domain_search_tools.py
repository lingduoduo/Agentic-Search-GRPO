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
    result, _, errors = await registry.invoke(
        "batch_search", {"queries": [{"query": "one"}, {}]}
    )
    assert not errors
    items = json.loads(result)["queries"]
    assert items[0]["query"] == "one" and "error" in items[1]
    assert registry.get("search_domain").citeable
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


def test_seeded_features_are_available_to_agents():
    from src.internal.tools.knowledge_base import seed_tools

    registry = ToolRegistry()
    seed_tools(registry)
    assert {"search_domain", "get_sub_domains", "extract_page", "batch_search"} <= {
        tool.name for tool in registry.agent_tools()
    }
