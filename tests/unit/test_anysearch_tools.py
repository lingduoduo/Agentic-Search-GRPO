"""Native AnySearch operations use the repository's ToolRegistry contracts."""

import json
from types import SimpleNamespace

import pytest

from src.internal.tools.anysearch import AnySearchError
from src.internal.tools.anysearch_tools import build_anysearch_tools
from src.internal.tools.base import ToolEffect
from src.internal.tools.registry import ToolRegistry


@pytest.fixture
def registry():
    calls = []

    async def search(query, **options):
        calls.append(("search", query, options))
        return {
            "data": {
                "results": [
                    {"title": "Quote", "content": "AAPL", "url": "https://example.test"}
                ]
            }
        }

    async def discover(domains):
        calls.append(("domains", domains))
        return {
            "data": {
                "domains": [
                    {
                        "domain": "finance",
                        "sub_domains": [{"sub_domain": "finance.quote"}],
                    }
                ]
            }
        }

    async def extract(url):
        calls.append(("extract", url))
        return {"data": {"title": "Page", "content": "Full page", "url": url}}

    async def batch(queries):
        calls.append(("batch", queries))
        return [
            await search("AAPL"),
            AnySearchError("rate limited", status=429, request_id="req2"),
        ]

    client = SimpleNamespace(
        search=search, get_sub_domains=discover, extract=extract, batch_search=batch
    )
    registry = ToolRegistry()
    for tool in build_anysearch_tools(client=client):
        registry.register(tool)
    return registry, calls


@pytest.mark.asyncio
async def test_native_search_is_citeable_json_without_query_hints(registry):
    registry, calls = registry
    response, _, errors = await registry.invoke(
        "anysearch_search",
        {"query": "AAPL", "tag": "finance.quote", "params": {"symbol": "AAPL"}},
    )
    assert not errors
    assert json.loads(response) == [
        {"title": "Quote", "content": "AAPL", "url": "https://example.test"}
    ]
    assert calls[0] == (
        "search",
        "AAPL",
        {"tag": "finance.quote", "params": {"symbol": "AAPL"}, "max_results": 5},
    )
    assert registry.get("anysearch_search").citeable


@pytest.mark.asyncio
async def test_capabilities_and_extract(registry):
    registry, calls = registry
    response, _, errors = await registry.invoke(
        "anysearch_get_sub_domains", {"domains": ["finance"]}
    )
    assert not errors
    assert (
        json.loads(response)["domains"][0]["sub_domains"][0]["sub_domain"]
        == "finance.quote"
    )
    response, _, errors = await registry.invoke(
        "anysearch_extract", {"url": "https://example.test"}
    )
    assert not errors
    assert json.loads(response) == [
        {"title": "Page", "content": "Full page", "url": "https://example.test"}
    ]
    assert registry.get("anysearch_extract").citeable
    assert not registry.get("anysearch_get_sub_domains").citeable


@pytest.mark.asyncio
async def test_batch_preserves_queries_errors_and_order(registry):
    registry, _ = registry
    queries = [{"query": "AAPL", "tag": "finance.quote"}, {"query": "broken"}]
    response, _, errors = await registry.invoke(
        "anysearch_batch_search", {"queries": queries}
    )
    assert not errors
    items = json.loads(response)["queries"]
    assert items[0]["query"] == "AAPL"
    assert items[0]["results"][0]["content"] == "AAPL"
    assert items[1] == {
        "query": "broken",
        "error": "rate limited",
        "status": 429,
        "request_id": "req2",
    }
    assert not registry.get("anysearch_batch_search").citeable


@pytest.mark.asyncio
async def test_invalid_or_failed_tools_return_json_errors():
    async def search(*args, **kwargs):
        raise AnySearchError("unavailable", status=503, request_id="req3")

    tools = build_anysearch_tools(client=SimpleNamespace(search=search))
    response, _, _ = await tools[0].execute("x", {"query": "x"})
    assert json.loads(response) == {
        "error": "unavailable",
        "status": 503,
        "request_id": "req3",
    }


def test_tools_are_opt_in_and_read_only(monkeypatch, registry):
    from src.internal.tools.knowledge_base import tool_knowledge_base

    monkeypatch.delenv("AGENTIC_SEARCH_ANYSEARCH_ENABLED", raising=False)
    assert not any(t.name.startswith("anysearch_") for t in tool_knowledge_base())
    monkeypatch.setenv("AGENTIC_SEARCH_ANYSEARCH_ENABLED", "true")
    enabled = [t for t in tool_knowledge_base() if t.name.startswith("anysearch_")]
    assert {t.name for t in enabled} == {
        "anysearch_search",
        "anysearch_get_sub_domains",
        "anysearch_extract",
        "anysearch_batch_search",
    }
    assert all(t.effect == ToolEffect.READ_ONLY for t in enabled)
