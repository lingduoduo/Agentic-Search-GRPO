"""MCP delegates domain features to the same repository service."""

import pytest

from src.internal.tools.search import DomainSearch
from src.internal.tools.search import SearchPage


@pytest.mark.asyncio
async def test_mcp_domain_features_share_service(monkeypatch):
    from src.internal.mcp_server.tools import search as module

    calls = []

    async def web(query, **kwargs):
        calls.append(query)
        return [SearchPage(title="Page", summary=query, url="https://example.test")]

    async def fetch(url, **kwargs):
        return "Full text"

    service = DomainSearch(web_search_fn=web, tools=[], fetch_fn=fetch)
    monkeypatch.setattr(module, "_domain_service", lambda: service)
    directory = await module.get_sub_domains(["academic"])
    assert directory["domains"][0]["sub_domains"][0]["sub_domain"] == "academic.web"
    assert calls == []
    result = await module.search_domain("battery", domain="academic")
    assert result["executed_query"] == "battery academic research"
    result = await module.batch_search(
        [{"query": "one"}, {"query": ""}], domain="finance"
    )
    assert result["queries"][0]["executed_query"] == "one finance"
    assert "error" in result["queries"][1]
    result = await module.extract_page("https://example.test")
    assert result[0]["content"] == "Full text"


@pytest.mark.asyncio
async def test_mcp_web_capability_uses_existing_provider(monkeypatch):
    from src.internal.mcp_server.tools import search as module

    calls = []

    async def search(query, **options):
        calls.append((query, options))
        return []

    monkeypatch.setattr(module, "search_tool", search)
    monkeypatch.setenv("MCP_WEB_SEARCH_PROVIDER", "serper")
    await module.search_domain("battery", domain="academic")
    assert calls == [
        ("battery academic research", {"provider": "serper", "page_size": 5})
    ]
