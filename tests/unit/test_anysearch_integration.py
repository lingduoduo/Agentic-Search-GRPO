"""AnySearch reuses the existing search result and tool contracts."""

import pytest

from src.internal.tools.anysearch import AnySearchClient, AnySearchError
from src.internal.tools.search import MultiQueryWebSearchTool, search_tool


@pytest.mark.asyncio
async def test_provider_normalizes_pages_and_tool_domain_once(monkeypatch):
    calls = []

    async def search(self, query, **options):
        calls.append((query, options))
        return {
            "data": {
                "results": [
                    {
                        "title": "Paper",
                        "url": "https://example.test/paper",
                        "content": "Details",
                        "metadata": {"year": 2026},
                    }
                ]
            }
        }

    monkeypatch.setattr(AnySearchClient, "search", search)
    tool = MultiQueryWebSearchTool(provider="anysearch", page_size=3)
    text, pages, metadata = await tool.execute(
        "test", {"queries": ["battery"], "domain": "academic"}
    )
    assert calls == [("battery academic research", {"max_results": 3})]
    assert "Details" in text
    assert pages[0].metadata == {"year": 2026}
    assert metadata["executed_queries"] == ["battery academic research"]


@pytest.mark.asyncio
async def test_provider_returns_error_page(monkeypatch):
    async def search(*args, **kwargs):
        raise AnySearchError("rate limited", status=429, request_id="req1")

    monkeypatch.setattr(AnySearchClient, "search", search)
    pages = await search_tool("battery", provider="anysearch")
    assert pages[0].error == "AnySearch: rate limited (request_id: req1)"


@pytest.mark.asyncio
async def test_anysearch_mcp_provider(monkeypatch):
    from src.internal.mcp_server.tools import search as mcp

    calls = []

    async def search(self, query, **options):
        calls.append(query)
        return {
            "data": {
                "results": [
                    {
                        "title": "Paper",
                        "url": "https://example.test/paper",
                        "snippet": "Details",
                    }
                ]
            }
        }

    monkeypatch.setattr(AnySearchClient, "search", search)
    monkeypatch.setenv("MCP_WEB_SEARCH_PROVIDER", "anysearch")
    result = await mcp.search_web("battery", domain="academic")
    assert calls == ["battery academic research"]
    assert result["results"][0]["snippet"] == "Details"
    assert result["executed_query"] == calls[0]


@pytest.mark.asyncio
async def test_provider_does_not_share_credential_dependent_cache(monkeypatch):
    from src.internal.cache import serving

    calls = []

    async def search(self, query, **options):
        calls.append(self.api_key)
        return {"data": {"results": [{"url": "https://example.test"}]}}

    monkeypatch.setattr(AnySearchClient, "search", search)
    serving.configure_serving_cache(60)
    try:
        for key in ("first-key", "second-key"):
            monkeypatch.setenv("ANYSEARCH_API_KEY", key)
            await search_tool("same query", provider="anysearch")
        assert calls == ["first-key", "second-key"]
    finally:
        serving.reset_serving_cache()


@pytest.mark.asyncio
async def test_unsupported_page_never_dispatches(monkeypatch):
    async def search(*args, **kwargs):
        pytest.fail("unsupported page reached provider")

    monkeypatch.setattr(AnySearchClient, "search", search)
    pages = await search_tool("query", provider="anysearch", page=2)
    assert "pagination" in pages[0].error
