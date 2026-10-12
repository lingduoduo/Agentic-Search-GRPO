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
    assert result["queries"][0]["executed_query"] == "one financial market data filings"
    assert "error" in result["queries"][1]
    result = await module.extract_page("https://example.test")
    assert result[0]["content"] == "Full text"


@pytest.mark.asyncio
async def test_mcp_web_capability_uses_existing_provider(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "1")
    from src.internal.mcp_server.tools import search as module
    from src.internal.tools.search import restricted_query

    calls = []

    async def search(query, **options):
        calls.append((query, options))
        return []

    monkeypatch.setattr(module, "search_tool", search)
    monkeypatch.setenv("MCP_WEB_SEARCH_PROVIDER", "serper")
    await module.search_domain("battery", domain="academic")
    assert calls == [
        (
            restricted_query("battery", "academic"),
            {"provider": "serper", "page_size": 5},
        ),
        ("battery academic research", {"provider": "serper", "page_size": 5}),
    ]


@pytest.mark.asyncio
async def test_mcp_search_web_scopes_then_falls_back(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "1")
    from src.internal.mcp_server.tools import search as module
    from src.internal.tools.search import restricted_query

    calls = []

    async def serper(query, *, page_size):
        calls.append(query)
        return []

    monkeypatch.setattr(module, "serper_dev_search", serper)
    monkeypatch.setenv("MCP_WEB_SEARCH_PROVIDER", "serper")
    result = await module.search_web("battery", domain="academic")
    assert calls == [
        restricted_query("battery", "academic"),
        "battery academic research",
    ]
    assert result["executed_query"] == restricted_query("battery", "academic")
    assert result["executed_queries"] == calls


@pytest.mark.asyncio
async def test_mcp_search_web_setting_off_issues_one_suffix_call(monkeypatch):
    from src.internal.mcp_server.tools import search as module

    calls = []

    async def serper(query, *, page_size):
        calls.append(query)
        return []

    monkeypatch.setattr(module, "serper_dev_search", serper)
    monkeypatch.setenv("MCP_WEB_SEARCH_PROVIDER", "serper")
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    result = await module.search_web("battery", domain="academic")
    assert calls == ["battery academic research"]
    assert result["executed_queries"] == calls


@pytest.mark.asyncio
async def test_mcp_search_web_malformed_setting_returns_the_error_payload(
    monkeypatch,
):
    from src.internal.mcp_server.tools import search as module

    calls = []

    async def serper(query, *, page_size):
        calls.append(query)
        return []

    monkeypatch.setattr(module, "serper_dev_search", serper)
    monkeypatch.setenv("MCP_WEB_SEARCH_PROVIDER", "serper")
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "disabled")
    result = await module.search_web("battery", domain="academic")
    assert result["error"].startswith("Web search failed:")
    assert result["results"] == [] and calls == []
    assert result["executed_queries"] == ["battery academic research"]


@pytest.mark.asyncio
async def test_mcp_search_web_domains_fans_out(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    monkeypatch.setenv("MCP_WEB_SEARCH_PROVIDER", "serper")
    from src.internal.mcp_server.tools import search as module

    calls = []

    async def serper(query, *, page_size):
        calls.append(query)
        return [SearchPage(title=query, summary="s", url=f"https://x.test/{query}")]

    monkeypatch.setattr(module, "serper_dev_search", serper)
    result = await module.search_web("rust", domains=["news", "code"])
    assert len(calls) == 2
    assert result["domains"] == ["news", "code"]
    assert sorted(result["executed_queries"]) == sorted(calls)
    assert len(result["results"]) == 2


@pytest.mark.asyncio
async def test_mcp_search_web_rejects_bad_domains_without_dispatch(monkeypatch):
    monkeypatch.setenv("MCP_WEB_SEARCH_PROVIDER", "serper")
    from src.internal.mcp_server.tools import search as module

    calls = []

    async def serper(query, *, page_size):
        calls.append(query)
        return []

    monkeypatch.setattr(module, "serper_dev_search", serper)
    too_many = await module.search_web(
        "rust", domains=["news", "code", "film", "academic"]
    )
    mixed = await module.search_web("rust", domain="news", domains=["code"])
    for payload in (too_many, mixed):
        assert payload["error"] and payload["results"] == []
    assert calls == []


@pytest.mark.asyncio
async def test_mcp_search_domain_passes_domains(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    from src.internal.mcp_server.tools import search as module

    calls = []

    async def web(query, **kwargs):
        calls.append(query)
        return [SearchPage(title=query, summary="s", url=f"https://x.test/{query}")]

    service = DomainSearch(web_search_fn=web, tools=[])
    monkeypatch.setattr(module, "_domain_service", lambda: service)
    result = await module.search_domain("rust", domains=["news", "code"])
    assert result["domains"] == ["news", "code"] and len(calls) == 2
    batch = await module.batch_search([{"query": "rust"}], domains=["news", "code"])
    assert batch["queries"][0]["domains"] == ["news", "code"]


@pytest.mark.asyncio
async def test_mcp_search_web_error_payload_lists_every_category_query(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    monkeypatch.setenv("MCP_WEB_SEARCH_PROVIDER", "serper")
    from src.internal.mcp_server.tools import search as module
    from src.internal.tools.search import prepare_domain_query

    async def serper(query, *, page_size):
        raise RuntimeError("down")

    monkeypatch.setattr(module, "serper_dev_search", serper)
    result = await module.search_web("rust", domains=["news", "code"])
    assert "error" in result
    assert result["executed_queries"] == [
        prepare_domain_query("rust", "news"),
        prepare_domain_query("rust", "code"),
    ]


@pytest.mark.asyncio
async def test_mcp_search_web_one_domains_entry_equals_domain(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    monkeypatch.setenv("MCP_WEB_SEARCH_PROVIDER", "serper")
    from src.internal.mcp_server.tools import search as module

    calls = []

    async def serper(query, *, page_size):
        calls.append(query)
        return [SearchPage(title=query, summary="s", url=f"https://x.test/{query}")]

    monkeypatch.setattr(module, "serper_dev_search", serper)
    single = await module.search_web("rust", domain="news")
    single_calls, calls[:] = list(calls), []
    listed = await module.search_web("rust", domains=["news"])
    assert listed == single and calls == single_calls


def _fake_resolver(selected, status="ok"):
    async def fake_resolve(query, domains):
        if domains == ["auto"]:
            return list(selected), {"selected": list(selected), "status": status}
        return domains, None

    return fake_resolve


@pytest.mark.asyncio
async def test_mcp_search_web_auto_reports_domains_and_auto_domains(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    monkeypatch.setenv("MCP_WEB_SEARCH_PROVIDER", "serper")
    from src.internal.mcp_server.tools import search as module

    calls = []

    async def serper(query, *, page_size):
        calls.append(query)
        return [SearchPage(title=query, summary="s", url=f"https://x.test/{query}")]

    monkeypatch.setattr(module, "serper_dev_search", serper)
    monkeypatch.setattr(
        module, "resolve_auto_domains", _fake_resolver(["news", "code"])
    )
    result = await module.search_web("rust", domains=["auto"])
    assert len(calls) == 2
    assert result["domains"] == ["news", "code"]
    assert result["auto_domains"] == {"selected": ["news", "code"], "status": "ok"}
    monkeypatch.setattr(module, "resolve_auto_domains", _fake_resolver(["news"]))
    result = await module.search_web("rust", domains=["auto"])
    assert result["domains"] == ["news"]
    assert result["auto_domains"]["selected"] == ["news"]
    plain = await module.search_web("rust", domains=["news", "code"])
    assert "auto_domains" not in plain


@pytest.mark.asyncio
async def test_mcp_auto_mixed_with_names_makes_no_provider_call(monkeypatch):
    monkeypatch.setenv("MCP_WEB_SEARCH_PROVIDER", "serper")
    from src.internal.mcp_server.tools import search as module
    from src.internal.tools.base import InvalidToolInput

    calls = []

    async def serper(query, *, page_size):
        calls.append(query)
        return []

    async def web(query, **kwargs):
        calls.append(query)
        return []

    monkeypatch.setattr(module, "serper_dev_search", serper)
    monkeypatch.setattr(
        module, "_domain_service", lambda: DomainSearch(web_search_fn=web, tools=[])
    )
    payload = await module.search_web("rust", domains=["auto", "news"])
    assert payload["error"] and payload["results"] == []
    with pytest.raises(InvalidToolInput):
        await module.search_domain("rust", domains=["auto", "news"])
    assert calls == []


@pytest.mark.asyncio
async def test_mcp_search_domain_auto_reports_auto_domains(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    from src.internal.mcp_server.tools import search as module

    async def web(query, **kwargs):
        return [SearchPage(title=query, summary="s", url=f"https://x.test/{query}")]

    monkeypatch.setattr(
        "src.internal.tools.search.resolve_auto_domains",
        _fake_resolver(["news", "code"]),
    )
    monkeypatch.setattr(
        module, "_domain_service", lambda: DomainSearch(web_search_fn=web, tools=[])
    )
    result = await module.search_domain("rust", domains=["auto"])
    assert result["domains"] == ["news", "code"]
    assert result["auto_domains"]["status"] == "ok"
    batch = await module.batch_search([{"query": "rust"}], domains=["auto"])
    assert batch["queries"][0]["auto_domains"]["selected"] == ["news", "code"]
