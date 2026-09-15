"""Domain hints through search tools, cache, and fallback."""

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
    _, _, meta = asyncio.run(
        tool.execute("i", {"queries": [" battery "], "domain": domain})
    )
    assert seen == (
        ["battery"] if domain == "general" else ["battery academic research"]
    )
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
        asyncio.run(
            MultiQueryWebSearchTool(search_fn=fake).execute(
                "i", {"queries": queries, "domain": "unknown"}
            )
        )


def test_single_query_domain(monkeypatch):
    seen = []

    async def fake(query, **kwargs):
        seen.append(query)
        return "formatted"

    monkeypatch.setattr("src.internal.tools.search.search_for_tool_string", fake)
    tool = build_search_tool()
    assert tool.schema.parameters["properties"]["domain"]["default"] == "general"
    assert asyncio.run(tool.execute("i", {"query": "patent", "domain": "ip"})) == (
        "formatted",
        "formatted",
        {},
    )
    assert seen == ["patent intellectual property"]


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
        browser_search_url="http://browser/retrieve",
        serpapi_fn=serp,
        browser_fn=browser,
    )
    tool = MultiQueryWebSearchTool(search_fn=cascade)
    asyncio.run(tool.execute("i", {"queries": ["battery"], "domain": "academic"}))
    assert seen == [
        ("serp", "battery academic research"),
        ("browser", "battery academic research"),
    ]
