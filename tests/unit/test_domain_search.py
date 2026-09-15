"""Domain features delegate to existing repository tools and search helpers."""

import asyncio
import json

import pytest

from src.internal.tools.base import FunctionTool, ToolEffect
from src.internal.tools.search import DomainSearch
from src.internal.tools.search import SearchPage
from src.internal.tools.search import AVAILABLE_DOMAINS


@pytest.fixture
def service():
    calls = []

    async def web(query, *, page_size, **kwargs):
        calls.append(("web", query, page_size))
        return [SearchPage(title="Page", summary="Body", url="https://example.test")]

    async def quote(symbol):
        calls.append(("quote", symbol))
        return json.dumps({"symbol": symbol, "price": 123})

    async def paper(query, limit=3):
        calls.append(("paper", query, limit))
        return json.dumps(
            [{"title": query, "content": "Abstract", "url": "https://paper.test"}]
        )

    async def fetch(url, *, max_length):
        calls.append(("fetch", url, max_length))
        return "Extracted text"

    tools = [
        FunctionTool(
            quote,
            name="get_stock_quote",
            description="Stock quote",
            parameters={
                "type": "object",
                "properties": {"symbol": {"type": "string", "description": "Ticker"}},
                "required": ["symbol"],
            },
            effect=ToolEffect.READ_ONLY,
        ),
        FunctionTool(
            paper,
            name="search_arxiv",
            description="Papers",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
            effect=ToolEffect.READ_ONLY,
            citeable=True,
        ),
    ]
    return DomainSearch(web_search_fn=web, tools=tools, fetch_fn=fetch), calls


def test_discovery_uses_real_schemas_and_all_domains(service):
    client, calls = service
    for domain in AVAILABLE_DOMAINS:
        directory = client.get_sub_domains([domain])["domains"][0]
        assert directory["domain"] == domain
        assert directory["sub_domains"][0]["sub_domain"] == f"{domain}.web"
    finance = client.get_sub_domains(["Finance"])["domains"][0]["sub_domains"]
    quote = next(item for item in finance if item["sub_domain"] == "finance.quote")
    assert quote["tool_name"] == "get_stock_quote"
    assert quote["query_parameter"] == "symbol"
    assert quote["parameters"]["properties"]["symbol"]["description"] == "Ticker"
    quote["parameters"]["properties"].clear()
    fresh = client.get_sub_domains(["finance"])["domains"][0]["sub_domains"][1]
    assert "symbol" in fresh["parameters"]["properties"]
    assert calls == []


@pytest.mark.asyncio
async def test_domain_query_routes_existing_web_search_once(service):
    client, calls = service
    result = await client.search("battery", domain="academic", max_results=3)
    assert calls == [("web", "battery academic research", 3)]
    assert result["tag"] == "academic.web"
    assert result["query"] == "battery"
    assert result["executed_query"] == "battery academic research"
    assert result["results"][0]["content"] == "Body"


@pytest.mark.asyncio
async def test_native_capability_uses_actual_tool_without_hints(service):
    client, calls = service
    result = await client.search("AAPL", tag="finance.quote")
    assert calls == [("quote", "AAPL")]
    assert result["results"] == {"symbol": "AAPL", "price": 123}
    result = await client.search(
        "battery",
        domain="academic",
        tag="academic.arxiv",
        params='{"limit":2}',
    )
    assert calls[-1] == ("paper", "battery", 2)
    assert result["results"][0]["content"] == "Abstract"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"domain": "unknown"},
        {"tag": "legal.nonexistent"},
        {"tag": "finance.quote", "domain": "code"},
        {"tag": "finance.quote", "params": {"bad": "value"}},
        {"tag": "finance.quote", "params": {"symbol": "MSFT"}},
        {"tag": "academic.arxiv", "params": {"limit": False}},
        {"domain": "general", "params": {"zone": "cn"}},
    ],
)
async def test_invalid_routes_and_params_fail_before_dispatch(service, options):
    client, calls = service
    with pytest.raises(ValueError):
        await client.search("AAPL", **options)
    assert calls == []


@pytest.mark.parametrize("domains", [[], ["general"] * 6, ["unknown"], "finance"])
def test_invalid_discovery_request(service, domains):
    with pytest.raises(ValueError):
        service[0].get_sub_domains(domains)


@pytest.mark.asyncio
async def test_extraction_reuses_existing_fetcher(service):
    client, calls = service
    assert await client.extract("https://example.test", max_length=100) == [
        {
            "title": "https://example.test",
            "content": "Extracted text",
            "url": "https://example.test",
        }
    ]
    assert calls == [("fetch", "https://example.test", 100)]
    with pytest.raises(ValueError):
        await client.extract("file:///etc/passwd")


@pytest.mark.asyncio
async def test_batch_concurrent_ordered_failures_and_overrides(service, monkeypatch):
    client, _ = service
    started = []
    ready = asyncio.Event()

    async def search(query, **options):
        started.append((query, options))
        if len(started) == 3:
            ready.set()
        await asyncio.wait_for(ready.wait(), timeout=1)
        if query == "broken":
            raise ValueError("failed")
        return {"query": query, "results": []}

    monkeypatch.setattr(client, "search", search)
    queries = [
        {"query": "one"},
        {"query": "broken"},
        {"query": "paper", "tag": "academic.arxiv"},
    ]
    results = await client.batch_search(queries, domain="finance", tag="finance.quote")
    assert [r["query"] for r in results] == ["one", "broken", "paper"]
    assert results[1]["error"] == "failed"
    assert started[0][1] == {"domain": "finance", "tag": "finance.quote"}
    assert started[2][1] == {"tag": "academic.arxiv"}
    assert queries[0] == {"query": "one"}
    for bad in ([], [{"query": "x"}] * 6):
        with pytest.raises(ValueError):
            await client.batch_search(bad)


@pytest.mark.asyncio
async def test_backend_error_is_not_reported_as_empty_results(service):
    client, _ = service

    async def broken(*args, **kwargs):
        return [SearchPage(error="no provider configured")]

    client.web_search_fn = broken
    with pytest.raises(ValueError, match="no provider configured"):
        await client.search("x")


@pytest.mark.asyncio
async def test_finance_route_reuses_real_stock_tool(monkeypatch):
    from src.internal.tools.public_data import market

    calls = []

    async def get_json(url, **kwargs):
        calls.append(url)
        return {
            "chart": {
                "result": [{"meta": {"regularMarketPrice": 123, "currency": "USD"}}]
            }
        }

    monkeypatch.setattr(market, "get_json", get_json)
    result = await DomainSearch().search("AAPL", tag="finance.quote")
    assert result["results"]["symbol"] == "AAPL"
    assert result["results"]["current_price"] == 123
    assert calls == [market.YAHOO_CHART_URL.format(symbol="AAPL")]


def test_discovery_excludes_side_effecting_and_arbitrary_tools():
    tools = [
        FunctionTool(lambda: "x", name="search", effect=ToolEffect.READ_ONLY),
        FunctionTool(
            lambda: "x", name="get_stock_quote", effect=ToolEffect.SIDE_EFFECTING
        ),
    ]
    directory = DomainSearch(tools=tools).get_sub_domains(["finance"])
    assert [item["sub_domain"] for item in directory["domains"][0]["sub_domains"]] == [
        "finance.web"
    ]


@pytest.mark.asyncio
async def test_batch_cancellation_cancels_workers(service, monkeypatch):
    client, _ = service
    started, cancelled = [], []
    ready = asyncio.Event()

    async def search(query, **kwargs):
        started.append(query)
        if len(started) == 2:
            ready.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(query)
            raise

    monkeypatch.setattr(client, "search", search)
    task = asyncio.create_task(
        client.batch_search([{"query": "one"}, {"query": "two"}])
    )
    await asyncio.wait_for(ready.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert set(cancelled) == {"one", "two"}
