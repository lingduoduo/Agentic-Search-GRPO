"""AnySearch sample consolidated into the repository's async tool contracts."""

import asyncio
import ast
from pathlib import Path

import pytest

from src.internal.tools.anysearch import (
    AnySearchClient,
    AnySearchError,
    normalize_search_item,
    parse_search_params,
)


def test_taxonomy_has_no_cli_or_import_side_effects():
    source = Path("src/internal/tools/search_domains.py").read_text()
    tree = ast.parse(source)
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert names == [
        "normalize_search_domain",
        "prepare_domain_query",
        "search_domain_parameter",
    ]
    assert "_load_env" not in source
    assert "sys.stdout" not in source
    assert "requests" not in source


@pytest.mark.parametrize(
    "params", ['{"symbol":"AAPL"}', "symbol=AAPL", "{symbol:AAPL}"]
)
def test_native_search_aliases(params):
    assert normalize_search_item(
        {
            "query": "AAPL",
            "domain": "Finance",
            "sub_domain": "finance.quote",
            "sub_domain_params": params,
            "max_results": 99,
        }
    ) == {
        "query": "AAPL",
        "tag": "finance.quote",
        "params": {"symbol": "AAPL"},
        "max_results": 10,
    }


@pytest.mark.parametrize(
    "item",
    [
        {"query": ""},
        {"query": "x", "domain": "finance"},
        {"query": "x", "domain": "code", "tag": "finance.quote"},
        {"query": "x", "tag": "finance.quote", "sub_domain": "finance.news"},
        {"query": "x", "tag": "unknown.quote"},
        {"query": "x", "tag": "finance"},
        {"query": "x", "params": "[1,2]"},
        {"query": "x", "params": "bad"},
        {"query": "x", "zone": "invalid"},
        {"query": "x", "max_results": True},
    ],
)
def test_bad_native_requests_rejected(item):
    with pytest.raises(ValueError):
        normalize_search_item(item)


def test_parameter_values_preserve_nested_json_and_commas():
    assert parse_search_params('{"q":"a,b","options":{"limit":2}}') == {
        "q": "a,b",
        "options": {"limit": 2},
    }
    with pytest.raises(ValueError):
        parse_search_params("q=a,bad")


class Response:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def json(self, **kwargs):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


@pytest.fixture
def http(monkeypatch):
    from src.internal.tools import anysearch

    calls = []
    state = {
        "body": {
            "code": 0,
            "data": {
                "results": [
                    {"title": "T", "url": "https://example.test", "content": "C"}
                ]
            },
        },
        "status": 200,
    }

    class Session:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def request(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            return Response(state["body"], state["status"])

    monkeypatch.setattr(anysearch.aiohttp, "ClientSession", Session)
    return calls, state


@pytest.mark.asyncio
async def test_client_search_uses_native_payload_and_explicit_config(http, monkeypatch):
    calls, _ = http
    monkeypatch.setenv("ANYSEARCH_API_KEY", "environment-key")
    client = AnySearchClient(api_key="explicit-key", base_url="https://service.test/")
    result = await client.search(
        "AAPL", domain="finance", tag="finance.quote", params={"symbol": "AAPL"}
    )
    method, url, kwargs = calls[0]
    assert (method, url) == ("POST", "https://service.test/v1/search")
    assert kwargs["headers"]["Authorization"] == "Bearer explicit-key"
    assert kwargs["json"] == {
        "query": "AAPL",
        "tag": "finance.quote",
        "params": {"symbol": "AAPL"},
    }
    assert result["data"]["results"][0]["title"] == "T"


@pytest.mark.asyncio
async def test_discovery_and_extract(http):
    calls, state = http
    state["body"] = {"data": {"domains": []}}
    client = AnySearchClient(api_key="")
    await client.get_sub_domains(["Finance", "academic"])
    assert calls[0][2]["params"] == [("domain", "finance"), ("domain", "academic")]
    assert "Authorization" not in calls[0][2]["headers"]
    state["body"] = {"data": {"content": "page"}}
    await client.extract("https://example.test/page")
    assert calls[-1][2]["json"] == {"url": "https://example.test/page"}
    for domains in ([], ["general"] * 6, ["unknown"]):
        with pytest.raises(ValueError):
            await client.get_sub_domains(domains)
    with pytest.raises(ValueError):
        await client.extract("file:///etc/passwd")
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,status",
    [
        ([], 200),
        (ValueError("not JSON"), 502),
        ({"code": 7, "message": "bad secret", "request_id": "req1"}, 429),
    ],
)
async def test_structured_error_never_leaks_key(http, body, status):
    _, state = http
    state.update(body=body, status=status)
    with pytest.raises(AnySearchError) as caught:
        await AnySearchClient(api_key="secret").search("x")
    assert "secret" not in str(caught.value)
    assert caught.value.status == status
    if status == 429:
        assert caught.value.request_id == "req1"


@pytest.mark.asyncio
async def test_batch_is_concurrent_ordered_and_failure_isolated(monkeypatch):
    client = AnySearchClient()
    started = []
    all_started = asyncio.Event()

    async def search(query, **kwargs):
        started.append(query)
        if len(started) == 3:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=1)
        if query == "broken":
            raise AnySearchError("failed")
        return {"data": {"query": query}}

    monkeypatch.setattr(client, "search", search)
    results = await client.batch_search(
        [{"query": "first"}, {"query": "broken"}, {"query": "last"}]
    )
    assert results[0]["data"]["query"] == "first"
    assert isinstance(results[1], AnySearchError)
    assert results[2]["data"]["query"] == "last"
    for items in ([], [{"query": "x"}] * 6):
        with pytest.raises(ValueError):
            await client.batch_search(items)


@pytest.mark.asyncio
async def test_invalid_batch_item_isolated_before_network(http):
    calls, _ = http
    results = await AnySearchClient().batch_search(
        [
            {"query": "valid"},
            {"query": "AAPL", "domain": "finance"},
        ]
    )
    assert len(calls) == 1
    assert isinstance(results[1], ValueError)


@pytest.mark.asyncio
async def test_transport_failures_are_redacted_and_sessions_close(monkeypatch):
    from src.internal.tools import anysearch

    closed = []

    class Session:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            closed.append(True)

        def request(self, *args, **kwargs):
            raise TimeoutError("secret")

    monkeypatch.setattr(anysearch.aiohttp, "ClientSession", Session)
    with pytest.raises(AnySearchError, match="timed out") as caught:
        await AnySearchClient(api_key="secret").search("x")
    assert "secret" not in str(caught.value)
    assert closed == [True]


@pytest.mark.asyncio
async def test_batch_shared_options_respect_per_item_routes(http):
    calls, _ = http
    items = [
        {"query": "AAPL"},
        {"query": "repo", "tag": "code.repository", "params": {"language": "python"}},
    ]
    results = await AnySearchClient().batch_search(
        items, domain="finance", sub_domain="finance.quote", params="symbol=AAPL"
    )
    assert all(isinstance(result, dict) for result in results)
    assert calls[0][2]["json"] == {
        "query": "AAPL",
        "tag": "finance.quote",
        "params": {"symbol": "AAPL"},
    }
    assert calls[1][2]["json"] == {
        "query": "repo",
        "tag": "code.repository",
        "params": {"language": "python"},
    }
    assert items[0] == {"query": "AAPL"}


@pytest.mark.asyncio
async def test_batch_cancellation_reaches_all_workers(monkeypatch):
    client = AnySearchClient()
    ready = asyncio.Event()
    started, cancelled = [], []

    async def search(query, **options):
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
