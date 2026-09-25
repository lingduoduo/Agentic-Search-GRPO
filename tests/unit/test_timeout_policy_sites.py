"""Each migrated site reads its value from the timeout policies."""

import asyncio
from contextlib import contextmanager

import pytest

from src.internal.configs.timeouts import load_timeout_policies, use_timeout_policies


@contextmanager
def overridden(overrides: dict):
    with use_timeout_policies(load_timeout_policies({}, overrides=overrides)) as p:
        yield p


def test_public_data_attempts_follow_the_policy(monkeypatch):
    from src.internal.tools.public_data import _http
    from tests.unit.test_public_data_http import _SequencedSession

    real_sleep = asyncio.sleep
    shared = _SequencedSession([503, 503, 503, 503, 503])
    _SequencedSession.calls = []

    class _Aiohttp:
        @staticmethod
        def ClientTimeout(total=None):
            return total

        @staticmethod
        def ClientSession(timeout=None):
            return shared

    monkeypatch.setattr(_http, "aiohttp", _Aiohttp)
    monkeypatch.setattr(_http.asyncio, "sleep", lambda s: real_sleep(0))
    with overridden({"tools": {"public_data": {"max_attempts": 5}}}):
        with pytest.raises(_http.PublicDataError):
            asyncio.run(_http.get_json("https://x.test"))
    assert (
        len(_SequencedSession.calls) == 5
    )  # 5 attempts, 2 backoff steps: last step reused


def test_public_data_timeout_follows_the_policy_and_kwarg_wins(monkeypatch):
    from src.internal.tools.public_data import _http

    seen = []

    async def fake_fetch(method, url, *, timeout_seconds, **kw):
        seen.append(timeout_seconds)
        return {}

    monkeypatch.setattr(_http, "_fetch", fake_fetch)
    with overridden({"tools": {"public_data": {"timeout_seconds": 3}}}):
        asyncio.run(_http.get_json("https://x.test"))
        asyncio.run(_http.get_json("https://x.test", timeout_seconds=9))
    assert seen == [3.0, 9]


def test_search_tool_uses_router_policy_and_kwarg_wins(monkeypatch):
    from src.internal.tools import search

    seen = []

    async def fake_retrieval(query, **kw):
        seen.append((kw["timeout_seconds"], kw["max_retries"]))
        return []

    monkeypatch.setattr(search, "retrieval_search", fake_retrieval)
    with overridden(
        {"tools": {"search_router": {"timeout_seconds": 4, "max_retries": 2}}}
    ):
        asyncio.run(search.search_tool("q"))
        asyncio.run(search.search_tool("q", timeout_seconds=6, max_retries=1))
    assert seen == [(4.0, 2), (6, 1)]


def test_retrieval_search_uses_client_policy(monkeypatch):
    from src.internal.tools import search

    configs = []

    class FakeClient:
        def __init__(self, config):
            configs.append(config)

        async def retrieve_one(self, *a, **k):
            return []

        async def aclose(self):
            pass

    monkeypatch.setattr(search, "SearchClient", FakeClient)
    with overridden(
        {"retrieval": {"client": {"timeout_seconds": 7, "max_retries": 2}}}
    ):
        asyncio.run(search.retrieval_search("q", search_url="http://x"))
    assert (configs[0].timeout_seconds, configs[0].max_retries) == (7.0, 2)


def test_google_search_uses_web_search_policy(monkeypatch):
    from src.internal.tools import search

    seen = []

    async def fake_get_json(url, *, params, timeout_seconds, **kw):
        seen.append(timeout_seconds)
        return {"items": []}

    monkeypatch.setattr(search, "_get_json", fake_get_json)
    with overridden({"tools": {"web_search": {"google_timeout_seconds": 3}}}):
        asyncio.run(search.google_custom_search("q", api_key="k", cse_id="c"))
    assert seen == [3.0]


def test_corpus_search_reports_router_attempts(monkeypatch):
    from src.internal.tools import routing_tools
    from src.internal.tools.search import SearchPage

    async def failing(*a, **k):
        return [SearchPage(error="down")]

    monkeypatch.setattr(routing_tools, "search_tool", failing)
    tool = routing_tools.build_search_routing_tool(search_url="http://x", top_k=3)
    with overridden({"tools": {"search_router": {"max_retries": 5}}}):
        text, _raw, meta = asyncio.run(tool.execute("i", {"query": "q"}))
    assert meta["failure"].provider_attempts == 5


def _openapi_schema() -> str:
    import json

    return json.dumps(
        {
            "openapi": "3.0.0",
            "info": {"title": "Demo", "description": "Demo API"},
            "servers": [{"url": "https://api.example.test"}],
            "paths": {
                "/ping": {
                    "get": {
                        "operationId": "ping",
                        "description": "Ping.",
                        "parameters": [],
                    }
                }
            },
        }
    )


class _FakeOpenApiResponse:
    headers = {"content-type": "application/json"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return {"ok": True}

    async def text(self):
        return "ok"


class _FakeOpenApiSession:
    def __init__(self, *, timeout):
        del timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def request(self, **kwargs):
        del kwargs
        return _FakeOpenApiResponse()


def test_openapi_tool_timeout_follows_policy(monkeypatch):
    from src.internal.tools import api as api_tools

    seen = []
    real = api_tools.aiohttp.ClientTimeout
    monkeypatch.setattr(
        api_tools.aiohttp,
        "ClientTimeout",
        lambda total: seen.append(total) or real(total=total),
    )
    monkeypatch.setattr(
        api_tools.aiohttp,
        "ClientSession",
        lambda *, timeout: _FakeOpenApiSession(timeout=timeout),
    )

    registry = api_tools.ApiToolRegistry()
    provider = registry.create_provider(name="demo", openapi_schema=_openapi_schema())
    tool = next(t for t in registry.build_tools(provider.id) if t.name == "ping")

    with overridden({"tools": {"openapi": {"timeout_seconds": 2}}}):
        asyncio.run(tool.execute("default", {}))

    assert seen == [2.0]


def test_mcp_client_passes_policy_timeouts(monkeypatch):
    pytest.importorskip("mcp")
    from contextlib import asynccontextmanager

    from src.internal.tools import mcp_client
    import mcp.client.streamable_http as streamable_http

    seen = []

    @asynccontextmanager
    async def fake_streamablehttp_client(url, **kwargs):
        seen.append(kwargs)
        raise RuntimeError("stop after recording")
        yield  # pragma: no cover

    monkeypatch.setattr(
        streamable_http, "streamablehttp_client", fake_streamablehttp_client
    )

    spec = mcp_client.McpServerSpec(name="local", url="http://x/")
    with overridden(
        {"tools": {"mcp": {"timeout_seconds": 3, "sse_read_timeout_seconds": 9}}}
    ):
        with pytest.raises(RuntimeError, match="stop after recording"):
            asyncio.run(_enter_and_exit(mcp_client._connect(spec)))

    assert seen[0]["timeout"] == 3.0
    assert seen[0]["sse_read_timeout"] == 9.0


async def _enter_and_exit(cm):
    async with cm as value:
        return value
