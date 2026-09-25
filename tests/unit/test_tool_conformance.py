import asyncio
import json

import pytest

from src.agents.core.state import TaskStatus
from src.internal.tools import (
    FailureCategory,
    InvalidToolInput,
    ResultKind,
    ToolErrorText,
)
from src.internal.tools.registry import validate_tool_contract
from src.internal.tools.search import (
    DomainSearch,
    MultiQueryWebSearchTool,
    SearchPage,
    build_domain_search_tools,
)


def assert_documents(text: str) -> list[dict]:
    items = json.loads(text)
    assert isinstance(items, list)
    for item in items:
        assert set(item) >= {"title", "content", "url"}
        assert all(isinstance(item[k], str) for k in ("title", "content", "url"))
    return items


def _run(tool, **arguments):
    async def go():
        instance = await tool.create()
        try:
            return await tool.execute(instance, arguments)
        finally:
            await tool.release(instance)

    return asyncio.run(go())


def test_public_data_tools_conform():
    from src.internal.tools.public_data import public_data_tools

    tools = public_data_tools()
    assert len(tools) == 9
    for tool in tools:
        assert validate_tool_contract(tool, source="function") == [], tool.name
    by_name = {t.name: t for t in tools}
    assert {n for n, t in by_name.items() if t.result_kind is ResultKind.DOCUMENTS} == {
        "search_wikipedia",
        "search_arxiv",
        "search_wayback",
    }
    assert {n for n, t in by_name.items() if not t.retries_internally} == {
        "search_nearby_places"
    }


def _corpus_tool(monkeypatch, pages):
    from src.internal.tools import routing_tools

    async def fake_search_tool(query, **kwargs):
        return pages

    monkeypatch.setattr(routing_tools, "search_tool", fake_search_tool)
    return routing_tools.build_search_routing_tool(
        search_url="http://x/retrieve", top_k=5
    )


def test_corpus_search_conforms_and_returns_documents(monkeypatch):
    tool = _corpus_tool(
        monkeypatch, [SearchPage(title="A", summary="a", url="http://a")]
    )
    assert validate_tool_contract(tool, source="function") == []
    assert tool.retries_internally is True
    response, _raw, meta = _run(tool, query="q")
    assert assert_documents(response)[0]["url"] == "http://a"
    assert meta == {}


def test_corpus_search_outage_is_a_typed_transient_failure(monkeypatch):
    tool = _corpus_tool(monkeypatch, [SearchPage(error="connection refused")])
    response, _raw, meta = _run(tool, query="q")
    assert isinstance(response, ToolErrorText)
    assert json.loads(response) == {"error": "connection refused"}
    assert meta["failure"].category is FailureCategory.TRANSIENT


def test_corpus_search_with_no_hits_is_an_empty_success(monkeypatch):
    tool = _corpus_tool(monkeypatch, [])
    response, _raw, meta = _run(tool, query="q")
    assert json.loads(response) == [] and meta == {}


def test_rag_routing_tool_conforms_and_lets_errors_raise(monkeypatch):
    from src.internal.tools import routing_tools

    tool = routing_tools.build_rag_routing_tool(
        llm=object(), search_url="http://x", top_k=3
    )
    assert validate_tool_contract(tool, source="function") == []
    assert tool.result_kind is ResultKind.JSON

    async def boom(*args, **kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr("src.context.answer_with_retrieval", boom)
    with pytest.raises(RuntimeError):
        _run(tool, query="q")


def _web_tool(pages_by_query):
    async def fake_search(query, **kwargs):
        return pages_by_query[query]

    return MultiQueryWebSearchTool(search_fn=fake_search)


def test_web_search_conforms():
    tool = _web_tool({})
    assert validate_tool_contract(tool, source="function") == []
    assert (tool.effect.value, tool.result_kind, tool.citeable) == (
        "read_only",
        ResultKind.DOCUMENTS,
        True,
    )


def test_web_search_returns_deduplicated_documents():
    tool = _web_tool(
        {
            "a": [
                SearchPage(title="A", summary="sa", url="http://1"),
                SearchPage(title="B", summary="sb", url="http://2"),
            ],
            "b": [SearchPage(title="A again", summary="x", url="http://1")],
        }
    )
    response, raw, meta = _run(tool, queries=["a", "b"])
    items = assert_documents(response)
    assert [i["url"] for i in items] == ["http://1", "http://2"]
    assert items[0] == {"title": "A", "content": "sa", "url": "http://1"}
    assert meta["queries"] == ["a", "b"]


def test_web_search_partial_failure_keeps_the_successes():
    tool = _web_tool(
        {
            "a": [SearchPage(title="A", summary="sa", url="http://1")],
            "b": [SearchPage(error="rate limited")],
        }
    )
    response, _raw, meta = _run(tool, queries=["a", "b"])
    assert "failure" not in meta
    assert [i["url"] for i in assert_documents(response)] == ["http://1"]


def test_web_search_total_failure_is_typed_unknown():
    tool = _web_tool(
        {
            "a": [SearchPage(error="no provider")],
            "b": [SearchPage(error="rate limited")],
        }
    )
    response, _raw, meta = _run(tool, queries=["a", "b"])
    assert isinstance(response, ToolErrorText)
    assert meta["failure"].category is FailureCategory.UNKNOWN
    assert json.loads(response) == {"error": "no provider"}


def test_web_search_missing_fields_become_empty_strings():
    tool = _web_tool({"a": [SearchPage(title="", summary="", url="")]})
    response, _raw, _meta = _run(tool, queries=["a"])
    assert assert_documents(response) == [{"title": "", "content": "", "url": ""}]


def test_web_search_runs_without_an_approval_callback():
    from src.agents import ToolAgentLoop, ToolAgentLoopConfig
    from tests.unit.test_tool_recovery_loop import _Manager, _Tokenizer

    tool = _web_tool({"q": [SearchPage(title="A", summary="s", url="http://1")]})
    tokenizer = _Tokenizer()
    manager = _Manager(
        tokenizer, ['{"name":"web_search","arguments":{"queries":["q"]}}', "done"]
    )
    loop = ToolAgentLoop(
        tokenizer, manager, [tool], ToolAgentLoopConfig(response_length=8192)
    )
    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {}))
    trace = [json.loads(line) for line in output.action_trace.splitlines()]
    assert trace[0]["status"] == str(TaskStatus.COMPLETED)


def test_domain_tools_conform():
    tools = {t.name: t for t in build_domain_search_tools(service=DomainSearch())}
    for tool in tools.values():
        assert validate_tool_contract(tool, source="function") == [], tool.name
    assert tools["extract_page"].result_kind is ResultKind.DOCUMENTS
    assert tools["search_domain"].citeable is False
    assert all(not t.retries_internally for t in tools.values())


def test_search_domain_bad_argument_is_invalid_input_with_the_same_text():
    tool = {t.name: t for t in build_domain_search_tools(service=DomainSearch())}[
        "search_domain"
    ]
    response, _raw, meta = _run(tool, query="q", max_results="many")
    assert meta["failure"].category is FailureCategory.INVALID_INPUT
    assert json.loads(response) == {
        "error": "InvalidToolInput: max_results must be an integer"
    }


def test_extract_page_dead_link_is_invalid_input():
    async def fetch(url, max_length):
        return "[fetch error] 404"

    tool = {
        t.name: t
        for t in build_domain_search_tools(service=DomainSearch(fetch_fn=fetch))
    }["extract_page"]
    _response, _raw, meta = _run(tool, url="http://dead.example")
    assert meta["failure"].category is FailureCategory.INVALID_INPUT


def _failing_capability(category):
    """A stand-in for the stock-quote capability whose call fails with *category*."""
    from src.internal.tools import FunctionTool, ToolEffect, ToolFailure
    from src.internal.tools.search import iter_capabilities

    tag, capability = next(
        (t, c) for t, c in iter_capabilities() if c.tool_name == "get_stock_quote"
    )

    async def fn(**kwargs):
        return ToolErrorText(
            json.dumps({"error": "invalid ticker symbol 'APPL'"}),
            ToolFailure(category, "m"),
        )

    tool = FunctionTool(
        fn,
        name="get_stock_quote",
        effect=ToolEffect.READ_ONLY,
        result_kind=ResultKind.JSON,
        parameters={
            "type": "object",
            "properties": {capability.query_parameter: {"type": "string"}},
        },
    )
    return DomainSearch(tools=[tool], web_search_fn=lambda *a, **k: None), tag


def test_capability_input_failure_surfaces_as_invalid_input():
    service, tag = _failing_capability(FailureCategory.INVALID_INPUT)
    with pytest.raises(InvalidToolInput, match="invalid ticker"):
        asyncio.run(service.search("APPL", tag=tag))


def test_capability_upstream_failure_is_not_invalid_input():
    service, tag = _failing_capability(FailureCategory.PERMANENT)
    with pytest.raises(ValueError) as caught:
        asyncio.run(service.search("APPL", tag=tag))
    assert not isinstance(caught.value, InvalidToolInput)


def test_mcp_client_tool_conforms_as_unspecified_text():
    from types import SimpleNamespace

    from src.internal.tools.mcp_client import McpServerSpec, _build_tool

    remote = SimpleNamespace(
        name="remote_op", description="d", inputSchema={"type": "object"}
    )
    tool = _build_tool(McpServerSpec(name="srv", url="http://mcp"), remote)
    assert validate_tool_contract(tool, source="mcp") == []
    assert (tool.effect.value, tool.result_kind) == ("unspecified", ResultKind.TEXT)


def test_openapi_tool_declares_json():
    from src.internal.tools.api import ApiRequestTool

    assert (
        ApiRequestTool.result_kind.fget(object.__new__(ApiRequestTool))
        is ResultKind.JSON
    )


class _Store:
    def add_user_memory(self, user_id, content):
        return None

    def update_user_memory(self, user_id, memory_id, content):
        return None

    def delete_user_memory(self, user_id, memory_id):
        return False


def test_memory_tools_conform_and_type_their_failures():
    from src.internal.memory.tools import build_memory_registry

    registry, _counts, _schemas = build_memory_registry(_Store(), "u1")
    for tool in registry.list_tools():
        assert validate_tool_contract(tool, source="function") == [], tool.name
        assert tool.result_kind is ResultKind.TEXT
    for name, args in (
        ("add_memory", {"content": ""}),
        ("update_memory", {"memory_id": "m", "content": "c"}),
        ("delete_memory", {"memory_id": "m"}),
    ):
        outcome = asyncio.run(registry.invoke_detailed(name, args))
        assert outcome.failure.category is FailureCategory.INVALID_INPUT, name


def test_memory_invoke_text_is_unchanged_for_not_found():
    from src.internal.memory.tools import build_memory_registry

    registry, _counts, _schemas = build_memory_registry(_Store(), "u1")
    response, _raw, errors = asyncio.run(
        registry.invoke("delete_memory", {"memory_id": "m"})
    )
    assert (response, errors) == ("memory not found", [])


def test_the_global_registry_is_strict():
    from src.internal.tools import FunctionTool, ToolEffect
    from src.internal.tools.registry import tool_registry

    with pytest.raises(ValueError, match="result_kind"):
        tool_registry.register(
            FunctionTool(lambda: "x", name="_undeclared", effect=ToolEffect.READ_ONLY)
        )
    assert tool_registry.get("_undeclared") is None


def test_every_production_tool_registers_strictly():
    from unittest.mock import MagicMock

    from src.internal.memory.tools import build_memory_registry
    from src.internal.tools import ToolRegistry
    from src.internal.tools.knowledge_base import seed_tools, tool_knowledge_base
    from src.internal.tools.routing_tools import build_search_routing_tool

    registry = ToolRegistry(strict=True)
    seed_tools(registry, tools=tool_knowledge_base(llm=MagicMock()))
    registry.register(
        build_search_routing_tool(search_url="http://x", top_k=5, name="search_bound")
    )
    build_memory_registry(
        _Store(), "u1"
    )  # strict internally; raises if a memory tool breaks the contract


def test_openapi_admin_endpoint_maps_contract_violation_to_422(tmp_path):
    from fastapi.testclient import TestClient

    from src.internal.auth import generate_user_jwt_token
    from src.internal.configs import AppSettings, AuthSettings
    from src.internal.db import UserRecord
    from src.internal.servers.web.app import SearchExperienceSettings, create_web_app
    from src.internal.tools.registry import tool_registry

    admin = "admin"
    settings = AppSettings(auth=AuthSettings(super_users=(admin,)))
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"),
        app_settings=settings,
    )
    app.state.auth_store.upsert_user(UserRecord(id=admin))
    token = generate_user_jwt_token(user_id=admin)

    def boom(self, openapi_json, *, name, headers=None, icon=None):
        raise ValueError("tool x: result_kind must be declared")

    original = tool_registry.__class__.register_from_openapi
    tool_registry.__class__.register_from_openapi = boom
    try:
        client = TestClient(app)
        resp = client.post(
            "/admin/tools/openapi",
            json={"name": "Bad", "openapi_json": "{}"},
            headers={"Authorization": f"Bearer {token}"},
        )
    finally:
        tool_registry.__class__.register_from_openapi = original

    assert resp.status_code == 422
    assert "result_kind" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_mcp_discovery_skips_a_non_conforming_remote_tool(monkeypatch, caplog):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from src.internal.tools import mcp_client
    from src.internal.tools.mcp_client import McpServerSpec, register_mcp_tools
    from src.internal.tools.registry import ToolRegistry

    good = SimpleNamespace(name="good_remote", description="d", inputSchema={})
    bad = SimpleNamespace(name="bad_remote", description="d", inputSchema={})

    class _FakeSession:
        async def initialize(self):
            return None

        async def list_tools(self):
            return SimpleNamespace(tools=[good, bad])

    @asynccontextmanager
    async def _connect(spec):
        yield _FakeSession()

    monkeypatch.setattr(mcp_client, "_connect", _connect)

    original_build_tool = mcp_client._build_tool

    def fake_build_tool(spec, remote):
        if remote.name == "bad_remote":
            from src.internal.tools import FunctionTool

            return FunctionTool(lambda: "x", name="bad_remote", result_kind=None)
        return original_build_tool(spec, remote)

    monkeypatch.setattr(mcp_client, "_build_tool", fake_build_tool)

    registry = ToolRegistry(strict=True)
    with caplog.at_level("WARNING"):
        count = await register_mcp_tools(
            registry, [McpServerSpec(name="local", url="http://x/")]
        )

    assert count == 1
    assert registry.get("good_remote") is not None
    assert registry.get("bad_remote") is None
    assert any("bad_remote" in record.message for record in caplog.records)
