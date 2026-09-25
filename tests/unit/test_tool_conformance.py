import asyncio
import json

import pytest

from src.agents.core.state import TaskStatus
from src.internal.tools import FailureCategory, ResultKind, ToolErrorText
from src.internal.tools.registry import validate_tool_contract
from src.internal.tools.search import MultiQueryWebSearchTool, SearchPage


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
