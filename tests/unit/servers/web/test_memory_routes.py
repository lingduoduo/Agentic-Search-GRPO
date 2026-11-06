"""Memory reaches the SEARCH and TOOL routes; the search_memory tool is bound
per request to a signed-in user only."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.agents.core.base import AgentLoopOutput
from src.internal.auth import generate_user_jwt_token
from src.internal.db.models import UserRecord
from src.internal.db.store import AgenticSearchStore
from src.internal.hooks.lifecycle import ToolCallContext
from src.internal.servers.web import app as web_app
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app
from src.internal.tools.base import Tool, ToolSchema

MEMORY = "User is allergic to peanuts"


class _StubSearchMemories(Tool):
    @property
    def name(self):
        return "search_memories"

    @property
    def schema(self):
        return ToolSchema(name="search_memories", description="mcp", parameters={})

    async def execute(self, instance_id, arguments):
        return "", None, {}


def _output():
    return AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        final_answer="done",
        action_trace="",
        trajectory_messages=[{"role": "assistant", "content": "done"}],
    )


def _patch_tool_loop(monkeypatch, captured):
    class FakeLoop:
        def __init__(self, *, tools, **kwargs):
            captured["tools"] = [t.name for t in tools]

        async def run(self, messages, sampling_params, **kwargs):
            captured["messages"] = messages
            on_tool_call = kwargs.get("on_tool_call")
            if on_tool_call is not None:
                await on_tool_call("lookup", {"q": "x"}, "result text")
            return _output()

    monkeypatch.setattr("src.agents.tool.ToolAgentLoop", FakeLoop)
    from src.internal.tools import tool_registry

    monkeypatch.setattr(
        tool_registry,
        "agent_tools",
        lambda user_present=True: [_StubSearchMemories()],
    )


def _client(store, signed_in):
    app = create_web_app(SearchExperienceSettings(), store=store)
    client = TestClient(app)
    if signed_in:
        client.cookies.set("fastapiusersauth", generate_user_jwt_token(user_id="u1"))
    return app, client


def _post_tool(app, client):
    app.state.search_agent_manager = object()
    app.state.search_agent_tokenizer = object()
    return client.post("/api/agent", json={"query": "weather?", "mode": "tool_agent"})


def _store():
    store = AgenticSearchStore(":memory:")
    store.upsert_user(UserRecord(id="u1"))
    store.add_user_memory("u1", MEMORY)
    return store


def test_tool_route_gets_memory_and_search_memory_tool(monkeypatch):
    captured: dict = {}
    _patch_tool_loop(monkeypatch, captured)
    store = _store()
    app, client = _client(store, signed_in=True)
    with client:
        resp = _post_tool(app, client)
    assert resp.status_code == 200
    system = captured["messages"][0]
    assert system["role"] == "system" and MEMORY in system["content"]
    assert "search_memory" in captured["tools"]
    assert "search_memories" not in captured["tools"]
    store.close()


def test_anonymous_tool_route_gets_no_memory_or_tool(monkeypatch):
    captured: dict = {}
    _patch_tool_loop(monkeypatch, captured)
    store = _store()
    app, client = _client(store, signed_in=False)
    with client:
        resp = _post_tool(app, client)
    assert resp.status_code == 200
    assert "search_memory" not in captured["tools"]
    assert MEMORY not in captured["messages"][0]["content"]
    store.close()


def test_tool_route_fires_after_tool_call_hook(monkeypatch):
    captured: dict = {}
    _patch_tool_loop(monkeypatch, captured)
    store = _store()
    app, client = _client(store, signed_in=True)
    seen: list[ToolCallContext] = []

    async def record(ctx):
        seen.append(ctx)

    app.state.lifecycle_hooks.on_after_tool_call(record)
    with client:
        resp = _post_tool(app, client)
    assert resp.status_code == 200
    assert [c.tool_name for c in seen] == ["lookup"]
    assert seen[0].user_id == "u1"
    store.close()


async def test_search_escalation_system_prompt_contains_memory(monkeypatch):
    captured: dict = {}

    class FakeSearchLoop:
        def __init__(self, *, search_config, **kwargs):
            captured["system_prompt"] = search_config.system_prompt

        async def run(self, messages, sampling_params, **kwargs):
            return SimpleNamespace(
                final_answer="a",
                control_flow_trace=[],
                num_turns=1,
                trajectory_messages=[],
            )

    monkeypatch.setattr("src.get_registered_agent_loop", lambda name: FakeSearchLoop)
    monkeypatch.setattr(web_app, "_search_agent_documents", lambda output: [])
    await web_app._run_search_direct_or_escalate(
        "q",
        manager=object(),
        tokenizer=object(),
        llm=None,
        search_url="http://x/retrieve",
        browser_search_url=None,
        rerank_url=None,
        top_k=3,
        filters=None,
        history=[],
        source_provider="web",
        user_memory=MEMORY,
    )
    assert MEMORY in captured["system_prompt"]
