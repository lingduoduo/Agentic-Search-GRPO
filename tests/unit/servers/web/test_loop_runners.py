"""Unit tests for the shared agent-loop runners in app.py.

Each runner builds + runs one loop and returns the canonical tuple
``(answer, citations, documents, intent, extra)`` consumed by both the
auto-route dispatcher and the explicit-mode chain.
"""

from __future__ import annotations

import types

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.search import AgenticRAGResult
from src.agents.core.base import AgentLoopOutput
from src.context.models import ContextDocument, SearchContextBundle, SearchFilters
from src.context.search import SearchResult
from src.internal.servers.web import app as web_app
from src.internal.tools import routing_tools as web_app_routing_tools


def _search_output(
    *, turns=None, rounds=None, num_turns=1, final_answer="grounded", trace=("e1",)
):
    if rounds is None:
        rounds = [list(turns)] if turns else []
    if turns is None:
        turns = [ctx for round_ctxs in rounds for ctx in round_ctxs]
    context = types.SimpleNamespace(turns=turns, rounds=rounds)
    return AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=num_turns,
        final_answer=final_answer,
        context=context,
        control_flow_trace=list(trace),
    )


@pytest.mark.asyncio
async def test_run_search_agent_returns_canonical_tuple(monkeypatch):
    ctx = types.SimpleNamespace(
        results=[SearchResult(contents="Title\nbody", url=None)]
    )
    monkeypatch.setattr(
        "src.agents.search.SearchAgentLoop.run",
        AsyncMock(return_value=_search_output(rounds=[[ctx]])),
    )
    answer, citations, documents, intent, extra = await web_app._run_search_agent(
        "q",
        manager=MagicMock(),
        tokenizer=MagicMock(),
        search_url="http://x/retrieve",
        top_k=5,
        on_turn=None,
        on_trace=None,
    )
    assert answer == "grounded"
    assert intent == "search"
    assert len(documents) == 1
    assert documents[0].citation == "[R1Q1D1]"
    assert citations == ["[R1Q1D1]"]
    # Source cards must show a provider label, not "Unknown": the search-agent
    # path retrieves from the local retrieval server.
    assert documents[0].metadata["source"] == "Local Retrieval"
    assert extra["num_turns"] == 1
    assert extra["control_flow_trace"] == ["e1"]


@pytest.mark.asyncio
async def test_run_agentic_rag_returns_canonical_tuple(monkeypatch):
    result = AgenticRAGResult(
        answer="synth",
        citations=["[D1]"],
        rounds_used=2,
        context=SearchContextBundle(query="q", documents=[]),
    )
    monkeypatch.setattr(
        "src.agents.search.agentic_rag.AgenticRAGLoop.run",
        AsyncMock(return_value=result),
    )
    answer, citations, documents, intent, extra = await web_app._run_agentic_rag(
        "q",
        llm=MagicMock(),
        search_url="http://x/retrieve",
        top_k=5,
        history=[],
    )
    assert answer == "synth"
    assert intent == "chat"
    assert citations == ["[D1]"]
    # F3: _run_agentic_rag now returns the chat-loop control-flow trace in extra.
    # The monkeypatched run() ignores the recorder, so the trace is empty here.
    assert extra == {"rounds_used": 2, "control_flow_trace": []}


@pytest.mark.asyncio
async def test_run_agentic_rag_labels_document_source(monkeypatch):
    # RAG documents carry raw retrieval metadata (no "source" key); the runner
    # must stamp a provider label so source cards don't render "Unknown".
    doc = ContextDocument(
        id="D1", title="T", content="c", url=None, score=0.5, metadata={}
    )
    result = AgenticRAGResult(
        answer="synth",
        citations=["[D1]"],
        rounds_used=1,
        context=SearchContextBundle(query="q", documents=[doc]),
    )
    monkeypatch.setattr(
        "src.agents.search.agentic_rag.AgenticRAGLoop.run",
        AsyncMock(return_value=result),
    )
    _, _, documents, _, _ = await web_app._run_agentic_rag(
        "q",
        llm=MagicMock(),
        search_url="http://x/retrieve",
        top_k=5,
        history=[],
    )
    assert documents[0].metadata["source"] == "Local Retrieval"
    # Citation id is preserved so answer links still resolve to the card.
    assert documents[0].citation == "[D1]"


def test_document_with_metadata_prefers_real_source_over_provider_label():
    # When the retrieval layer supplies a real per-document source, it wins over
    # the generic provider label; source_provider still records the fetcher.
    doc = ContextDocument(
        id="D1",
        title="T",
        content="c",
        url=None,
        score=0.5,
        metadata={"source": "Team Wiki"},
    )
    labeled = web_app._document_with_metadata(
        doc, source_provider="retrieval", query="q", entry_point="rag"
    )
    assert labeled.metadata["source"] == "Team Wiki"
    assert labeled.metadata["source_provider"] == "retrieval"


def test_document_with_metadata_falls_back_to_provider_label():
    doc = ContextDocument(
        id="D1", title="T", content="c", url=None, score=0.5, metadata={}
    )
    labeled = web_app._document_with_metadata(
        doc, source_provider="retrieval", query="q", entry_point="rag"
    )
    assert labeled.metadata["source"] == "Local Retrieval"


@pytest.mark.asyncio
async def test_run_agentic_rag_threads_access_filters_into_loop(monkeypatch):
    filters = SearchFilters(access_acl=["user:alice"])
    observed = {}

    async def fake_run(self, question, **kwargs):
        observed["filters"] = self.config.filters
        return AgenticRAGResult(
            answer="synth",
            citations=[],
            rounds_used=1,
            context=SearchContextBundle(query=question, documents=[]),
        )

    monkeypatch.setattr("src.agents.search.agentic_rag.AgenticRAGLoop.run", fake_run)
    await web_app._run_agentic_rag(
        "q",
        llm=MagicMock(),
        search_url="http://x/retrieve",
        top_k=5,
        filters=filters,
        history=[],
    )

    assert observed["filters"] is filters


@pytest.mark.asyncio
async def test_run_agentic_rag_forwards_on_claim_to_the_loop(monkeypatch):
    """_run_agentic_rag must forward on_claim into AgenticRAGLoop.run.

    This pins the hop at app.py's _run_agentic_rag -> rag_loop.run(on_claim=...)
    call. The SSE-layer tests (test_sse_streaming.py) stub _run_agentic_rag
    wholesale, so they cannot catch on_claim being silently dropped between
    _run_agentic_rag and the loop; this test is the layer that can.
    """
    observed = {}

    async def fake_run(self, question, **kwargs):
        observed["on_claim"] = kwargs.get("on_claim")
        return AgenticRAGResult(
            answer="synth",
            citations=[],
            rounds_used=1,
            context=SearchContextBundle(query=question, documents=[]),
        )

    monkeypatch.setattr("src.agents.search.agentic_rag.AgenticRAGLoop.run", fake_run)
    callback = lambda text: None  # noqa: E731

    await web_app._run_agentic_rag(
        "q",
        llm=MagicMock(),
        search_url="http://x/retrieve",
        top_k=5,
        history=[],
        on_claim=callback,
    )

    assert observed["on_claim"] is callback


@pytest.mark.asyncio
async def test_run_tool_agent_exposes_assistant_fallback(monkeypatch):
    output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        final_answer="",  # empty → callers decide policy
        action_trace="",
        trajectory_messages=[{"role": "assistant", "content": "fallback text"}],
    )
    monkeypatch.setattr(
        "src.agents.tool.tool_calling.ToolAgentLoop.run",
        AsyncMock(return_value=output),
    )
    answer, citations, documents, intent, extra = await web_app._run_tool_agent(
        "q",
        manager=MagicMock(),
        tokenizer=MagicMock(),
        search_url="http://x/retrieve",
        history=[],
        resolved=types.SimpleNamespace(tool_agent_parser="hermes"),
        on_turn=None,
        with_search_tool=False,
    )
    assert answer == ""  # runner applies no fallback
    assert extra["_assistant_fallback"] == "fallback text"
    assert extra["tool_calls"] == []
    assert extra["num_turns"] == 1


# ---------------------------------------------------------------------------
# Tool selection: the agent must see one unambiguous corpus-search tool and no
# answer-generating tool. Four near-identical retrieval tools made a small local
# model answer from memory (or narrate the RAG tool) instead of calling one.
# ---------------------------------------------------------------------------


def _capture_tool_agent_loop(monkeypatch):
    """Patch ToolAgentLoop so the runner's tools/messages can be inspected."""
    captured: dict = {}
    output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        final_answer="ok",
        action_trace="",
        trajectory_messages=[],
    )

    class _SpyLoop:
        def __init__(self, *, tokenizer, server_manager, tools, config):
            captured["tools"] = list(tools)
            captured["config"] = config

        async def run(
            self, messages, sampling_params, *, on_turn=None, on_approval=None
        ):
            captured["messages"] = list(messages)
            captured["sampling_params"] = dict(sampling_params)
            return output

    monkeypatch.setattr("src.agents.tool.ToolAgentLoop", _SpyLoop)
    return captured


async def _run(monkeypatch, *, history=None, with_search_tool=True):
    captured = _capture_tool_agent_loop(monkeypatch)
    await web_app._run_tool_agent(
        "q",
        manager=MagicMock(),
        tokenizer=MagicMock(),
        search_url="http://x/retrieve",
        history=history or [],
        resolved=types.SimpleNamespace(tool_agent_parser="json"),
        on_turn=None,
        with_search_tool=with_search_tool,
    )
    return captured


@pytest.mark.asyncio
@pytest.mark.parametrize("with_search_tool", [True, False])
async def test_tool_agent_sees_exactly_one_corpus_search_tool(
    monkeypatch, with_search_tool
):
    from src.internal.tools.knowledge_base import seed_tools, tool_knowledge_base
    from src.internal.tools.registry import ToolRegistry

    registry = ToolRegistry()
    seed_tools(registry, tools=tool_knowledge_base(llm=MagicMock()))
    monkeypatch.setattr("src.internal.tools.tool_registry", registry)

    captured = await _run(monkeypatch, with_search_tool=with_search_tool)
    names = [t.name for t in captured["tools"]]

    # The globally-seeded corpus search is unfiltered (built at process start,
    # no request identity), so it never reaches the agent. with_search_tool
    # controls whether a request-bound replacement is added instead.
    assert names.count("search") == (1 if with_search_tool else 0)
    assert "search_routing_tool" not in names  # renamed to plain `search`
    assert "rag_routing_tool" not in names  # answers instead of being a tool
    assert "web_search" in names  # distinct capability, kept


@pytest.mark.asyncio
async def test_tool_agent_keeps_externally_registered_tools(monkeypatch):
    from src.internal.tools.base import FunctionTool
    from src.internal.tools.registry import ToolRegistry

    async def _noop() -> str:
        return ""

    registry = ToolRegistry()
    registry.register(
        FunctionTool(
            fn=_noop,
            name="jira_create_issue",
            description="Create a Jira issue.",
            parameters={"type": "object", "properties": {}},
        )
    )
    monkeypatch.setattr("src.internal.tools.tool_registry", registry)

    captured = await _run(monkeypatch)
    assert "jira_create_issue" in [t.name for t in captured["tools"]]


@pytest.mark.asyncio
async def test_tool_agent_prepends_tool_use_system_prompt(monkeypatch):
    from src.internal.servers.web.tool_agent_runner import TOOL_AGENT_SYSTEM_PROMPT

    captured = await _run(monkeypatch)
    assert captured["messages"][0] == {
        "role": "system",
        "content": TOOL_AGENT_SYSTEM_PROMPT,
    }
    assert captured["messages"][-1] == {"role": "user", "content": "q"}


@pytest.mark.asyncio
async def test_tool_agent_does_not_override_an_existing_system_message(monkeypatch):
    history = [types.SimpleNamespace(role="system", content="caller system prompt")]
    captured = await _run(monkeypatch, history=history)
    roles = [m["role"] for m in captured["messages"]]
    assert roles.count("system") == 1
    assert captured["messages"][0]["content"] == "caller system prompt"


# ---------------------------------------------------------------------------
# Answer budget: a 512-token cap cut answers mid-word. The cap is shared across
# turns, so a tool call spends part of the same budget the answer needs.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_budget_is_configurable_and_applied_to_both_caps(monkeypatch):
    captured = _capture_tool_agent_loop(monkeypatch)
    await web_app._run_tool_agent(
        "q",
        manager=MagicMock(),
        tokenizer=MagicMock(),
        search_url="http://x/retrieve",
        history=[],
        resolved=types.SimpleNamespace(
            tool_agent_parser="json", tool_agent_max_tokens=1536
        ),
        on_turn=None,
        with_search_tool=True,
    )
    # max_tokens caps one generation; response_length caps the whole rollout and
    # truncates each response. Both scale with the configured budget, but they
    # are not equal — see test_rollout_budget_leaves_room_for_tool_responses.
    assert captured["sampling_params"]["max_tokens"] == 1536
    assert captured["config"].response_length >= 1536


@pytest.mark.asyncio
async def test_answer_budget_falls_back_when_config_lacks_the_field(monkeypatch):
    captured = _capture_tool_agent_loop(monkeypatch)
    await web_app._run_tool_agent(
        "q",
        manager=MagicMock(),
        tokenizer=MagicMock(),
        search_url="http://x/retrieve",
        history=[],
        resolved=types.SimpleNamespace(tool_agent_parser="json"),
        on_turn=None,
        with_search_tool=True,
    )
    from src.internal.servers.web.tool_agent_runner import TOOL_AGENT_MAX_TOKENS

    assert captured["sampling_params"]["max_tokens"] == TOOL_AGENT_MAX_TOKENS


@pytest.mark.asyncio
async def test_rollout_budget_leaves_room_for_tool_responses(monkeypatch):
    # response_length budgets model output AND the tool responses fed back in.
    # Sizing it to max_tokens alone let a few large tool results exhaust it
    # before the model wrote its answer, ending the run on <tool_call> markup.
    captured = _capture_tool_agent_loop(monkeypatch)
    await web_app._run_tool_agent(
        "q",
        manager=MagicMock(),
        tokenizer=MagicMock(),
        search_url="http://x/retrieve",
        history=[],
        resolved=types.SimpleNamespace(
            tool_agent_parser="json", tool_agent_max_tokens=1024
        ),
        on_turn=None,
        with_search_tool=True,
    )
    assert captured["sampling_params"]["max_tokens"] == 1024
    assert captured["config"].response_length > 1024


def test_generation_timeout_is_configurable():
    # The wall-clock stop, not the token budget, is what cuts long answers on
    # slow local hardware. It was hardcoded at 120s inside the server manager.
    from src.internal.configs.app_configs import load_app_settings

    assert load_app_settings({}).generation_timeout_seconds == 120.0
    assert (
        load_app_settings(
            {"AGENTIC_SEARCH_GENERATION_TIMEOUT": "600"}
        ).generation_timeout_seconds
        == 600.0
    )


@pytest.mark.asyncio
async def test_tool_agent_withholds_user_scoped_tools_without_a_user(monkeypatch):
    from src.internal.tools.base import FunctionTool
    from src.internal.tools.registry import ToolRegistry

    async def _noop() -> str:
        return ""

    registry = ToolRegistry()
    registry.register(
        FunctionTool(
            _noop,
            name="save_memory",
            description="Save a memory.",
            parameters={"type": "object"},
        ),
        user_scoped=True,
    )
    monkeypatch.setattr("src.internal.tools.tool_registry", registry)

    captured = _capture_tool_agent_loop(monkeypatch)
    await web_app._run_tool_agent(
        "q",
        manager=MagicMock(),
        tokenizer=MagicMock(),
        search_url="http://x/retrieve",
        history=[],
        resolved=types.SimpleNamespace(tool_agent_parser="json"),
        on_turn=None,
        with_search_tool=False,
        user_present=False,
    )
    assert "save_memory" not in [t.name for t in captured["tools"]]


@pytest.mark.asyncio
async def test_auto_routed_tool_strategy_defaults_to_no_user_scoped_tools(
    monkeypatch,
):
    """Closes a hole where a client-supplied ``request.user_id`` (unauthenticated)
    could reach ``_run_tool_agent`` as ``user_present=True``. ``_run_auto_routed``
    now takes ``user_present`` directly (not a user id) and defaults it to
    ``False``, so a caller that omits it — e.g. one that never resolved an
    authenticated user — fails closed instead of leaking user-scoped tools.
    """
    from src.internal.servers.web.intent import RouteDecision, RouteStrategy
    from src.internal.tools.base import FunctionTool
    from src.internal.tools.registry import ToolRegistry

    async def _noop() -> str:
        return ""

    registry = ToolRegistry()
    registry.register(
        FunctionTool(
            _noop,
            name="save_memory",
            description="Save a memory.",
            parameters={"type": "object"},
        ),
        user_scoped=True,
    )
    monkeypatch.setattr("src.internal.tools.tool_registry", registry)
    monkeypatch.setattr(
        "src.internal.servers.web.app.recognize_intent",
        lambda *a, **k: RouteDecision(RouteStrategy.TOOL),
    )

    captured = _capture_tool_agent_loop(monkeypatch)
    await web_app._run_auto_routed(
        "q",
        llm=MagicMock(),
        manager=MagicMock(),
        tokenizer=MagicMock(),
        search_url="http://x/retrieve",
        browser_search_url=None,
        rerank_url=None,
        top_k=5,
        filters=None,
        history=[],
        resolved=types.SimpleNamespace(tool_agent_parser="json"),
        # user_present intentionally omitted: this is the "no authenticated
        # user" path a client-supplied request.user_id must not bypass.
    )
    assert "save_memory" not in [t.name for t in captured["tools"]]


@pytest.mark.asyncio
async def test_the_seeded_corpus_search_never_reaches_the_agent(monkeypatch):
    # The globally-seeded instance is built where no request identity exists,
    # so it is unfiltered. With with_search_tool=False the agent must get no
    # corpus search at all rather than that one.
    from src.internal.tools.registry import ToolRegistry
    from src.internal.tools.routing_tools import build_search_routing_tool

    registry = ToolRegistry()
    registry.register(build_search_routing_tool(search_url="http://seeded/", top_k=5))
    monkeypatch.setattr("src.internal.tools.tool_registry", registry)

    captured = _capture_tool_agent_loop(monkeypatch)
    await web_app._run_tool_agent(
        "q",
        manager=MagicMock(),
        tokenizer=MagicMock(),
        search_url="http://request/retrieve",
        history=[],
        resolved=types.SimpleNamespace(tool_agent_parser="json"),
        on_turn=None,
        with_search_tool=False,
    )
    assert "search" not in [t.name for t in captured["tools"]]


@pytest.mark.asyncio
async def test_the_request_bound_corpus_search_carries_the_filters(monkeypatch):
    from src.context.models import SearchFilters

    built = {}
    real_builder = web_app_routing_tools.build_search_routing_tool

    def _spy(**kwargs):
        built.update(kwargs)
        return real_builder(**kwargs)

    monkeypatch.setattr(web_app_routing_tools, "build_search_routing_tool", _spy)

    filters = SearchFilters(access_acl=["public", "user:userA"])
    captured = _capture_tool_agent_loop(monkeypatch)
    await web_app._run_tool_agent(
        "q",
        manager=MagicMock(),
        tokenizer=MagicMock(),
        search_url="http://request/retrieve",
        history=[],
        resolved=types.SimpleNamespace(tool_agent_parser="json"),
        on_turn=None,
        with_search_tool=True,
        filters=filters,
    )
    assert built["filters"] is filters
    assert built["search_url"] == "http://request/retrieve"
    assert "search" in [t.name for t in captured["tools"]]


@pytest.mark.asyncio
async def test_auto_routed_tool_strategy_gets_a_filtered_corpus_search(monkeypatch):
    """The auto route's TOOL branch gets a corpus search bound to this request.

    Its comment has always said "run ToolAgentLoop with the real registered
    tools", and it passed ``with_search_tool=False`` to mean "use what the
    registry holds rather than synthesising one". That stopped being true when
    the seeded corpus search became non-agent-callable: the branch then got no
    corpus search at all. Passing the flag restores the capability with a
    request-bound, ACL-filtered instance rather than the unfiltered seeded one.
    """
    from src.context.models import SearchFilters
    from src.internal.servers.web.intent import RouteDecision, RouteStrategy

    built = {}
    real_builder = web_app_routing_tools.build_search_routing_tool

    def _spy(**kwargs):
        built.update(kwargs)
        return real_builder(**kwargs)

    monkeypatch.setattr(web_app_routing_tools, "build_search_routing_tool", _spy)
    monkeypatch.setattr(
        "src.internal.servers.web.app.recognize_intent",
        lambda *a, **k: RouteDecision(RouteStrategy.TOOL),
    )

    filters = SearchFilters(access_acl=["public", "user:userA"])
    captured = _capture_tool_agent_loop(monkeypatch)
    await web_app._run_auto_routed(
        "q",
        llm=MagicMock(),
        manager=MagicMock(),
        tokenizer=MagicMock(),
        search_url="http://request/retrieve",
        browser_search_url=None,
        rerank_url=None,
        top_k=5,
        filters=filters,
        history=[],
        resolved=types.SimpleNamespace(tool_agent_parser="json"),
        user_present=True,
    )

    assert "search" in [t.name for t in captured["tools"]]
    # The instance offered is the request-bound one, carrying this caller's ACL
    # — not the globally-seeded instance, which has neither.
    assert built["filters"] is filters
    assert built["search_url"] == "http://request/retrieve"
