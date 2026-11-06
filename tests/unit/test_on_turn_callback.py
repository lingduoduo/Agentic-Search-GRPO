"""Tests that on_turn callback is invoked correctly in agent loops."""

from __future__ import annotations
import pytest
from src.agents.search import SearchAgentLoop, SearchAgentLoopConfig
from src.agents.tool import ToolAgentLoop, ToolAgentLoopConfig
from src.agents.tool.tool_calling import ApprovalDecision
from src.agents.core.state import PerformanceMetrics, TaskStatus, ToolExecutionResult
from src.context.search import SearchResult
from src.internal.tools.parsers import FunctionCall


class _Tok:
    """Minimal tokenizer that encodes text as UTF-8 bytes and decodes back."""

    chat_template = "dummy"

    def apply_chat_template(
        self, messages, add_generation_prompt=True, tokenize=False, tools=None
    ):
        if tokenize:
            return [1]
        return "DUMMY"

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=True):
        return bytes(ids).decode("utf-8", errors="replace")


class _MultiManager:
    """Returns successive byte-encoded responses per generation call."""

    def __init__(self, responses: list[str]):
        self._queue = list(responses)
        self._idx = 0

    async def generate(self, request_id, prompt_ids, sampling_params):
        text = self._queue[self._idx % len(self._queue)]
        self._idx += 1
        return list(text.encode("utf-8"))


# ---------------------------------------------------------------------------
# SearchAgentLoop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_agent_on_turn_called_after_search_round(monkeypatch):
    cfg = SearchAgentLoopConfig(
        max_turns=5,
        require_sufficient_evidence_before_answer=False,
        allow_internal_knowledge_answer=False,
        search_url="http://unused",
    )
    manager = _MultiManager(
        [
            "<search>FAISS indexing</search>",
            "<answer>FAISS is fast.</answer>",
        ]
    )
    loop = SearchAgentLoop(tokenizer=_Tok(), server_manager=manager, search_config=cfg)

    dummy = SearchResult(contents="FAISS doc", score=0.9, title="FAISS", url=None)

    async def fake_retrieve_many(queries, retriever=None):
        return [[dummy] for _ in queries]

    monkeypatch.setattr(loop, "_retrieve_many", fake_retrieve_many)

    called: list[tuple] = []

    async def spy(turn, tool_name, doc_count):
        called.append((turn, tool_name, doc_count))

    await loop.run(
        [{"role": "user", "content": "What is FAISS?"}],
        {},
        on_turn=spy,
    )

    # At least one progress event for the search round
    search_calls = [(t, n, c) for t, n, c in called if n == "search_routing_tool"]
    assert len(search_calls) >= 1, "on_turn should be called after a search round"
    assert search_calls[0][2] >= 1, "doc_count should reflect results found"

    # Final 'writing answer' event
    writing_calls = [(t, n, c) for t, n, c in called if n is None]
    assert len(writing_calls) >= 1, "on_turn(turn, None, 0) should fire before answer"


@pytest.mark.asyncio
async def test_search_agent_on_turn_not_required(monkeypatch):
    """on_turn=None should not raise."""
    cfg = SearchAgentLoopConfig(
        max_turns=2,
        require_sufficient_evidence_before_answer=False,
        allow_internal_knowledge_answer=True,
        search_url="http://unused",
    )
    manager = _MultiManager(
        ["<search_decision>answer</search_decision><answer>Direct.</answer>"]
    )
    loop = SearchAgentLoop(tokenizer=_Tok(), server_manager=manager, search_config=cfg)

    output = await loop.run([{"role": "user", "content": "q"}], {})
    assert output is not None


# ---------------------------------------------------------------------------
# ToolAgentLoop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_agent_on_turn_called_after_tool_execution(monkeypatch):
    final_answer = "The result is 42."
    parse_count = 0

    class _ToolTok:
        chat_template = "dummy"

        def apply_chat_template(
            self, messages, add_generation_prompt=True, tokenize=True, tools=None
        ):
            if tokenize:
                return [1, 2, 3]
            return "DUMMY"

        def encode(self, text):
            return list(text.encode("utf-8"))

        def decode(self, ids, skip_special_tokens=True):
            return final_answer

    gen_count = 0

    class _Mgr:
        async def generate(self, request_id, prompt_ids, sampling_params):
            nonlocal gen_count
            gen_count += 1
            return list(final_answer.encode("utf-8"))

    loop = ToolAgentLoop(
        tokenizer=_ToolTok(),
        server_manager=_Mgr(),
        tools=[],
        config=ToolAgentLoopConfig(max_assistant_turns=3),
    )

    async def fake_extract(response_ids):
        nonlocal parse_count
        parse_count += 1
        if parse_count == 1:
            return ("", [FunctionCall(name="my_tool", arguments="{}")])
        return (final_answer, [])

    loop.tool_parser.extract_tool_calls = fake_extract

    async def fake_call_tool(tool_call):
        return ToolExecutionResult(
            tool_name=tool_call.name,
            status=TaskStatus.COMPLETED,
            result="tool output",
            performance=PerformanceMetrics(execution_time=0.01),
        )

    monkeypatch.setattr(loop, "_call_tool", fake_call_tool)

    called: list[tuple] = []

    async def spy(turn, tool_name, doc_count):
        called.append((turn, tool_name, doc_count))

    await loop.run(
        [{"role": "user", "content": "Calculate 6 × 7."}],
        {},
        on_turn=spy,
    )

    assert len(called) == 1, (
        "on_turn should be called once after the tool execution batch"
    )
    assert called[0][1] == "my_tool"


@pytest.mark.asyncio
async def test_tool_agent_on_tool_call_receives_name_args_and_response(monkeypatch):
    final_answer = "Done."

    class _ToolTok:
        chat_template = "dummy"

        def apply_chat_template(
            self, messages, add_generation_prompt=True, tokenize=True, tools=None
        ):
            return [1, 2, 3] if tokenize else "DUMMY"

        def encode(self, text):
            return list(text.encode("utf-8"))

        def decode(self, ids, skip_special_tokens=True):
            return final_answer

    class _Mgr:
        async def generate(self, request_id, prompt_ids, sampling_params):
            return list(final_answer.encode("utf-8"))

    loop = ToolAgentLoop(
        tokenizer=_ToolTok(),
        server_manager=_Mgr(),
        tools=[],
        config=ToolAgentLoopConfig(max_assistant_turns=3),
    )
    parse_count = 0

    async def fake_extract(response_ids):
        nonlocal parse_count
        parse_count += 1
        if parse_count == 1:
            return ("", [FunctionCall(name="my_tool", arguments='{"q": 1}')])
        return (final_answer, [])

    loop.tool_parser.extract_tool_calls = fake_extract

    async def fake_call_tool(tool_call, *args, **kwargs):
        return ToolExecutionResult(
            tool_name=tool_call.name,
            status=TaskStatus.COMPLETED,
            result="tool output",
            arguments={"q": 1},
            performance=PerformanceMetrics(execution_time=0.01),
        )

    monkeypatch.setattr(loop, "_call_tool", fake_call_tool)

    calls: list[tuple] = []

    async def spy(name, args, text):
        calls.append((name, args, text))

    async def boom(name, args, text):
        raise RuntimeError("callback failure must not fail the turn")

    async def approve(request):
        return ApprovalDecision.APPROVE

    await loop.run(
        [{"role": "user", "content": "q"}],
        {},
        on_tool_call=spy,
        on_approval=approve,
    )
    assert calls == [("my_tool", {"q": 1}, "tool output")]

    parse_count = 0
    out = await loop.run(
        [{"role": "user", "content": "q"}],
        {},
        on_tool_call=boom,
        on_approval=approve,
    )
    assert out is not None
