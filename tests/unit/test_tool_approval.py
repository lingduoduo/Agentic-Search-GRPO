from __future__ import annotations

import asyncio
import json
from datetime import timezone

import pytest

from src.agents import ApprovalDecision, ToolAgentLoop, ToolAgentLoopConfig
from src.agents.core.state import TaskStatus
from src.internal.tools import FunctionTool, ToolEffect


class _Tokenizer:
    chat_template = ""

    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(item) for item in ids)

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True, tokenize=True
    ):
        text = "\n".join(message.get("content", "") for message in messages)
        return self.encode(text) if tokenize else text


class _Manager:
    def __init__(self, tokenizer, responses):
        self.tokenizer = tokenizer
        self.responses = iter(responses)
        self.prompts = []

    async def generate(self, request_id, prompt_ids, sampling_params):
        self.prompts.append(self.tokenizer.decode(prompt_ids))
        return self.tokenizer.encode(next(self.responses))


def _trace(output):
    return [json.loads(line) for line in (output.action_trace or "").splitlines()]


def _loop(tools, responses, **config):
    tokenizer = _Tokenizer()
    manager = _Manager(tokenizer, responses)
    return ToolAgentLoop(
        tokenizer, manager, tools, ToolAgentLoopConfig(response_length=4096, **config)
    ), manager


def test_approval_contract_and_missing_callback_fail_closed():
    assert [item.value for item in ApprovalDecision] == ["approve", "deny", "expired"]
    executions = []

    @FunctionTool.from_fn(effect=ToolEffect.SIDE_EFFECTING)
    def mutate(value: int):
        executions.append(value)
        return value

    loop, _ = _loop([mutate], ['{"name":"mutate","arguments":{"value":3}}', "done"])
    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {}))

    result = _trace(output)[0]
    assert executions == []
    assert result["status"] == str(TaskStatus.SKIPPED)
    assert result["error_code"] == "approval_denied"
    assert output.final_answer == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "expected_executions", "expected_error"),
    [
        (None, [], "approval_denied"),
        (ApprovalDecision.APPROVE, [5], None),
        (ApprovalDecision.DENY, [], "approval_denied"),
    ],
)
async def test_unspecified_tools_require_approval(
    decision, expected_executions, expected_error
):
    executions = []

    @FunctionTool.from_fn()
    def legacy_tool(value: int):
        executions.append(value)
        return value

    requests = []

    async def decide(request):
        requests.append(request)
        return decision

    loop, _ = _loop(
        [legacy_tool],
        ['{"name":"legacy_tool","arguments":{"value":5}}', "done"],
    )
    output = await loop.run(
        [{"role": "user", "content": "go"}],
        {},
        on_approval=decide if decision is not None else None,
    )

    assert executions == expected_executions
    assert _trace(output)[0]["error_code"] == expected_error
    assert len(requests) == (decision is not None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "code", "executions"),
    [
        (ApprovalDecision.APPROVE, None, [7]),
        (ApprovalDecision.DENY, "approval_denied", []),
        (ApprovalDecision.EXPIRED, "approval_expired", []),
    ],
)
async def test_approval_decisions(decision, code, executions):
    actual = []

    @FunctionTool.from_fn(effect=ToolEffect.SIDE_EFFECTING)
    def mutate(value: int):
        actual.append(value)
        return value

    requests = []

    async def approve(request):
        requests.append(request)
        return decision

    loop, _ = _loop([mutate], ['{"name":"mutate","arguments":{"value":7}}', "done"])
    output = await loop.run(
        [{"role": "user", "content": "go"}], {}, on_approval=approve
    )

    assert actual == executions
    assert requests[0].tool_name == "mutate"
    assert requests[0].arguments == {"value": 7}
    assert requests[0].approval_id
    assert requests[0].created_at.tzinfo is timezone.utc
    assert requests[0].expires_at > requests[0].created_at
    result = _trace(output)[0]
    assert result["error_code"] == code
    counter = {
        ApprovalDecision.APPROVE: "tool_approvals_approved",
        ApprovalDecision.DENY: "tool_approvals_denied",
        ApprovalDecision.EXPIRED: "tool_approvals_expired",
    }[decision]
    assert output.metrics[counter] == 1


@pytest.mark.asyncio
async def test_callback_error_and_timeout_skip_execution():
    executions = []

    @FunctionTool.from_fn(effect=ToolEffect.SIDE_EFFECTING)
    def mutate():
        executions.append(1)

    async def raises(request):
        raise RuntimeError("broker unavailable")

    loop, _ = _loop([mutate], ['{"name":"mutate","arguments":{}}', "done"])
    errored = await loop.run(
        [{"role": "user", "content": "go"}], {}, on_approval=raises
    )
    assert _trace(errored)[0]["error_code"] == "approval_denied"
    assert errored.metrics["tool_approval_errors"] == 1

    async def waits(request):
        await asyncio.Event().wait()

    loop, _ = _loop(
        [mutate],
        ['{"name":"mutate","arguments":{}}', "done"],
        approval_timeout_seconds=0.001,
    )
    expired = await loop.run([{"role": "user", "content": "go"}], {}, on_approval=waits)
    assert executions == []
    assert _trace(expired)[0]["error_code"] == "approval_expired"
    assert expired.metrics["tool_approvals_expired"] == 1


@pytest.mark.asyncio
async def test_batch_preflight_resolves_before_any_tool_executes_and_skips_continue():
    executions = []
    both_requested = asyncio.Event()
    gates = {"first": asyncio.Event(), "second": asyncio.Event()}
    requests = []

    @FunctionTool.from_fn(effect=ToolEffect.READ_ONLY)
    def read():
        executions.append("read")
        return "read"

    @FunctionTool.from_fn(effect=ToolEffect.SIDE_EFFECTING)
    def first():
        executions.append("first")
        return "first"

    @FunctionTool.from_fn(effect=ToolEffect.SIDE_EFFECTING)
    def second():
        executions.append("second")
        return "second"

    async def decide(request):
        requests.append(request)
        if len(requests) == 2:
            both_requested.set()
        await gates[request.tool_name].wait()
        return (
            ApprovalDecision.APPROVE
            if request.tool_name == "first"
            else ApprovalDecision.DENY
        )

    calls = json.dumps(
        [
            {"name": "read", "arguments": {}},
            {"name": "first", "arguments": {}},
            {"name": "second", "arguments": {}},
        ]
    )
    loop, manager = _loop([read, first, second], [calls, "done"])
    task = asyncio.create_task(
        loop.run([{"role": "user", "content": "go"}], {}, on_approval=decide)
    )
    await asyncio.wait_for(both_requested.wait(), 1)
    assert executions == []
    gates["first"].set()
    await asyncio.sleep(0)
    assert executions == []
    gates["second"].set()
    output = await task

    assert sorted(executions) == ["first", "read"]
    assert len({request.approval_id for request in requests}) == 2
    assert "approval_denied" in manager.prompts[-1]
    assert output.final_answer == "done"


@pytest.mark.asyncio
async def test_final_answer_recorded_on_assistant_turn_cap():
    """A cap firing on the first generated turn must still surface final_answer."""
    loop, _ = _loop([], ["the answer"], max_assistant_turns=1)
    output = await loop.run([{"role": "user", "content": "go"}], {})

    assert output.final_answer == "the answer"
    assert output.trajectory_messages[-1] == {
        "role": "assistant",
        "content": "the answer",
    }


@pytest.mark.asyncio
async def test_final_answer_recorded_on_response_length_cap():
    """A response-length cap on the first turn must still surface final_answer."""
    tokenizer = _Tokenizer()
    manager = _Manager(tokenizer, ["hello"])
    # "hello" encodes to 5 tokens; response_length=5 lets the full answer through
    # and then trips the length cap on the same turn.
    loop = ToolAgentLoop(tokenizer, manager, [], ToolAgentLoopConfig(response_length=5))
    output = await loop.run([{"role": "user", "content": "go"}], {})

    assert output.final_answer == "hello"
    assert output.trajectory_messages[-1]["content"] == "hello"


@pytest.mark.asyncio
async def test_failed_tool_feeds_error_back_and_continues():
    @FunctionTool.from_fn(effect=ToolEffect.READ_ONLY)
    def broken():
        raise ValueError("nope")

    loop, manager = _loop([broken], ['{"name":"broken","arguments":{}}', "recovered"])
    output = await loop.run([{"role": "user", "content": "go"}], {})
    # Failure is recorded; an unclassified exception on a read-only tool is not
    # retried (only FailureCategory.TRANSIENT is) and degrades the tool...
    assert _trace(output)[0]["status"] == str(TaskStatus.FAILED)
    assert _trace(output)[0]["error_code"] == "tool_unavailable"
    # ...fed back into the next prompt as an unavailable notice...
    assert '"status": "unavailable"' in manager.prompts[-1]
    # ...the loop continued (prompted again) and produced the recovery answer.
    assert len(manager.prompts) == 2
    assert output.final_answer == "recovered"
