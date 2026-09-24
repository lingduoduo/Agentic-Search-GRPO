import asyncio
import json

from src.agents import ToolAgentLoop, ToolAgentLoopConfig
from src.agents.core.state import TaskStatus
from src.agents.tool.recovery import RecoveryPolicy
from src.internal.tools import (
    FailureCategory,
    FunctionTool,
    ToolEffect,
    ToolErrorText,
    ToolFailure,
)

FAST = RecoveryPolicy(backoff=(0.0, 0.0))


class _Tokenizer:
    chat_template = ""

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True, tokenize=True
    ):
        text = "\n".join(m.get("content", "") for m in messages)
        return self.encode(text) if tokenize else text


class _Manager:
    def __init__(self, tokenizer, responses):
        self.tokenizer = tokenizer
        self.responses = iter(responses)
        self.prompts = []

    async def generate(self, request_id, prompt_ids, sampling_params):
        self.prompts.append(self.tokenizer.decode(prompt_ids))
        return self.tokenizer.encode(next(self.responses))


def _loop(tools, responses, policy=FAST):
    tokenizer = _Tokenizer()
    manager = _Manager(tokenizer, responses)
    return (
        ToolAgentLoop(
            tokenizer,
            manager,
            tools,
            ToolAgentLoopConfig(response_length=8192),
            recovery_policy=policy,
        ),
        manager,
    )


def _trace(output):
    return [json.loads(line) for line in (output.action_trace or "").splitlines()]


def _flaky(fail_times, category=FailureCategory.TRANSIENT):
    calls = []

    @FunctionTool.from_fn(name="lookup", effect=ToolEffect.READ_ONLY)
    async def lookup():
        calls.append(1)
        if len(calls) <= fail_times:
            return ToolErrorText(
                '{"error": "x"}',
                ToolFailure(category, "upstream temporarily unavailable"),
            )
        return {"ok": True}

    return lookup, calls


CALL = '{"name":"lookup","arguments":{}}'


def test_transient_read_only_failure_recovers_on_retry():
    tool, calls = _flaky(1)
    loop, _ = _loop([tool], [CALL, "done"])
    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {}))
    [result] = _trace(output)
    assert len(calls) == 2
    assert result["status"] == str(TaskStatus.COMPLETED)
    assert result["retry_count"] == 1
    assert output.tool_recovery == {
        "outcome": "recovered",
        "needs_user": False,
        "retries": 1,
        "degraded": [],
        "escalations": [],
    }


def test_exhausted_retries_mark_the_tool_unavailable_and_short_circuit():
    tool, calls = _flaky(99)
    loop, manager = _loop([tool], [CALL, CALL, "answered from what I have"])
    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {}))
    first, second = _trace(output)
    assert len(calls) == 3  # 1 + max_retries; the second model call never executes
    assert (first["status"], first["error_code"]) == (
        str(TaskStatus.FAILED),
        "tool_unavailable",
    )
    assert (second["status"], second["error_code"]) == (
        str(TaskStatus.SKIPPED),
        "tool_unavailable",
    )
    assert '"status": "unavailable"' in manager.prompts[1]
    assert output.tool_recovery["degraded"] == ["lookup"]
    assert output.tool_recovery["outcome"] == "degraded"
    assert output.final_answer == "answered from what I have"


def test_permanent_read_only_failure_degrades_without_retry():
    tool, calls = _flaky(99, FailureCategory.PERMANENT)
    loop, _ = _loop([tool], [CALL, "done"])
    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {}))
    assert len(calls) == 1
    assert output.tool_recovery["degraded"] == ["lookup"]


class _NoJitter(RecoveryPolicy):
    """Deterministic delays: the loop never passes a jitter source."""

    def decide(self, *args, **kwargs):
        return super().decide(*args, uniform=lambda low, high: 1.0)


def test_concurrent_retries_share_and_never_overspend_the_budget():
    tool, calls = _flaky(99)
    # Each retry costs exactly 0.04 s; the run has 0.06 s. The first call to
    # reserve takes 0.04, leaving 0.02 — too little for any further retry.
    policy = _NoJitter(backoff=(0.04, 0.04), retry_budget=0.06)
    loop, _ = _loop([tool], [f"[{CALL},{CALL}]", "done"], policy=policy)
    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {}))
    assert output.tool_recovery["retries"] == 1


def test_invalid_arguments_still_go_back_to_the_model():
    @FunctionTool.from_fn(
        name="lookup",
        effect=ToolEffect.READ_ONLY,
        parameters={
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        },
    )
    def lookup(n):
        return n

    loop, _ = _loop([lookup], [CALL, "done"])
    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {}))
    [result] = _trace(output)
    assert result["error_code"] == "invalid_arguments"
    assert output.tool_recovery is None


def test_no_failures_means_no_recovery_summary():
    tool, _ = _flaky(0)
    loop, _ = _loop([tool], [CALL, "done"])
    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {}))
    assert output.tool_recovery is None
