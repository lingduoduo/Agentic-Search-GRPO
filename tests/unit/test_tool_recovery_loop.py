import asyncio
import json

from src.agents import (
    ApprovalDecision,
    EscalationDecision,
    ToolAgentLoop,
    ToolAgentLoopConfig,
)
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


def test_a_wrong_argument_feeds_back_the_adapter_text_and_the_fix_runs():
    calls = []

    @FunctionTool.from_fn(
        name="quote",
        effect=ToolEffect.READ_ONLY,
        parameters={
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    )
    async def quote(symbol):
        calls.append(symbol)
        if symbol == "APPL":
            return ToolErrorText(
                json.dumps({"error": "invalid ticker symbol 'APPL'"}),
                ToolFailure(
                    FailureCategory.INVALID_INPUT, "upstream rejected the input"
                ),
            )
        return {"symbol": symbol, "price": 1}

    loop, manager = _loop(
        [quote],
        [
            '{"name":"quote","arguments":{"symbol":"APPL"}}',
            '{"name":"quote","arguments":{"symbol":"AAPL"}}',
            "done",
        ],
    )
    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {}))
    first, second = _trace(output)
    assert calls == ["APPL", "AAPL"]  # the corrected call is not skipped
    assert first["error_code"] == "invalid_arguments"
    assert second["status"] == str(TaskStatus.COMPLETED)
    assert "invalid ticker symbol 'APPL'" in manager.prompts[1]
    assert output.tool_recovery is None
    assert output.final_answer == "done"


def test_no_failures_means_no_recovery_summary():
    tool, _ = _flaky(0)
    loop, _ = _loop([tool], [CALL, "done"])
    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {}))
    assert output.tool_recovery is None


def _writer(fail_times):
    calls = []

    @FunctionTool.from_fn(name="send", effect=ToolEffect.SIDE_EFFECTING)
    async def send():
        calls.append(1)
        if len(calls) <= fail_times:
            return ToolErrorText(
                "Error: boom",
                ToolFailure(FailureCategory.UNKNOWN, "remote tool reported an error"),
            )
        return "sent"

    return send, calls


SEND = '{"name":"send","arguments":{}}'


async def _approve(request):
    return ApprovalDecision.APPROVE


def _run(tool, responses, on_escalation, **config):
    tokenizer = _Tokenizer()
    manager = _Manager(tokenizer, responses)
    loop = ToolAgentLoop(
        tokenizer,
        manager,
        [tool],
        ToolAgentLoopConfig(response_length=8192, **config),
        recovery_policy=FAST,
    )
    return asyncio.run(
        loop.run(
            [{"role": "user", "content": "go"}],
            {},
            on_approval=_approve,
            on_escalation=on_escalation,
        )
    )


def _answering(*decisions):
    seen = []
    queue = list(decisions)

    async def on_escalation(request):
        seen.append(request)
        return queue.pop(0)

    return on_escalation, seen


def test_side_effecting_failure_escalates_and_retry_runs_it_again():
    tool, calls = _writer(1)
    on_escalation, seen = _answering(EscalationDecision.RETRY)
    output = _run(tool, [SEND, "done"], on_escalation)
    assert len(calls) == 2 and len(seen) == 1
    assert (
        seen[0].tool_name == "send"
        and seen[0].category == "unknown"
        and seen[0].attempts == 1
    )
    assert _trace(output)[0]["status"] == str(TaskStatus.COMPLETED)
    assert output.tool_recovery["escalations"] == [
        {"tool": "send", "category": "unknown", "attempts": 1, "decision": "retry"}
    ]


def test_skip_degrades_and_the_run_continues():
    tool, calls = _writer(99)
    on_escalation, _ = _answering(EscalationDecision.SKIP)
    output = _run(tool, [SEND, "carried on"], on_escalation)
    assert len(calls) == 1
    assert output.final_answer == "carried on"
    assert output.tool_recovery["degraded"] == ["send"]


def test_cancel_stops_with_the_fixed_answer():
    tool, _ = _writer(99)
    on_escalation, _ = _answering(EscalationDecision.CANCEL)
    output = _run(tool, [SEND, "never generated"], on_escalation)
    assert (
        output.final_answer
        == "Stopped: send failed (unknown); nothing further was attempted."
    )
    assert output.tool_recovery["outcome"] == "cancelled"
    assert output.tool_recovery["needs_user"] is False


def test_expired_stops_unresolved():
    tool, _ = _writer(99)
    on_escalation, _ = _answering(EscalationDecision.EXPIRED)
    output = _run(tool, [SEND, "never generated"], on_escalation)
    assert (
        output.final_answer
        == "send failed and was not retried; the action may not have completed."
    )
    assert output.tool_recovery["needs_user"] is True


def test_expired_after_a_user_retry_reports_the_attempts():
    tool, calls = _writer(99)
    on_escalation, _ = _answering(EscalationDecision.RETRY, EscalationDecision.EXPIRED)
    output = _run(tool, [SEND, "never generated"], on_escalation)
    assert len(calls) == 2
    assert (
        output.final_answer
        == "send failed after 2 attempts; the action may not have completed."
    )


def test_no_callback_stops_unresolved_without_asking():
    tool, calls = _writer(99)
    output = _run(tool, [SEND, "never generated"], None)
    assert len(calls) == 1
    assert output.tool_recovery["outcome"] == "unresolved"
    assert output.tool_recovery["escalations"][0]["decision"] == "no_callback"


def test_callback_that_raises_or_hangs_is_unresolved():
    tool, _ = _writer(99)

    async def boom(request):
        raise RuntimeError("socket gone")

    boom_output = _run(tool, [SEND, "x"], boom)
    assert boom_output.tool_recovery["needs_user"] is True
    # Pinned to exactly one escalation so a raise -> RETRY mutant (which would
    # keep retrying until the per-run cap kicks in) fails here rather than
    # being masked by the cap forcing the same needs_user=True outcome later.
    assert len(boom_output.tool_recovery["escalations"]) == 1
    assert boom_output.tool_recovery["escalations"][0]["decision"] == "expired"

    async def hang(request):
        await asyncio.sleep(10)

    output = _run(tool, [SEND, "x"], hang, escalation_timeout_seconds=0.05)
    assert output.tool_recovery["needs_user"] is True
    assert len(output.tool_recovery["escalations"]) == 1
    assert output.tool_recovery["escalations"][0]["decision"] == "expired"


def test_bad_callback_return_value_is_treated_as_expired():
    tool, _ = _writer(99)

    async def bad(request):
        return None

    output = _run(tool, [SEND, "x"], bad)
    assert output.tool_recovery["needs_user"] is True
    assert len(output.tool_recovery["escalations"]) == 1
    assert output.tool_recovery["escalations"][0]["decision"] == "expired"


def test_concurrent_retry_is_blocked_once_another_call_cancels():
    tool, calls = _writer(99)
    cancelled = asyncio.Event()

    async def on_escalation(request):
        # Whichever concurrent call's escalation resolves first wins CANCEL;
        # the other must see the stopped run and not replay its side effect.
        if not cancelled.is_set():
            cancelled.set()
            return EscalationDecision.CANCEL
        return EscalationDecision.RETRY

    output = _run(tool, [f"[{SEND},{SEND}]", "never generated"], on_escalation)
    assert len(calls) == 2  # one execution per call, no post-cancel retry
    assert (
        output.final_answer
        == "Stopped: send failed (unknown); nothing further was attempted."
    )
    assert output.tool_recovery["outcome"] == "cancelled"


def test_escalations_are_capped_per_run():
    tool, calls = _writer(99)
    on_escalation, seen = _answering(*[EscalationDecision.RETRY] * 10)
    output = _run(tool, [SEND, "x"], on_escalation, max_escalations=3)
    assert len(seen) == 3
    assert len(calls) == 4  # first attempt + 3 user-authorised retries
    assert output.tool_recovery["outcome"] == "unresolved"
    assert output.tool_recovery["escalations"][-1]["decision"] == "cap"
    assert (
        output.final_answer
        == "send failed after 4 attempts; the action may not have completed."
    )


def test_denied_approval_is_never_escalated():
    tool, calls = _writer(99)
    on_escalation, seen = _answering(EscalationDecision.RETRY)
    tokenizer = _Tokenizer()
    manager = _Manager(tokenizer, [SEND, "done"])
    loop = ToolAgentLoop(
        tokenizer, manager, [tool], ToolAgentLoopConfig(response_length=8192)
    )

    async def deny(request):
        return ApprovalDecision.DENY

    asyncio.run(
        loop.run(
            [{"role": "user", "content": "go"}],
            {},
            on_approval=deny,
            on_escalation=on_escalation,
        )
    )
    assert calls == [] and seen == []
