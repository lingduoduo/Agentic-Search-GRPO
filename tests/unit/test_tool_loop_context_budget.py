"""ToolAgentLoop must respect ``prompt_length`` for the whole run, not just turn 1.

Two defects, both in how the loop manages its token-space context:

1. ``run()`` built ``prompt_ids`` once, then only ever grew it —
   ``prompt_ids + response_ids`` and ``prompt_ids + tool_response_ids`` — with no
   further bound. A multi-turn tool conversation therefore sailed past
   ``prompt_length`` and left the backend to cope.
2. The one crop it did perform kept the *tail*
   (``[-self.prompt_length:]``), which discards the system prompt and the tool
   schemas first — exactly the tokens the model needs to keep calling tools
   correctly. ``AgentLoopBase._crop_prompt_ids`` exists to preserve that prefix
   and was not used here.

The loop cannot crop mid-run: ``run()`` recovers the prompt/response split from
``len(prompt_ids) - len(response_mask)``, so dropping tokens the mask already
covers would desync the two. It stops instead.
"""

from __future__ import annotations

import asyncio

from src.agents import ToolAgentLoop, ToolAgentLoopConfig
from src.internal.tools.base import FunctionTool

_BIG = "X" * 400


async def _bulky() -> str:
    """A tool whose response is large relative to the test's prompt budget."""
    return _BIG


def _tool() -> FunctionTool:
    return FunctionTool(
        _bulky, name="bulky", description="returns a lot", parameters={"type": "object"}
    )


class _Tokenizer:
    """One token per character, so budgets are countable by hand."""

    chat_template = ""

    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(item) for item in ids)

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True, tokenize=True
    ):
        text = "\n".join(message.get("content", "") for message in messages)
        if not tokenize:
            return text
        return self.encode(text)


class _Manager:
    """Replies with a tool call every turn, so the loop never stops on its own."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prompts_seen: list[list[int]] = []

    async def generate(self, request_id, prompt_ids, sampling_params):
        self.prompts_seen.append(list(prompt_ids))
        return self.tokenizer.encode('{"name": "bulky", "arguments": {}}')


def _run(config: ToolAgentLoopConfig, messages):
    tokenizer = _Tokenizer()
    manager = _Manager(tokenizer)
    loop = ToolAgentLoop(tokenizer, manager, [_tool()], config)
    output = asyncio.run(loop.run(messages, sampling_params={}))
    return output, manager


def test_the_prompt_stays_within_budget_across_turns():
    """Every prompt handed to the backend must fit the configured budget."""
    config = ToolAgentLoopConfig(
        prompt_length=600,
        response_length=100_000,
        max_assistant_turns=50,
        max_user_turns=50,
    )

    _, manager = _run(config, [{"role": "user", "content": "go"}])

    assert manager.prompts_seen, "expected at least one generation"
    oversized = [len(p) for p in manager.prompts_seen if len(p) > config.prompt_length]
    assert not oversized, (
        f"prompt exceeded prompt_length={config.prompt_length}: {oversized}"
    )


def test_the_run_still_ends_and_reports_its_turns():
    """Stopping on the prompt budget must be an ordinary exit, not a hang."""
    config = ToolAgentLoopConfig(
        prompt_length=600,
        response_length=100_000,
        max_assistant_turns=50,
        max_user_turns=50,
    )

    output, manager = _run(config, [{"role": "user", "content": "go"}])

    # Far below the turn caps: the prompt budget is what ended this run.
    assert len(manager.prompts_seen) < config.max_assistant_turns
    assert output.final_answer is not None
    # The tool did run, and its result is preserved in the transcript even
    # though the budget stopped its tokens being appended.
    assert any(m["role"] == "tool" for m in output.trajectory_messages)


def test_an_over_budget_prompt_keeps_the_system_prefix():
    """Over budget, the system prompt survives and the oldest turns are dropped."""
    tokenizer = _Tokenizer()
    loop = ToolAgentLoop(
        tokenizer, _Manager(tokenizer), [_tool()], ToolAgentLoopConfig(prompt_length=60)
    )
    system = "SYSTEM-RULES"

    ids = loop._build_prompt_ids_with_tools_sync(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "A" * 200},
        ]
    )

    assert len(ids) <= 60
    assert tokenizer.decode(ids).startswith(system), (
        f"system prefix was cropped away: {tokenizer.decode(ids)[:40]!r}"
    )
