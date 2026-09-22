"""Unit tests for system-preserving prompt-id truncation."""

from __future__ import annotations

from src.agents.core.base import AgentLoopBase, AgentLoopConfig, _crop_prompt_ids
from tests.unit.test_agent_loop import DummyServerManager, DummyTokenizerWithEncode


def test_under_budget_unchanged():
    full = [1, 2, 3]
    assert _crop_prompt_ids(full, [9], 10) == full
    assert _crop_prompt_ids(full, [9], 0) == full  # budget <= 0 → unchanged


def test_no_system_tail_crop():
    full = list(range(10))
    assert _crop_prompt_ids(full, [], 4) == [6, 7, 8, 9]


def test_system_preserved_over_budget():
    system = [100, 101]
    full = list(range(20))  # far over budget
    out = _crop_prompt_ids(full, system, 6)
    assert len(out) == 6
    assert out[:2] == system
    assert out[2:] == full[-(6 - 2) :]  # recent tail fills the rest


def test_system_larger_than_budget_degenerate():
    system = [1, 2, 3, 4, 5]
    full = list(range(50))
    out = _crop_prompt_ids(full, system, 3)
    assert out == system[-3:]


def test_build_prompt_ids_sync_keeps_system_prefix():
    loop = AgentLoopBase(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
        config=AgentLoopConfig(prompt_length=40),
    )
    system_content = "SYSTEM RULES"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "u" * 200},  # forces the crop
    ]
    ids = loop._build_prompt_ids_sync(messages)
    assert len(ids) == 40
    assert ids[: len(system_content)] == [ord(c) for c in system_content]


# ---------------------------------------------------------------------------
# Over budget, the prompt must stay well-formed and say what it dropped
# ---------------------------------------------------------------------------


class ChatTemplateTokenizer:
    """A tokenizer with a real-shaped chat template, one token per character.

    Character-level so budgets are countable by hand, and the rendered text can
    be read back out of the token ids to check the prompt's structure.
    """

    chat_template = "chatml"

    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True, tokenize=False
    ):
        out = "".join(
            f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n"
            for m in messages
        )
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return self.encode(out) if tokenize else out


SYSTEM = "RULES: cite evidence."


def _buffer(n_turns: int, filler: int = 60) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM}]
    for i in range(1, n_turns + 1):
        messages.append({"role": "user", "content": f"q{i} " + "x" * filler})
        messages.append({"role": "assistant", "content": f"a{i} " + "y" * filler})
    return messages


def _loop(budget: int) -> AgentLoopBase:
    return AgentLoopBase(
        tokenizer=ChatTemplateTokenizer(),
        server_manager=DummyServerManager([]),
        config=AgentLoopConfig(prompt_length=budget),
    )


def _render(loop: AgentLoopBase, ids: list[int]) -> str:
    return loop.tokenizer.decode(ids)


def test_an_over_budget_prompt_has_no_headless_fragment():
    """The old token slice glued a mid-word fragment onto the system message.

    Everything after the system block must begin at a role marker, or the model
    reads a truncated assistant turn as part of its own instructions.
    """
    loop = _loop(budget=400)
    text = _render(loop, loop._build_prompt_ids_sync(_buffer(6)))

    head, _, tail = text.partition("<|im_end|>\n")
    assert head.startswith("<|im_start|>system"), text[:80]
    assert tail.startswith("<|im_start|>"), (
        f"content after the system block is headless: {tail[:70]!r}"
    )


def test_an_over_budget_prompt_has_balanced_role_markers():
    """One unclosed <|im_start|> is expected: the trailing generation cue."""
    loop = _loop(budget=400)
    text = _render(loop, loop._build_prompt_ids_sync(_buffer(6)))

    assert text.endswith("<|im_start|>assistant\n")
    body = text[: -len("<|im_start|>assistant\n")]
    assert body.count("<|im_start|>") == body.count("<|im_end|>"), (
        f"unbalanced role markers in:\n{body[:200]!r}"
    )


def test_whole_messages_are_dropped_oldest_first():
    loop = _loop(budget=400)
    text = _render(loop, loop._build_prompt_ids_sync(_buffer(6)))

    assert SYSTEM in text, "the system message must always survive"
    assert "q6 " in text, "the newest turn must survive"
    assert "q1 " not in text, "the oldest turn should have been dropped"


def test_the_number_of_dropped_messages_is_reported():
    """Losing context silently is the defect; the count is the signal."""
    loop = _loop(budget=400)

    loop._build_prompt_ids_sync(_buffer(6))

    assert loop.prompt_messages_dropped > 0
    assert loop.prompt_hard_truncated is False, (
        "dropping whole messages is not a hard truncation"
    )


def test_under_budget_drops_nothing_and_reports_nothing():
    loop = _loop(budget=100_000)

    text = _render(loop, loop._build_prompt_ids_sync(_buffer(6)))

    assert "q1 " in text and "q6 " in text
    assert loop.prompt_messages_dropped == 0
    assert loop.prompt_hard_truncated is False


def test_a_single_oversized_message_is_hard_truncated_and_flagged():
    """When even system + newest cannot fit, a raw slice is unavoidable.

    That is the one case that can still malform the prompt, so it is reported
    separately rather than folded into the dropped-message count.
    """
    loop = _loop(budget=120)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "z" * 5000},
    ]

    ids = loop._build_prompt_ids_sync(messages)

    assert len(ids) <= 120
    assert loop.prompt_hard_truncated is True
    assert SYSTEM in _render(loop, ids), "the system prefix still survives"


# ---------------------------------------------------------------------------
# The signal must reach a channel something actually reads
# ---------------------------------------------------------------------------


def test_search_loop_surfaces_context_loss_in_metrics():
    """AgentLoopOutput.truncated is read by nothing; metrics are read by reward.py.

    A flag no consumer looks at is not visibility, so the budget accounting goes
    where the rest of the run's numbers already go.
    """
    import asyncio

    from src.agents.search import SearchAgentLoop, SearchAgentLoopConfig

    tokenizer = DummyTokenizerWithEncode()
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(
            [tokenizer.encode("<answer>done</answer>")] * 4
        ),
        search_config=SearchAgentLoopConfig(
            max_turns=2,
            prompt_length=80,
            require_sufficient_evidence_before_answer=False,
        ),
    )
    history = [
        {"role": "user", "content": "old question " + "x" * 200},
        {"role": "assistant", "content": "old answer " + "y" * 200},
        {"role": "user", "content": "the current question"},
    ]

    output = asyncio.run(loop.run(history, {"temperature": 0.0}))

    assert "prompt_messages_dropped" in output.metrics
    assert output.metrics["prompt_messages_dropped"] > 0


def test_tool_loop_surfaces_context_loss_in_metrics():
    import asyncio

    from src.agents import ToolAgentLoop, ToolAgentLoopConfig

    tokenizer = ChatTemplateTokenizer()

    class _Server:
        async def generate(self, request_id, prompt_ids, sampling_params):
            return tokenizer.encode("all done")

    loop = ToolAgentLoop(
        tokenizer,
        _Server(),
        [],
        ToolAgentLoopConfig(prompt_length=90, response_length=500),
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "older turn " + "x" * 300},
        {"role": "user", "content": "newest turn"},
    ]

    output = asyncio.run(loop.run(messages, {"temperature": 0.0}))

    assert "prompt_messages_dropped" in output.metrics
    assert output.metrics["prompt_messages_dropped"] > 0


def test_metrics_report_zero_when_nothing_was_dropped():
    """The keys must always be present, or a consumer cannot tell 0 from absent."""
    import asyncio

    from src.agents.search import SearchAgentLoop, SearchAgentLoopConfig

    tokenizer = DummyTokenizerWithEncode()
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(
            [tokenizer.encode("<answer>done</answer>")] * 4
        ),
        search_config=SearchAgentLoopConfig(
            max_turns=2,
            prompt_length=100_000,
            require_sufficient_evidence_before_answer=False,
        ),
    )

    output = asyncio.run(loop.run([{"role": "user", "content": "hi"}], {}))

    assert output.metrics["prompt_messages_dropped"] == 0.0
    assert output.metrics["prompt_hard_truncated"] == 0.0


# ---------------------------------------------------------------------------
# Fitting must not cost a render per dropped message
# ---------------------------------------------------------------------------


def test_fitting_a_long_buffer_takes_a_bounded_number_of_renders():
    """Dropping one message and re-rendering is O(dropped) renders per build.

    Measured on a 40-turn buffer at a 1024-token budget: 77 renders and 1.1s for
    a single prompt build, against 28ms for the old raw slice. Since the render
    itself is O(buffer), the loop was O(n^2) per turn and O(n^3) per run.

    Token counts are accumulated from the newest message backwards instead, so
    the number of full renders does not grow with the number dropped.
    """
    loop = _loop(budget=400)
    messages = _buffer(40)

    renders = {"n": 0}
    original = loop._render_prompt_ids

    def counted(msgs):
        renders["n"] += 1
        return original(msgs)

    loop._render_prompt_ids = counted
    ids = loop._build_prompt_ids_sync(messages)

    assert len(ids) <= 400
    assert loop.prompt_messages_dropped > 50, "this buffer must be well over budget"
    assert renders["n"] <= 4, (
        f"{renders['n']} full renders to fit one prompt; must not scale with "
        f"the {loop.prompt_messages_dropped} messages dropped"
    )


def test_bounded_fitting_still_keeps_system_and_newest():
    """The cheap path must preserve the guarantees the slow one had."""
    loop = _loop(budget=400)
    text = _render(loop, loop._build_prompt_ids_sync(_buffer(40)))

    assert SYSTEM in text
    assert "q40 " in text
    assert "q1 " not in text
    _, _, tail = text.partition("<|im_end|>\n")
    assert tail.startswith("<|im_start|>")
