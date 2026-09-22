"""Repo-native agent loop helpers for search-oriented generation."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .control_flow_trace import ControlFlowEvent

logger = logging.getLogger(__name__)

# Matches the first recognised action tag in a model response.
# Covers all tags used by SearchAgentLoop; callers can override via action_re.
_DEFAULT_ACTION_RE: re.Pattern[str] = re.compile(
    r"<(plan|search_decision|subquestions|searches|search|fetch|answer)>(.*?)</\1>",
    re.DOTALL,
)

_DEFAULT_TERMINAL_ACTIONS: frozenset[str] = frozenset({"answer"})


def _crop_prompt_ids(
    full_ids: list[int], system_ids: list[int], budget: int
) -> list[int]:
    """Truncate `full_ids` to `budget` tokens, preserving the system prefix.

    Under budget → returned unchanged. Over budget → keep `system_ids` at the
    front and fill the remaining budget with the tail of `full_ids` (the most
    recent tokens, ending with the generation cue).

    **Last resort only.** This slices a rendered chat template at an arbitrary
    token offset, so the surviving tail almost always starts mid-message: the
    model receives a headless fragment of a truncated turn glued directly onto
    the system message, plus an orphaned end-of-turn marker closing a block that
    was never opened. Prefer ``AgentLoopBase._fit_messages_to_budget``, which
    drops whole messages and keeps the prompt well-formed. This remains for the
    one case that cannot: a single message that alone exceeds the budget.
    """
    if budget <= 0 or len(full_ids) <= budget:
        return full_ids
    if not system_ids:
        return full_ids[-budget:]
    if len(system_ids) >= budget:
        return system_ids[-budget:]
    return system_ids + full_ids[-(budget - len(system_ids)) :]


# Signature: (turn_number, tool_name_or_None, doc_count) -> awaitable
OnTurnCallback = Callable[[int, "str | None", int], Awaitable[None]]
# One decoded chunk of the answer as the model produces it. Optional everywhere:
# passing None keeps the non-streaming path, which is what every trainer and
# offline script uses.
OnTokenCallback = Callable[[str], Awaitable[None]]


def _guarded_token_callback(on_token: "OnTokenCallback") -> "OnTokenCallback":
    """Wrap *on_token* so a failing consumer cannot abort the generation.

    The callback is a delivery detail -- an SSE client that has gone away. By
    the time tokens flow the run has already done the expensive work, so losing
    the answer to a disconnected browser would be the wrong trade. Failures are
    swallowed per token and the run completes normally.
    """

    async def guarded(text: str) -> None:
        try:
            await on_token(text)
        except Exception:  # noqa: BLE001 - delivery failure, not a run failure
            pass

    return guarded


_REGISTERED_AGENT_LOOPS: dict[str, type["AgentLoopBase"]] = {}


def register(name: str):
    """Register an agent loop class by name."""

    def decorator(cls: type["AgentLoopBase"]) -> type["AgentLoopBase"]:
        _REGISTERED_AGENT_LOOPS[name] = cls
        return cls

    return decorator


def get_registered_agent_loop(name: str) -> type["AgentLoopBase"]:
    """Return a registered agent loop class by name."""

    try:
        return _REGISTERED_AGENT_LOOPS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown agent loop: {name!r}") from exc


def list_registered_agent_loops() -> list[str]:
    """List registered agent loop names."""

    return sorted(_REGISTERED_AGENT_LOOPS)


# The four AgentLoopBase loops the registry consolidates. AgenticRAGLoop and the
# retrieval-pipeline modes are deliberately NOT here (different contract).
CANONICAL_AGENT_NAMES: frozenset[str] = frozenset(
    {"plain_generation", "single_turn_agent", "search_agent", "tool_agent"}
)

# CLI/web aliases → canonical registry name. Canonical names map to themselves.
_AGENT_ALIASES: dict[str, str] = {
    "single": "plain_generation",
    "search": "search_agent",
    "tool": "tool_agent",
    "plain_generation": "plain_generation",
    "single_turn_agent": "single_turn_agent",
    "search_agent": "search_agent",
    "tool_agent": "tool_agent",
}

# Guard the two hand-maintained literals against silent drift: every alias must
# resolve to a canonical name, and every canonical name must be reachable.
assert set(_AGENT_ALIASES.values()) == CANONICAL_AGENT_NAMES


def resolve_agent_name(name: str) -> str:
    """Resolve a CLI/web alias or canonical name to a canonical registry loop name.

    Raises KeyError for names that are not registry loops (e.g. chat_loop,
    search_tool, hybrid_search, chat_once) so callers keep dispatching those on
    their existing non-registry paths.
    """
    try:
        return _AGENT_ALIASES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown agent loop alias: {name!r}. Known: {sorted(_AGENT_ALIASES)}"
        ) from exc


@dataclass(frozen=True)
class AgentLoopConfig:
    prompt_length: int = 4096
    response_length: int = 512


@dataclass(frozen=True)
class RolloutStep:
    """One state → action step in a policy rollout trajectory.

    Maps directly to the RL triple:
        state (prompt_ids) → policy.generate() → action (response_ids) → terminal/continue

    action_type is the XML tag the model emitted: "search", "answer", "fetch",
    "plan", etc.  is_terminal=True means the trajectory should stop after this step.
    """

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    action_type: str | None
    action_content: str
    is_terminal: bool


@dataclass
class AgentLoopOutput:
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    num_turns: int
    metrics: dict[str, float] = field(default_factory=dict)
    request_id: str | None = None
    group_id: str | None = None
    rollout_index: int | None = None
    context: Any | None = None  # AgentContext when produced by SearchAgentLoop
    trajectory_messages: list[dict[str, Any]] = field(default_factory=list)
    action_trace: str | None = None
    final_answer: str | None = None  # Content of the last <answer> tag, if any
    control_flow_trace: list[ControlFlowEvent] = field(default_factory=list)
    # True when a generation was cut short by the wall-clock stop rather than
    # finishing. The answer is a fragment; callers should say so.
    truncated: bool = False


@contextmanager
def simple_timer(name: str, metrics: dict[str, float]):
    """Small timing helper for per-step metrics."""

    start = time.perf_counter()
    try:
        yield
    finally:
        metrics[name] = time.perf_counter() - start


class AgentLoopBase:
    """Minimal async agent loop base used inside this repository."""

    def __init__(
        self,
        tokenizer: Any,
        server_manager: Any,
        config: AgentLoopConfig | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.server_manager = server_manager
        self.config = config or AgentLoopConfig()
        self.loop = loop
        self.prompt_length = self.config.prompt_length
        self.response_length = self.config.response_length
        # Set when any generation in this run was cut short by the wall clock.
        self.generation_truncated = False
        # Context-budget accounting for this run. Whole messages dropped to fit
        # the prompt budget, and whether a single message had to be sliced --
        # reported because losing context invisibly is how a run silently
        # answers from less than the caller believes it had.
        self.prompt_messages_dropped = 0
        self.prompt_hard_truncated = False
        self._cached_message_overhead: int | None = None

    async def get_loop(self) -> asyncio.AbstractEventLoop:
        if self.loop is None:
            self.loop = asyncio.get_running_loop()
        return self.loop

    async def build_prompt_ids(self, messages: list[dict[str, Any]]) -> list[int]:
        """Tokenize a chat message list using the best tokenizer API available."""

        event_loop = await self.get_loop()
        return await event_loop.run_in_executor(
            None, lambda: self._build_prompt_ids_sync(messages)
        )

    def _encode_system_prefix(self, messages: list[dict[str, Any]]) -> list[int]:
        """Token ids of the leading system message's content, or [] if none.

        Uses the raw content (not a lone chat-template render): it is simpler,
        avoids template-specific behavior when a system message is rendered
        alone, and still preserves the instruction text when the prompt is
        cropped over budget.
        """
        if not messages or messages[0].get("role") != "system":
            return []
        if not hasattr(self.tokenizer, "encode"):
            return []
        return list(self.tokenizer.encode(messages[0].get("content", "")))

    def _render_prompt_ids(self, messages: list[dict[str, Any]]) -> list[int]:
        """Render `messages` to token ids, with no regard for the budget."""
        chat_template = getattr(self.tokenizer, "chat_template", "__missing__")
        if hasattr(self.tokenizer, "apply_chat_template") and chat_template is not None:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            return list(self.tokenizer.encode(prompt_text))
        if hasattr(self.tokenizer, "encode"):
            joined = "\n".join(message.get("content", "") for message in messages)
            return list(self.tokenizer.encode(joined))
        raise TypeError(
            "tokenizer must implement apply_chat_template(...) or encode(...)."
        )

    def _estimate_message_tokens(self, message: dict[str, Any]) -> int:
        """Conservative token cost of one message, without a template render.

        Deliberately an over-estimate: the per-message overhead is measured with
        the generation cue included, so a suffix chosen from these numbers fits
        the budget with room to spare and the final render almost never needs
        correcting. Over-estimating costs one extra dropped message; under-
        estimating costs another full render.
        """
        content = str(message.get("content", ""))
        if not hasattr(self.tokenizer, "encode"):
            # No encoder: fall back to a coarse character heuristic.
            return len(content) // 4 + self._message_overhead()
        return len(self.tokenizer.encode(content)) + self._message_overhead()

    def _message_overhead(self) -> int:
        """Per-message template overhead, rendered once and cached."""
        if self._cached_message_overhead is None:
            try:
                self._cached_message_overhead = len(
                    self._render_prompt_ids([{"role": "user", "content": ""}])
                )
            except Exception:  # noqa: BLE001 - estimation only; never fail a run
                self._cached_message_overhead = 8
        return self._cached_message_overhead

    def _fit_messages_to_budget(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[int]]:
        """Drop whole oldest messages until the rendered prompt fits the budget.

        Returns the messages kept and their rendered ids. A leading system
        message and the newest message are always kept, so the result can still
        exceed the budget -- the caller slices that case and flags it.

        Dropping whole messages rather than slicing tokens is what keeps the
        prompt well-formed: every surviving turn has both of its role markers.

        The kept suffix is chosen from per-message estimates rather than by
        dropping one message and re-rendering. That loop cost one full render per
        dropped message -- measured at 77 renders and 1.1s for a single build on a
        40-turn buffer at a 1024-token budget, against 28ms for a raw slice.
        Since each render is itself O(buffer), it was O(n^2) per turn.
        """
        prompt_ids = self._render_prompt_ids(messages)
        budget = self.prompt_length
        if budget <= 0 or len(prompt_ids) <= budget:
            return messages, prompt_ids

        lead = messages[:1] if messages and messages[0].get("role") == "system" else []
        rest = list(messages[len(lead) :])
        if len(rest) <= 1:
            return messages, prompt_ids

        # Walk backwards from the newest message, keeping what fits.
        used = sum(self._estimate_message_tokens(m) for m in lead)
        used += self._estimate_message_tokens(rest[-1])
        keep = 1
        for message in reversed(rest[:-1]):
            cost = self._estimate_message_tokens(message)
            if used + cost > budget:
                break
            used += cost
            keep += 1

        kept = lead + rest[-keep:]
        prompt_ids = self._render_prompt_ids(kept)
        # The estimate ignores render-boundary effects, so correct downward if it
        # was optimistic. Conservative estimation makes this rare, not impossible.
        while len(prompt_ids) > budget and len(kept) > len(lead) + 1:
            kept = lead + kept[len(lead) + 1 :]
            prompt_ids = self._render_prompt_ids(kept)

        self.prompt_messages_dropped += len(messages) - len(kept)
        return kept, prompt_ids

    def _build_prompt_ids_sync(self, messages: list[dict[str, Any]]) -> list[int]:
        system_ids = self._encode_system_prefix(messages)
        kept, prompt_ids = self._fit_messages_to_budget(messages)
        if self.prompt_length > 0 and len(prompt_ids) > self.prompt_length:
            # Even the system message plus the newest one overflows. Slicing is
            # the only option left, and it is the one case that can still hand
            # the model a malformed prompt -- so say so rather than counting it
            # as an ordinary dropped message.
            self.prompt_hard_truncated = True
            logger.warning(
                "Prompt budget %d exceeded by a single message (%d tokens); "
                "hard-truncating. The prompt may be malformed.",
                self.prompt_length,
                len(prompt_ids),
            )
            return _crop_prompt_ids(prompt_ids, system_ids, self.prompt_length)
        if self.prompt_messages_dropped:
            logger.warning(
                "Dropped %d message(s) to fit the %d-token prompt budget; %d kept.",
                self.prompt_messages_dropped,
                self.prompt_length,
                len(kept),
            )
        return prompt_ids

    async def generate_response_ids(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str | None = None,
        on_token: "OnTokenCallback | None" = None,
    ) -> list[int]:
        """Generate response token ids, optionally streaming decoded chunks.

        ``on_token`` is used only when the configured manager implements
        ``generate_stream``. Managers are structurally typed (``ServerManager``
        is a Protocol), so probing for the method rather than adding a required
        argument keeps every existing implementation and test double working --
        including the ones in the training stack, which never stream.
        """
        request_id = request_id or uuid4().hex
        streamer = getattr(self.server_manager, "generate_stream", None)
        if on_token is not None and streamer is not None:
            generate_result = streamer(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                on_token=_guarded_token_callback(on_token),
            )
        else:
            generate_result = self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
        response_ids = (
            await generate_result
            if inspect.isawaitable(generate_result)
            else generate_result
        )
        # Managers that can cut a generation short report it here; those that
        # cannot (OpenAI-compatible, test doubles) simply lack the hook.
        pop_truncated = getattr(self.server_manager, "pop_truncated", None)
        if pop_truncated is not None and pop_truncated(request_id):
            self.generation_truncated = True
        return list(response_ids)[: self.response_length]

    def record_prompt_budget_metrics(self, metrics: dict[str, float]) -> None:
        """Copy this run's context-budget accounting into *metrics*.

        Always writes both keys, including zeros: a consumer cannot distinguish
        "nothing was dropped" from "this loop does not report" otherwise.
        ``metrics`` is the channel because it is the one the reward functions and
        the run's own numbers already travel on -- ``AgentLoopOutput.truncated``
        is set by one loop and read by nothing.
        """
        metrics["prompt_messages_dropped"] = float(self.prompt_messages_dropped)
        metrics["prompt_hard_truncated"] = 1.0 if self.prompt_hard_truncated else 0.0

    def _record_tool_stage(self, name: str, args: dict[str, Any], result: Any) -> None:
        # Deferred import: src.internal.servers.web's package __init__ imports
        # agent-loop modules (via app.py), so a top-level import here would be
        # circular.
        from src.internal.servers.web import request_capture as _capture

        _capture.record_stage(
            "tool", name, {"name": name, "args": args, "result": result}
        )

    def build_response_mask(self, response_ids: list[int]) -> list[int]:
        return [1] * len(response_ids)

    def decode_response_ids(self, response_ids: list[int]) -> str:
        """Decode model-generated response ids when the tokenizer supports it."""
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return ""

    async def _generate_text(
        self,
        messages: list[dict[str, Any]],
        *,
        metrics: dict[str, float],
        metric_name: str,
        request_id: str,
        sampling_params: dict[str, Any],
        on_token: "OnTokenCallback | None" = None,
    ) -> tuple[list[int], list[int], str]:
        with simple_timer(metric_name, metrics):
            prompt_ids = await self.build_prompt_ids(messages)
            response_ids = await self.generate_response_ids(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                on_token=on_token,
            )
        return prompt_ids, response_ids, self.decode_response_ids(response_ids)

    async def generate_rollout_step(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        *,
        action_re: re.Pattern[str] = _DEFAULT_ACTION_RE,
        terminal_actions: frozenset[str] = _DEFAULT_TERMINAL_ACTIONS,
        request_id: str | None = None,
    ) -> RolloutStep:
        """Sample one policy action and classify it for the RL loop.

        Analogous to actor_rollout_wg.generate_sequences — the per-step call that
        drives the trajectory:
            state (prompt_ids) → generate → classify action → RolloutStep

        The caller decides whether to continue (is_terminal=False) or stop
        (is_terminal=True) based on the returned action_type.
        """
        response_ids = await self.generate_response_ids(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=request_id,
        )
        response_text = self.decode_response_ids(response_ids)
        match = action_re.search(response_text)
        action_type = match.group(1) if match else None
        action_content = match.group(2).strip() if match else ""
        return RolloutStep(
            prompt_ids=list(prompt_ids),
            response_ids=list(response_ids),
            response_mask=self.build_response_mask(response_ids),
            action_type=action_type,
            action_content=action_content,
            is_terminal=action_type in terminal_actions if action_type else True,
        )

    async def run(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        *,
        on_turn: "OnTurnCallback | None" = None,
    ) -> AgentLoopOutput:
        raise NotImplementedError
