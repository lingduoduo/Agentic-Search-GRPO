"""Generic multi-turn function/tool-calling agent loop.

The agent generates a response, the ToolParser extracts function calls from it,
the tools are executed in parallel, and their results are injected back as
``{"role": "tool", ...}`` messages before the next generation step.

This loop is intentionally not tied to search-agent XML actions. Search may be
registered as one function tool, but the same loop can call calculators,
databases, file tools, or any other JSON-schema-described function.

Supported tool-call formats are controlled by ``ToolAgentLoopConfig.tool_parser_format``:
    - ``"hermes"``  — NousResearch Hermes 2.5 / 3
    - ``"llama3"``  — Meta Llama 3.1 / 3.2
    - ``"json"``    — generic JSON fallback

Usage::

    from src import ToolAgentLoop, ToolAgentLoopConfig
    from src import FunctionTool

    @FunctionTool.from_fn(description="Search the web", parameters={...})
    async def search(query: str) -> str:
        ...

    loop = ToolAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        tools=[search],
        config=ToolAgentLoopConfig(tool_parser_format="json"),
    )
    output = await loop.run(messages, sampling_params)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Awaitable, Callable
from uuid import uuid4

from src.agents.core.base import (
    AgentLoopBase,
    AgentLoopConfig,
    AgentLoopOutput,
    OnTurnCallback,
    _crop_prompt_ids,
    register,
    simple_timer,
)
from src.agents.core.state import PerformanceMetrics, TaskStatus, ToolExecutionResult
from src.internal.tools.base import Tool, ToolEffect
from src.internal.tools.parsers import FunctionCall, ToolParser
from src.internal.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("AGENTIC_SEARCH_LOG_LEVEL", "WARN"))


def _as_token_ids(templated: Any) -> list[int]:
    """Normalise an ``apply_chat_template(tokenize=True)`` result to token ids.

    transformers 5.x returns a ``BatchEncoding`` (a Mapping) where 4.x returned
    a flat list of ints. Iterating the Mapping yields its keys, so without this
    the loop passes ``["input_ids", "attention_mask"]`` to the model backend.
    """
    if isinstance(templated, Mapping):
        return list(templated["input_ids"])
    return list(templated)


class ApprovalDecision(str, Enum):
    APPROVE = "approve"
    DENY = "deny"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class ToolApprovalRequest:
    approval_id: str
    tool_name: str
    arguments: dict[str, Any]
    created_at: datetime
    expires_at: datetime


ToolApprovalCallback = Callable[[ToolApprovalRequest], Awaitable[ApprovalDecision]]


def _truncation_footer(shown: int, total: int, omitted: int) -> str:
    """Tell the model what it is not seeing, in its own message."""
    return f"\n...{shown} of {total} results shown, {omitted} omitted for length."


def _fit_json_array(text: str, limit: int) -> str | None:
    """Trim a JSON array to the leading items that fit within *limit*.

    Ranked tool results are ordered best-first, so dropping items from the end
    keeps what matters and leaves the model valid JSON rather than a fragment
    that starts mid-object.

    Returns None when *text* is not a non-empty JSON list, or when not even one
    item fits. The caller then falls back to character slicing: a readable
    prefix of one large item beats a valid but empty array.
    """
    try:
        items = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if not isinstance(items, list) or not items:
        return None

    # ensure_ascii=False throughout: bare json.dumps would re-escape every
    # non-ASCII character to \uXXXX (one character becomes six), which both
    # inflates the size accounting below and defeats the purpose of trimming
    # for content that already arrived as real UTF-8 text.
    compact = json.dumps(items, ensure_ascii=False)
    if len(compact) <= limit:
        # Re-serializing without the original's whitespace was enough. This
        # must be checked before the loop below: that loop charges every
        # candidate the cost of a footer, so a small indented array could
        # otherwise be rejected outright even though all of it fits.
        return compact

    total = len(items)
    kept: list[Any] = []
    for item in items:
        candidate = kept + [item]
        footer = _truncation_footer(len(candidate), total, total - len(candidate))
        if len(json.dumps(candidate, ensure_ascii=False) + footer) > limit:
            break
        kept = candidate

    if not kept:
        return None
    return json.dumps(kept, ensure_ascii=False) + _truncation_footer(
        len(kept), total, total - len(kept)
    )


def _slice_text(text: str, limit: int, side: str) -> str:
    """Character-slice *text*, keeping the side named by the config.

    The marker is appended outside the slice, so the result can exceed
    *limit* by its length (up to 17 chars for "middle") — unlike the JSON
    path above, which budgets its footer inside the limit.
    """
    if side == "left":
        return text[:limit] + "...(truncated)"
    if side == "right":
        return "(truncated)..." + text[-limit:]
    half = limit // 2
    return text[:half] + "...(truncated)..." + text[-half:]


def _truncate_tool_text(text: str, limit: int, side: str) -> str:
    """Bound one tool response, preferring whole JSON items over a raw slice."""
    if len(text) <= limit:
        return text
    fitted = _fit_json_array(text, limit)
    if fitted is not None:
        return fitted
    return _slice_text(text, limit, side)


@dataclass(frozen=True)
class ToolAgentLoopConfig(AgentLoopConfig):
    """Configuration for ToolAgentLoop.

    Inherits ``prompt_length`` and ``response_length`` from AgentLoopConfig.
    """

    max_user_turns: int = 10
    max_assistant_turns: int = 10
    max_parallel_calls: int = 4
    max_tool_response_length: int = 2048
    # Fallback policy for a tool response that exceeds
    # max_tool_response_length and is NOT a JSON array (arrays are trimmed by
    # whole items instead — see _fit_json_array). Defaults to keeping the
    # start: tool results are ranked best-first, so dropping the tail loses
    # the least.
    #   "left"   — keep the start, append "...(truncated)"
    #   "right"  — prepend "(truncated)...", keep the end
    #   "middle" — keep equal halves from start and end
    tool_response_truncate_side: str = "left"
    tool_parser_format: str = "json"
    approval_timeout_seconds: float = 60.0


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    """Multi-turn agent loop that executes generic tool calls from the model.

    Each iteration:
      1. Tokenise the current message history (including tool schemas).
      2. Generate a response.
      3. Parse tool calls from the response with the configured ToolParser.
      4. If tool calls are found: execute them in parallel, append the results
         as ``{"role": "tool"}`` messages, and continue.
      5. Stop when no tool calls are found, a turn limit is reached, or the
         response budget is exhausted.

    ``response_mask`` is 1 for model-generated tokens and 0 for tool-response
    tokens injected back into the prompt — matching the VERL rollout convention.
    """

    def __init__(
        self,
        tokenizer: Any,
        server_manager: Any,
        tools: list[Tool] | None = None,
        config: ToolAgentLoopConfig | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        cfg = config or ToolAgentLoopConfig()
        super().__init__(
            tokenizer=tokenizer,
            server_manager=server_manager,
            config=cfg,
            loop=loop,
        )
        self.tool_config = cfg
        _tools = list(tools or [])
        # Per-loop registry: the single tool lookup + execution path (invoke).
        self._registry = ToolRegistry()
        for _t in _tools:
            self._registry.register(_t)
        self.tool_schemas: list[dict[str, Any]] = [
            t.schema.to_dict() for t in self._registry.list_tools()
        ]
        self.tool_parser: ToolParser = ToolParser.get_tool_parser(
            cfg.tool_parser_format, tokenizer
        )
        # Baseline token count produced by apply_chat_template for an empty
        # conversation — used to strip the template prefix when re-tokenising
        # tool responses so they don't double-count the system prompt.
        self._template_prefix_len: int = self._measure_template_prefix()

    def _measure_template_prefix(self) -> int:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return 0
        try:
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                add_generation_prompt=False,
                tokenize=True,
            )
            return len(_as_token_ids(ids))
        except Exception as exc:
            logger.debug("Could not measure template prefix length: %s", exc)
            return 0

    def _build_prompt_ids_with_tools_sync(
        self, messages: list[dict[str, Any]]
    ) -> list[int]:
        """Like _build_prompt_ids_sync but injects tool schemas into the template."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            ids = self.tokenizer.apply_chat_template(
                messages,
                tools=self.tool_schemas or None,
                add_generation_prompt=True,
                tokenize=True,
            )
            # Crop like the base loop does, preserving the system prefix. A bare
            # tail slice drops the system prompt and the tool schemas first --
            # the tokens the model most needs in order to keep calling tools.
            return _crop_prompt_ids(
                _as_token_ids(ids),
                self._encode_system_prefix(messages),
                self.prompt_length,
            )
        # Fallback: no tool schema injection
        return self._build_prompt_ids_sync(messages)

    def _truncate_tool_response(self, text: str) -> str:
        return _truncate_tool_text(
            text,
            self.tool_config.max_tool_response_length,
            self.tool_config.tool_response_truncate_side,
        )

    async def _call_tool(self, tool_call: FunctionCall) -> ToolExecutionResult:
        """Execute one tool call via the per-loop registry; return a structured result.

        ``ToolRegistry.invoke`` is the single execution path — it looks up the
        tool, validates arguments against the schema, and runs the
        create/execute/release lifecycle. This adapts its ``(response, raw,
        errors)`` tuple into the loop's ``ToolExecutionResult``.
        """
        start = time.perf_counter()
        args = tool_call.parsed_arguments()
        status = TaskStatus.FAILED
        result: Any = None
        error_code: str | None = None
        error_message: str | None = None
        try:
            response, _raw, errors = await self._registry.invoke(tool_call.name, args)
            if errors:
                error_code = (
                    "tool_not_found"
                    if self._registry.get(tool_call.name) is None
                    else "invalid_arguments"
                )
                error_message = "; ".join(errors)
            else:
                result = response
                status = TaskStatus.COMPLETED
                self._record_tool_stage(tool_call.name, args, result)
        except Exception as exc:
            logger.exception("Error executing tool %r: %s", tool_call.name, exc)
            error_code = type(exc).__name__
            error_message = str(exc)
        elapsed = time.perf_counter() - start
        return ToolExecutionResult(
            tool_name=tool_call.name,
            status=status,
            result=result,
            arguments=args,
            performance=PerformanceMetrics(
                execution_time=elapsed,
                success_rate=1.0 if status is TaskStatus.COMPLETED else 0.0,
            ),
            error_code=error_code,
            error_message=error_message,
        )

    async def _request_approval(
        self,
        tool_call: FunctionCall,
        on_approval: ToolApprovalCallback | None,
        metrics: dict[str, float],
    ) -> ApprovalDecision:
        tool = self._registry.get(tool_call.name)
        if tool is not None and tool.effect is ToolEffect.READ_ONLY:
            return ApprovalDecision.APPROVE

        metrics["tool_approvals_requested"] += 1
        if on_approval is None:
            metrics["tool_approvals_denied"] += 1
            return ApprovalDecision.DENY

        created_at = datetime.now(timezone.utc)
        request = ToolApprovalRequest(
            approval_id=uuid4().hex,
            tool_name=tool_call.name,
            arguments=tool_call.parsed_arguments(),
            created_at=created_at,
            expires_at=created_at
            + timedelta(seconds=self.tool_config.approval_timeout_seconds),
        )
        try:
            decision = await asyncio.wait_for(
                on_approval(request), timeout=self.tool_config.approval_timeout_seconds
            )
        except asyncio.TimeoutError:
            metrics["tool_approvals_expired"] += 1
            return ApprovalDecision.EXPIRED
        except asyncio.CancelledError:
            metrics["tool_approvals_cancelled"] += 1
            raise
        except Exception:
            logger.exception("Approval callback failed for tool %r", tool_call.name)
            metrics["tool_approval_errors"] += 1
            metrics["tool_approvals_denied"] += 1
            return ApprovalDecision.DENY

        metrics[
            "tool_approvals_approved"
            if decision is ApprovalDecision.APPROVE
            else "tool_approvals_expired"
            if decision is ApprovalDecision.EXPIRED
            else "tool_approvals_denied"
        ] += 1
        return decision

    @staticmethod
    def _skipped_tool_result(
        tool_call: FunctionCall, decision: ApprovalDecision
    ) -> ToolExecutionResult:
        error_code = (
            "approval_expired"
            if decision is ApprovalDecision.EXPIRED
            else "approval_denied"
        )
        return ToolExecutionResult(
            tool_name=tool_call.name,
            status=TaskStatus.SKIPPED,
            result=None,
            arguments=tool_call.parsed_arguments(),
            performance=PerformanceMetrics(execution_time=0.0, success_rate=0.0),
            error_code=error_code,
            error_message="Tool execution skipped because approval was not granted.",
        )

    @staticmethod
    def _tool_message_content(result: ToolExecutionResult) -> str:
        """Serialize a tool result into the content of a role:"tool" message.

        FAILED results are fed back to the model (with the error) so it can
        self-correct, rather than aborting the run. COMPLETED and SKIPPED
        formats are unchanged.
        """
        if result.status is TaskStatus.COMPLETED:
            return str(result.result)
        if result.status is TaskStatus.SKIPPED:
            return json.dumps({"status": "skipped", "error_code": result.error_code})
        payload = {"status": "failed", "error_code": result.error_code}
        if result.error_message:
            payload["error_message"] = result.error_message
        return json.dumps(payload)

    async def run(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        *,
        on_turn: "OnTurnCallback | None" = None,
        on_approval: ToolApprovalCallback | None = None,
    ) -> AgentLoopOutput:
        metrics: dict[str, float] = {
            "tool_approvals_requested": 0,
            "tool_approvals_approved": 0,
            "tool_approvals_denied": 0,
            "tool_approvals_expired": 0,
            "tool_approvals_cancelled": 0,
            "tool_approval_errors": 0,
        }
        request_id = uuid4().hex
        event_loop = await self.get_loop()
        prompt_ids: list[int] = await event_loop.run_in_executor(
            None,
            lambda: self._build_prompt_ids_with_tools_sync(messages),
        )
        response_mask: list[int] = []
        working_messages: list[dict[str, Any]] = list(messages)
        tool_results: list[ToolExecutionResult] = []
        user_turns = 0
        assistant_turns = 0
        final_answer: str | None = None

        while True:
            # ── generate ─────────────────────────────────────────────────
            with simple_timer("generate_sequences", metrics):
                response_ids = await self.generate_response_ids(
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    request_id=f"{request_id}_{assistant_turns}",
                )

            prompt_ids = prompt_ids + response_ids
            response_mask.extend([1] * len(response_ids))
            assistant_turns += 1

            # ── parse tool calls ──────────────────────────────────────────
            # Record the assistant answer BEFORE the stopping checks so a
            # turn/length cap firing this turn still surfaces the model's
            # generated response in final_answer / trajectory_messages.
            assistant_content, tool_calls = await self.tool_parser.extract_tool_calls(
                response_ids
            )
            working_messages.append({"role": "assistant", "content": assistant_content})
            final_answer = assistant_content

            # ── stopping conditions ───────────────────────────────────────
            if len(response_mask) >= self.response_length:
                break
            if (
                self.tool_config.max_assistant_turns
                and assistant_turns >= self.tool_config.max_assistant_turns
            ):
                break
            if (
                self.tool_config.max_user_turns
                and user_turns >= self.tool_config.max_user_turns
            ):
                break

            if not tool_calls:
                break

            # Resolve the whole batch before any tool may execute.
            truncated_calls = tool_calls[: self.tool_config.max_parallel_calls]
            decisions = await asyncio.gather(
                *[
                    self._request_approval(tc, on_approval, metrics)
                    for tc in truncated_calls
                ]
            )

            # ── execute approved tools in parallel ────────────────────────
            with simple_timer("tool_calls", metrics):
                tool_execution_results = await asyncio.gather(
                    *[
                        self._call_tool(tc)
                        if decision is ApprovalDecision.APPROVE
                        else asyncio.sleep(
                            0, result=self._skipped_tool_result(tc, decision)
                        )
                        for tc, decision in zip(truncated_calls, decisions)
                    ]
                )
            tool_results.extend(tool_execution_results)
            if on_turn is not None:
                for r in tool_execution_results:
                    await on_turn(assistant_turns, r.tool_name, 0)

            tool_responses = [
                {
                    "role": "tool",
                    "content": self._truncate_tool_response(
                        self._tool_message_content(result)
                    ),
                }
                for result in tool_execution_results
            ]
            working_messages.extend(tool_responses)

            # ── re-tokenise tool responses and append ─────────────────────
            tool_response_ids: list[int] = await event_loop.run_in_executor(
                None,
                lambda: _as_token_ids(
                    self.tokenizer.apply_chat_template(
                        list(tool_responses), add_generation_prompt=True, tokenize=True
                    )
                ),
            )
            # Strip the template prefix to avoid re-including the system prompt tokens.
            tool_response_ids = tool_response_ids[self._template_prefix_len :]

            if len(response_mask) + len(tool_response_ids) >= self.response_length:
                break
            # Same guard for the prompt budget. This is the only path back to
            # generate(), so checking the next prompt's size here is what keeps
            # the whole run within prompt_length -- the initial build is cropped,
            # but from then on prompt_ids only ever grows. Stopping rather than
            # cropping is deliberate: the teardown below recovers the
            # prompt/response split from len(prompt_ids) - len(response_mask),
            # so dropping tokens the mask already covers would desync the two.
            # The tool results are already in working_messages, so the
            # transcript keeps them either way.
            if len(prompt_ids) + len(tool_response_ids) > self.prompt_length:
                break

            prompt_ids = prompt_ids + tool_response_ids
            response_mask.extend([0] * len(tool_response_ids))
            user_turns += 1

        # Split accumulated prompt_ids back into prompt / response portions.
        n = len(prompt_ids) - len(response_mask)
        final_prompt_ids = prompt_ids[:n]
        final_response_ids = prompt_ids[n:]

        return AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=final_response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
            request_id=request_id,
            trajectory_messages=working_messages,
            action_trace="\n".join(
                json.dumps(r.to_dict(), default=str) for r in tool_results
            )
            or None,
            final_answer=final_answer,
            truncated=self.generation_truncated,
        )
