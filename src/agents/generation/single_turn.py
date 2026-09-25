"""One-shot retrieval-assisted generation loop.

``SingleTurnAgentLoop`` is a compact RAG loop for cases that need at most one
retrieval step. It supports two flows:

- model-directed: generate ``<search>query</search>`` or ``<answer>text</answer>``;
  search only when the model requests it, then generate the final answer.
- forced RAG: retrieve using the user question first, inject ``<information>``,
  then generate one answer.

Use ``PlainGenerationLoop`` for no-retrieval smoke tests, and ``SearchAgentLoop``
for multi-turn XML search trajectories used by RL/SFT workflows.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from src.agents.core.base import (
    AgentLoopBase,
    AgentLoopConfig,
    AgentLoopOutput,
    OnTurnCallback,
    register,
    simple_timer,
)
from src.context.search import AgentContext, citation_prefix
from src.context.retrieval.client import SearchClient, SearchClientConfig
from src.internal.configs.timeouts import get_timeout_policies

logger = logging.getLogger(__name__)
logger.setLevel(
    os.getenv("AGENTIC_SEARCH_LOG_LEVEL", os.getenv("VERL_LOGGING_LEVEL", "WARN"))
)


def _client_policy():
    return get_timeout_policies().retrieval.client


# Only <search> and <answer> are valid in the one-shot tool flow.
# <plan>, <fetch>, <searches> etc. belong to the multi-turn ReAct path.
_ACTION_RE: re.Pattern[str] = re.compile(
    r"<(?P<tag>search|answer)>(?P<content>.*?)</(?P=tag)>",
    re.DOTALL,
)


def _parse_first_action(text: str) -> tuple[str, str] | None:
    """Return (tag, content) for the first <search> or <answer> tag, or None."""
    m = _ACTION_RE.search(text)
    return (m.group("tag"), m.group("content").strip()) if m else None


def _extract_answer_text(text: str) -> str | None:
    """Return the content of the first <answer>…</answer> tag, or None."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None


@dataclass(frozen=True)
class SingleTurnAgentLoopConfig(AgentLoopConfig):
    """Configuration for the one-shot retrieve-then-answer agent.

    Two modes
    ---------
    ``force_search=False`` (default — tool-augmented):
        The model generates first.  If it emits ``<search>query</search>``
        and ``use_retrieval=True``, one retrieval call is made and the model
        generates its final ``<answer>``.  If the model emits ``<answer>``
        directly, no retrieval happens.

    ``force_search=True`` (pre-retrieval RAG):
        Retrieves unconditionally from the user query *before* first
        generation.  The model receives the evidence in its prompt and
        generates one response.  This is the classic RAG pattern and the
        original behaviour of this class.
    """

    search_url: str = "http://localhost:8000/retrieve"
    topk: int = 5
    search_timeout_seconds: float = field(
        default_factory=lambda: _client_policy().timeout_seconds
    )
    search_max_retries: int = field(
        default_factory=lambda: _client_policy().max_retries
    )
    fetch_url: str | None = None
    use_retrieval: bool = True
    # True → classic pre-retrieval RAG (retrieve before first generation).
    # False (default) → tool-augmented: model decides to search via <search> tag.
    force_search: bool = False
    # Accept a direct <answer> with no preceding <search> call.
    allow_direct_answer: bool = True
    system_prompt: str | None = (
        "You are a retrieval-augmented assistant. "
        "Use the provided <information> evidence when answering the question."
    )
    evidence_intro: str = (
        "Retrieved evidence for the question is below.\n"
        "Use it if helpful, and answer in one response.\n"
    )
    obs_template: str = "\n\n<information>\n{content}\n</information>\n\n"


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """One-shot retrieval-assisted generation.

    Default (tool-augmented one-shot)::

        user: "Who won the Nobel Prize in Physics 2024?"
        model: <search>Nobel Prize Physics 2024</search>
        → retrieve → inject <information>…</information>
        model: <answer>Hopfield and Hinton.</answer>

    Classic RAG mode (``force_search=True``)::

        user: "Who won the Nobel Prize in Physics 2024?"
        → retrieve unconditionally
        → inject evidence into prompt
        model: "Hopfield and Hinton."

    For CLI ``--mode single``, use ``PlainGenerationLoop`` through
    ``examples.run_agentic_search``. For project-native search-agent trajectories,
    use ``SearchAgentLoop``.
    """

    def __init__(
        self,
        tokenizer: Any,
        server_manager: Any,
        config: AgentLoopConfig | SingleTurnAgentLoopConfig | None = None,
        loop: Any | None = None,
    ) -> None:
        cfg = config or SingleTurnAgentLoopConfig()
        if not isinstance(cfg, SingleTurnAgentLoopConfig):
            cfg = SingleTurnAgentLoopConfig(
                prompt_length=cfg.prompt_length,
                response_length=cfg.response_length,
            )
        super().__init__(
            tokenizer=tokenizer,
            server_manager=server_manager,
            config=AgentLoopConfig(
                prompt_length=cfg.prompt_length,
                response_length=cfg.response_length,
            ),
            loop=loop,
        )
        self.single_turn_config = cfg
        self._search_client: SearchClient | None = None

    # ── private helpers ──────────────────────────────────────────────────────

    def _maybe_prepend_system(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        cfg = self.single_turn_config
        if not cfg.system_prompt:
            return messages
        msgs = list(messages)
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {
                "role": "system",
                "content": f"{cfg.system_prompt}\n\n{msgs[0].get('content', '')}".strip(),
            }
        else:
            msgs.insert(0, {"role": "system", "content": cfg.system_prompt})
        return msgs

    def _extract_query(self, messages: list[dict[str, Any]]) -> str:
        """Return the last user message content (used in force_search mode)."""
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", "")).strip()
        return str(messages[-1].get("content", "")).strip() if messages else ""

    def _get_search_client(self) -> SearchClient:
        if self._search_client is None:
            cfg = self.single_turn_config
            self._search_client = SearchClient(
                SearchClientConfig(
                    url=cfg.search_url,
                    topk=cfg.topk,
                    timeout_seconds=cfg.search_timeout_seconds,
                    max_retries=cfg.search_max_retries,
                    fetch_url=cfg.fetch_url,
                )
            )
        return self._search_client

    async def _retrieve(self, query: str) -> list:
        """Call the retrieval server; returns a flat list of SearchResult."""
        search_client = self._get_search_client()
        retrieve_one = getattr(search_client, "retrieve_one", None)
        if callable(retrieve_one):
            return await retrieve_one(query, topk=self.single_turn_config.topk)
        retrieved = await search_client.retrieve(
            [query],
            topk=self.single_turn_config.topk,
        )
        return retrieved[0] if retrieved else []

    def _build_force_search_messages(
        self,
        prompt_messages: list[dict[str, Any]],
        agent_ctx: AgentContext,
    ) -> list[dict[str, Any]]:
        """Pre-retrieval RAG: append evidence as a user message (force_search mode)."""
        cfg = self.single_turn_config
        if not agent_ctx.turns:
            return prompt_messages
        evidence = agent_ctx.turns[0].to_information_block(
            citation_prefix=citation_prefix(1, 1)
        )
        obs = cfg.obs_template.format(content=evidence).strip()
        return prompt_messages + [
            {"role": "user", "content": f"{cfg.evidence_intro}{obs}"}
        ]

    def _build_tool_obs_messages(
        self,
        prompt_messages: list[dict[str, Any]],
        search_action_text: str,
        agent_ctx: AgentContext,
    ) -> list[dict[str, Any]]:
        """Tool-augmented: append the search action + observation as chat turns."""
        cfg = self.single_turn_config
        evidence = (
            agent_ctx.turns[0].to_information_block(
                citation_prefix=citation_prefix(1, 1)
            )
            if agent_ctx.turns
            else ""
        )
        obs = cfg.obs_template.format(content=evidence).strip()
        return prompt_messages + [
            {"role": "assistant", "content": search_action_text},
            {"role": "user", "content": obs},
        ]

    def _make_output(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        response_ids: list[int],
        num_turns: int,
        metrics: dict[str, float],
        agent_ctx: AgentContext,
        trajectory_messages: list[dict[str, Any]],
        final_answer: str | None,
    ) -> AgentLoopOutput:
        metrics.setdefault("search_rounds", 0.0)
        metrics["rounds_used"] = metrics["search_rounds"]
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=self.build_response_mask(response_ids),
            num_turns=num_turns,
            metrics=metrics,
            request_id=request_id,
            context=agent_ctx,
            trajectory_messages=trajectory_messages,
            action_trace=final_answer,
            final_answer=final_answer,
        )

    # ── main entry point ─────────────────────────────────────────────────────

    async def run(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        *,
        on_turn: "OnTurnCallback | None" = None,
    ) -> AgentLoopOutput:
        metrics: dict[str, float] = {}
        request_id = uuid4().hex
        agent_ctx = AgentContext()
        cfg = self.single_turn_config

        prompt_messages = self._maybe_prepend_system(list(messages))

        try:
            if cfg.force_search:
                return await self._run_force_search(
                    prompt_messages=prompt_messages,
                    original_messages=messages,
                    agent_ctx=agent_ctx,
                    metrics=metrics,
                    request_id=request_id,
                    sampling_params=sampling_params,
                )
            return await self._run_tool_augmented(
                prompt_messages=prompt_messages,
                agent_ctx=agent_ctx,
                metrics=metrics,
                request_id=request_id,
                sampling_params=sampling_params,
            )
        finally:
            close = getattr(self._search_client, "aclose", None)
            if callable(close):
                await close()

    # ── mode implementations ─────────────────────────────────────────────────

    async def _run_force_search(
        self,
        *,
        prompt_messages: list[dict[str, Any]],
        original_messages: list[dict[str, Any]],
        agent_ctx: AgentContext,
        metrics: dict[str, float],
        request_id: str,
        sampling_params: dict[str, Any],
    ) -> AgentLoopOutput:
        """Pre-retrieval RAG: retrieve → inject → generate (classic mode)."""
        cfg = self.single_turn_config

        if cfg.use_retrieval:
            query = self._extract_query(original_messages)
            if query:
                with simple_timer("retrieve", metrics):
                    results = await self._retrieve(query)
                agent_ctx.add_turn(query, results)
                metrics["search_rounds"] = 1.0
                prompt_messages = self._build_force_search_messages(
                    prompt_messages, agent_ctx
                )

        prompt_ids, response_ids, response_text = await self._generate_text(
            prompt_messages,
            metrics=metrics,
            metric_name="generate_sequences",
            request_id=request_id,
            sampling_params=sampling_params,
        )
        final_answer = _extract_answer_text(response_text) or response_text
        trajectory = prompt_messages + [{"role": "assistant", "content": response_text}]

        return self._make_output(
            request_id=request_id,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            num_turns=1,
            metrics=metrics,
            agent_ctx=agent_ctx,
            trajectory_messages=trajectory,
            final_answer=final_answer,
        )

    async def _run_tool_augmented(
        self,
        *,
        prompt_messages: list[dict[str, Any]],
        agent_ctx: AgentContext,
        metrics: dict[str, float],
        request_id: str,
        sampling_params: dict[str, Any],
    ) -> AgentLoopOutput:
        """Tool-augmented one-shot: generate → maybe <search> → <answer>."""
        cfg = self.single_turn_config

        # Step 1: first generation — model decides to search or answer directly.
        prompt_ids, first_ids, first_text = await self._generate_text(
            prompt_messages,
            metrics=metrics,
            metric_name="generate_sequences",
            request_id=request_id,
            sampling_params=sampling_params,
        )
        action = _parse_first_action(first_text)

        if action and action[0] == "search" and cfg.use_retrieval:
            # Step 2: model chose to search — retrieve and inject observation.
            query = action[1]
            with simple_timer("retrieve", metrics):
                results = await self._retrieve(query)
            agent_ctx.add_turn(query, results)
            metrics["search_rounds"] = 1.0

            # Step 3: inject observation and generate the final <answer>.
            obs_messages = self._build_tool_obs_messages(
                prompt_messages, first_text, agent_ctx
            )
            _, final_ids, final_text = await self._generate_text(
                obs_messages,
                metrics=metrics,
                metric_name="generate_sequences_2",
                request_id=request_id,
                sampling_params=sampling_params,
            )
            final_answer = _extract_answer_text(final_text) or final_text
            trajectory = obs_messages + [{"role": "assistant", "content": final_text}]
            return self._make_output(
                request_id=request_id,
                prompt_ids=prompt_ids,
                response_ids=final_ids,
                num_turns=2,
                metrics=metrics,
                agent_ctx=agent_ctx,
                trajectory_messages=trajectory,
                final_answer=final_answer,
            )

        # Model answered directly (or search disabled / no search tag found).
        final_answer = _extract_answer_text(first_text) or first_text
        trajectory = prompt_messages + [{"role": "assistant", "content": first_text}]
        return self._make_output(
            request_id=request_id,
            prompt_ids=prompt_ids,
            response_ids=first_ids,
            num_turns=1,
            metrics=metrics,
            agent_ctx=agent_ctx,
            trajectory_messages=trajectory,
            final_answer=final_answer,
        )
