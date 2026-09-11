"""FastAPI app for a browser-based search and agent experience."""

from __future__ import annotations

import asyncio
import json as _json
import logging
import uuid as _uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.internal.auth import AuthenticatedUser
from src.internal.servers._auth import caller_may_use_session, make_require_admin
from src.internal.configs import AppSettings
from src.internal.configs import load_app_settings
from src.internal.llm.interfaces import LLMConfig
from src.internal.llm.providers import OpenAICompatibleLLM
from src.internal.utils.embedding_gate import (
    gate_embedder,
    make_cosine_fn,
    search_direct_cos_min,
)
from src.internal.utils.text_processing import levenshtein_lt2, normalize_for_match
from src.internal.search.process_search_query import run_expanded_search
from src.internal.servers.secondary_llm_flows import expand_keywords
from src.internal.servers.secondary_llm_flows.query_expansion import (
    with_temporal_context,
)
from src.agents.search import AgenticRAGConfig, AgenticRAGLoop
from src.agents.core.base import OnTurnCallback
from src.agents.core.control_flow_trace import (
    ControlFlowEvent,
    ControlFlowRecorder,
    EventSink,
)
from src.context import LLMClient
from src.context import answer_with_retrieval
from src.context import build_context_bundle
from src.context.models import AnswerGenerationResult
from src.context.models import ContextDocument
from src.context.models import SearchFilters
from src.context.search import SearchResult, citation_key
from src.internal.access.capabilities import resolve_capabilities
from src.internal.db import AgenticSearchStore
from src.internal.hooks import HookPoint
from src.internal.hooks import HookRegistry
from src.internal.hooks import HookSoftFailed
from src.internal.hooks import execute_hook
from src.internal.search.models import CandidateSet
from src.internal.search.models import GeneratedAnswer
from src.internal.search.models import RankedEvidence
from src.internal.search.pipeline import SearchPipeline
from src.internal.search.ranking import DefaultRankingStage
from src.internal.search.stages import RerankHTTPRankingStage
from src.internal.servers.admin_surface.api import create_admin_surface_router
from src.internal.servers.analytics.api import create_analytics_router
from src.internal.servers.web import request_capture as _capture
from src.internal.servers.web.auth_check import PUBLIC_ENDPOINT_SPECS
from src.internal.servers.web.auth_check import check_router_auth
from src.internal.servers.billing.api import create_billing_router
from src.internal.servers.tools.api import create_tools_router
from src.internal.servers.enterprise_settings.api import (
    create_enterprise_settings_routers,
)
from src.internal.servers.evals.api import create_evals_router
from src.internal.servers.features.hooks.api import create_hooks_router
from src.internal.servers.license.api import create_license_router
from src.internal.servers.manage.standard_answer import create_manage_router
from src.internal.servers.middleware.latency_logging import (
    add_latency_logging_middleware,
)
from src.internal.servers.middleware.license_enforcement import (
    add_license_enforcement_middleware,
)
from src.internal.servers.middleware.tenant_tracking import (
    add_api_server_tenant_id_middleware,
)
from src.internal.servers.middleware.tier_gate import add_tier_gate_middleware
from src.internal.servers.oauth.api import create_oauth_router
from src.internal.servers.query_and_chat.chat_backend import create_chat_router
from src.internal.servers.query_and_chat.query_backend import (
    basic_router as query_basic_router,
)
from src.internal.servers.query_and_chat.search_backend import create_search_router
from src.internal.servers.query_history.api import create_query_history_router
from src.internal.servers.reporting.api import create_reporting_router
from src.internal.servers.scim.api import create_scim_router
from src.internal.servers.scim.api import register_scim_exception_handlers
from src.internal.servers.web.seeding import seed_db
from src.internal.tools.knowledge_base import seed_tools, tool_knowledge_base
from src.internal.tools.registry import tool_registry
from src.internal.servers.web.tool_approval import (
    ApprovalConflict,
    ApprovalExpired,
    ApprovalForbidden,
    ApprovalNotFound,
    ToolApprovalBroker,
)
from src.internal.servers.settings.api import create_settings_router
from src.internal.servers.tenants.api import router as tenants_router
from src.internal.servers.token_rate_limits.api import create_token_rate_limits_router
from src.internal.servers.user_group.api import create_user_group_router
from src.internal.servers.users.api import create_users_router
from src.internal.servers.users.api import _require_auth
from src.internal.servers.users.api import resolve_active_user
from src.internal.tools import SearchPage
from src.internal.tools import fetch_pages_concurrently
from src.internal.tools import search_tool

from .static import APP_CSS
from .static import APP_HTML
from .static import APP_JS
from .intent import (
    RouteStrategy,
    recognize_intent,
)
from .tool_agent_runner import (
    NO_LOCAL_MODEL_MESSAGE,
    ToolCallView,
    _run_tool_agent,
)
from src.internal.cache.interface import get_cache_backend
from src.internal.cache.serving import configure_serving_cache, reset_serving_cache
from src.internal.memory.working import (
    MAX_HISTORY_MESSAGES,
    load_working_memory,
    schedule_compression,
)
from src.internal.observability import stage_metrics as _stage_metrics
from src.internal.observability.stage_metrics import STAGE_LATENCY

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchExperienceSettings:
    """Runtime settings for the browser search app."""

    search_url: str = "http://localhost:8001/retrieve"
    top_k: int = 5
    db_path: str | Path = ":memory:"
    browser_search_url: str | None = None
    rerank_url: str | None = None
    # When False (default), the retrieval URL is resolved server-side and any
    # client-supplied search_url is ignored — this closes the SSRF hole where a
    # client could point the backend at an arbitrary URL. Enable only for local
    # dev/debugging via AGENTIC_SEARCH_ALLOW_CLIENT_RETRIEVAL_URL=true.
    allow_client_search_url: bool = False
    # Dev-only observability console. Off by default; never enable in prod.
    debug_panels: bool = False
    # Refuse anonymous /api/memory callers instead of pooling them into the
    # shared default_user bucket. Off by default: the CLI is documented as
    # working unauthenticated for local research use.
    memory_require_auth: bool = False
    # Summarize turns that fall off the history tail into a system message the
    # next turn sees, using the configured LLM client. Off by default; with no
    # LLM client the flag is inert.
    memory_compression: bool = False
    # Seconds a retrieval row, web-provider page or rerank score stays in the
    # process-local serving cache. 0 disables it. The lifespan configures it.
    search_cache_ttl: int = 300

    @classmethod
    def from_app_settings(
        cls,
        settings: AppSettings | None = None,
    ) -> "SearchExperienceSettings":
        import os

        def _flag(name: str) -> bool:
            return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}

        app_settings = settings or load_app_settings()
        return cls(
            search_url=app_settings.services.retrieval_url,
            rerank_url=app_settings.services.rerank_url,
            top_k=app_settings.services.web_top_k,
            db_path=app_settings.services.web_db_path,
            allow_client_search_url=_flag("AGENTIC_SEARCH_ALLOW_CLIENT_RETRIEVAL_URL"),
            debug_panels=_flag("AGENTIC_SEARCH_DEBUG_PANELS"),
            memory_require_auth=_flag("AGENTIC_SEARCH_MEMORY_REQUIRE_AUTH"),
            memory_compression=_flag("AGENTIC_SEARCH_MEMORY_COMPRESSION"),
            search_cache_ttl=app_settings.services.search_cache_ttl_seconds,
        )


class SessionCreateRequest(BaseModel):
    title: str | None = None
    user_id: str | None = None


class ToolApprovalDecisionRequest(BaseModel):
    decision: Literal["approve", "deny"]


class ToolApprovalDecisionResponse(BaseModel):
    id: str
    decision: str


class ChatMessageView(BaseModel):
    role: str
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)


class ChatSessionView(BaseModel):
    id: str
    title: str | None = None
    user_id: str | None = None
    messages: list[ChatMessageView] = Field(default_factory=list)


class SourceDocumentView(BaseModel):
    id: str
    citation: str
    title: str
    content: str
    url: str | None = None
    score: float = 0.0
    metadata: dict[str, object] = Field(default_factory=dict)


class AgentExperienceRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str | None = None
    user_id: str | None = None
    search_url: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    source_provider: str = Field(
        default="auto",
        description=(
            "'auto' (default — internal retrieval first, then SerpAPI and the "
            "configured browser fallback when evidence is weak or empty), "
            "'retrieval', 'serpapi', 'google', 'browser', or 'all'. An explicit "
            "value other "
            "than 'auto' forces a search against that single provider (dev only)."
        ),
    )
    mode: str | None = Field(
        default=None,
        description=(
            "Optional explicit mode override: 'search_tool', 'hybrid_search', 'chat_once', "
            "'chat_loop', 'search_agent', 'tool_agent'. When None, intent is auto-detected."
        ),
    )
    route: str | None = Field(
        default=None,
        description=(
            "Optional route chosen by the user after a clarification: 'chat', "
            "'search', or 'tool'. Skips the router and dispatches directly."
        ),
    )


class ControlFlowEventView(BaseModel):
    sequence: int
    timestamp: str
    turn: int
    component: str
    action: str
    status: str
    duration_ms: int | None = None
    details: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class AgentExperienceResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[str]
    documents: list[SourceDocumentView]
    messages: list[ChatMessageView]
    hook_metadata: dict[str, object] = Field(default_factory=dict)
    intent: str = "chat"  # "search" | "chat" | "tool" | "clarify"
    clarification: dict | None = None
    tool_calls: list[ToolCallView] = Field(default_factory=list)
    control_flow_trace: list[ControlFlowEventView] = Field(default_factory=list)


class QueryProcessingHookResponse(BaseModel):
    query: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


def _control_flow_event_view(event: ControlFlowEvent) -> ControlFlowEventView:
    return ControlFlowEventView(**event.to_dict())


async def _request_tool_approval(
    broker: ToolApprovalBroker,
    owner_user_id: str,
    approval_request,
    queue: asyncio.Queue[dict],
):
    registered = asyncio.Event()
    publish_task: asyncio.Task[None] | None = None

    def publish(view) -> None:
        nonlocal publish_task
        publish_task = asyncio.create_task(
            queue.put({"type": "approval_required", "approval": asdict(view)})
        )
        registered.set()

    request_task = asyncio.create_task(
        broker.request(owner_user_id, approval_request, on_registered=publish)
    )
    registration_task = asyncio.create_task(registered.wait())
    try:
        done, _ = await asyncio.wait(
            (request_task, registration_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if request_task in done and not registered.is_set():
            return await request_task

        await registration_task
        assert publish_task is not None
        done, _ = await asyncio.wait(
            (request_task, publish_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if request_task in done and not publish_task.done():
            return await request_task
        await publish_task
        return await request_task
    finally:
        tracked_tasks = [request_task, registration_task]
        if publish_task is not None:
            tracked_tasks.append(publish_task)
        for task in tracked_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tracked_tasks, return_exceptions=True)


@dataclass(frozen=True)
class _HybridSearchResult:
    executed_queries: list[str]
    documents: list[ContextDocument]
    status: str = "ok"  # "ok" | "empty" | "unreachable"
    ranking: dict | None = None


class _RankedDocumentList(list[ContextDocument]):
    """List-compatible ranked documents carrying the ranking stage facts."""

    def __init__(self, documents: list[ContextDocument], ranking: dict) -> None:
        super().__init__(documents)
        self.ranking = ranking


def _ranking_metadata(
    documents: list[ContextDocument], *, fallback_operations: list[str]
) -> dict:
    ranking = getattr(documents, "ranking", None)
    if ranking is not None:
        return dict(ranking)
    return {
        "operations": fallback_operations,
        "candidate_count": len(documents),
    }


def _register_routers(
    app: FastAPI,
    db: AgenticSearchStore,
    settings: AppSettings,
    search_url: str,
    debug_panels: bool = False,
    llm: LLMClient | None = None,
    memory_require_auth: bool = False,
    memory_compression: bool = False,
) -> None:
    """Attach all API routers and exception handlers to *app*."""

    # --- User auth ---
    app.include_router(create_users_router(db, settings))

    # --- Core search & chat ---
    app.include_router(
        create_chat_router(db, llm=llm, memory_compression=memory_compression)
    )
    app.include_router(create_search_router(db, search_url=search_url))

    from src.internal.servers.query_and_chat.tool_backend import create_tool_router

    app.include_router(
        create_tool_router(
            db,
            search_url=search_url,
            resolved=settings,
            llm=llm,
            memory_compression=memory_compression,
        )
    )
    app.include_router(query_basic_router)
    app.include_router(create_query_history_router(db, settings))

    app.include_router(create_tools_router(settings))

    # --- Users & groups ---
    app.include_router(create_user_group_router(db, settings))
    app.include_router(tenants_router)

    # --- Admin: enterprise settings, evals, hooks ---
    ee_admin_router, ee_basic_router = create_enterprise_settings_routers(settings)
    app.include_router(ee_admin_router)
    app.include_router(ee_basic_router)
    app.include_router(create_evals_router(settings, search_url=search_url, db=db))
    app.include_router(create_hooks_router(db, settings))

    # --- Admin: license, billing ---
    app.include_router(create_license_router(settings))
    app.include_router(create_billing_router(settings))

    # --- Admin: rate limits, standard answers, reporting ---
    app.include_router(create_admin_surface_router(db, settings))
    app.include_router(create_token_rate_limits_router(db, settings))
    app.include_router(create_manage_router(db, settings))
    app.include_router(create_reporting_router(db, settings))

    # --- Admin: OAuth, settings, SCIM ---
    app.include_router(create_oauth_router(settings))
    app.include_router(create_settings_router(settings))
    app.include_router(create_scim_router(db, settings))
    register_scim_exception_handlers(app)

    # --- Analytics ---
    app.include_router(create_analytics_router(db, settings))

    # --- Retrieval feedback ---
    from src.internal.servers.retrieval.feedback_router import create_feedback_router

    app.include_router(create_feedback_router(db))

    # --- Process telemetry (admin) ---
    from src.internal.servers.web.metrics_router import create_metrics_router

    app.include_router(create_metrics_router(db, settings))

    # --- Memory ---
    from src.internal.memory.router import create_memory_router

    app.include_router(create_memory_router(db, llm, require_auth=memory_require_auth))

    # --- Dev console (gated; dev-only observability) ---
    if debug_panels:
        from src.internal.servers.web.debug_router import create_debug_router

        app.include_router(
            create_debug_router(search_url=search_url, llm=llm, db=db),
            dependencies=[Depends(make_require_admin(settings))],
        )


class _WebHybridRetrievalStage:
    """Adapt the established hybrid/provider policy to normalized candidates."""

    def __init__(
        self,
        *,
        llm,
        search_url,
        browser_search_url,
        rerank_url,
        source_provider,
    ) -> None:
        self._llm = llm
        self._search_url = search_url
        self._browser_search_url = browser_search_url
        self._rerank_url = rerank_url
        self._source_provider = source_provider

    async def retrieve(self, query, history, filters, top_k) -> CandidateSet:
        result = await _run_hybrid_search(
            query,
            llm=self._llm,
            search_url=self._search_url,
            browser_search_url=self._browser_search_url,
            rerank_url=self._rerank_url,
            top_k=top_k,
            filters=filters,
            source_provider=self._source_provider,
        )
        candidates = [
            SearchResult(
                contents=document.content,
                score=document.score,
                title=document.title,
                url=document.url,
                metadata=document.metadata,
            )
            for document in result.documents
        ]
        return CandidateSet(
            query=query,
            candidates=candidates,
            provider=self._source_provider,
            filters=filters,
            metadata={
                "status": result.status,
                "executed_queries": result.executed_queries,
            },
        )


class _WebEvidenceStage:
    """Preserve the already-ranked hybrid result while normalizing its IDs."""

    async def rank(self, query, candidates, top_k) -> RankedEvidence:
        documents = DefaultRankingStage.documents(candidates)[:top_k]
        return RankedEvidence(
            query=query,
            evidence=documents,
            metadata={
                "operations": ["hybrid_ranking"],
                "executed_queries": candidates.metadata.get("executed_queries", []),
            },
        )


class _WebSearchAnswerStage:
    """Keep the existing deterministic search answer behind inference protocol."""

    def __init__(self, source_provider: str) -> None:
        self._source_provider = source_provider

    async def generate(self, query, history, evidence) -> GeneratedAnswer:
        answer = _search_only_answer(
            "Search",
            queries=evidence.metadata.get("executed_queries", [query]),
            documents=evidence.evidence,
            source_provider=self._source_provider,
        )
        return GeneratedAnswer(
            answer=answer,
            citations=[document.citation for document in evidence.evidence],
        )


async def _auto_search_pipeline(
    query: str,
    *,
    llm,
    search_url: str,
    browser_search_url,
    rerank_url,
    top_k: int,
    filters,
    history: list,
    source_provider: str,
    extra: dict,
) -> tuple:
    """Run the shared stage composer as the grounded degraded fallback."""
    pipeline = SearchPipeline(
        _WebHybridRetrievalStage(
            llm=llm,
            search_url=search_url,
            browser_search_url=browser_search_url,
            rerank_url=rerank_url,
            source_provider=source_provider,
        ),
        _WebEvidenceStage(),
        _WebSearchAnswerStage(source_provider),
    )
    result = await pipeline.run(query, history, filters, top_k, source_provider)
    result[4].update(extra)
    return result


# ---------------------------------------------------------------------------
# Loop runners — single source of truth for building + running each agent loop.
# Each returns the canonical (answer, citations, documents, intent, extra) tuple
# consumed by both _run_auto_routed and the explicit-mode chain. Runners hold no
# capability policy: degradation lives in _run_auto_routed and the explicit-mode
# 400 guards live at those call sites.
# ---------------------------------------------------------------------------


def _search_agent_documents(output) -> list[ContextDocument]:
    """Extract documents from a SearchAgentLoop output, preserving each result's
    real ``[RxQyDz]`` citation label so answer markers resolve to source cards.

    Rounds/queries/docs are enumerated 1-based to match
    ``SearchAgentLoop._format_round_information`` (the labels the model cited).
    Dedup is intentionally skipped here: every cited label needs its own card so
    no citation dangles, even when the same doc is retrieved under two queries.
    """
    documents: list[ContextDocument] = []
    rounds = getattr(output.context, "rounds", None) or []
    for round_idx, round_ctxs in enumerate(rounds, 1):
        for query_idx, ctx in enumerate(round_ctxs, 1):
            for doc_idx, result in enumerate(ctx.results, 1):
                key = citation_key(round_idx, query_idx, doc_idx)
                base = ContextDocument.from_search_result(result, index=doc_idx)
                documents.append(
                    ContextDocument(
                        id=key,
                        title=base.title,
                        content=base.content,
                        url=base.url,
                        score=base.score,
                        metadata=base.metadata,
                    )
                )
    return documents


async def _run_search_agent(
    query: str,
    *,
    manager,
    tokenizer,
    search_url: str,
    top_k: int,
    history: list | None = None,
    # A SearchFilters, not the wire dict: SearchAgentLoop serialises it for the
    # request and enforces it on the results, so a backend that ignores the
    # field cannot put an unreadable document into the model's context.
    filters: "SearchFilters | None" = None,
    allow_internal_knowledge_answer: bool = True,
    on_turn=None,
    on_trace=None,
) -> tuple:
    """Run the multi-turn SearchAgentLoop. Assumes a local model is configured."""
    from src import get_registered_agent_loop, resolve_agent_name
    from src.agents.search import SearchAgentLoopConfig

    loop_cls = get_registered_agent_loop(resolve_agent_name("search_agent"))
    loop = loop_cls(
        tokenizer=tokenizer,
        server_manager=manager,
        search_config=SearchAgentLoopConfig(
            search_url=search_url,
            topk=top_k,
            max_turns=3,
            filters=filters,
            allow_internal_knowledge_answer=allow_internal_knowledge_answer,
        ),
    )
    output = await loop.run(
        _build_search_agent_messages(query, history or []),
        sampling_params={"temperature": 0.0, "max_tokens": 256},
        on_turn=on_turn,
        on_trace=on_trace,
    )
    documents = [
        _document_with_metadata(
            document,
            source_provider="retrieval",
            query=query,
            entry_point="search_agent",
        )
        for document in _search_agent_documents(output)
    ]
    extra = {
        "control_flow_trace": output.control_flow_trace,
        "num_turns": output.num_turns,
    }
    return (
        output.final_answer or "",
        [d.citation for d in documents],
        documents,
        "search",
        extra,
    )


async def _run_agentic_rag(
    query: str,
    *,
    llm,
    search_url: str,
    top_k: int,
    filters=None,
    history: list,
    user_memory: str | None = None,
    on_claim=None,
) -> tuple:
    """Run the AgenticRAGLoop (decompose + HyDE). Assumes an LLM is configured."""
    rag_loop = AgenticRAGLoop(
        AgenticRAGConfig(
            max_rounds=3,
            topk=top_k,
            retrieval_url=search_url,
            filters=filters,
        ),
        llm=llm,
    )
    from uuid import uuid4

    recorder = ControlFlowRecorder(uuid4().hex)
    rag = await rag_loop.run(
        query,
        chat_history=history,
        recorder=recorder,
        user_memory=user_memory,
        on_claim=on_claim,
    )
    documents = [
        _document_with_metadata(
            document,
            source_provider="retrieval",
            query=query,
            entry_point="agentic_rag",
        )
        for document in rag.context.documents
    ]
    return (
        rag.answer,
        rag.citations,
        documents,
        "chat",
        {
            "rounds_used": rag.rounds_used,
            "control_flow_trace": recorder.snapshot(),
        },
    )


def _finalize_response(
    db,
    session_id: str,
    *,
    query: str,
    answer: str,
    citations: list,
    documents: list,
    intent: str,
    hook_metadata: dict,
    extra: dict,
    mode: str,
) -> "AgentExperienceResponse":
    """Persist the assistant turn and build the response. Single response tail.

    ``extra`` carries loop-specific data on the canonical tuple. ``tool_calls``
    and ``control_flow_trace`` are response fields (not metadata); everything else
    in ``extra`` is merged into the persisted metadata and the response's
    ``hook_metadata``. The private ``_assistant_fallback`` key is dropped.
    """
    extra = dict(extra)
    extra.pop("_assistant_fallback", None)
    tool_calls = extra.pop("tool_calls", [])
    clarification = extra.pop("clarification", None)
    raw_trace = extra.pop("control_flow_trace", None)
    trace_views = (
        [_control_flow_event_view(event) for event in raw_trace] if raw_trace else []
    )
    pipeline_stages = _capture.pipeline_stage_summary(
        query,
        citations=citations,
        documents=documents,
        metadata=extra,
    )
    metadata = {
        "citations": citations,
        "document_ids": [d.id for d in documents],
        "hooks": hook_metadata,
        "mode": mode,
        "intent": intent,
        "pipeline_stages": pipeline_stages,
        **extra,
    }
    if trace_views:
        metadata["control_flow_trace"] = [view.model_dump() for view in trace_views]
    stage_metrics = _stage_metrics.current()
    if stage_metrics is not None:
        # Retrieval time/docs and generation time/tokens for this turn, kept
        # apart so a slow or bad turn can be attributed to one side.
        metadata["stage_metrics"] = stage_metrics.snapshot()
    _capture.record_pipeline_stages(pipeline_stages)
    _capture.record_stage(
        "final",
        "answer",
        {
            "answer": answer,
            "citations": citations,
            "documents": [d.id for d in documents],
            "intent": intent,
            "route": extra.get("route"),
            "route_degraded": extra.get("route_degraded"),
        },
    )
    db.add_chat_message(session_id, role="assistant", content=answer, metadata=metadata)
    messages = [
        ChatMessageView(role=m.role, content=m.content, metadata=m.metadata)
        for m in db.list_chat_messages(session_id)
    ]
    return AgentExperienceResponse(
        session_id=session_id,
        answer=answer,
        citations=citations,
        documents=[_document_view(d) for d in documents],
        messages=messages,
        hook_metadata={**hook_metadata, **extra},
        intent=intent,
        clarification=clarification,
        tool_calls=tool_calls,
        control_flow_trace=trace_views,
    )


def _direct_gate_decision(
    query: str,
    docs: list[ContextDocument],
    *,
    cos_min: float,
    cosine_fn: Callable[[str, str], float | None],
) -> tuple[bool, str, float, float | None]:
    """Tiered strong/weak gate over the rank-1 retrieval result.

    exact title match → direct; typo (Levenshtein<2) confirmed by cosine → direct;
    semantic cosine > cos_min → direct; otherwise escalate. Backend-independent.
    """
    top_score = max((d.score or 0.0 for d in docs), default=0.0)
    if not docs:
        return False, "weak", top_score, None

    top = docs[0]
    q = normalize_for_match(query)
    t = normalize_for_match(top.title)

    if q and q == t:
        return True, "exact", top_score, None

    passage = f"{top.title} {top.content}".strip()

    if q and t and levenshtein_lt2(q, t):
        cosine = cosine_fn(query, passage)
        if cosine is not None and cosine > cos_min:
            return True, "fuzzy", top_score, cosine
        return False, "weak", top_score, cosine

    cosine = cosine_fn(query, passage)
    if cosine is not None and cosine > cos_min:
        return True, "semantic", top_score, cosine
    return False, "weak", top_score, cosine


def _enforce_access(documents: list, filters) -> list:
    """Drop documents the caller may not see.

    Filters are sent to the retrieval server, and `demo.py`, `hybrid.py` and the
    full RetrievalService all honour them — but a third-party backend need not,
    so passing filters down is not enforcement on its own and this route checks
    the returned documents itself. Documents that declare no ACL are public (see
    ``SearchFilters.matches``).
    """
    if filters is None or not documents:
        return documents
    matches = getattr(filters, "matches", None)
    if matches is None:
        return documents
    return [d for d in documents if matches(getattr(d, "metadata", None) or {})]


def _filters_payload(filters):
    """Serialise ``filters`` for a helper that posts them straight over the wire.

    ``search_tool`` and the hybrid leg need a plain dict, not the
    ``SearchFilters`` object ``_enforce_access`` uses for its ``.matches()``
    check. Convert when given the object; pass a dict through unchanged.

    **Do not wrap the value handed to ``SearchAgentLoopConfig``.** That loop
    holds the object on purpose: it serialises for its own request *and*
    enforces on the results, before the documents reach the model's context.
    Passing it a payload here would silently disable that enforcement — the
    unpaired-serialisation shape this repo has already been burned by twice."""
    to_payload = getattr(filters, "to_payload", None)
    return to_payload() if to_payload else filters


async def _run_search_direct_or_escalate(
    query: str,
    *,
    manager,
    tokenizer,
    llm,
    search_url: str,
    browser_search_url,
    rerank_url,
    top_k: int,
    filters,
    history: list,
    source_provider: str,
    on_turn=None,
) -> tuple:
    """Direct retrieval first; return docs when the query matches, else escalate.

    A tiered match gate (`_direct_gate_decision`) compares the query to the
    rank-1 result: exact title match, or a typo (Levenshtein<2) confirmed by
    cosine, or semantic cosine > threshold → return the ranked docs with a
    non-LLM summary (no agent loop). Otherwise escalate to the SearchAgentLoop
    (local model) or the degraded pipeline, preserving today's behavior.
    """

    # Signing in narrows results; it does not change which route runs. Filters
    # used to divert the whole query to `_auto_search_pipeline` because the
    # direct and agent paths took no `filters` and would have retrieved across
    # every user's documents. Both thread them end-to-end now (#407) and every
    # call below passes `filters`, so that diversion only made an authenticated
    # query slower and differently-answered than the same query anonymously.

    async def _escalate(top_score: float, reason: str) -> tuple:
        escalate_extra = {
            "search_mode": "escalated",
            "top_score": top_score,
            "escalate_reason": reason,
        }
        has_local_model = manager is not None and tokenizer is not None
        if has_local_model:
            answer, _citations, docs, intent, run_extra = await _run_search_agent(
                query,
                manager=manager,
                tokenizer=tokenizer,
                search_url=search_url,
                top_k=top_k,
                history=history,
                # The object, not the payload: the loop serialises it for the
                # request and enforces it on the results, before the documents
                # reach the model's context.
                filters=filters,
                allow_internal_knowledge_answer=False,
                on_turn=on_turn,
                on_trace=None,
            )
            run_extra.update(escalate_extra)
            docs = _enforce_access(docs, filters)
            # Citations are rebuilt from the surviving documents: `citations`
            # as returned still names anything `_enforce_access` just dropped.
            return (
                answer,
                [d.citation for d in docs],
                docs,
                intent,
                run_extra,
            )

        escalate_extra["route_degraded"] = "no_local_model"
        return await _auto_search_pipeline(
            query,
            llm=llm,
            search_url=search_url,
            browser_search_url=browser_search_url,
            rerank_url=rerank_url,
            top_k=top_k,
            filters=filters,
            history=history,
            source_provider=source_provider,
            extra=escalate_extra,
        )

    # Explicit non-default source: honor it via the existing escalation path
    # (no direct-first local retrieval short-circuit).
    if source_provider not in ("auto", "retrieval"):
        return await _escalate(0.0, "explicit_source")

    internal_unreachable = False
    try:
        documents = await _run_direct_search(
            query,
            source_provider="retrieval",
            search_url=search_url,
            rerank_url=rerank_url,
            top_k=top_k,
            filters=_filters_payload(filters),
        )
    except Exception:
        internal_unreachable = True
        documents = []
    documents = _enforce_access(documents, filters)
    real = [d for d in documents if not d.metadata.get("error")]
    if documents and not real:
        internal_unreachable = True
    is_strong, tier, top_score, cosine = await asyncio.to_thread(
        _direct_gate_decision,
        query,
        real,
        cos_min=search_direct_cos_min(),
        cosine_fn=make_cosine_fn(gate_embedder()),
    )
    _capture.record_stage(
        "search",
        "direct_retrieval",
        {
            "query": query,
            "top_k": top_k,
            "top_score": top_score,
            "documents": [
                {"id": d.id, "title": d.title, "score": d.score} for d in real
            ],
        },
    )

    if is_strong:
        ranking = _ranking_metadata(documents, fallback_operations=["direct_ranking"])
        ranking["operations"] = [*ranking["operations"], "sufficiency_gate"]
        _capture.record_stage(
            "search",
            "sufficiency",
            {"mode": "direct", "tier": tier, "cosine": cosine, "top_score": top_score},
        )
        answer = _search_only_answer(
            "Direct retrieval",
            queries=[query],
            documents=real,
            source_provider="retrieval",
        )
        return (
            answer,
            [d.citation for d in real],
            real,
            "search",
            {
                "search_mode": "direct",
                "tier": tier,
                "top_score": top_score,
                "source_provider": "retrieval",
                "retrieval_query": query,
                "ranking": ranking,
                "inference": {"mode": "deterministic", "model": None},
            },
        )

    _capture.record_stage(
        "search",
        "sufficiency",
        {"mode": "escalated", "tier": tier, "cosine": cosine, "top_score": top_score},
    )

    if source_provider == "auto":
        external_providers = ["serpapi"]
        if browser_search_url:
            external_providers.append("browser")
        providers_attempted = 1
        unreachable_providers = 1 if internal_unreachable else 0

        for provider in external_providers:
            providers_attempted += 1
            try:
                external_documents = await _run_direct_search(
                    query,
                    source_provider=provider,
                    search_url=search_url,
                    browser_search_url=(
                        browser_search_url if provider == "browser" else None
                    ),
                    rerank_url=rerank_url,
                    top_k=top_k,
                    filters=None,
                )
            except Exception as exc:
                logger.warning("%s fallback failed for %r: %s", provider, query, exc)
                unreachable_providers += 1
                external_documents = []

            external_real = [
                doc for doc in external_documents if not doc.metadata.get("error")
            ]
            if external_documents and not external_real:
                unreachable_providers += 1
            if external_real:
                _capture.record_stage(
                    "search",
                    "external_fallback",
                    {"provider": provider, "documents": len(external_real)},
                )
                answer = _search_only_answer(
                    "External search",
                    queries=[query],
                    documents=external_real,
                    source_provider=provider,
                )
                return (
                    answer,
                    [doc.citation for doc in external_real],
                    external_real,
                    "search",
                    {
                        "search_mode": "external_fallback",
                        "external_provider": provider,
                        "source_provider": provider,
                        "retrieval_query": query,
                        "top_score": top_score,
                        "ranking": _ranking_metadata(
                            external_documents,
                            fallback_operations=["direct_ranking"],
                        ),
                        "inference": {"mode": "deterministic", "model": None},
                    },
                )

        _capture.record_stage(
            "search",
            "external_fallback",
            {"provider": "none", "documents": 0},
        )
        all_unreachable = unreachable_providers == providers_attempted
        return (
            (
                "No sources are reachable right now. Please try again shortly."
                if all_unreachable
                else f"No results found for: {query}"
            ),
            [],
            [],
            "search",
            {
                "search_mode": "external_empty",
                "top_score": top_score,
            },
        )

    return await _escalate(top_score, "weak_retrieval")


async def _run_auto_routed(
    query: str,
    *,
    app_settings: AppSettings | None = None,
    llm,
    manager,
    tokenizer,
    search_url: str,
    browser_search_url,
    rerank_url,
    top_k: int,
    filters,
    history: list,
    resolved,
    source_provider: str = "retrieval",
    on_turn=None,
    on_approval=None,
    on_claim=None,
    user_memory: str | None = None,
    user_present: bool = False,
    forced_route: RouteStrategy | None = None,
) -> tuple:
    """3-way agentic routing. Returns (answer, citations, documents, intent, extra).

    `recognize_intent` picks one of {chat, search, tool}, or asks the user when no
    signal in the cascade dominates; dispatch is capability-aware and degrades
    gracefully when a required backend (local model vs LLM client) is
    unavailable. `extra["route"]` records the chosen strategy and
    `extra["route_degraded"]` records any degradation.
    """
    extra: dict = {}
    explicit_source = source_provider != "auto"
    has_local_model = manager is not None and tokenizer is not None

    if forced_route is not None:
        strategy = forced_route
        extra["route_mechanism"] = "user_selected"
    else:
        decision = recognize_intent(
            query,
            llm=llm,
            explicit_source=explicit_source,
            settings=app_settings,
        )
        extra.update(decision.metadata)
        if decision.clarification is not None:
            extra["route"] = "clarify"
            extra["clarification"] = {
                "question": decision.clarification.question,
                "options": [
                    {"route": option.route, "label": option.label}
                    for option in decision.clarification.options
                ],
            }
            return decision.clarification.question, [], [], "clarify", extra
        strategy = decision.strategy
    extra["route"] = strategy.value

    # ---- TOOL: run ToolAgentLoop with the real registered tools ----
    # ``with_search_tool=True`` so this branch gets a corpus search bound to
    # this request, carrying the caller's ACL. It used to pass False, meaning
    # "use whatever the registry holds rather than synthesising one" — but the
    # seeded corpus search is built at process start with no identity, so it is
    # now registered non-agent-callable and this branch would otherwise get no
    # corpus search at all.
    if strategy is RouteStrategy.TOOL:
        if has_local_model:
            result = None
            try:
                result = await _run_tool_agent(
                    query,
                    manager=manager,
                    tokenizer=tokenizer,
                    search_url=search_url,
                    history=history,
                    resolved=resolved,
                    on_turn=on_turn,
                    on_approval=on_approval,
                    with_search_tool=True,
                    user_present=user_present,
                    filters=filters,
                )
            except Exception as exc:
                logger.warning("ToolAgentLoop failed, degrading to RAG: %s", exc)
            if result is not None:
                answer, citations, documents, intent, run_extra = result
                run_extra.pop("_assistant_fallback", None)
                if answer.strip():
                    extra.update(run_extra)
                    return answer, citations, documents, intent, extra
            extra["route_degraded"] = "tool_unavailable"
        else:
            extra["route_degraded"] = "no_local_model"
        strategy = RouteStrategy.CHAT

    # ---- SEARCH: direct retrieval first, escalate to the agent loop if weak ----
    if strategy is RouteStrategy.SEARCH:
        (
            answer,
            citations,
            documents,
            intent,
            run_extra,
        ) = await _run_search_direct_or_escalate(
            query,
            manager=manager,
            tokenizer=tokenizer,
            llm=llm,
            search_url=search_url,
            browser_search_url=browser_search_url,
            rerank_url=rerank_url,
            top_k=top_k,
            filters=filters,
            history=history,
            source_provider=source_provider,
            on_turn=on_turn,
        )
        extra.update(run_extra)
        return answer, citations, documents, intent, extra

    # ---- CHAT: grounded synthesis via AgenticRAGLoop (degrade to pipeline) ----
    # Memory injection is intentionally scoped to this CHAT/AgenticRAG path (and
    # the classic RAG path); SEARCH/TOOL routes do not receive user_memory.
    if strategy is RouteStrategy.CHAT:
        if llm is not None:
            answer, citations, documents, intent, run_extra = await _run_agentic_rag(
                query,
                llm=llm,
                search_url=search_url,
                top_k=top_k,
                filters=filters,
                history=history,
                user_memory=user_memory,
                on_claim=on_claim,
            )
            extra.update(run_extra)
            return answer, citations, documents, intent, extra
        extra.setdefault("route_degraded", "no_llm")
        return await _auto_search_pipeline(
            query,
            llm=llm,
            search_url=search_url,
            browser_search_url=browser_search_url,
            rerank_url=rerank_url,
            top_k=top_k,
            filters=filters,
            history=history,
            source_provider=source_provider,
            extra=extra,
        )


def create_web_app(
    settings: SearchExperienceSettings | None = None,
    *,
    app_settings: AppSettings | None = None,
    store: AgenticSearchStore | None = None,
    llm: LLMClient | None = None,
    hook_registry: HookRegistry | None = None,
) -> FastAPI:
    """Create the user-facing web app.

    The app serves a dependency-free HTML/JS interface and a small JSON API that
    runs the repo's retrieval-grounded answer pipeline. Chat state is persisted
    through `AgenticSearchStore`, which defaults to an in-memory SQLite DB.
    """
    load_dotenv()
    resolved = app_settings or load_app_settings()
    if resolved.auth.dev_admin_bypass:
        logger.warning(
            "ADMIN AUTH BYPASSED via AGENTIC_SEARCH_DEV_ADMIN — every admin "
            "endpoint accepts unauthenticated requests. Dev only; never set "
            "this in production."
        )
    settings = settings or SearchExperienceSettings.from_app_settings(resolved)
    owns_store = store is None
    db = store or AgenticSearchStore(settings.db_path)
    if llm is None:
        import os

        api_key = resolved.llm.api_key or os.environ.get("OPENAI_API_KEY")
        if api_key:
            llm = OpenAICompatibleLLM(
                LLMConfig(
                    model_provider=resolved.llm.model_provider,
                    model_name=resolved.llm.model_name,
                    api_key=api_key,
                    api_base=resolved.llm.api_base,
                    max_input_tokens=resolved.llm.max_input_tokens,
                )
            )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Fail before seeding, background discovery, or loading a model.
        check_router_auth(_app, PUBLIC_ENDPOINT_SPECS)
        seed_db(db)
        seed_tools(
            tool_registry,
            tools=tool_knowledge_base(
                search_url=resolved.services.retrieval_url, llm=llm
            ),
        )
        # Pull tools in from configured MCP servers. Off unless
        # AGENTIC_SEARCH_MCP_SERVERS is set; unreachable servers are skipped.
        #
        # Deliberately a background task rather than an await: our own MCP server
        # authenticates by calling back into this process's /me, which does not
        # accept connections until lifespan startup finishes. Awaiting here would
        # make discovery fail against exactly the server we ship.
        from src.internal.tools.mcp_client import parse_mcp_servers, register_mcp_tools

        mcp_specs = parse_mcp_servers(
            resolved.mcp_servers,
            token=resolved.mcp_token,
            agent_exclude=resolved.mcp_agent_exclude,
        )
        _app.state.mcp_discovery_task = (
            asyncio.create_task(register_mcp_tools(tool_registry, mcp_specs))
            if mcp_specs
            else None
        )
        _app.state.search_agent_manager = None
        _app.state.search_agent_tokenizer = None
        if resolved.search_agent_server_url:
            try:
                from transformers import AutoTokenizer
                from src.model.serving import build_server_manager

                model = resolved.search_agent_model or "Qwen/Qwen2.5-1.5B-Instruct"
                tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                manager = build_server_manager(
                    tokenizer,
                    server_url=resolved.search_agent_server_url,
                    model=model,
                )
                _app.state.search_agent_tokenizer = tokenizer
                _app.state.search_agent_manager = manager
                logger.info(
                    "search_agent: connected to OpenAI-compatible server at %s (model %s)",
                    resolved.search_agent_server_url,
                    model,
                )
            except Exception:
                logger.exception(
                    "search_agent: failed to init OpenAI server manager — mode will return 400"
                )
        elif resolved.search_agent_model:
            try:
                from transformers import AutoTokenizer
                from src.model.serving import build_server_manager

                tokenizer = AutoTokenizer.from_pretrained(
                    resolved.search_agent_model,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                manager = build_server_manager(
                    tokenizer,
                    model=resolved.search_agent_model,
                    device=resolved.search_agent_device,
                    allow_unsafe_mps=True,
                    local_files_only=True,
                    generation_timeout_seconds=resolved.generation_timeout_seconds,
                )
                _app.state.search_agent_tokenizer = tokenizer
                _app.state.search_agent_manager = manager
                logger.info(
                    "search_agent: loaded %s on %s",
                    resolved.search_agent_model,
                    resolved.search_agent_device,
                )
            except Exception:
                logger.exception(
                    "search_agent: failed to load model — mode will return 400"
                )
        try:
            gate_embedder()  # warm the direct-gate e5 model (no-op when SEMANTIC=0)
        except Exception:
            logger.exception("direct-gate: embedder warmup failed")
        # Process-local cache behind SearchClient.retrieve, the web providers in
        # search_tool, and RerankHTTPRankingStage. Owned here so nothing is
        # cached in a process that never started the web app.
        configure_serving_cache(settings.search_cache_ttl)
        try:
            yield
        finally:
            reset_serving_cache()
            if owns_store:
                db.close()

    app = FastAPI(title="Agentic Search Web", lifespan=lifespan)
    app.state.auth_store = db
    app.state.tool_approval_broker = ToolApprovalBroker(
        resolved.tool_approval_timeout_seconds
    )
    import os as _os
    from src.internal.servers.web.request_capture_store import RequestCaptureStore

    app.state.request_captures = RequestCaptureStore(
        max_size=int(_os.environ.get("AGENTIC_SEARCH_REQUEST_CAPTURE_MAX", "20"))
    )
    add_api_server_tenant_id_middleware(app, resolved)
    add_license_enforcement_middleware(app, resolved)
    add_tier_gate_middleware(app, resolved)
    # Cheap enough to run everywhere: one monotonic pair and a bounded deque
    # append per request. /api/debug/latency reads what it records.
    add_latency_logging_middleware(app, logging.LoggerAdapter(logger, {}))

    _register_routers(
        app,
        db,
        resolved,
        search_url=settings.search_url,
        debug_panels=settings.debug_panels,
        llm=llm,
        memory_require_auth=settings.memory_require_auth,
        memory_compression=settings.memory_compression,
    )

    frontend_dist = _frontend_dist_path()

    @app.get("/health")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    def _app_shell() -> str:
        if frontend_dist:
            return (frontend_dist / "index.html").read_text(encoding="utf-8")
        return APP_HTML

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _app_shell()

    # The frontend routes these client-side. They are listed explicitly rather
    # than served from a catch-all so a mistyped API path stays a 404 instead of
    # silently returning HTML. Keep in sync with ROUTES in web/src/router.tsx.
    @app.get("/assist", response_class=HTMLResponse)
    @app.get("/search", response_class=HTMLResponse)
    @app.get("/chat", response_class=HTMLResponse)
    @app.get("/tools", response_class=HTMLResponse)
    def spa_page() -> str:
        return _app_shell()

    @app.get("/assets/app.css")
    def stylesheet() -> Response:
        return Response(APP_CSS, media_type="text/css")

    @app.get("/assets/app.js")
    def javascript() -> Response:
        return Response(APP_JS, media_type="application/javascript")

    if frontend_dist:
        app.mount(
            "/assets",
            StaticFiles(directory=frontend_dist / "assets"),
            name="frontend-assets",
        )

    @app.post("/api/sessions")
    def create_session(
        request: SessionCreateRequest, http_request: Request
    ) -> ChatSessionView:
        caller = _optional_user_from_request(http_request, db)
        session = db.create_chat_session(
            user_id=caller.id if caller else None,
            title=request.title or "Search session",
            metadata={"source": "web"},
        )
        return ChatSessionView(
            id=session.id,
            title=session.title,
            user_id=session.user_id,
            messages=[],
        )

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str, http_request: Request) -> ChatSessionView:
        session = db.get_chat_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if not caller_may_use_session(
            session, _optional_user_from_request(http_request, db)
        ):
            # 404 rather than 403: a 403 confirms the id exists, which is the
            # one bit an id-guessing caller does not already have.
            raise HTTPException(status_code=404, detail="Session not found")
        return ChatSessionView(
            id=session.id,
            title=session.title,
            user_id=session.user_id,
            messages=[
                ChatMessageView(
                    role=message.role,
                    content=message.content,
                    metadata=message.metadata,
                )
                for message in db.list_chat_messages(session.id)
            ],
        )

    async def _run_agent_impl(
        request: AgentExperienceRequest,
        http_request: Request,
        *,
        request_id: str,
        on_turn: "OnTurnCallback | None" = None,
        on_trace: EventSink | None = None,
        on_approval=None,
        on_token=None,
        on_claim=None,
    ) -> AgentExperienceResponse:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query is required")
        hook_metadata: dict[str, object] = {}

        auth_user = _optional_user_from_request(http_request, db)
        capabilities = resolve_capabilities(auth_user, db)
        # Attribution and entitlement both come from the authenticated caller.
        user_id = capabilities.user_id
        # Memory-augmented generation: a signed-in caller's stored memories are
        # injected because they are signed in. Anonymous callers have none.
        user_memory = capabilities.memory_preamble or None
        hook_result = execute_hook(
            hook_point=HookPoint.QUERY_PROCESSING,
            payload={"query": query, "user_id": user_id},
            response_type=QueryProcessingHookResponse,
            registry=hook_registry,
        )
        if isinstance(hook_result, QueryProcessingHookResponse):
            if hook_result.query and hook_result.query.strip():
                query = hook_result.query.strip()
            hook_metadata = hook_result.metadata
        elif isinstance(hook_result, HookSoftFailed):
            hook_metadata = {"query_processing_hook_error": hook_result.error_message}

        mode_str = request.mode.strip().lower() if request.mode else None
        normalized_mode = _normalize_agent_mode(mode_str) if mode_str else None
        forced_route = _normalize_route(request.route) if request.route else None

        session_request = _copy_agent_request(request, user_id=user_id)
        session_id = _ensure_session(db, session_request, auth_user)
        working = load_working_memory(
            db,
            session_id,
            keep_last=MAX_HISTORY_MESSAGES,
            cache=get_cache_backend() if settings.memory_compression else None,
        )
        history = working.messages
        db.add_chat_message(session_id, role="user", content=query)

        # Resolve the retrieval URL server-side. A client-supplied search_url is
        # honored only when explicitly allowed (dev/debug); otherwise it is
        # ignored so the backend, not the browser, decides what it will fetch.
        search_url = (
            request.search_url
            if settings.allow_client_search_url and request.search_url
            else settings.search_url
        )
        top_k = request.top_k or settings.top_k
        # Always filtered: anonymous means ["public"], not "unfiltered".
        filters = SearchFilters(access_acl=capabilities.access_acl)

        manager = getattr(http_request.app.state, "search_agent_manager", None)
        tokenizer = getattr(http_request.app.state, "search_agent_tokenizer", None)

        capture_on = (
            getattr(http_request.app.state, "request_captures", None) is not None
        )
        capture_token = (
            _capture.start_capture(request_id, query)
            if capture_on and settings.debug_panels
            else None
        )
        # Always on, unlike the capture: the retrieval/generation split every
        # path's choke points write into, persisted with the turn and rolled
        # into the per-stage window /api/admin/metrics reads.
        stage_token = _stage_metrics.start_request()

        try:
            try:
                if normalized_mode is None:
                    # Auto-routing path
                    (
                        answer,
                        citations,
                        documents,
                        intent,
                        extra,
                    ) = await _run_auto_routed(
                        query,
                        app_settings=resolved,
                        llm=llm,
                        manager=manager,
                        tokenizer=tokenizer,
                        search_url=search_url,
                        browser_search_url=settings.browser_search_url,
                        rerank_url=settings.rerank_url,
                        top_k=top_k,
                        filters=filters,
                        history=history,
                        resolved=resolved,
                        source_provider=_normalize_source_provider(
                            request.source_provider
                        ),
                        on_turn=on_turn,
                        on_approval=on_approval,
                        on_claim=on_claim,
                        user_memory=user_memory,
                        user_present=capabilities.user_present,
                        forced_route=forced_route,
                    )
                    _cap = _capture.active()
                    if _cap is not None:
                        _cap.route = extra.get("route")
                        _cap.route_degraded = extra.get("route_degraded")
                    return _finalize_response(
                        db,
                        session_id,
                        query=query,
                        answer=answer,
                        citations=citations,
                        documents=documents,
                        intent=intent,
                        hook_metadata=hook_metadata,
                        extra=extra,
                        mode="auto",
                    )

                # Explicit mode — pin the dispatch, then share the runners + tail.
                mode = normalized_mode

                if mode == "search_tool":
                    source_provider = _normalize_source_provider(
                        request.source_provider
                    )
                    documents = await _run_direct_search(
                        query,
                        source_provider=source_provider,
                        search_url=search_url,
                        browser_search_url=settings.browser_search_url,
                        rerank_url=settings.rerank_url,
                        top_k=top_k,
                        filters=_filters_payload(filters),
                    )
                    # Serializing the filters for the wire and enforcing on the
                    # result are one change: the server may ignore what it was
                    # sent, so sending without checking is a silent leak.
                    documents = _enforce_access(documents, filters)
                    answer = _search_only_answer(
                        "Direct search tool",
                        queries=[query],
                        documents=documents,
                        source_provider=source_provider,
                    )
                    return _finalize_response(
                        db,
                        session_id,
                        query=query,
                        answer=answer,
                        citations=[doc.citation for doc in documents],
                        documents=documents,
                        intent="search",
                        hook_metadata=hook_metadata,
                        extra={
                            "source_provider": source_provider,
                            "retrieval_query": query,
                            "ranking": _ranking_metadata(
                                documents, fallback_operations=["direct_ranking"]
                            ),
                            "inference": {"mode": "deterministic", "model": None},
                        },
                        mode=mode,
                    )

                if mode == "hybrid_search":
                    source_provider = _normalize_source_provider(
                        request.source_provider
                    )
                    search_result = await _run_hybrid_search(
                        query,
                        llm=llm,
                        search_url=search_url,
                        browser_search_url=settings.browser_search_url,
                        rerank_url=settings.rerank_url,
                        top_k=top_k,
                        filters=filters,
                        source_provider=source_provider,
                    )
                    answer = _search_only_answer(
                        "Hybrid search",
                        queries=search_result.executed_queries,
                        documents=search_result.documents,
                        source_provider=source_provider,
                    )
                    return _finalize_response(
                        db,
                        session_id,
                        query=query,
                        answer=answer,
                        citations=[doc.citation for doc in search_result.documents],
                        documents=search_result.documents,
                        intent="search",
                        hook_metadata=hook_metadata,
                        extra={
                            "source_provider": source_provider,
                            "retrieval_query": query,
                            "executed_queries": search_result.executed_queries,
                            "ranking": search_result.ranking
                            or {
                                "operations": ["hybrid_ranking"],
                                "candidate_count": len(search_result.documents),
                            },
                            "inference": {"mode": "deterministic", "model": None},
                        },
                        mode=mode,
                    )

                if mode == "chat_loop":
                    (
                        answer,
                        citations,
                        documents,
                        intent,
                        extra,
                    ) = await _run_agentic_rag(
                        query,
                        llm=llm,
                        search_url=search_url,
                        top_k=top_k,
                        filters=filters,
                        history=history,
                        user_memory=user_memory,
                        on_claim=on_claim,
                    )
                    return _finalize_response(
                        db,
                        session_id,
                        query=query,
                        answer=answer,
                        citations=citations,
                        documents=documents,
                        intent=intent,
                        hook_metadata=hook_metadata,
                        extra=extra,
                        mode=mode,
                    )

                if mode == "search_agent":
                    if manager is None or tokenizer is None:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                "search_agent mode is not configured. "
                                "Set SEARCH_AGENT_MODEL in .env and restart the server."
                            ),
                        )
                    (
                        answer,
                        citations,
                        documents,
                        intent,
                        extra,
                    ) = await _run_search_agent(
                        query,
                        manager=manager,
                        tokenizer=tokenizer,
                        search_url=search_url,
                        top_k=top_k,
                        history=history,
                        # The object, not the payload: the loop serialises it for the
                        # request and enforces it on the results, before the documents
                        # reach the model's context.
                        filters=filters,
                        on_turn=on_turn,
                        on_trace=on_trace,
                    )
                    # Same pairing as above: the loop's retrieval went out with
                    # a filters payload the server is free to ignore, so the
                    # documents it came back with are checked here.
                    documents = _enforce_access(documents, filters)
                    citations = [doc.citation for doc in documents]
                    return _finalize_response(
                        db,
                        session_id,
                        query=query,
                        answer=answer,
                        citations=citations,
                        documents=documents,
                        intent=intent,
                        hook_metadata=hook_metadata,
                        extra=extra,
                        mode=mode,
                    )

                if mode == "tool_agent":
                    if manager is None or tokenizer is None:
                        raise HTTPException(
                            status_code=400,
                            detail=NO_LOCAL_MODEL_MESSAGE,
                        )
                    answer, citations, documents, intent, extra = await _run_tool_agent(
                        query,
                        manager=manager,
                        tokenizer=tokenizer,
                        search_url=search_url,
                        history=history,
                        resolved=resolved,
                        on_turn=on_turn,
                        on_approval=on_approval,
                        with_search_tool=True,
                        user_present=capabilities.user_present,
                        filters=filters,
                    )
                    # Explicit mode falls back to the last assistant message on an
                    # empty final answer (the auto-route instead degrades to RAG).
                    answer = answer or extra.pop("_assistant_fallback", "")
                    return _finalize_response(
                        db,
                        session_id,
                        query=query,
                        answer=answer,
                        citations=citations,
                        documents=documents,
                        intent=intent,
                        hook_metadata=hook_metadata,
                        extra=extra,
                        mode=mode,
                    )

                result = await answer_with_retrieval(
                    query,
                    llm=llm,
                    chat_history=history,
                    search_url=search_url,
                    top_k=top_k,
                    filters=filters,
                    user_memory=user_memory,
                )
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception("Agent dispatch error: %s", exc)
                detail = (
                    str(exc)
                    if str(exc).strip()
                    else "Unexpected error during agent dispatch"
                )
                raise HTTPException(status_code=502, detail=detail) from exc

            documents = [
                _document_with_metadata(
                    document,
                    source_provider="retrieval",
                    query=query,
                    entry_point="rag",
                )
                for document in result.context.documents
            ]
            return _finalize_response(
                db,
                session_id,
                query=query,
                answer=result.answer,
                citations=result.citations,
                documents=documents,
                intent="chat",
                hook_metadata=hook_metadata,
                extra={},
                mode=mode,
            )
        finally:
            schedule_compression(
                working,
                session_id=session_id,
                llm=llm,
                enabled=settings.memory_compression,
            )
            STAGE_LATENCY.record(_stage_metrics.finish_request(stage_token))
            cap = _capture.active()
            if cap is not None:
                cap.finish()
                http_request.app.state.request_captures.put(cap.snapshot())
            if capture_token is not None:
                _capture.reset_capture(capture_token)

    @app.post("/api/agent")
    async def run_agent(
        request: AgentExperienceRequest,
        http_request: Request,
    ) -> AgentExperienceResponse:
        request_id = _uuid.uuid4().hex
        return await _run_agent_impl(
            request, http_request, request_id=request_id, on_turn=None
        )

    @app.post(
        "/api/agent/approvals/{approval_id}",
        response_model=ToolApprovalDecisionResponse,
    )
    async def decide_tool_approval(
        approval_id: str,
        request: ToolApprovalDecisionRequest,
        http_request: Request,
    ) -> ToolApprovalDecisionResponse:
        from src.agents.tool import ApprovalDecision

        auth_user = _require_auth(http_request, db)
        try:
            await http_request.app.state.tool_approval_broker.decide(
                approval_id,
                auth_user.id,
                ApprovalDecision(request.decision),
            )
        except ApprovalForbidden as exc:
            raise HTTPException(status_code=403, detail="Approval forbidden") from exc
        except ApprovalNotFound as exc:
            raise HTTPException(status_code=404, detail="Approval not found") from exc
        except ApprovalConflict as exc:
            raise HTTPException(
                status_code=409, detail="Approval already decided"
            ) from exc
        except ApprovalExpired as exc:
            raise HTTPException(status_code=410, detail="Approval expired") from exc
        return ToolApprovalDecisionResponse(id=approval_id, decision=request.decision)

    @app.post("/api/agent/stream")
    async def stream_agent(
        request: AgentExperienceRequest,
        http_request: Request,
    ) -> StreamingResponse:
        """Stream agent progress as Server-Sent Events.

        Emits:
          {"type": "progress", "turn": N, "text": "..."}  — per-turn progress
          {"type": "claim",    "text": "..."}             — one verified claim
          {"type": "answer",   "text": "..."}             — final answer text
          {"type": "done",     "session_id": "...", "citations": [...], "documents": [...], "intent": "..."}
          {"type": "error",    "detail": "..."}           — on failure
        """

        def _sse(data: dict) -> str:
            return f"data: {_json.dumps(data)}\n\n"

        queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=100)
        dropped_trace_events = 0
        dropped_claim_events = 0
        auth_user = _optional_user_from_request(http_request, db)
        request_id = _uuid.uuid4().hex
        loop = asyncio.get_running_loop()

        async def on_turn(turn: int, tool_name: "str | None", doc_count: int) -> None:
            text = (
                f"{tool_name} · {doc_count} docs" if tool_name else "writing answer..."
            )
            await queue.put({"type": "progress", "turn": turn, "text": text})

        def _offer(item: dict) -> None:
            # The terminal answer event still carries the full text, so a
            # dropped claim doesn't lose data — but it's worth counting.
            nonlocal dropped_claim_events
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                dropped_claim_events += 1

        def on_claim(text: str) -> None:
            # Called from the generate_answer worker thread (see #547's
            # asyncio.to_thread offload in AgenticRAGLoop.run), so hop back to
            # the loop before touching the queue — put_nowait is not
            # thread-safe.
            loop.call_soon_threadsafe(_offer, {"type": "claim", "text": text})

        async def on_trace(event: ControlFlowEvent) -> None:
            nonlocal dropped_trace_events
            try:
                queue.put_nowait(
                    {
                        "type": "trace",
                        "event": _control_flow_event_view(event).model_dump(),
                    }
                )
            except asyncio.QueueFull:
                dropped_trace_events += 1

        async def on_approval(approval_request):
            return await _request_tool_approval(
                http_request.app.state.tool_approval_broker,
                auth_user.id,
                approval_request,
                queue,
            )

        async def _generate():
            task = asyncio.create_task(
                _run_agent_impl(
                    request,
                    http_request,
                    request_id=request_id,
                    on_turn=on_turn,
                    on_trace=on_trace,
                    on_approval=on_approval if auth_user is not None else None,
                    on_claim=on_claim,
                )
            )
            try:
                while not task.done():
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=0.05)
                        yield _sse(item)
                    except asyncio.TimeoutError:
                        continue
                while not queue.empty():
                    yield _sse(queue.get_nowait())
                result: AgentExperienceResponse = task.result()
                yield _sse({"type": "answer", "text": result.answer})
                yield _sse(
                    {
                        "type": "done",
                        "request_id": request_id,
                        "session_id": result.session_id,
                        "citations": result.citations,
                        "documents": [d.model_dump() for d in result.documents],
                        "intent": result.intent,
                        "clarification": result.clarification,
                        "route": result.hook_metadata.get("route"),
                        "route_degraded": result.hook_metadata.get("route_degraded"),
                        "tool_calls": [tc.model_dump() for tc in result.tool_calls],
                        "control_flow_trace": [
                            event.model_dump() for event in result.control_flow_trace
                        ],
                    }
                )
                if dropped_trace_events:
                    logger.warning(
                        "dropped %d live control-flow trace events",
                        dropped_trace_events,
                    )
                if dropped_claim_events:
                    logger.warning(
                        "dropped %d live claim events",
                        dropped_claim_events,
                    )
            except BaseException as exc:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                if isinstance(exc, asyncio.CancelledError):
                    return  # client disconnected
                if isinstance(exc, HTTPException):
                    yield _sse({"type": "error", "detail": exc.detail})
                else:
                    yield _sse({"type": "error", "detail": str(exc)})

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def _ensure_session(
    store: AgenticSearchStore,
    request: AgentExperienceRequest,
    caller: "AuthenticatedUser | None" = None,
) -> str:
    if request.session_id:
        existing = store.get_chat_session(request.session_id)
        if existing is not None:
            if not caller_may_use_session(existing, caller):
                # Reusing someone else's session is worse than reading it: the
                # handler loads its history into the model context and appends
                # the caller's message to it.
                raise HTTPException(status_code=404, detail="Session not found")
            return request.session_id
    # Body-supplied identities must never attribute a conversation to another user.
    user_id = caller.id if caller and not caller.is_anonymous else None
    session = store.create_chat_session(
        user_id=user_id,
        title=request.query[:80],
        metadata={"source": "web"},
        session_id=request.session_id,
    )
    return session.id


def _copy_agent_request(
    request: AgentExperienceRequest,
    *,
    user_id: str | None,
) -> AgentExperienceRequest:
    model_copy = getattr(request, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"user_id": user_id})
    return request.copy(update={"user_id": user_id})


_MODE_ALIASES = {
    "standard": "chat_once",
    "agentic_rag": "chat_loop",
}
_VALID_AGENT_MODES = {
    "search_tool",
    "hybrid_search",
    "chat_once",
    "chat_loop",
    "search_agent",
    "tool_agent",
}


def _trim_history(history: list, max_messages: int = MAX_HISTORY_MESSAGES) -> list:
    """Keep only the tail of chat history to avoid overflowing the LLM context."""
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


# Search mode stacks long <information> observations on top of history each turn,
# so cap threaded history tighter than MAX_HISTORY_MESSAGES.
SEARCH_AGENT_HISTORY_MESSAGES = 6


def _build_search_agent_messages(query: str, history: list) -> list[dict[str, str]]:
    """Build the SearchAgentLoop message buffer: capped prior turns + the query.

    A leading ``system`` message is the working-memory summary of turns that
    already fell off the tail; it is kept ahead of the cap. The rest of the
    history is capped to the last ``SEARCH_AGENT_HISTORY_MESSAGES`` messages
    and mapped to ``{"role", "content"}`` dicts; the current user query is
    appended last. ``SearchAgentLoop._with_system_prompt`` prepends the system
    prompt.
    """
    leading = history[:1] if history and history[0].role == "system" else []
    capped = _trim_history(
        history[len(leading) :], max_messages=SEARCH_AGENT_HISTORY_MESSAGES
    )
    messages = [{"role": m.role, "content": m.content} for m in [*leading, *capped]]
    messages.append({"role": "user", "content": query})
    return messages


def _normalize_agent_mode(mode: str) -> str:
    requested = mode.strip().lower()
    normalized = _MODE_ALIASES.get(requested, requested)
    if normalized not in _VALID_AGENT_MODES:
        valid = ", ".join(sorted(_VALID_AGENT_MODES))
        raise HTTPException(status_code=422, detail=f"mode must be one of: {valid}")
    return normalized


_VALID_ROUTES = {"chat", "search", "tool"}


def _normalize_route(route: str) -> RouteStrategy:
    requested = route.strip().lower()
    if requested not in _VALID_ROUTES:
        valid = ", ".join(sorted(_VALID_ROUTES))
        raise HTTPException(status_code=422, detail=f"route must be one of: {valid}")
    return RouteStrategy(requested)


_SOURCE_PROVIDER_ALIASES = {
    "local": "retrieval",
    "direct": "retrieval",
    "serp": "serpapi",
    "web": "all",
}
_VALID_SOURCE_PROVIDERS = {
    "auto",
    "retrieval",
    "serpapi",
    "browser",
    "all",
}
_SOURCE_PROVIDER_LABELS = {
    "auto": "Auto (internal + web)",
    "retrieval": "Local Retrieval",
    "serpapi": "SerpAPI",
    "browser": "Browser Retrieval",
    "all": "All Active Sources",
}


def _normalize_source_provider(source_provider: str) -> str:
    requested = source_provider.strip().lower()
    normalized = _SOURCE_PROVIDER_ALIASES.get(requested, requested)
    if normalized not in _VALID_SOURCE_PROVIDERS:
        valid = ", ".join(sorted(_VALID_SOURCE_PROVIDERS))
        raise HTTPException(
            status_code=422,
            detail=f"source_provider must be one of: {valid}",
        )
    return normalized


# Default provider set when the user does not pick a source: fan out to internal
# RAG + fast web search, merged. Browser is excluded (too slow for the default).
_DEFAULT_FANOUT_PROVIDERS = ["retrieval", "serpapi"]


def _source_providers_for(source_provider: str) -> list[str]:
    if source_provider == "all":
        return ["retrieval", "serpapi", "browser"]
    if source_provider == "auto":
        return list(_DEFAULT_FANOUT_PROVIDERS)
    return [source_provider]


_WEB_PROVIDERS = {"serpapi"}


def _is_web_provider(source_provider: str) -> bool:
    """Returns True for providers that return URL snippets needing full-page fetch."""
    return source_provider in _WEB_PROVIDERS


def _tool_provider_for(source_provider: str):
    return "retrieval" if source_provider == "browser" else source_provider


def _source_label(source_provider: str) -> str:
    return _SOURCE_PROVIDER_LABELS.get(source_provider, source_provider)


async def _run_direct_search(
    query: str,
    *,
    source_provider: str,
    search_url: str,
    browser_search_url: str | None = None,
    rerank_url: str | None = None,
    top_k: int,
    filters: dict | None = None,
) -> list[ContextDocument]:
    # Over-fetch so MMR has candidates beyond top_k to diversify from.
    fetch_k = top_k * 2
    documents: list[ContextDocument] = []
    for provider in _source_providers_for(source_provider):
        if provider == "browser":
            if browser_search_url:
                browser_docs = await _run_browser_search(
                    query,
                    browser_search_url=browser_search_url,
                    top_k=fetch_k,
                    existing_count=len(documents),
                )
                documents.extend(browser_docs)
            continue
        pages = await search_tool(
            query,
            provider=provider,
            search_url=search_url,
            page_size=fetch_k,
            # Access filters apply to the internal corpus only (search_tool
            # forwards them for the 'retrieval' provider and ignores them for
            # web providers).
            filters=filters,
        )
        if _is_web_provider(provider):
            pages = await fetch_pages_concurrently(pages, max_chars=2000)
        documents.extend(
            _documents_from_search_pages(
                pages,
                source_provider=provider,
                query=query,
                start_index=len(documents) + 1,
            )
        )
    # Sidecar: add browser when it isn't already the primary or part of "all".
    # "auto" (internal + serpapi) deliberately excludes the slow browser.
    if browser_search_url and source_provider not in {"browser", "all", "auto"}:
        browser_docs = await _run_browser_search(
            query,
            browser_search_url=browser_search_url,
            top_k=fetch_k,
            existing_count=len(documents),
        )
        documents.extend(browser_docs)
    return await _rank_documents(documents, query, rerank_url, top_k)


async def _run_browser_search(
    query: str,
    *,
    browser_search_url: str,
    top_k: int,
    existing_count: int,
) -> list[ContextDocument]:
    """Call a running browser.py /retrieve server and convert results to ContextDocuments."""
    try:
        pages = await search_tool(
            query,
            provider="retrieval",
            search_url=browser_search_url,
            page_size=top_k,
        )
    except Exception as exc:
        logger.warning("Browser search failed for %r: %s", query, exc)
        return []
    return _documents_from_search_pages(
        pages,
        source_provider="browser",
        query=query,
        start_index=existing_count + 1,
    )


async def _rerank_documents(
    docs: list[ContextDocument],
    query: str,
    rerank_url: str,
) -> list[ContextDocument]:
    """Compatibility wrapper around the shared ranking policy."""
    return await _rank_documents(docs, query, rerank_url, len(docs))


async def _rank_documents(
    documents: list[ContextDocument],
    query: str,
    rerank_url: str | None,
    top_k: int,
) -> list[ContextDocument]:
    """Adapt web documents to the shared default ranking policy."""
    candidates = CandidateSet(
        query=query,
        provider="web",
        candidates=[
            SearchResult(
                contents=document.content,
                title=document.title,
                url=document.url,
                score=document.score,
                metadata=document.metadata,
            )
            for document in documents
        ],
    )
    reranker = (
        RerankHTTPRankingStage(
            rerank_url,
            document_contents=lambda candidate: (
                f"{candidate.title}\n{candidate.contents}"
            ),
            send_top_k=False,
        )
        if rerank_url
        else None
    )
    ranked = await DefaultRankingStage(reranker=reranker).rank(query, candidates, top_k)
    return _RankedDocumentList(ranked.evidence, ranked.metadata)


def _provider_error_doc(provider: str, message: str) -> ContextDocument:
    """A placeholder doc marking a provider that failed/timed out. Filtered from
    results; its presence signals 'unreachable' when no real docs were found."""
    return ContextDocument(
        id="D0",
        title="Search error",
        content=message,
        url=None,
        score=0.0,
        metadata={
            "source_provider": provider,
            "source": _source_label(provider),
            "error": True,
        },
    )


async def _finalize_hybrid(
    documents: list[ContextDocument],
    *,
    executed_queries: list[str],
    query: str,
    rerank_url: str | None,
    top_k: int,
) -> _HybridSearchResult:
    real = [d for d in documents if not d.metadata.get("error")]
    errored = [d for d in documents if d.metadata.get("error")]
    if not real:
        status = "unreachable" if errored else "empty"
        return _HybridSearchResult(
            executed_queries=executed_queries,
            documents=[],
            status=status,
            ranking={"operations": [], "candidate_count": 0},
        )
    diversified = await _rank_documents(real, query, rerank_url, top_k)
    ranking = _ranking_metadata(diversified, fallback_operations=["hybrid_ranking"])
    return _HybridSearchResult(
        executed_queries=executed_queries,
        documents=_reindex_documents(diversified),
        status="ok" if diversified else "empty",
        ranking=ranking,
    )


async def _run_hybrid_search(
    query: str,
    *,
    llm: LLMClient | None,
    search_url: str,
    browser_search_url: str | None = None,
    rerank_url: str | None = None,
    top_k: int,
    filters: SearchFilters | None,
    source_provider: str,
) -> _HybridSearchResult:
    if source_provider == "retrieval":
        # Path A — corpus retrieval with its own query expansion pipeline.
        # Over-fetch so MMR has candidates beyond top_k to diversify from.
        search_result = await run_expanded_search(
            query,
            llm=llm,
            search_url=search_url,
            top_k=top_k * 2,
            filters=filters,
            expand=True,
        )
        context = build_context_bundle(
            query,
            search_result.results,
            max_documents=top_k * 2,
        )
        diversified = await _rank_documents(context.documents, query, None, top_k)
        ranking = _ranking_metadata(diversified, fallback_operations=["hybrid_ranking"])
        return _HybridSearchResult(
            executed_queries=search_result.executed_queries,
            documents=[
                _document_with_metadata(
                    doc,
                    source_provider=source_provider,
                    query=query,
                    entry_point="hybrid_search",
                )
                for doc in diversified
            ],
            status="ok" if diversified else "empty",
            ranking=ranking,
        )

    if source_provider == "browser":
        # Path B-browser — playwright browser server; no query expansion (too slow).
        browser_docs: list[ContextDocument] = []
        if browser_search_url:
            browser_docs = await _run_browser_search(
                query,
                browser_search_url=browser_search_url,
                top_k=top_k * 2,
                existing_count=0,
            )
        diversified_b = await _rank_documents(browser_docs, query, rerank_url, top_k)
        ranking = _ranking_metadata(
            diversified_b, fallback_operations=["hybrid_ranking"]
        )
        return _HybridSearchResult(
            executed_queries=[query],
            documents=_reindex_documents(diversified_b),
            status="ok" if diversified_b else "empty",
            ranking=ranking,
        )

    executed_queries = _expanded_queries(query, llm)

    async def _fetch_provider(provider: str) -> list[ContextDocument]:
        if provider == "browser":
            if not browser_search_url:
                return []
            # IDs are globally reassigned by _finalize_hybrid -> _reindex_documents, so starting at 0 here is safe.
            return await _run_browser_search(
                query,
                browser_search_url=browser_search_url,
                top_k=top_k * 2,
                existing_count=0,
            )
        page_lists: list[list[SearchPage]] = list(
            await asyncio.gather(
                *[
                    search_tool(
                        expanded_query,
                        provider=provider,
                        search_url=search_url,
                        page_size=top_k,
                        timeout_seconds=5,
                        max_retries=1,
                        **(
                            {"filters": _filters_payload(filters)}
                            if provider == "retrieval"
                            else {}
                        ),
                    )
                    for expanded_query in executed_queries
                ]
            )
        )
        if _is_web_provider(provider):
            all_pages = [p for pages in page_lists for p in pages]
            enriched = await fetch_pages_concurrently(all_pages, max_chars=2000)
            it = iter(enriched)
            page_lists = [list(islice(it, len(pages))) for pages in page_lists]
        docs: list[ContextDocument] = []
        for expanded_query, pages in zip(executed_queries, page_lists):
            docs.extend(
                _documents_from_search_pages(
                    pages,
                    source_provider=provider,
                    query=expanded_query,
                    start_index=len(docs) + 1,
                    entry_point="hybrid_search",
                )
            )
        if provider == "retrieval":
            # Paired with the `_filters_payload` sent above: the retrieval
            # server may ignore the `filters` field (demo.py and hybrid.py do),
            # so what it returned is checked here rather than by each caller of
            # `_run_hybrid_search` — one of which was missed and leaked.
            docs = _enforce_access(docs, filters)
        return docs

    async def _fetch_provider_guarded(provider: str) -> list[ContextDocument]:
        try:
            return await asyncio.wait_for(_fetch_provider(provider), timeout=8.0)
        except Exception as exc:  # timeout or provider error → mark unreachable
            logger.warning("Provider %s failed/timed out: %s", provider, exc)
            return [_provider_error_doc(provider, str(exc))]

    def _has_usable(docs: list[ContextDocument]) -> bool:
        return any(not doc.metadata.get("error") for doc in docs)

    async def _fetch_web_with_fallback() -> list[ContextDocument]:
        """Cascade the web leg for 'auto': try fast web search (SerpAPI) first;
        if it yields no usable results (missing key, error, or empty), fall back
        to the browser provider. Keeps internal+web hybrid useful without a key."""
        web_docs = await _fetch_provider_guarded("serpapi")
        if _has_usable(web_docs):
            return web_docs
        if browser_search_url:
            browser_docs = await _fetch_provider_guarded("browser")
            if _has_usable(browser_docs):
                return browser_docs
        return web_docs  # propagate error doc so status reflects 'unreachable'

    if source_provider == "auto":
        # Local retrieval ∥ (SerpAPI → browser fallback); MMR picks top_k below.
        legs = await asyncio.gather(
            _fetch_provider_guarded("retrieval"),
            _fetch_web_with_fallback(),
        )
    else:
        legs = await asyncio.gather(
            *[
                _fetch_provider_guarded(provider)
                for provider in _source_providers_for(source_provider)
            ]
        )
    documents = [doc for docs in legs for doc in docs]
    return await _finalize_hybrid(
        documents,
        executed_queries=executed_queries,
        query=query,
        rerank_url=rerank_url,
        top_k=top_k,
    )


def _expanded_queries(query: str, llm: LLMClient | None) -> list[str]:
    if llm is None:
        expansions = []
    else:
        try:
            expansions = expand_keywords(query, llm)
        except Exception:
            logger.exception("Query expansion failed for hybrid web search")
            expansions = []
    queries = [query] + [e for e in expansions if e != query]
    temporal = with_temporal_context(query)
    if temporal != query and temporal not in queries:
        queries.append(temporal)
    return queries


def _documents_from_search_pages(
    pages: list[SearchPage],
    *,
    source_provider: str,
    query: str,
    start_index: int = 1,
    entry_point: str = "search_tool",
) -> list[ContextDocument]:
    documents: list[ContextDocument] = []
    for offset, page in enumerate(pages):
        index = start_index + offset
        documents.append(
            ContextDocument(
                id=f"D{index}",
                title=page.title
                or ("Search error" if page.error else f"Result {index}"),
                content=page.error or page.summary or "No summary available.",
                url=page.url or None,
                score=page.score,
                metadata={
                    # Retrieval metadata first so the labels below win on a
                    # key clash, but the document's own acl survives.
                    **(page.metadata or {}),
                    "entry_point": entry_point,
                    "source": _source_label(source_provider),
                    "source_provider": source_provider,
                    "query": query,
                    "error": bool(page.error),
                },
            )
        )
    return documents


def _document_with_metadata(
    document: ContextDocument,
    *,
    source_provider: str,
    query: str,
    entry_point: str,
) -> ContextDocument:
    return ContextDocument(
        id=document.id,
        title=document.title,
        content=document.content,
        url=document.url,
        score=document.score,
        metadata={
            **document.metadata,
            "entry_point": entry_point,
            # Prefer the document's real per-corpus/per-doc source when the
            # retrieval layer supplied one; otherwise fall back to the provider
            # label ("Local Retrieval"). source_provider always records which
            # provider fetched the document.
            "source": document.metadata.get("source") or _source_label(source_provider),
            "source_provider": source_provider,
            "query": query,
        },
    )


def _reindex_documents(documents: list[ContextDocument]) -> list[ContextDocument]:
    return [
        ContextDocument(
            id=f"D{index}",
            title=document.title,
            content=document.content,
            url=document.url,
            score=document.score,
            metadata={**document.metadata, "mmr_rank": index},
        )
        for index, document in enumerate(documents, 1)
    ]


def _search_only_answer(
    label: str,
    *,
    queries: list[str],
    documents: list[ContextDocument],
    source_provider: str,
) -> str:
    query_lines = "\n".join(f"- {query}" for query in queries)
    if not documents:
        return (
            f"{label} returned no results from {_source_label(source_provider)}.\n\n"
            f"Executed queries:\n{query_lines}"
        )
    citation_list = ", ".join(doc.citation for doc in documents)
    return (
        f"{label} returned {len(documents)} result(s) from "
        f"{_source_label(source_provider)}.\n\n"
        f"Executed queries:\n{query_lines}\n\n"
        f"Open the Sources panel to inspect {citation_list}."
    )


def _response_from_result(
    session_id: str,
    result: AnswerGenerationResult,
    messages: list[ChatMessageView],
    *,
    hook_metadata: dict[str, object] | None = None,
) -> AgentExperienceResponse:
    return AgentExperienceResponse(
        session_id=session_id,
        answer=result.answer,
        citations=result.citations,
        documents=[_document_view(document) for document in result.context.documents],
        messages=messages,
        hook_metadata=hook_metadata or {},
    )


def _document_view(document: ContextDocument) -> SourceDocumentView:
    return SourceDocumentView(
        id=document.id,
        citation=document.citation,
        title=document.title,
        content=document.content,
        url=document.url,
        score=document.score,
        metadata=document.metadata,
    )


def _optional_user_from_request(
    request: Request, store: AgenticSearchStore
) -> AuthenticatedUser | None:
    return resolve_active_user(request, store)


def _frontend_dist_path() -> Path | None:
    # parents[4], counting up from src/internal/servers/web/app.py: web →
    # servers → internal → src → the repository root, where `npm run build`
    # writes web/dist. parents[3] is `src/`, whose web/dist nothing creates, so
    # this returned None for every build and the backend quietly served the
    # inline fallback shell instead of the app.
    dist = Path(__file__).resolve().parents[4] / "web" / "dist"
    if (dist / "index.html").exists() and (dist / "assets").is_dir():
        return dist
    return None


app = create_web_app()
