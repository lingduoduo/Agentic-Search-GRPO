"""Tool-agent API router — the tool engine's own conversational surface.

Parallels search_backend/chat_backend. Endpoints:
  POST /tool/send-tool-message  — run ToolAgentLoop, stream progress + tool calls
  GET  /tool/tool-history       — past sessions for the caller (session proxy)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request

from src.context.models import ModelUnavailableError, SearchFilters
from src.internal.access.capabilities import resolve_capabilities
from src.internal.cache.interface import get_cache_backend
from src.internal.db import AgenticSearchStore
from src.internal.memory.working import (
    MAX_HISTORY_MESSAGES,
    load_working_memory,
    schedule_compression,
)
from src.internal.servers._auth import caller_may_use_session
from src.internal.servers.sse import sse_frame
from src.internal.servers.sse import sse_response
from src.internal.servers.query_and_chat.models import (
    SendToolMessageRequest,
    ToolAgentMessageResponse,
    ToolHistoryResponse,
    ToolSessionSummary,
)
from src.internal.servers.users.api import resolve_active_user

logger = logging.getLogger(__name__)

_DEGRADED_TOOL_INTRO = (
    "The tool model is temporarily unavailable, so this answer comes straight "
    "from a search of your documents."
)
_DEGRADED_SNIPPET_CHARS = 200


def _degraded_tool_answer(documents: list) -> str:
    """Render the corpus results into the answer itself: /tool has no Sources
    panel and no documents field, so a count and [Dn] labels would be empty."""
    if not documents:
        return f"{_DEGRADED_TOOL_INTRO} The search found no matching documents."
    lines = [_DEGRADED_TOOL_INTRO, ""]
    for index, doc in enumerate(documents, 1):
        heading = doc.title or "Untitled"
        if doc.url:
            heading += f" ({doc.url})"
        lines.append(f"{index}. {heading}")
        snippet = " ".join((doc.content or "").split())
        if snippet:
            if len(snippet) > _DEGRADED_SNIPPET_CHARS:
                snippet = snippet[:_DEGRADED_SNIPPET_CHARS].rstrip() + "…"
            lines.append(f"   {snippet}")
    return "\n".join(lines)


def create_tool_router(
    store: AgenticSearchStore,
    *,
    search_url: str = "http://localhost:8000/retrieve",
    resolved,
    llm=None,
    memory_compression: bool = False,
    memory_auto_curate: bool = False,
    memory_history_tokens: int | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/tool", tags=["tool"])

    def _model_backend(request: Request):
        manager = getattr(request.app.state, "search_agent_manager", None)
        tokenizer = getattr(request.app.state, "search_agent_tokenizer", None)
        return manager, tokenizer

    def _ensure_session(body: SendToolMessageRequest, user) -> str:
        if body.session_id:
            existing = store.get_chat_session(body.session_id)
            if existing is not None:
                if not caller_may_use_session(existing, user):
                    raise HTTPException(
                        status_code=404, detail="Chat session not found"
                    )
                return body.session_id
        session = store.create_chat_session(
            user_id=user.id if user else None,
            title=body.message[:80],
            metadata={"source": "tool"},
            session_id=body.session_id,
        )
        return session.id

    @router.post("/send-tool-message", response_model=None)
    async def send_tool_message(body: SendToolMessageRequest, http_request: Request):
        # Deferred to call time: tool_agent_runner lives inside src.internal.servers.web,
        # whose package __init__ eagerly imports app.py, and app.py's _register_routers
        # imports this module back to mount the router. A module-level import here would
        # deadlock that cycle when this module is the import entry point (e.g. in tests).
        from src.internal.servers.web.tool_agent_runner import (
            NO_LOCAL_MODEL_MESSAGE,
            _run_tool_agent,
        )

        manager, tokenizer = _model_backend(http_request)
        if manager is None or tokenizer is None:
            raise HTTPException(status_code=400, detail=NO_LOCAL_MODEL_MESSAGE)

        # Entitlement comes from the resolved user, never from the request
        # body. `capabilities` is the one place that mapping happens; deriving
        # "is there a user?" by hand here is how the two spellings drift.
        user = resolve_active_user(http_request, store)
        capabilities = resolve_capabilities(user, store)
        session_id = _ensure_session(body, user)
        working = load_working_memory(
            store,
            session_id,
            keep_last=MAX_HISTORY_MESSAGES,
            token_budget=memory_history_tokens,
            cache=get_cache_backend() if memory_compression else None,
        )
        history = working.messages
        store.add_chat_message(session_id, role="user", content=body.message)
        # The answer comes from the local model, so the remote llm summarizing
        # now contends with nothing; one site covers both branches.
        schedule_compression(
            working,
            session_id=session_id,
            llm=llm,
            enabled=memory_compression,
            store=store,
            user_id=capabilities.user_id,
            auto_curate=memory_auto_curate,
        )

        async def _run(on_turn=None, on_approval=None, on_escalation=None):
            answer, _citations, documents, _intent, extra = await _run_tool_agent(
                body.message,
                manager=manager,
                tokenizer=tokenizer,
                search_url=search_url,
                history=history,
                resolved=resolved,
                on_turn=on_turn,
                on_approval=on_approval,
                on_escalation=on_escalation,
                with_search_tool=body.run_search_tool,
                user_present=capabilities.user_present,
                filters=SearchFilters(access_acl=capabilities.access_acl),
            )
            answer = answer or extra.pop("_assistant_fallback", "")
            tool_calls = extra.get("tool_calls", [])
            return (
                answer,
                tool_calls,
                extra.get("num_turns", 0),
                bool(extra.get("truncated", False)),
                extra.get("tool_recovery"),
            )

        async def _search_only_answer() -> str:
            # Lazy for the same cycle as tool_agent_runner above.
            from src.internal.servers.web.app import _auto_search_pipeline
            from src.internal.servers.web.tool_agent_runner import (
                _CORPUS_SEARCH_TOP_K,
            )

            _answer, _citations, documents, *_ = await _auto_search_pipeline(
                body.message,
                # Not the model that just failed: query expansion would call it.
                llm=None,
                search_url=search_url,
                browser_search_url=None,
                rerank_url=resolved.services.rerank_url,
                top_k=_CORPUS_SEARCH_TOP_K,
                filters=SearchFilters(access_acl=capabilities.access_acl),
                history=history,
                # Corpus-only: an outage never sends the query to a web provider.
                source_provider="retrieval",
                extra={"route_degraded": "model_unavailable"},
            )
            # The shared search-only text points at a Sources panel and cites
            # [Dn]; /tool has neither, and this answer is saved to history, so
            # it must carry what was found itself.
            return _degraded_tool_answer(documents)

        if not body.stream:
            try:
                answer, tool_calls, num_turns, truncated, tool_recovery = await _run()
                store.add_chat_message(session_id, role="assistant", content=answer)
                return ToolAgentMessageResponse(
                    session_id=session_id,
                    answer=answer,
                    tool_calls=[tc.model_dump() for tc in tool_calls],
                    num_turns=num_turns,
                    truncated=truncated,
                    tool_recovery=tool_recovery,
                )
            except ModelUnavailableError as exc:
                logger.warning("Model unavailable, degrading /tool to search: %s", exc)
                try:
                    answer = await _search_only_answer()
                except Exception as fallback_exc:  # noqa: BLE001
                    logger.exception("Search-only fallback failed: %r", body.message)
                    return ToolAgentMessageResponse(
                        session_id=session_id, answer="", error=str(fallback_exc)
                    )
                store.add_chat_message(session_id, role="assistant", content=answer)
                return ToolAgentMessageResponse(
                    session_id=session_id, answer=answer, degraded="model_unavailable"
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Tool agent failed for: %r", body.message)
                return ToolAgentMessageResponse(
                    session_id=session_id, answer="", error=str(exc)
                )

        async def _gen() -> AsyncGenerator[str, None]:
            queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=100)

            async def on_turn(turn: int, tool_name, doc_count: int) -> None:
                text = (
                    f"{tool_name} · {doc_count} docs"
                    if tool_name
                    else "writing answer..."
                )
                await queue.put({"type": "progress", "turn": turn, "text": text})

            broker = getattr(http_request.app.state, "tool_approval_broker", None)
            on_approval = None
            if user is not None and not user.is_anonymous and broker is not None:
                from src.internal.servers.web.app import _request_tool_approval

                async def on_approval(approval_request):
                    return await _request_tool_approval(
                        broker, user.id, approval_request, queue
                    )

            escalation_broker = getattr(
                http_request.app.state, "tool_escalation_broker", None
            )
            on_escalation = None
            if (
                user is not None
                and not user.is_anonymous
                and escalation_broker is not None
            ):
                from src.internal.servers.web.app import _request_tool_escalation

                async def on_escalation(escalation_request):
                    return await _request_tool_escalation(
                        escalation_broker, user.id, escalation_request, queue
                    )

            task = asyncio.create_task(
                _run(
                    on_turn=on_turn,
                    on_approval=on_approval,
                    on_escalation=on_escalation,
                )
            )
            try:
                while not task.done():
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=0.05)
                        yield sse_frame(item)
                    except asyncio.TimeoutError:
                        continue
                while not queue.empty():
                    yield sse_frame(queue.get_nowait())

                answer, tool_calls, num_turns, truncated, tool_recovery = task.result()
                store.add_chat_message(session_id, role="assistant", content=answer)
                for tc in tool_calls:
                    yield sse_frame({"type": "tool_call", **tc.model_dump()})
                yield sse_frame({"type": "answer", "text": answer})
                yield sse_frame(
                    {
                        "type": "done",
                        "session_id": session_id,
                        "tool_calls": [tc.model_dump() for tc in tool_calls],
                        "num_turns": num_turns,
                        "truncated": truncated,
                        "tool_recovery": tool_recovery,
                    }
                )
            except ModelUnavailableError as exc:
                logger.warning("Model unavailable, degrading /tool to search: %s", exc)
                try:
                    answer = await _search_only_answer()
                except Exception as fallback_exc:  # noqa: BLE001
                    logger.exception("Search-only fallback failed: %r", body.message)
                    yield sse_frame({"type": "error", "detail": str(fallback_exc)})
                    return
                store.add_chat_message(session_id, role="assistant", content=answer)
                yield sse_frame({"type": "answer", "text": answer})
                yield sse_frame(
                    {
                        "type": "done",
                        "session_id": session_id,
                        "tool_calls": [],
                        "num_turns": 0,
                        "truncated": False,
                        "tool_recovery": None,
                        "degraded": "model_unavailable",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Streaming tool agent failed for: %r", body.message)
                yield sse_frame({"type": "error", "detail": str(exc)})

        return sse_response(_gen())

    @router.get("/tool-history")
    def tool_history(
        limit: int = 100,
        filter_days: int | None = None,
        http_request: Request = None,
    ) -> ToolHistoryResponse:
        if limit <= 0 or limit > 1000:
            raise HTTPException(
                status_code=400, detail="limit must be between 1 and 1000"
            )
        if filter_days is not None and filter_days <= 0:
            raise HTTPException(status_code=400, detail="filter_days must be > 0")

        user = resolve_active_user(http_request, store) if http_request else None
        if user is None or user.is_anonymous:
            return ToolHistoryResponse(sessions=[])

        sessions = store.list_sessions_for_user(
            user.id, limit=limit, filter_days=filter_days
        )
        return ToolHistoryResponse(
            sessions=[
                ToolSessionSummary(
                    session_id=s.id, title=s.title or s.id, created_at=s.created_at
                )
                for s in sessions
                if s.created_at
            ]
        )

    return router


__all__ = ["create_tool_router"]
