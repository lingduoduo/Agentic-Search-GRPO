"""Chat session management router for Agentic Search.

Provides CRUD endpoints for chat sessions backed by AgenticSearchStore.
The send-message flow is handled by POST /api/agent in the web app.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import StreamingResponse

from src.context import ChatMessage
from src.internal.auth import AuthenticatedUser
from src.internal.servers._auth import caller_may_use_session
from src.internal.db import AgenticSearchStore
from src.internal.servers.query_and_chat.models import ChatFeedbackRequest
from src.internal.servers.query_and_chat.models import ChatMessageDetail
from src.internal.servers.query_and_chat.models import ChatMessageResponse
from src.internal.servers.query_and_chat.models import ChatRenameRequest
from src.internal.servers.query_and_chat.models import ChatSessionCreationRequest
from src.internal.servers.query_and_chat.models import ChatSessionDetailResponse
from src.internal.servers.query_and_chat.models import ChatSessionDetails
from src.internal.servers.query_and_chat.models import ChatSessionsResponse
from src.internal.servers.query_and_chat.models import RenameChatSessionResponse
from src.internal.servers.query_and_chat.models import SendChatMessageRequest
from src.internal.servers.users.api import resolve_active_user

logger = logging.getLogger(__name__)
_MAX_HISTORY_MESSAGES = 40


async def _run_plain_chat(
    message: str,
    *,
    manager,
    tokenizer,
    history: list,
    on_turn=None,
) -> str:
    """Delegate to the plain-chat runner.

    Deferred import: plain_chat_runner lives inside src.internal.servers.web,
    whose package __init__ eagerly imports app.py, and app.py's
    _register_routers imports this module back to mount the router. A
    module-level import here would deadlock that cycle when this module is
    the import entry point (e.g. in tests). Defined as a real function here
    (rather than imported directly into the endpoint) so tests can
    monkeypatch chat_backend._run_plain_chat.
    """
    from src.internal.servers.web.plain_chat_runner import (
        _run_plain_chat as _impl,
    )

    return await _impl(
        message,
        manager=manager,
        tokenizer=tokenizer,
        history=history,
        on_turn=on_turn,
    )


def create_chat_router(store: AgenticSearchStore) -> APIRouter:
    """Return an APIRouter for chat session endpoints bound to *store*."""

    router = APIRouter(prefix="/chat", tags=["chat"])

    def _get_user(request: Request) -> AuthenticatedUser | None:
        return resolve_active_user(request, store)

    def _session_or_404(session_id: str, request: Request):
        """Fetch a session the caller is entitled to, or 404.

        404 rather than 403 on refusal: a 403 confirms the id exists, which is
        the one bit an id-guessing caller does not already have.

        These endpoints duplicate the `/api/sessions` surface and shipped with
        the same missing check, which is why the guard now lives in
        `servers/_auth.py` rather than in either router.
        """
        session = store.get_chat_session(session_id)
        if session is None or not caller_may_use_session(session, _get_user(request)):
            raise HTTPException(status_code=404, detail="Chat session not found")
        return session

    @router.get("/get-user-chat-sessions")
    def get_user_chat_sessions(
        request: Request,
        page_size: int = 50,
    ) -> ChatSessionsResponse:
        user = _get_user(request)
        if user is None or user.is_anonymous:
            return ChatSessionsResponse(sessions=[], has_more=False)
        sessions = store.list_sessions_for_user(user.id, limit=page_size + 1)
        has_more = len(sessions) > page_size
        return ChatSessionsResponse(
            sessions=[
                ChatSessionDetails(
                    id=s.id,
                    title=s.title,
                    created_at=s.created_at or "",
                    updated_at=s.updated_at or "",
                )
                for s in sessions[:page_size]
            ],
            has_more=has_more,
        )

    @router.get("/get-chat-session/{session_id}")
    def get_chat_session(
        session_id: str,
        request: Request,
    ) -> ChatSessionDetailResponse:
        session = _session_or_404(session_id, request)
        messages = store.list_chat_messages(session_id)
        return ChatSessionDetailResponse(
            session_id=session_id,
            title=session.title,
            messages=[
                ChatMessageDetail(
                    id=m.id,
                    session_id=m.session_id,
                    role=m.role,
                    content=m.content,
                    created_at=m.created_at or "",
                )
                for m in messages
            ],
        )

    @router.post("/create-chat-session")
    def create_new_chat_session(
        req: ChatSessionCreationRequest,
        request: Request,
    ) -> dict[str, str]:
        user = _get_user(request)
        session = store.create_chat_session(
            user_id=user.id if user and not user.is_anonymous else None,
            title=req.title,
        )
        return {"chat_session_id": session.id}

    @router.put("/rename-chat-session")
    def rename_chat_session(
        req: ChatRenameRequest,
        request: Request,
    ) -> RenameChatSessionResponse:
        _session_or_404(req.chat_session_id, request)
        found = store.update_chat_session_title(req.chat_session_id, req.name)
        if not found:
            raise HTTPException(status_code=404, detail="Chat session not found")
        return RenameChatSessionResponse(new_name=req.name)

    @router.delete("/delete-chat-session/{session_id}")
    def delete_chat_session_by_id(
        session_id: str,
        request: Request,
    ) -> None:
        _session_or_404(session_id, request)
        found = store.delete_chat_session(session_id)
        if not found:
            raise HTTPException(status_code=404, detail="Chat session not found")

    @router.post("/create-chat-message-feedback")
    def create_chat_feedback(feedback: ChatFeedbackRequest, request: Request) -> None:
        message = store.get_chat_message(feedback.chat_message_id)
        if message is None:
            raise HTTPException(status_code=404, detail="Chat message not found")
        _session_or_404(message.session_id, request)
        found = store.upsert_message_feedback(
            feedback.chat_message_id,
            feedback.is_positive,
            getattr(feedback, "feedback_text", None),
        )
        if not found:
            logger.warning(
                "Feedback for unknown message %s ignored", feedback.chat_message_id
            )

    @router.post("/send-chat-message", response_model=None)
    async def send_chat_message(body: SendChatMessageRequest, http_request: Request):
        from src.internal.servers.web.tool_agent_runner import NO_LOCAL_MODEL_MESSAGE

        manager = getattr(http_request.app.state, "search_agent_manager", None)
        tokenizer = getattr(http_request.app.state, "search_agent_tokenizer", None)
        if manager is None or tokenizer is None:
            raise HTTPException(status_code=400, detail=NO_LOCAL_MODEL_MESSAGE)

        user = _get_user(http_request)
        user_id = user.id if user and not user.is_anonymous else None
        if body.session_id and store.get_chat_session(body.session_id):
            _session_or_404(body.session_id, http_request)
            session_id = body.session_id
        else:
            session_id = store.create_chat_session(
                user_id=user_id,
                title=body.message[:80],
                metadata={"source": "chat"},
                session_id=body.session_id,
            ).id
        history = [
            ChatMessage(role=m.role, content=m.content)
            for m in store.list_chat_messages(session_id)
        ][-_MAX_HISTORY_MESSAGES:]
        store.add_chat_message(session_id, role="user", content=body.message)

        if not body.stream:
            try:
                answer = await _run_plain_chat(
                    body.message, manager=manager, tokenizer=tokenizer, history=history
                )
                store.add_chat_message(session_id, role="assistant", content=answer)
                return ChatMessageResponse(session_id=session_id, answer=answer)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Chat failed for: %r", body.message)
                return ChatMessageResponse(
                    session_id=session_id, answer="", error=str(exc)
                )

        async def _gen() -> AsyncGenerator[str, None]:
            def _sse(data: dict) -> str:
                return f"data: {_json.dumps(data)}\n\n"

            # Tokens are produced inside the run and consumed here, so they need
            # a hand-off. Unbounded on purpose: bounding it would mean either
            # dropping answer text or back-pressuring generation itself, and
            # neither is acceptable for the user-visible answer. The producer is
            # rate-limited by the model, which is far slower than the drain.
            tokens: asyncio.Queue = asyncio.Queue()
            _SENTINEL = object()

            async def on_token(text: str) -> None:
                await tokens.put(text)

            async def _run() -> str:
                try:
                    return await _run_plain_chat(
                        body.message,
                        manager=manager,
                        tokenizer=tokenizer,
                        history=history,
                        on_token=on_token,
                    )
                finally:
                    await tokens.put(_SENTINEL)

            task = asyncio.create_task(_run())
            try:
                while True:
                    item = await tokens.get()
                    if item is _SENTINEL:
                        break
                    yield _sse({"type": "token", "text": item})
                answer = await task
                store.add_chat_message(session_id, role="assistant", content=answer)
                # The authoritative text still arrives whole: a client that
                # ignores `token` events is unaffected, and any divergence
                # between the streamed chunks and the final answer is visible
                # rather than silent.
                yield _sse({"type": "answer", "text": answer})
                yield _sse({"type": "done", "session_id": session_id})
            except Exception as exc:  # noqa: BLE001
                if not task.done():
                    task.cancel()
                logger.exception("Streaming chat failed for: %r", body.message)
                yield _sse({"type": "error", "detail": str(exc)})

        return StreamingResponse(_gen(), media_type="text/event-stream")

    return router


__all__ = ["create_chat_router"]
