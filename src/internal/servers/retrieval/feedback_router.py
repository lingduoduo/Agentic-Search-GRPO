"""POST /api/feedback — persists thumbs_up / thumbs_down signals."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.internal.db import AgenticSearchStore
from src.internal.servers._auth import caller_may_use_session
from src.internal.servers.users.api import resolve_active_user


FeedbackTarget = Literal["retrieval", "generation", "overall"]


class FeedbackRequest(BaseModel):
    session_id: str
    signal: Literal["thumbs_up", "thumbs_down"]
    # What the signal is about: the documents that were retrieved, the answer
    # written from them, or the turn as a whole. A thumbs-down on "retrieval"
    # says the sources were wrong; on "generation" that the answer was wrong
    # given the sources.
    target: FeedbackTarget = "overall"
    note: str | None = None
    source: str | None = None
    parent_feedback_id: str | None = None
    correlation_id: str | None = None


class FeedbackResponse(BaseModel):
    ok: bool


def create_feedback_router(db: AgenticSearchStore) -> APIRouter:
    """Return a router with POST /api/feedback."""
    router = APIRouter(tags=["feedback"])

    @router.post("/api/feedback", response_model=FeedbackResponse)
    def submit_feedback(
        request: FeedbackRequest, http_request: Request
    ) -> FeedbackResponse:
        session = db.get_chat_session(request.session_id)
        if session is not None and not caller_may_use_session(
            session, resolve_active_user(http_request, db)
        ):
            raise HTTPException(status_code=404, detail="Session not found")
        db.save_retrieval_feedback(
            request.session_id,
            request.signal,
            target=request.target,
            note=request.note,
            source=request.source,
            parent_feedback_id=request.parent_feedback_id,
            correlation_id=request.correlation_id,
        )
        return FeedbackResponse(ok=True)

    return router
