"""POST /api/feedback — persists thumbs_up / thumbs_down signals."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from src.internal.db import AgenticSearchStore


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
    def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
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
