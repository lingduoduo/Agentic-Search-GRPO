"""Evaluation API router.

The previous version dispatched an eval run to Celery
(EVAL_RUN_TASK) and required a cloud superuser token.

This repo has no Celery worker. The eval endpoint runs
``run_expanded_search`` synchronously and returns structured results,
making it useful for local quality measurement without any task queue.
Admin access (AppSettings.auth.super_users) is required.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter
from fastapi import Depends
from pydantic import BaseModel
from pydantic import Field

from src.internal.auth import AuthenticatedUser
from src.internal.configs import AppSettings
from src.internal.db import AgenticSearchStore
from src.context.models import SearchFilters
from src.internal.search.process_search_query import run_expanded_search
from src.internal.servers._auth import make_require_admin

logger = logging.getLogger(__name__)

# `asyncio` holds only weak references to running tasks, so a fire-and-forget
# background eval could be garbage-collected mid-flight -- silently, since the
# log line it writes on completion is the route's only observable output. Hold
# a strong reference for the task's lifetime and drop it when it finishes.
_background_tasks: set[asyncio.Task] = set()


def _run_detached(coro) -> asyncio.Task:
    task = asyncio.ensure_future(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class EvalConfigurationOptions(BaseModel):
    """Parameters for a single evaluation run."""

    query: str = Field(..., min_length=1)
    num_hits: int = Field(default=5, ge=1, le=50)
    run_query_expansion: bool = False
    filters: SearchFilters | None = None


class EvalResultDoc(BaseModel):
    title: str | None
    url: str | None
    score: float
    content_preview: str


class EvalRunResult(BaseModel):
    success: bool
    query: str
    executed_queries: list[str]
    num_results: int
    results: list[EvalResultDoc]
    error: str | None = None


class EvalRunAck(BaseModel):
    """Returned immediately when an eval is queued (Celery mode) or completed."""

    success: bool
    message: str | None = None


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


class EvalsSummary(BaseModel):
    thumbs_up_rate: float
    ctr: float
    rated_queries: int
    # Per feedback target (retrieval / generation / overall): rated count and
    # thumbs-up rate, so retrieval complaints read apart from answer complaints.
    by_target: dict[str, dict[str, float]] = Field(default_factory=dict)


def create_evals_router(
    app_settings: AppSettings,
    *,
    search_url: str = "http://localhost:8000/retrieve",
    db: "AgenticSearchStore | None" = None,
    require_admin: object = None,
) -> APIRouter:
    """Return an APIRouter for evaluation endpoints bound to *app_settings*."""

    router = APIRouter(tags=["evals"])

    _require_admin = (
        require_admin if require_admin is not None else make_require_admin(app_settings)
    )

    @router.post("/evals/eval_run", response_model=EvalRunResult)
    async def eval_run(
        request: EvalConfigurationOptions,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> EvalRunResult:
        """Run a search evaluation synchronously and return structured results.

        Unlike the previous version (Celery async), this endpoint blocks
        until the search completes and returns results directly — suitable for
        local quality measurement and CI pipelines.
        """
        logger.info("Eval run: query=%r num_hits=%d", request.query, request.num_hits)
        try:
            result = await run_expanded_search(
                request.query,
                search_url=search_url,
                top_k=request.num_hits,
                filters=request.filters,
                expand=request.run_query_expansion,
            )
            return EvalRunResult(
                success=True,
                query=request.query,
                executed_queries=result.executed_queries,
                num_results=len(result.results),
                results=[
                    EvalResultDoc(
                        title=r.title,
                        url=r.url,
                        score=r.score,
                        content_preview=r.contents[:200],
                    )
                    for r in result.results
                ],
            )
        except Exception as exc:
            logger.exception("Eval run failed for query %r", request.query)
            return EvalRunResult(
                success=False,
                query=request.query,
                executed_queries=[request.query],
                num_results=0,
                results=[],
                error=str(exc),
            )

    @router.post("/evals/eval_run_ack")
    async def eval_run_ack(
        request: EvalConfigurationOptions,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> EvalRunAck:
        """Fire-and-forget variant: queues the eval and returns immediately.

        The search runs in a background task so the response is instant.
        Check server logs for results.
        """

        async def _run_background() -> None:
            try:
                result = await run_expanded_search(
                    request.query,
                    search_url=search_url,
                    top_k=request.num_hits,
                    filters=request.filters,
                    expand=request.run_query_expansion,
                )
                logger.info(
                    "Background eval complete: query=%r hits=%d",
                    request.query,
                    len(result.results),
                )
            except Exception as exc:
                logger.error("Background eval failed: %s", exc)

        _run_detached(_run_background())
        return EvalRunAck(success=True, message="Eval queued.")

    @router.get("/api/admin/evals/summary", response_model=EvalsSummary)
    def evals_summary(
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> EvalsSummary:
        """Return aggregate retrieval feedback metrics (admin only)."""
        if db is None:
            return EvalsSummary(thumbs_up_rate=0.0, ctr=0.0, rated_queries=0)
        summary = db.get_feedback_summary()
        return EvalsSummary(
            thumbs_up_rate=float(summary["thumbs_up_rate"]),
            ctr=float(summary["ctr"]),
            rated_queries=int(summary["rated_queries"]),
            by_target=dict(summary.get("by_target", {})),
        )

    return router


__all__ = [
    "EvalConfigurationOptions",
    "EvalRunAck",
    "EvalRunResult",
    "create_evals_router",
]
