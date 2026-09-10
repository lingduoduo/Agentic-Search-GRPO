"""GET /api/admin/metrics — the process's own telemetry, readable in production.

The route-latency window and the per-stage (retrieval vs generation) window
are recorded in every web process, but until now were readable only behind
the dev-only debug router. This admin-gated read is the production surface
for them, alongside the feedback summary split by target.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.internal.auth import AuthenticatedUser
from src.internal.configs import AppSettings
from src.internal.db import AgenticSearchStore
from src.internal.observability.stage_metrics import STAGE_LATENCY
from src.internal.servers._auth import make_require_admin
from src.internal.servers.middleware.latency_logging import ROUTE_LATENCY


def create_metrics_router(db: AgenticSearchStore, settings: AppSettings) -> APIRouter:
    router = APIRouter(tags=["metrics"])
    require_admin = make_require_admin(settings)

    @router.get("/api/admin/metrics")
    def metrics(_: AuthenticatedUser = Depends(require_admin)) -> dict:
        """Route latency, per-stage latency, and feedback rates by target."""
        return {
            "routes": ROUTE_LATENCY.snapshot(),
            "stages": STAGE_LATENCY.snapshot(),
            "feedback": db.get_feedback_summary(),
        }

    return router
