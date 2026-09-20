"""FastAPI app wrapping RetrievalService.

Replaces retrieval_server.py in M3. During M1-M2 both run in parallel.
"""

from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: dict | None = None


class SearchResultItem(BaseModel):
    doc_id: str
    title: str
    text: str
    url: str | None = None
    score: float
    metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    retrieval_mode: str
    executed_queries: list[str]
    latency_ms: float


def _to_item(r: RetrievalResult) -> SearchResultItem:
    return SearchResultItem(
        doc_id=r.doc_id,
        title=r.title,
        text=r.text,
        url=r.url,
        score=r.score,
        metadata=r.metadata,
    )


def create_app(service: RetrievalService | None = None) -> FastAPI:
    from src.internal.servers.retrieval.eval_router import create_eval_router

    _service = service or RetrievalService.from_env()
    _backend_name = os.environ.get("RETRIEVAL_BACKEND", "local")
    app = FastAPI(title="Retrieval Service", version="1.0")

    @app.middleware("http")
    async def add_api_version_header(request, call_next):
        response = await call_next(request)
        response.headers["Retrieval-API-Version"] = "1.0"
        return response

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "backend": _backend_name}

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        t0 = time.monotonic()
        results, mode = _service.search(
            request.query, top_k=request.top_k, filters=request.filters
        )
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        logger.info(
            "retrieval query completed",
            extra={
                "retrieval_mode": mode,
                "latency_ms": latency_ms,
                "top_k": request.top_k,
            },
        )
        return SearchResponse(
            results=[_to_item(r) for r in results],
            retrieval_mode=mode,
            executed_queries=[request.query],
            latency_ms=latency_ms,
        )

    # Internal eval endpoints (no auth in dev; pass require_admin in prod)
    app.include_router(create_eval_router(_service, require_admin=None))

    # Optimization admin endpoints
    from src.internal.servers.retrieval.optimize_router import create_optimize_router

    app.include_router(create_optimize_router())

    class RetrievalConfigPatch(BaseModel):
        rrf_k: int | None = None
        mmr_lambda: float | None = None
        result_cache_ttl: int | None = None

    @app.get("/api/admin/retrieval/stats")
    def retrieval_stats() -> dict:
        stats: dict = {
            "backend": _backend_name,
            "query_expansion_enabled": os.environ.get(
                "QUERY_EXPANSION_ENABLED", "false"
            ),
        }
        if hasattr(_service, "_result_cache") and _service._result_cache is not None:
            stats.update(_service._result_cache.stats())
        return stats

    @app.patch("/api/admin/retrieval/config")
    def patch_config(patch: RetrievalConfigPatch) -> dict:
        applied: list[str] = []
        if patch.rrf_k is not None:
            os.environ["RRF_K"] = str(patch.rrf_k)
            applied.append("rrf_k")
        if patch.mmr_lambda is not None:
            os.environ["MMR_LAMBDA"] = str(patch.mmr_lambda)
            applied.append("mmr_lambda")
        if (
            patch.result_cache_ttl is not None
            and hasattr(_service, "_result_cache")
            and _service._result_cache
        ):
            _service._result_cache._ttl = patch.result_cache_ttl
            applied.append("result_cache_ttl")
        return {"applied": applied}

    return app
