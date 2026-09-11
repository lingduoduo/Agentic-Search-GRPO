"""Dev-console debug router — read-only observability for backend servers.

Mounted only when DEBUG_PANELS is enabled (see create_web_app). Proxies the
retrieval server's per-mode /internal/search/* endpoints so the browser can
inspect sparse/dense/hybrid/graph results without cross-origin calls.
The web app mounts this router with an admin dependency; custom embeddings
must attach equivalent authorization when including it.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field, field_validator

from src.internal.retrieval.query_transform_factory import (
    build_query_transform_pipeline_from_env,
)
from src.internal.observability.metric_taxonomy import flatten_metrics, group_metrics
from src.internal.observability.stage_metrics import STAGE_LATENCY
from src.internal.servers.middleware.latency_logging import ROUTE_LATENCY

_MODES = {"sparse", "dense", "hybrid", "graph"}


class DebugRetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=10, le=200)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    over_fetch: int = Field(default=2, ge=1, le=4)
    rerank: bool = False


class DebugQueryTransformRequest(BaseModel):
    query: str = Field(..., min_length=1)
    filters: dict | None = None


class DebugToolDiscoverRequest(BaseModel):
    query: str = Field(..., min_length=1)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


def _retrieval_base(search_url: str) -> str:
    """Strip a trailing /retrieve so we can address /internal/search/*."""
    return search_url.rstrip("/").removesuffix("/retrieve").rstrip("/")


def create_debug_router(
    *,
    search_url: str,
    http_client: httpx.Client | None = None,
    llm: object | None = None,
    db: object | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/debug", tags=["debug"])
    base = _retrieval_base(search_url)
    client = http_client or httpx.Client(timeout=15.0)

    @router.get("/eval-results")
    def eval_results() -> dict:
        """Read-only listing of evaluation result files.

        Scans AGENTIC_SEARCH_EVAL_RESULTS_DIR (default data/eval/) for *.json and
        returns each file's finite numeric metrics (nested dicts dotted, two
        levels deep) as ``metrics``, plus the same numbers bucketed by the
        shared taxonomy as ``groups`` (retrieval / generation / reward /
        latency / other), newest first. Confined to the configured directory;
        never raises.
        """
        import json
        import os
        from pathlib import Path

        results_dir = Path(
            os.environ.get("AGENTIC_SEARCH_EVAL_RESULTS_DIR", "data/eval")
        )
        if not results_dir.is_dir():
            return {"results": []}

        out: list[dict] = []
        for path in sorted(results_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                mtime = path.stat().st_mtime
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            metrics = flatten_metrics(data)
            out.append(
                {
                    "name": path.name,
                    "modified": mtime,
                    "metrics": metrics,
                    "groups": group_metrics(metrics),
                }
            )
        out.sort(key=lambda r: r["modified"], reverse=True)
        return {"results": out}

    @router.get("/latency")
    def latency() -> dict:
        """Per-route request latency: count, errors, p50/p95/max, slowest first.

        Reads the rolling window the latency middleware fills. Complements
        /api/debug/requests, which shows one request's stages but cannot say
        which route is slow or how often. ``stages`` splits recent requests
        into their retrieval and generation shares so a slow route can be
        attributed to one side or the other.
        """
        return {"routes": ROUTE_LATENCY.snapshot(), "stages": STAGE_LATENCY.snapshot()}

    @router.get("/tools")
    def tools() -> dict:
        """Registered tools plus the discovery catalog grouped by server.

        Reads the process-wide ``tool_registry`` singleton (seeded at web
        startup). Empty registry → empty lists, never 500.
        """
        from dataclasses import asdict

        from src.internal.tools.registry import tool_registry
        from src.internal.tools.semantic_router import catalog_from_registry

        return {
            "registered": tool_registry.all_summaries(),
            "catalog": [asdict(s) for s in catalog_from_registry(tool_registry)],
        }

    @router.post("/tools/discover")
    def tools_discover(req: DebugToolDiscoverRequest) -> dict:
        """Rank tools for *query* via the semantic router (TF-IDF, no LLM)."""
        from src.internal.tools.semantic_router import (
            SemanticRouter,
            default_tool_catalog,
        )

        return SemanticRouter(default_tool_catalog()).get_routing_details(req.query)

    @router.post("/query-transform")
    def query_transform(body: DebugQueryTransformRequest) -> dict:
        """Run *only* the query-transform pipeline (no retrieval).

        Returns the variants + merged filters + per-leg breakdown. When no
        pipeline is configured (no LLM / no QT_* flags) the factory returns None;
        we report ``active=false`` with the original query unchanged — never 500.
        """
        pipe = build_query_transform_pipeline_from_env(llm)
        if pipe is None:
            return {
                "original": body.query,
                "variants": [body.query],
                "merged_filters": {},
                "active": False,
                "legs": {},
            }
        bundle = pipe.transform(body.query, body.filters)
        return {
            "original": bundle.original,
            "variants": bundle.retrieval_variants(),
            "merged_filters": bundle.merged_filters,
            "active": True,
            "legs": {
                "sub_queries": bundle.sub_queries,
                "multi_query": bundle.multi_query,
                "rewrite": bundle.rewrite,
                "hyde_text": bundle.hyde_text,
                "step_back": bundle.step_back,
                "keywords": bundle.keywords,
            },
        }

    @router.get("/health")
    def health() -> dict:
        """Reachability of the configured servers. Never raises — up/down each."""
        servers = [{"name": "web", "url": "self", "status": "up"}]
        try:
            r = client.get(f"{base}/health", timeout=3.0)
            status = "up" if r.status_code == 200 else "down"
        except Exception:
            status = "down"
        servers.append({"name": "retrieval", "url": base, "status": status})
        return {"servers": servers}

    @router.post("/retrieval/{mode}")
    def retrieval(mode: str, body: DebugRetrievalRequest) -> Response:
        if mode not in _MODES:
            return Response(
                content=f'{{"detail":"unknown mode {mode!r}"}}',
                status_code=404,
                media_type="application/json",
            )
        payload: dict = {
            "query": body.query,
            "top_k": body.top_k,
            "rerank": body.rerank,
        }
        if mode == "hybrid":
            payload.update(
                rrf_k=body.rrf_k,
                mmr_lambda=body.mmr_lambda,
                over_fetch=body.over_fetch,
            )
        upstream = client.post(f"{base}/internal/search/{mode}", json=payload)
        # Pass the upstream status through verbatim — a down/misconfigured
        # retrieval server must not be masked as a generic 200/500.
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type="application/json",
        )

    @router.get("/requests")
    async def list_requests(request: Request) -> dict:
        """Summaries of recent captured runs (newest first). Empty when capture off."""
        store = getattr(request.app.state, "request_captures", None)
        return {"requests": store.list() if store is not None else []}

    @router.get("/request/{request_id}")
    async def get_request(request_id: str, request: Request) -> Response:
        """Full raw stage snapshot for one run; 404 if evicted or capture off."""
        import json as _json

        store = getattr(request.app.state, "request_captures", None)
        snap = store.get(request_id) if store is not None else None
        if snap is None:
            return Response(
                content=f'{{"detail":"unknown request {request_id!r}"}}',
                status_code=404,
                media_type="application/json",
            )
        return Response(
            content=_json.dumps(snap), status_code=200, media_type="application/json"
        )

    return router
