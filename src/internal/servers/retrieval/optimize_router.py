"""Admin endpoints for offline parameter optimization.

POST /internal/optimize/bm25-tune       → BM25Params JSON
POST /internal/optimize/fusion-weights  → FusionWeights JSON
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.internal.retrieval.bm25_tuner import BM25Tuner
from src.internal.retrieval.fusion_learner import FusionLearner

logger = logging.getLogger(__name__)


class BM25TuneRequest(BaseModel):
    qa_pairs_path: str
    k1_range: list[float] | None = None
    b_range: list[float] | None = None


class FusionWeightsRequest(BaseModel):
    qa_pairs_path: str
    w_sparse_range: list[float] | None = None


def _make_service(*_args: Any) -> object:
    from src.internal.retrieval.service import RetrievalService

    return RetrievalService.from_env()


def create_optimize_router(require_admin: Callable | None = None) -> APIRouter:
    deps = [Depends(require_admin)] if require_admin is not None else []
    router = APIRouter(prefix="/internal/optimize", dependencies=deps)

    @router.post("/bm25-tune")
    def bm25_tune(req: BM25TuneRequest) -> dict:
        def factory(k1, b):
            return _make_service(k1, b)

        tuner = BM25Tuner(factory)
        params = tuner.grid_search(
            req.qa_pairs_path,
            k1_range=req.k1_range,
            b_range=req.b_range,
        )
        return asdict(params)

    @router.post("/fusion-weights")
    def fusion_weights(req: FusionWeightsRequest) -> dict:
        def factory(w_s, w_d):
            return _make_service(w_s, w_d)

        learner = FusionLearner(factory)
        weights = learner.fit(req.qa_pairs_path, w_sparse_range=req.w_sparse_range)
        return asdict(weights)

    return router
