"""Routing data model: targets, routes, and the per-query routing decision."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RetrieverTarget(str, Enum):
    SPARSE = "sparse"
    DENSE = "dense"
    HYBRID = "hybrid"
    METADATA = "metadata"
    SQL = "sql"
    GRAPH = "graph"
    API = "api"


@dataclass(frozen=True)
class Route:
    name: str
    description: str
    sources: tuple[str, ...]
    retriever: RetrieverTarget


@dataclass(frozen=True)
class RouteDecision:
    domain: str
    sources: list[str] = field(default_factory=list)
    retriever: RetrieverTarget = RetrieverTarget.HYBRID
    confidence: float = 1.0
    strategy: str = "heuristic"
