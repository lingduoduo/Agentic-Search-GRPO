"""Per-query router: heuristic default with optional logical/semantic strategies."""

from __future__ import annotations

import logging

from .registry import RouteRegistry
from .route import RetrieverTarget, Route, RouteDecision

logger = logging.getLogger(__name__)

_SQL_CUES = (
    "how many",
    "how much",
    "count of",
    "number of",
    "total ",
    "sum of",
    "average ",
    "avg ",
    "per year",
    "per month",
    "per ",
    "group by",
    "most ",
    "least ",
    "top ",
    "ranked by",
    "aggregate",
)
_GRAPH_CUES = (
    "connected to",
    "related to",
    "relationship between",
    "related entities",
    "linked to",
    "associated with",
    "path between",
    "neighbors of",
    "depends on",
)
_API_CUES = (
    "current ",
    "latest ",
    "real-time",
    "real time",
    "right now",
    "today's",
    "live ",
    "as of now",
    "up to date",
    "up-to-date",
)


def _matches(query: str, cues: tuple[str, ...]) -> bool:
    q = query.lower()
    return any(cue in q for cue in cues)


class Router:
    def __init__(
        self,
        registry: RouteRegistry,
        llm: object | None = None,
        embedder: object | None = None,
        logical: bool = False,
        semantic: bool = False,
    ) -> None:
        self._registry = registry
        self._llm = llm
        self._embedder = embedder
        self._logical = logical
        self._semantic = semantic

    def route(self, query: str) -> RouteDecision:
        if self._logical and self._llm is not None:
            try:
                return self._logical_route(query)
            except Exception as exc:
                logger.warning("logical route failed, falling back: %s", exc)
        if self._semantic and self._embedder is not None:
            try:
                return self._semantic_route(query)
            except Exception as exc:
                logger.warning("semantic route failed, falling back: %s", exc)
        return self._heuristic(query)

    def _decision(
        self, route: Route, *, confidence: float, strategy: str
    ) -> RouteDecision:
        return RouteDecision(
            domain=route.name,
            sources=list(route.sources),
            retriever=route.retriever,
            confidence=confidence,
            strategy=strategy,
        )

    def _heuristic(self, query: str) -> RouteDecision:
        if _matches(query, _SQL_CUES):
            target = RetrieverTarget.SQL
        elif _matches(query, _GRAPH_CUES):
            target = RetrieverTarget.GRAPH
        elif _matches(query, _API_CUES):
            target = RetrieverTarget.API
        else:
            return self._decision(
                self._registry.default(), confidence=0.5, strategy="heuristic"
            )
        route = self._registry.by_retriever(target)
        if route is None:  # target not registered → safe default
            return self._decision(
                self._registry.default(), confidence=0.5, strategy="heuristic"
            )
        return self._decision(route, confidence=0.7, strategy="heuristic")

    def _logical_route(self, query: str) -> RouteDecision:
        from src.context.models import ChatMessage

        names = ", ".join(r.name for r in self._registry.routes)
        catalog = "\n".join(
            f"- {r.name}: {r.description}" for r in self._registry.routes
        )
        prompt = (
            "Choose the single best route for the user's query.\n"
            f"Routes:\n{catalog}\n\n"
            f"Answer with exactly one route name from: {names}.\n"
            f"Query: {query}\nRoute:"
        )
        resp = self._llm.complete([ChatMessage(role="user", content=prompt)])
        parts = (getattr(resp, "text", None) or str(resp)).strip().lower().split()
        label = parts[0] if parts else ""
        route = self._registry.get(label)
        if route is None:
            return self._heuristic(query)
        return self._decision(route, confidence=0.9, strategy="logical")

    def _semantic_route(self, query: str) -> RouteDecision:
        from .semantic_router import cosine_route

        route, score = cosine_route(query, self._registry.routes, self._embedder)
        return self._decision(
            route, confidence=round(float(score), 4), strategy="semantic"
        )
