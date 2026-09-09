"""Dependency-free types shared by serving-time intent recognition."""

from __future__ import annotations

from dataclasses import dataclass, field
from src.shared_configs.intent import RouteStrategy as RouteStrategy


@dataclass(frozen=True)
class ClarificationOption:
    """One route the user can choose when recognition cannot decide."""

    route: str
    label: str


@dataclass(frozen=True)
class Clarification:
    """A question to ask instead of guessing a route."""

    question: str
    options: tuple[ClarificationOption, ...]


@dataclass(frozen=True)
class RouteDecision:
    """A recognized route with optional clarification and diagnostics."""

    strategy: RouteStrategy
    clarification: Clarification | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class IntentModelDecision:
    """A valid similarity prediction with serving diagnostics."""

    strategy: RouteStrategy
    confidence: float
    latency_ms: float
    modules: tuple[str, ...] = ()
    composite: bool = False
    margin: float = 0.0
    abstain_reason: str | None = None
