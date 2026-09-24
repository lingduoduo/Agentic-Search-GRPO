"""Shared runtime state models for agent orchestration.

These containers keep request, retrieval, and tool execution state explicit
without pulling in training dependencies. They are intentionally small, slotted
dataclasses because agent loops may create many of them during rollout
generation.

Scope note: this module holds only state the loops actually keep. It previously
also carried a planning/routing vocabulary -- ``Plan``, ``PlanStep``,
``TaskNode``, ``TaskType``, ``ToolType``, ``RouteDecision``,
``RetrievedDocument``, ``ToolCall``, ``ToolResult`` -- that no loop ever wrote
to, reachable only through package re-exports. Routing lives in
``src/internal/routing/``; planning, where it exists, is the search loop's
subquestion tracking. Re-adding a type here is a claim that a loop maintains it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..context.search import SearchResult

__all__ = [
    "TaskStatus",
    "UserRequest",
    "PerformanceMetrics",
    "ToolExecutionResult",
    "AgentState",
    "Retriever",
    "Citation",
]


class TaskStatus(Enum):
    """Terminal status of one executed tool call.

    Only outcomes, deliberately: a tool may be retried inside one call; the
    status is the final outcome, so there is no PENDING/RUNNING/RETRYING to
    observe. A tool is invoked inline by ``ToolRegistry.invoke`` and reports
    how it finished.
    """

    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(slots=True)
class UserRequest:
    user_id: str
    channel: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PerformanceMetrics:
    execution_time: float = 0.0
    cost_estimate: float = 0.0
    memory_usage: float = 0.0
    success_rate: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolExecutionResult:
    tool_name: str
    status: TaskStatus
    result: Any
    arguments: dict[str, Any] = field(default_factory=dict)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    error_code: str | None = None
    error_message: str | None = None
    optimization_suggestions: list[str] = field(default_factory=list)
    retry_count: int = 0

    @property
    def success(self) -> bool:
        return self.status is TaskStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Retriever(Enum):
    """The retriever backends the policy can choose between per search."""

    WEB = "web"
    VECTOR_DB = "vector_db"


@dataclass(slots=True)
class Citation:
    """A reference linking an answer claim back to a retrieved document."""

    doc_id: str
    marker: str
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentState:
    """Retrieval state for one agent run.

    Every field here is written by a loop or a component: ``SearchAgentLoop``
    constructs it, ``EvidenceJudge`` sets the score, ``SearchTool`` and
    ``RerankerTool`` record rounds, ``AnswerGenerator`` sets citations.
    """

    request_id: str
    user_request: UserRequest
    question: str = ""
    previous_queries: list[str] = field(default_factory=list)
    retrieved_docs: list[SearchResult] = field(default_factory=list)
    evidence_score: float = 0.0
    search_rounds: int = 0
    citations: list[Citation] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.question:
            self.question = self.user_request.message

    def record_search_round(self, queries: list[str], docs: list[SearchResult]) -> None:
        for query in queries:
            if query not in self.previous_queries:
                self.previous_queries.append(query)
        self.retrieved_docs.extend(docs)
        self.search_rounds += 1

    def record_rerank(self, reordered_docs: list[SearchResult]) -> None:
        self.retrieved_docs = list(reordered_docs)

    def set_evidence(self, score: float) -> None:
        self.evidence_score = max(0.0, min(1.0, score))

    def set_citations(self, citations: list[Citation]) -> None:
        self.citations = list(citations)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
