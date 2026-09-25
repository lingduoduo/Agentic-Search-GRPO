"""Models for retrieval-grounded chat and answer generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol

from src.context.search import SearchResult
from src.internal.configs.timeouts import get_timeout_policies

from .enums import AgentBehavior
from .enums import AnswerStyle
from .enums import SearchType
from .structured_output import StructuredCompletionMetadata


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    raw: object | None = None
    structured: StructuredCompletionMetadata = field(
        default_factory=StructuredCompletionMetadata
    )


class LLMClient(Protocol):
    def complete(
        self, messages: list[ChatMessage], **kwargs: object
    ) -> LLMResponse | str:
        """Return an LLM completion for chat-style messages."""


class ModelUnavailableError(RuntimeError):
    """The model could not be reached or refused service (connect error,
    timeout, 5xx/429, circuit open). Not raised for request/config errors."""


class LLMTimeoutError(ModelUnavailableError):
    """The LLM call exceeded its timeout."""


@dataclass(frozen=True)
class SearchFilters:
    source_types: list[str] | None = None
    document_sets: list[str] | None = None
    tags: dict[str, str] | None = None
    access_acl: list[str] | None = None
    time_cutoff: datetime | None = None

    def matches(self, metadata: dict[str, object] | None) -> bool:
        if not metadata:
            return not (
                self.source_types
                or self.document_sets
                or self.tags
                or self.access_acl
                or self.time_cutoff
            )
        if self.source_types and metadata.get("source_type") not in self.source_types:
            return False
        if self.document_sets:
            doc_sets = metadata.get("document_sets", metadata.get("document_set"))
            if not set(self.document_sets).intersection(_metadata_values(doc_sets)):
                return False
        if self.tags:
            tags = metadata.get("tags", metadata)
            if not isinstance(tags, dict):
                return False
            for key, value in self.tags.items():
                if tags.get(key) != value:
                    return False
        if self.access_acl:
            doc_acl = _metadata_acl_values(metadata)
            # Documents with no ACL metadata are treated as public (open access).
            # Only enforce when the document explicitly declares its ACL.
            if doc_acl and not doc_acl.intersection(self.access_acl):
                return False
        if self.time_cutoff:
            updated_at = metadata.get("updated_at")
            if isinstance(updated_at, datetime):
                if updated_at < self.time_cutoff:
                    return False
        return True

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.source_types:
            payload["source_types"] = self.source_types
        if self.document_sets:
            payload["document_sets"] = self.document_sets
        if self.tags:
            payload["tags"] = self.tags
        if self.access_acl:
            payload["access_acl"] = self.access_acl
        if self.time_cutoff:
            payload["time_cutoff"] = self.time_cutoff.isoformat()
        return payload


@dataclass(frozen=True)
class SearchRequest:
    query: str
    provider: SearchType = SearchType.RETRIEVAL
    top_k: int = 5
    filters: SearchFilters | None = None

    def validate(self) -> None:
        if not self.query.strip():
            raise ValueError("query is required.")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")


@dataclass(frozen=True)
class ContextDocument:
    id: str
    title: str
    content: str
    url: str | None = None
    score: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_search_result(
        cls,
        result: SearchResult,
        *,
        index: int,
        metadata: dict[str, object] | None = None,
    ) -> "ContextDocument":
        title, content = split_title_and_content(result)
        return cls(
            id=f"D{index}",
            title=title or f"Document {index}",
            content=content,
            url=result.url,
            score=result.score,
            metadata=metadata or result.metadata,
        )

    @property
    def citation(self) -> str:
        return f"[{self.id}]"


@dataclass(frozen=True)
class ContextSection:
    center: ContextDocument
    documents: list[ContextDocument]
    combined_content: str


@dataclass(frozen=True)
class EvidenceSnippet:
    citation: str
    title: str
    text: str
    score: float
    document: ContextDocument
    section: ContextSection | None = None


@dataclass(frozen=True)
class CitationVerdict:
    """Grounding verdict for a single [Dx]/[Tx] citation in a generated answer."""

    citation: str
    document_found: bool
    overlap_score: float
    is_grounded: bool
    sentence: str


@dataclass(frozen=True)
class GroundingReport:
    """Per-answer report produced by GroundingVerifier."""

    verdicts: list[CitationVerdict]
    answer_clean: str

    @property
    def dangling_citations(self) -> list[str]:
        return [v.citation for v in self.verdicts if not v.document_found]

    @property
    def ungrounded_citations(self) -> list[str]:
        return [v.citation for v in self.verdicts if not v.is_grounded]

    @property
    def grounding_rate(self) -> float:
        if not self.verdicts:
            return 1.0
        return sum(1 for v in self.verdicts if v.is_grounded) / len(self.verdicts)


@dataclass(frozen=True)
class EvidenceSource:
    """Normalized retrieval or tool evidence available to answer generation."""

    id: str
    text: str
    title: str
    url: str | None = None
    provenance: str = "retrieval"
    tool_name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AnswerClaim:
    text: str
    evidence_ids: list[str]


@dataclass(frozen=True)
class AnswerDraft:
    claims: list[AnswerClaim]
    missing_information: list[str] = field(default_factory=list)
    abstain: bool = False


@dataclass(frozen=True)
class ClaimVerdict:
    claim: AnswerClaim
    supported: bool
    overlap_scores: dict[str, float] = field(default_factory=dict)
    reason: str | None = None


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    PARTIAL = "partial"
    ABSTAINED = "abstained"


@dataclass(frozen=True)
class VerificationResult:
    verdicts: list[ClaimVerdict]
    supported_claims: list[AnswerClaim]
    unsupported_claims: list[AnswerClaim]
    unknown_evidence_ids: list[str]
    confidence: float
    retry_occurred: bool = False
    status: VerificationStatus = VerificationStatus.ABSTAINED


@dataclass(frozen=True)
class SearchContextBundle:
    query: str
    documents: list[ContextDocument]
    sections: list[ContextSection] = field(default_factory=list)

    def to_context_text(self, *, max_chars_per_doc: int = 1200) -> str:
        if not self.documents:
            return "No retrieved context."
        blocks = []
        for doc in self.documents:
            url = f"\nURL: {doc.url}" if doc.url else ""
            content = doc.content[:max_chars_per_doc].strip()
            blocks.append(f"{doc.citation} {doc.title}{url}\n{content}")
        return "\n\n".join(blocks)


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str
    messages: list[ChatMessage]


@dataclass(frozen=True)
class AgentBehaviorConfig:
    behavior: AgentBehavior = AgentBehavior.RESEARCH
    answer_style: AnswerStyle = AnswerStyle.CONCISE
    require_citations: bool = True
    max_search_rounds: int = 3
    max_context_docs: int = 5


@dataclass(frozen=True)
class GroundedGenerationConfig:
    enabled: bool = True
    max_retries: int = field(
        default_factory=lambda: get_timeout_policies().llm.grounded_max_retries
    )
    overlap_threshold: float = 0.15


@dataclass(frozen=True)
class AnswerGenerationRequest:
    question: str
    context: SearchContextBundle
    chat_history: list[ChatMessage] = field(default_factory=list)
    behavior: AgentBehaviorConfig = field(default_factory=AgentBehaviorConfig)
    verify_grounding: bool = False
    evidence: list[EvidenceSource] | None = None
    grounded_generation: GroundedGenerationConfig = field(
        default_factory=GroundedGenerationConfig
    )
    evidence_sufficiency: float | None = None
    user_memory: str | None = None


@dataclass(frozen=True)
class GenerationTimings:
    """First-response latencies for one grounded generation.

    Both are None when they did not happen: no first token when the provider
    does not stream, no first claim when none was ever committed.
    """

    llm_first_token_ms: float | None = None
    time_to_first_claim_ms: float | None = None


@dataclass(frozen=True)
class AnswerGenerationResult:
    answer: str
    citations: list[str]
    context: SearchContextBundle
    prompt: PromptBundle
    grounding_report: GroundingReport | None = None
    confidence: float | None = None
    verification_status: VerificationStatus | None = None
    abstained: bool = False
    tool_evidence: list[EvidenceSource] = field(default_factory=list)
    retry_count: int = 0
    structured_output_requested: bool = False
    structured_output_applied: bool = False
    structured_output_downgraded: bool = False
    structured_output_category: str | None = None
    timings: "GenerationTimings | None" = None


def split_title_and_content(result: SearchResult) -> tuple[str, str]:
    content = result.contents.strip()
    first_line, _, rest = content.partition("\n")
    title = result.title or first_line.strip().strip('"')
    body = rest.strip() or content
    return title, body


def _metadata_values(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(part).strip() for part in value if str(part).strip()}
    return {str(value)}


def _metadata_acl_values(metadata: dict[str, object]) -> set[str]:
    acl_values = _metadata_values(metadata.get("acl"))
    tags = metadata.get("tags")
    if isinstance(tags, dict):
        acl_values.update(_metadata_values(tags.get("acl")))
    return acl_values
