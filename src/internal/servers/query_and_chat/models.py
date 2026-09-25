"""Request/response models for the search and query API.

These models mirror the structure of the query_and_chat models
but are built on top of the types that exist in this repo:
  - SearchResult  (src/context/search.py)
  - SearchFilters (src/context/models.py)
  - SearchQueryResult (src/search/process_search_query.py)

repo-specific concepts (StandardAnswer, InferenceSection, BaseFilters,
SearchDoc) are replaced with repo-native equivalents or omitted.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from src.context.models import SearchFilters
from src.context.search import SearchResult


# ---------------------------------------------------------------------------
# Search flow classification
# ---------------------------------------------------------------------------


class SearchFlowClassificationRequest(BaseModel):
    user_query: str


class SearchFlowClassificationResponse(BaseModel):
    is_search_flow: bool


# ---------------------------------------------------------------------------
# Send search query
# ---------------------------------------------------------------------------


class SendSearchQueryRequest(BaseModel):
    search_query: str
    filters: SearchFilters | None = None
    run_query_expansion: bool = False
    num_hits: int = Field(default=10, ge=1, le=100)
    stream: bool = False


# ---------------------------------------------------------------------------
# Search document view
# ---------------------------------------------------------------------------


class SearchDocWithContent(BaseModel):
    """A search result with optional full content, suitable for API responses."""

    title: str | None = None
    url: str | None = None
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_search_result(cls, result: SearchResult) -> "SearchDocWithContent":
        return cls(
            title=result.title,
            url=result.url,
            content=result.contents,
            score=result.score,
            metadata=result.metadata,
        )


# ---------------------------------------------------------------------------
# Search response
# ---------------------------------------------------------------------------


class SearchFullResponse(BaseModel):
    all_executed_queries: list[str]
    search_docs: list[SearchDocWithContent]
    llm_selected_doc_ids: list[str] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Search history
# ---------------------------------------------------------------------------


class SearchQueryResponse(BaseModel):
    """Represents one past search interaction (session title as the query)."""

    query: str
    query_expansions: list[str] | None = None
    created_at: datetime


class SearchHistoryResponse(BaseModel):
    search_queries: list[SearchQueryResponse]


# ---------------------------------------------------------------------------
# Chat message request (used by the LLM loop and process_message)
# ---------------------------------------------------------------------------


class SendMessageRequest(BaseModel):
    """Request payload for sending a user message in a chat session."""

    message: str
    chat_session_id: Any | None = None
    persona_id: int | None = None
    search_doc_ids: list[str] | None = None
    file_descriptors: list[Any] = []
    parent_message_id: int | None = None
    prompt_id: int | None = None
    retrieval_options: Any | None = None
    regenerate: bool = False
    use_existing_user_message: bool = False
    no_ai_answer: bool = False


class Placement(BaseModel):
    """Token placement metadata for streaming packets."""

    turn_index: int = 0
    tab_index: int | None = None


# ---------------------------------------------------------------------------
# Chat session / message models
# ---------------------------------------------------------------------------


class ChatSessionCreationRequest(BaseModel):
    title: str | None = None


class ChatSessionDetails(BaseModel):
    id: str
    title: str | None
    created_at: str
    updated_at: str


class ChatSessionsResponse(BaseModel):
    sessions: list[ChatSessionDetails]
    has_more: bool = False


class ChatMessageDetail(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    created_at: str


class ChatSessionDetailResponse(BaseModel):
    session_id: str
    title: str | None
    messages: list[ChatMessageDetail]


class ChatRenameRequest(BaseModel):
    chat_session_id: str
    name: str


class RenameChatSessionResponse(BaseModel):
    new_name: str


class ChatFeedbackRequest(BaseModel):
    chat_message_id: str
    is_positive: bool | None = None
    feedback_text: str | None = None


# ---------------------------------------------------------------------------
# Tool-agent message / history
# ---------------------------------------------------------------------------


class SendToolMessageRequest(BaseModel):
    session_id: str | None = None
    message: str
    run_search_tool: bool = True
    stream: bool = True


class ToolAgentMessageResponse(BaseModel):
    session_id: str
    answer: str
    tool_calls: list[dict] = Field(default_factory=list)
    num_turns: int = 0
    # The answer is a fragment: a generation hit the wall-clock stop.
    truncated: bool = False
    tool_recovery: dict | None = None
    # "model_unavailable": the model was down and `answer` is the corpus-only
    # search-only fallback.
    degraded: str | None = None
    error: str | None = None


class ToolSessionSummary(BaseModel):
    session_id: str
    title: str
    created_at: datetime


class ToolHistoryResponse(BaseModel):
    sessions: list[ToolSessionSummary] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Chat message (direct LLM, no retrieval/tools)
# ---------------------------------------------------------------------------


class SendChatMessageRequest(BaseModel):
    session_id: str | None = None
    message: str
    stream: bool = True


class ChatMessageResponse(BaseModel):
    session_id: str
    answer: str
    # "model_unavailable": the model was down and `answer` is a fixed notice.
    degraded: str | None = None
    error: str | None = None


__all__ = [
    "ChatFeedbackRequest",
    "ChatMessageDetail",
    "ChatRenameRequest",
    "ChatSessionCreationRequest",
    "ChatMessageResponse",
    "ChatSessionDetailResponse",
    "ChatSessionDetails",
    "ChatSessionsResponse",
    "RenameChatSessionResponse",
    "SendChatMessageRequest",
    "SearchDocWithContent",
    "SearchFlowClassificationRequest",
    "SearchFlowClassificationResponse",
    "SearchFullResponse",
    "SearchHistoryResponse",
    "SearchQueryResponse",
    "SendSearchQueryRequest",
    "SendToolMessageRequest",
    "ToolAgentMessageResponse",
    "ToolHistoryResponse",
    "ToolSessionSummary",
]
