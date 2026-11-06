"""Dataclass records for the local Agentic Search metadata store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.shared_configs.enums import EmbeddingProvider

JsonObject = dict[str, Any]


@dataclass(slots=True)
class UserRecord:
    """User identity used for access control and chat ownership."""

    id: str
    email: str | None = None
    name: str | None = None
    metadata: JsonObject = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class UserMemoryRecord:
    """A sanitized memory stored for one user."""

    id: str
    user_id: str
    memory_text: str
    metadata: JsonObject = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    category: str = "general"


@dataclass(slots=True)
class MemoryIndexNodeRecord:
    """Summary of one memory category (or the root) for one user."""

    user_id: str
    node: str
    summary: str
    memory_count: int
    dirty: bool
    updated_at: str | None = None


@dataclass(slots=True)
class UserProfileEntryRecord:
    """One entry of an LLM-consolidated user profile."""

    id: str
    user_id: str
    topic: str
    subtopic: str
    content: str
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class MemoryTrajectoryRecord:
    """Audit record of one memory-curation run."""

    id: str
    user_id: str
    session_id: str | None
    model: str
    trajectory: JsonObject = field(default_factory=dict)
    created_at: str | None = None


@dataclass(slots=True)
class GroupRecord:
    """Named group and its current user membership."""

    id: str
    name: str
    user_ids: list[str] = field(default_factory=list)
    metadata: JsonObject = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class ChatSessionRecord:
    """Conversation session metadata."""

    id: str
    user_id: str | None = None
    title: str | None = None
    metadata: JsonObject = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class ChatMessageRecord:
    """Single chat message in a conversation session."""

    id: str
    session_id: str
    role: str
    content: str
    metadata: JsonObject = field(default_factory=dict)
    created_at: str | None = None


@dataclass(slots=True)
class HookRecord:
    """Persisted configuration for an outbound webhook."""

    id: str
    name: str
    hook_point: str
    endpoint_url: str
    api_key: str | None = None
    fail_strategy: str = "soft"
    timeout_seconds: float = 5.0
    is_active: bool = True
    is_reachable: bool | None = None
    creator_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


# ---------------------------------------------------------------------------
# Chat ORM stubs used by the chat pipeline
# These are lightweight dataclass stand-ins for the full SQLAlchemy models
# that would exist in a production Postgres deployment.
# ---------------------------------------------------------------------------


@dataclass
class Persona:
    """Configuration for a chat persona / assistant."""

    id: int = 0
    name: str = "Assistant"
    system_prompt: str = ""
    task_prompt: str = ""
    datetime_aware: bool = True
    replace_base_system_prompt: bool = False
    tools: list = field(default_factory=list)
    user_files: list = field(default_factory=list)


@dataclass
class ChatSession:
    """A single chat session."""

    id: Any = None
    user_id: str | None = None
    persona_id: int | None = None
    description: str = ""
    project_id: int | None = None
    llm_override: Any = None
    persona: Any = None


@dataclass
class ChatMessage:
    """One message within a chat session."""

    id: int = 0
    session_id: Any = None
    message: str = ""
    message_type: str = "user"
    token_count: int = 0
    chat_session_id: Any = None
    reasoning_tokens: str | None = None
    is_clarification: bool = False
    processing_duration_seconds: float | None = None
    citations: dict | None = None
    files: list | None = None


@dataclass
class User:
    """Application user record."""

    id: str = ""
    email: str | None = None
    name: str | None = None
    is_anonymous: bool = False
    use_memories: bool = True


@dataclass
class UserFile:
    """A file record owned by a user."""

    id: Any = None
    user_id: str | None = None
    filename: str = ""


@dataclass
class ToolCall:
    """Persisted record for a single LLM tool call."""

    id: int = 0
    chat_message_id: int = 0
    tool_name: str = ""
    tool_arguments: dict = field(default_factory=dict)
    tool_result: str = ""
    tool_call_id: str = ""
    parent_tool_call_id: int | None = None
    tab_index: int = 0
    turn_number: int = 0


@dataclass
class SlackContext:
    """Slack-specific context for bot responses."""

    channel_id: str = ""
    thread_ts: str | None = None


@dataclass
class SearchSettings:
    """Settings for an embedding model used during indexing and search."""

    id: int = 0
    model_name: str | None = None
    normalize: bool = True
    query_prefix: str | None = None
    passage_prefix: str | None = None
    api_key: str | None = None
    provider_type: EmbeddingProvider | None = None
    api_url: str | None = None
    api_version: str | None = None
    deployment_name: str | None = None
    reduced_dimension: int | None = None
