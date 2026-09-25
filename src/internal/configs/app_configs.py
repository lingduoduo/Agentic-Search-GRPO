"""Typed environment configuration for Agentic Search.

The repo mostly passes explicit dataclass configs into individual components.
This module adds one small shared layer for values that naturally come from the
process environment: service URLs, default ports, auth secrets, and permission
sync cadence.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

from src.shared_configs.intent import (
    DEFAULT_MIN_MODULE_SCORE,
    DEFAULT_MIN_ROUTE_MARGIN,
    DEFAULT_TOP_K,
)

from .timeouts import TimeoutPolicies, get_timeout_policies, load_timeout_policies

EnvMapping = Mapping[str, str]

DEFAULT_RETRIEVAL_URL = "http://localhost:8001/retrieve"
DEFAULT_FETCH_URL = "http://localhost:8001/fetch"
DEFAULT_WEB_DB_PATH = ":memory:"
DEFAULT_AUTH_SECRET = "agentic-search-dev-secret"


@dataclass(frozen=True)
class ServiceSettings:
    """Network and persistence defaults for local services."""

    retrieval_url: str = DEFAULT_RETRIEVAL_URL
    fetch_url: str | None = None
    rerank_url: str | None = None
    web_db_path: str | Path = DEFAULT_WEB_DB_PATH
    web_top_k: int = 5
    # TTL of the process-local serving cache (retrieval rows, web-provider
    # pages, rerank scores) the web app configures at startup; 0 disables it.
    search_cache_ttl_seconds: int = 300
    # Seconds past the TTL an expired entry is kept to answer when the live
    # call fails (served labelled ``stale``); 0 disables the fallback.
    search_cache_stale_seconds: int = 3600
    retrieval_host: str = "0.0.0.0"
    retrieval_port: int = 8001
    web_host: str = "0.0.0.0"
    web_port: int = 8080


@dataclass(frozen=True)
class AuthSettings:
    """Authentication-related process settings."""

    environment: str = "development"
    workload_issuer: str | None = None
    workload_audience: str | None = None
    workload_subjects: dict[str, dict] = field(default_factory=dict, hash=False)
    secret: str = DEFAULT_AUTH_SECRET
    jwt_public_key_url: str | None = None
    super_users: tuple[str, ...] = ()
    super_api_key: str | None = None
    # Dev-only: when True, admin endpoints treat every request as a dev admin
    # (no token needed) so the local admin dashboard works without minting a
    # cookie. Default off. NEVER enable in production.
    dev_admin_bypass: bool = False

    def __post_init__(self) -> None:
        if self.environment not in {"development", "test", "production"}:
            raise ValueError(
                "AGENTIC_SEARCH_ENVIRONMENT must be development, test or production"
            )
        if self.environment == "production" and (
            not self.secret.strip()
            or self.secret == DEFAULT_AUTH_SECRET
            or self.dev_admin_bypass
        ):
            raise ValueError(
                "Production requires a non-development auth secret and disables dev admin bypass"
            )
        configured = (
            self.jwt_public_key_url,
            self.workload_issuer,
            self.workload_audience,
            self.workload_subjects,
        )
        if any(configured):
            if not all(configured):
                raise ValueError(
                    "Workload identity requires JWKS URL, issuer, audience and subject mappings"
                )
            for value in (self.jwt_public_key_url, self.workload_issuer):
                parsed = urlparse(value)
                if (
                    parsed.scheme != "https"
                    or not parsed.hostname
                    or parsed.username
                    or parsed.password
                    or parsed.fragment
                ):
                    raise ValueError(
                        "Workload identity URLs must use HTTPS without credentials or fragments"
                    )
        if not isinstance(self.workload_subjects, dict):
            raise ValueError("Workload subjects must be a JSON object")
        for subject, identity in self.workload_subjects.items():
            if (
                not isinstance(subject, str)
                or not subject
                or not isinstance(identity, dict)
            ):
                raise ValueError("Invalid workload subject mapping")
            if set(identity) - {"user_id", "group_ids", "tenant_id"}:
                raise ValueError(
                    "Workload mappings only accept user_id, group_ids and tenant_id"
                )
            if not isinstance(identity.get("user_id"), str) or not identity["user_id"]:
                raise ValueError("Workload mapping requires a local user_id")
            groups = identity.get("group_ids", [])
            if not isinstance(groups, list) or any(
                not isinstance(g, str) or not g for g in groups
            ):
                raise ValueError(
                    "Workload group_ids must be a list of nonempty strings"
                )
            if identity.get("tenant_id") is not None and (
                not isinstance(identity["tenant_id"], str) or not identity["tenant_id"]
            ):
                raise ValueError("Workload tenant_id must be a nonempty string")


@dataclass(frozen=True)
class PermissionSyncSettings:
    """Connector permission sync cadence, in seconds."""

    default_doc_sync_frequency: int = 5 * 60
    num_workers: int = 2
    doc_sync_frequency_by_source: dict[str, int] = field(
        default_factory=dict, hash=False, compare=False
    )
    group_sync_frequency_by_source: dict[str, int] = field(
        default_factory=dict, hash=False, compare=False
    )
    anonymous_access_is_public_by_source: dict[str, bool] = field(
        default_factory=dict, hash=False, compare=False
    )

    def doc_sync_frequency(self, source: str) -> int:
        return self.doc_sync_frequency_by_source.get(
            source.lower(),
            self.default_doc_sync_frequency,
        )

    def group_sync_frequency(self, source: str) -> int | None:
        return self.group_sync_frequency_by_source.get(source.lower())

    def anonymous_access_is_public(self, source: str) -> bool:
        return self.anonymous_access_is_public_by_source.get(source.lower(), False)


@dataclass(frozen=True)
class TelemetrySettings:
    posthog_api_key: str | None = None
    posthog_host: str = "https://us.i.posthog.com"
    posthog_debug_logs_enabled: bool = False


@dataclass(frozen=True)
class LLMSettings:
    """Default LLM provider used when no per-persona override is set.

    Env vars (all optional – defaults to a local Ollama-compatible endpoint):
      GEN_AI_MODEL_PROVIDER  e.g. "openai", "anthropic", "ollama_chat"
      GEN_AI_MODEL_VERSION   e.g. "gpt-4o-mini", "claude-3-5-haiku-20241022"
      GEN_AI_API_KEY         provider API key (omit for local endpoints)
      GEN_AI_API_BASE        override base URL (e.g. "http://localhost:11434/v1")
      GEN_AI_MAX_INPUT_TOKENS  context window size (default 8192)
    """

    model_provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    api_key: str | None = None
    api_base: str | None = None
    max_input_tokens: int = 8192


@dataclass(frozen=True)
class VectorDbSettings:
    """Vector-DB feature flags (vestigial: the Weaviate backend was removed;
    kept for AppSettings compatibility)."""

    disable_vector_db: bool = False
    multi_tenant: bool = False

    # Indexing behaviour
    max_chunks_per_doc_batch: int = 512
    enable_multipass_indexing: bool = False


@dataclass(frozen=True)
class AppSettings:
    """Top-level process settings for Agentic Search."""

    services: ServiceSettings = field(default_factory=ServiceSettings)
    auth: AuthSettings = field(default_factory=AuthSettings)
    permissions: PermissionSyncSettings = field(default_factory=PermissionSyncSettings)
    telemetry: TelemetrySettings = field(default_factory=TelemetrySettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    vector_db: VectorDbSettings = field(default_factory=VectorDbSettings)
    license_enforcement_enabled: bool = False
    cloud_data_plane_url: str | None = None
    # Billing / Stripe configuration (optional; only needed when using billing endpoints)
    stripe_publishable_key_override: str | None = None
    stripe_publishable_key_url: str | None = None
    web_domain: str = "http://localhost:8080"
    dev_mode: bool = False
    search_agent_model: str | None = None
    search_agent_device: str = "mps"
    search_agent_server_url: str | None = None
    tool_agent_parser: str = "json"
    tool_approval_timeout_seconds: float = 60.0
    # Remote MCP servers to pull tools from, as "name=url, name=url".
    # Empty (the default) leaves the feature off.
    mcp_servers: str | None = None
    mcp_token: str | None = None
    # Remote tool names no agent loop may be offered (they re-enter an agent).
    # Unset uses mcp_client.DEFAULT_AGENT_EXCLUDE.
    mcp_agent_exclude: str | None = None
    # Remote tool names backed by per-user storage. Unset uses
    # mcp_client.DEFAULT_USER_SCOPED.
    mcp_user_scoped: str | None = None
    # Token budget for one tool-agent run; caps a generation and the run total.
    tool_agent_max_tokens: int = 1024
    # Wall-clock cap on one local generation. The binding limit on slow local
    # hardware: MPS runs a 1.5B model at a few tokens/sec, so 120s cuts a long
    # answer mid-word long before the token budget is reached. 0 disables it.
    generation_timeout_seconds: float = 120.0
    intent_index_path: Path | None = None
    # Evaluate only requests reaching similarity; record but do not serve
    # predictions when shadow mode is enabled.
    intent_shadow_mode: bool = False
    # The route margin is the sole abstention gate. Module scores are diagnostics.
    intent_min_route_margin: float = DEFAULT_MIN_ROUTE_MARGIN
    intent_min_module_score: float = DEFAULT_MIN_MODULE_SCORE
    # Neighbors averaged per route, selected jointly with the route margin.
    intent_top_k: int = DEFAULT_TOP_K
    route_clarification: bool = True
    timeouts: TimeoutPolicies = field(default_factory=get_timeout_policies)

    def __post_init__(self) -> None:
        for name in (
            "intent_min_route_margin",
            "intent_min_module_score",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be a finite cosine similarity between 0 and 1"
                )
        if self.intent_top_k <= 0:
            raise ValueError("intent_top_k must be a positive integer")


def load_app_settings(env: EnvMapping | None = None) -> AppSettings:
    """Load settings from an environment mapping.

    Passing an explicit mapping makes tests deterministic and keeps import-time
    configuration side-effect free.
    """

    source = env if env is not None else os.environ
    timeouts = load_timeout_policies(source)
    intent_min_route_margin = get_env_float(
        source, "AGENTIC_SEARCH_INTENT_MIN_ROUTE_MARGIN", DEFAULT_MIN_ROUTE_MARGIN
    )
    intent_min_module_score = get_env_float(
        source, "AGENTIC_SEARCH_INTENT_MIN_MODULE_SCORE", DEFAULT_MIN_MODULE_SCORE
    )
    intent_top_k = get_env_int(source, "AGENTIC_SEARCH_INTENT_TOP_K", DEFAULT_TOP_K)
    intent_shadow_mode = get_env_bool(
        source, "AGENTIC_SEARCH_INTENT_SHADOW_MODE", False
    )
    intent_index_path_value = get_env_str(
        source, "AGENTIC_SEARCH_INTENT_INDEX_PATH", None
    )
    search_cache_stale_seconds = get_env_int(
        source, "AGENTIC_SEARCH_SEARCH_CACHE_STALE_SECONDS", 3600
    )
    if search_cache_stale_seconds < 0:
        raise ValueError(
            "AGENTIC_SEARCH_SEARCH_CACHE_STALE_SECONDS must not be negative."
        )
    return AppSettings(
        services=ServiceSettings(
            retrieval_url=get_env_str(
                source, "AGENTIC_SEARCH_RETRIEVAL_URL", DEFAULT_RETRIEVAL_URL
            ),
            fetch_url=get_env_str(source, "AGENTIC_SEARCH_FETCH_URL", None),
            rerank_url=get_env_str(source, "AGENTIC_SEARCH_RERANK_URL", None),
            web_db_path=get_env_str(
                source, "AGENTIC_SEARCH_WEB_DB_PATH", DEFAULT_WEB_DB_PATH
            ),
            web_top_k=get_env_int(source, "AGENTIC_SEARCH_WEB_TOP_K", 5),
            search_cache_ttl_seconds=get_env_int(
                source, "AGENTIC_SEARCH_SEARCH_CACHE_TTL", 300
            ),
            search_cache_stale_seconds=search_cache_stale_seconds,
            retrieval_host=get_env_str(
                source, "AGENTIC_SEARCH_RETRIEVAL_HOST", "0.0.0.0"
            ),
            retrieval_port=get_env_int(source, "AGENTIC_SEARCH_RETRIEVAL_PORT", 8001),
            web_host=get_env_str(source, "AGENTIC_SEARCH_WEB_HOST", "0.0.0.0"),
            web_port=get_env_int(source, "AGENTIC_SEARCH_WEB_PORT", 8080),
        ),
        auth=AuthSettings(
            environment=source.get("AGENTIC_SEARCH_ENVIRONMENT", "development"),
            workload_issuer=get_env_str(source, "AGENTIC_SEARCH_WORKLOAD_ISSUER"),
            workload_audience=get_env_str(source, "AGENTIC_SEARCH_WORKLOAD_AUDIENCE"),
            workload_subjects=json.loads(
                source.get("AGENTIC_SEARCH_WORKLOAD_SUBJECTS") or "{}"
            ),
            secret=get_env_str(
                source, "AGENTIC_SEARCH_AUTH_SECRET", DEFAULT_AUTH_SECRET
            ),
            jwt_public_key_url=get_env_str(
                source, "AGENTIC_SEARCH_JWT_PUBLIC_KEY_URL", None
            ),
            super_users=tuple(get_env_json_list(source, "AGENTIC_SEARCH_SUPER_USERS")),
            super_api_key=get_env_str(source, "AGENTIC_SEARCH_SUPER_API_KEY", None),
            dev_admin_bypass=get_env_bool(source, "AGENTIC_SEARCH_DEV_ADMIN", False),
        ),
        permissions=load_permission_sync_settings(source),
        llm=LLMSettings(
            model_provider=get_env_str(source, "GEN_AI_MODEL_PROVIDER", "openai"),
            model_name=get_env_str(source, "GEN_AI_MODEL_VERSION", "gpt-4o-mini"),
            api_key=get_env_str(source, "GEN_AI_API_KEY", None),
            api_base=get_env_str(source, "GEN_AI_API_BASE", None),
            max_input_tokens=get_env_int(source, "GEN_AI_MAX_INPUT_TOKENS", 8192),
        ),
        telemetry=TelemetrySettings(
            posthog_api_key=get_env_str(source, "POSTHOG_API_KEY", None),
            posthog_host=get_env_str(
                source, "POSTHOG_HOST", "https://us.i.posthog.com"
            ),
            posthog_debug_logs_enabled=get_env_bool(
                source, "POSTHOG_DEBUG_LOGS_ENABLED", False
            ),
        ),
        vector_db=VectorDbSettings(
            disable_vector_db=get_env_bool(source, "DISABLE_VECTOR_DB", False),
            multi_tenant=get_env_bool(source, "MULTI_TENANT", False),
            max_chunks_per_doc_batch=get_env_int(
                source, "MAX_CHUNKS_PER_DOC_BATCH", 512
            ),
            enable_multipass_indexing=get_env_bool(
                source, "ENABLE_MULTIPASS_INDEXING", False
            ),
        ),
        license_enforcement_enabled=get_env_bool(
            source, "AGENTIC_SEARCH_LICENSE_ENFORCEMENT_ENABLED", False
        ),
        cloud_data_plane_url=get_env_str(
            source, "AGENTIC_SEARCH_CLOUD_DATA_PLANE_URL", None
        ),
        stripe_publishable_key_override=get_env_str(
            source, "STRIPE_PUBLISHABLE_KEY_OVERRIDE", None
        ),
        stripe_publishable_key_url=get_env_str(
            source, "STRIPE_PUBLISHABLE_KEY_URL", None
        ),
        web_domain=get_env_str(source, "WEB_DOMAIN", "http://localhost:8080"),
        dev_mode=get_env_bool(source, "DEV_MODE", False),
        search_agent_model=get_env_str(source, "SEARCH_AGENT_MODEL", None),
        search_agent_device=get_env_str(source, "SEARCH_AGENT_DEVICE", "mps"),
        search_agent_server_url=get_env_str(source, "SEARCH_AGENT_SERVER_URL", None),
        tool_agent_parser=get_env_str(source, "TOOL_AGENT_PARSER", "json"),
        tool_approval_timeout_seconds=timeouts.tool_loop.approval_timeout_seconds,
        mcp_servers=get_env_str(source, "AGENTIC_SEARCH_MCP_SERVERS", None),
        mcp_token=get_env_str(source, "AGENTIC_SEARCH_MCP_TOKEN", None),
        mcp_agent_exclude=get_env_str(source, "AGENTIC_SEARCH_MCP_AGENT_EXCLUDE", None),
        mcp_user_scoped=get_env_str(source, "AGENTIC_SEARCH_MCP_USER_SCOPED", None),
        tool_agent_max_tokens=get_env_int(source, "TOOL_AGENT_MAX_TOKENS", 1024),
        generation_timeout_seconds=timeouts.llm.local_generation_timeout_seconds,
        intent_index_path=(
            Path(intent_index_path_value) if intent_index_path_value else None
        ),
        intent_shadow_mode=intent_shadow_mode,
        intent_min_route_margin=intent_min_route_margin,
        intent_min_module_score=intent_min_module_score,
        intent_top_k=intent_top_k,
        route_clarification=get_env_bool(
            source, "AGENTIC_SEARCH_ROUTE_CLARIFICATION", True
        ),
        timeouts=timeouts,
    )


def load_permission_sync_settings(
    env: EnvMapping | None = None,
) -> PermissionSyncSettings:
    source = env if env is not None else os.environ
    return PermissionSyncSettings(
        default_doc_sync_frequency=get_env_int(
            source,
            "AGENTIC_SEARCH_DEFAULT_PERMISSION_DOC_SYNC_FREQUENCY",
            get_env_int(source, "DEFAULT_PERMISSION_DOC_SYNC_FREQUENCY", 5 * 60),
        ),
        num_workers=get_env_int(
            source,
            "AGENTIC_SEARCH_NUM_PERMISSION_WORKERS",
            get_env_int(source, "NUM_PERMISSION_WORKERS", 2),
        ),
        doc_sync_frequency_by_source={
            "confluence": get_env_int(
                source, "CONFLUENCE_PERMISSION_DOC_SYNC_FREQUENCY", 30 * 60
            ),
            "jira": get_env_int(source, "JIRA_PERMISSION_DOC_SYNC_FREQUENCY", 30 * 60),
            "github": get_env_int(
                source, "GITHUB_PERMISSION_DOC_SYNC_FREQUENCY", 5 * 60
            ),
            "slack": get_env_int(source, "SLACK_PERMISSION_DOC_SYNC_FREQUENCY", 5 * 60),
            "teams": get_env_int(source, "TEAMS_PERMISSION_DOC_SYNC_FREQUENCY", 5 * 60),
            "sharepoint": get_env_int(
                source, "SHAREPOINT_PERMISSION_DOC_SYNC_FREQUENCY", 30 * 60
            ),
        },
        group_sync_frequency_by_source={
            "confluence": get_env_int(
                source, "CONFLUENCE_PERMISSION_GROUP_SYNC_FREQUENCY", 30 * 60
            ),
            "jira": get_env_int(
                source, "JIRA_PERMISSION_GROUP_SYNC_FREQUENCY", 30 * 60
            ),
            "google_drive": get_env_int(
                source, "GOOGLE_DRIVE_PERMISSION_GROUP_SYNC_FREQUENCY", 5 * 60
            ),
            "github": get_env_int(
                source, "GITHUB_PERMISSION_GROUP_SYNC_FREQUENCY", 5 * 60
            ),
            "sharepoint": get_env_int(
                source, "SHAREPOINT_PERMISSION_GROUP_SYNC_FREQUENCY", 5 * 60
            ),
        },
        anonymous_access_is_public_by_source={
            "confluence": get_env_bool(
                source, "CONFLUENCE_ANONYMOUS_ACCESS_IS_PUBLIC", False
            ),
        },
    )


def get_env_str(env: EnvMapping, name: str, default: str | None = None) -> str | None:
    value = env.get(name)
    if value is None or value == "":
        return default
    return value


def get_env_int(env: EnvMapping, name: str, default: int) -> int:
    value = env.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def get_env_float(env: EnvMapping, name: str, default: float) -> float:
    value = env.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float.") from exc


def get_env_bool(env: EnvMapping, name: str, default: bool = False) -> bool:
    value = env.get(name)
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean.")


def get_env_json_list(env: EnvMapping, name: str) -> list[str]:
    value = env.get(name)
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a JSON list.") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{name} must be a JSON list.")
    return [str(item) for item in parsed]


__all__ = [
    "AppSettings",
    "AuthSettings",
    "DEFAULT_AUTH_SECRET",
    "DEFAULT_FETCH_URL",
    "DEFAULT_RETRIEVAL_URL",
    "DEFAULT_WEB_DB_PATH",
    "LLMSettings",
    "PermissionSyncSettings",
    "ServiceSettings",
    "TelemetrySettings",
    "VectorDbSettings",
    "get_env_bool",
    "get_env_float",
    "get_env_int",
    "get_env_json_list",
    "get_env_str",
    "load_app_settings",
    "load_permission_sync_settings",
    "MULTI_TENANT",
]

INTEGRATION_TESTS_MODE: bool = False

MULTI_TENANT: bool = os.environ.get("MULTI_TENANT", "").lower() in {"1", "true", "yes"}
