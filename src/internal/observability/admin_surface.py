"""Admin and observability summary helpers."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.internal.configs import AppSettings
from src.internal.db import AgenticSearchStore
from src.internal.tools import tool_registry


def citeable_tool_count() -> int:
    return sum(1 for t in tool_registry.list_tools() if t.citeable)


class AdminSurfaceMetric(BaseModel):
    label: str
    value: str
    detail: str


class AdminSurfaceCard(BaseModel):
    key: str
    title: str
    status: str
    tone: str = Field(pattern="^(good|watch|neutral)$")
    description: str
    items: list[str]


class AdminSurfaceSummary(BaseModel):
    health_label: str = "Operational readiness"
    health_score: int
    metrics: list[AdminSurfaceMetric]
    sections: list[AdminSurfaceCard]


def build_admin_surface_summary(
    store: AgenticSearchStore,
    settings: AppSettings,
) -> AdminSurfaceSummary:
    """Build one compact admin/observability snapshot from local state."""

    users = store.list_users()
    groups = store.list_groups()
    hooks = store.list_hooks()
    scim_tokens = store.list_scim_tokens()
    scim_user_mappings = sum(
        1 for user in users if store.get_scim_user_mapping(user.id)
    )
    scim_group_mappings = sum(
        1 for group in groups if store.get_scim_group_mapping(group.id)
    )

    active_hooks = sum(1 for hook in hooks if hook.is_active)

    # Only the "auth" card below can report a "watch" tone now that the
    # connectors/indexing signals are gone; base the score on it.
    health_score = 100 if settings.auth.super_users else 80

    return AdminSurfaceSummary(
        health_score=health_score,
        metrics=[
            AdminSurfaceMetric(
                label="Users/groups",
                value=f"{len(users)}/{len(groups)}",
                detail=f"{scim_user_mappings + scim_group_mappings} SCIM mapped",
            ),
            AdminSurfaceMetric(
                label="Tools/actions",
                value=str(len(tool_registry.list_tools())),
                detail=f"{active_hooks} active hooks",
            ),
        ],
        sections=[
            AdminSurfaceCard(
                key="access",
                title="Users and groups",
                status="Synced" if scim_tokens else "Local",
                tone="good" if scim_tokens else "neutral",
                description="Internal groups, external group mappings, and document ACLs.",
                items=[
                    f"{len(users)} users, {len(groups)} groups",
                    f"{scim_user_mappings} user mappings, {scim_group_mappings} group mappings",
                ],
            ),
            AdminSurfaceCard(
                key="auth",
                title="Auth controls",
                status="Enforced" if settings.auth.super_users else "Dev",
                tone="good" if settings.auth.super_users else "watch",
                description="SSO, API keys, tenant gates, role policies, and token limits.",
                items=[
                    f"{len(settings.auth.super_users)} super users",
                    "Super API key configured"
                    if settings.auth.super_api_key
                    else "No super API key",
                ],
            ),
            AdminSurfaceCard(
                key="models",
                title="Model settings",
                status="Ready",
                tone="neutral",
                description="Primary LLM, reasoning model, reranker, and embedding settings.",
                items=[
                    f"{settings.llm.model_provider}/{settings.llm.model_name}",
                    f"{settings.llm.max_input_tokens} max input tokens",
                ],
            ),
            AdminSurfaceCard(
                key="tools",
                title="Tools and actions",
                status="Governed" if active_hooks else "Ready",
                tone="good" if active_hooks else "neutral",
                description="Custom actions, OpenAPI tools, MCP integrations, and policies.",
                items=[
                    f"{citeable_tool_count()} citeable tools",
                    f"{active_hooks}/{len(hooks)} hooks active",
                ],
            ),
            AdminSurfaceCard(
                key="analytics",
                title="Analytics",
                status="Live",
                tone="good",
                description="Sessions, citations, answer quality, latency, and usage trends.",
                items=[
                    "Session analytics endpoint enabled",
                    "User activity endpoint enabled",
                ],
            ),
            AdminSurfaceCard(
                key="enterprise",
                title="Enterprise controls",
                status="Active"
                if settings.license_enforcement_enabled
                else "Available",
                tone="good" if settings.license_enforcement_enabled else "neutral",
                description="Licensing, tenant isolation, audit hooks, and data controls.",
                items=[
                    "License enforcement on"
                    if settings.license_enforcement_enabled
                    else "License enforcement off",
                    "Cloud data plane configured"
                    if settings.cloud_data_plane_url
                    else "Local data plane",
                ],
            ),
        ],
    )
