"""Path allowlists and tier checks for optional feature gating."""

from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        pass


class Tier(StrEnum):
    FREE = "free"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


_TIER_RANK = {
    Tier.FREE: 0,
    Tier.BUSINESS: 1,
    Tier.ENTERPRISE: 2,
}


LICENSE_ENFORCEMENT_ALLOWED_PREFIXES: frozenset[str] = frozenset(
    {
        "/auth",
        "/health",
        "/ready",
        "/metrics",
        "/me",
        "/settings",
        "/license",
        "/billing",
        "/admin/billing",
        "/api/sessions",
        "/assets",
    }
)


PATH_PREFIX_MIN_TIER: dict[str, Tier] = {
    "/admin/api-key": Tier.BUSINESS,
    "/admin/chat-sessions": Tier.BUSINESS,
    "/admin/query-history": Tier.BUSINESS,
    "/analytics/admin": Tier.BUSINESS,
    "/manage/admin/user-group": Tier.BUSINESS,
    "/admin/hooks": Tier.ENTERPRISE,
    "/admin/token-rate-limits": Tier.ENTERPRISE,
    "/analytics": Tier.ENTERPRISE,
    "/evals": Tier.ENTERPRISE,
    "/scim": Tier.ENTERPRISE,
}


def normalize_path(path: str) -> str:
    if not path:
        return "/"
    normalized = path if path.startswith("/") else f"/{path}"
    return normalized.rstrip("/") or "/"


def path_matches_prefix(path: str, prefix: str) -> bool:
    normalized_path = normalize_path(path)
    normalized_prefix = normalize_path(prefix)
    return normalized_path == normalized_prefix or normalized_path.startswith(
        f"{normalized_prefix}/"
    )


def is_license_enforcement_exempt(path: str) -> bool:
    return any(
        path_matches_prefix(path, prefix)
        for prefix in LICENSE_ENFORCEMENT_ALLOWED_PREFIXES
    )


def required_tier_for_path(path: str) -> Tier:
    """Return the minimum tier for the longest matching path prefix."""

    matches = [
        (prefix, tier)
        for prefix, tier in PATH_PREFIX_MIN_TIER.items()
        if path_matches_prefix(path, prefix)
    ]
    if not matches:
        return Tier.FREE
    return max(matches, key=lambda item: len(normalize_path(item[0])))[1]


def tier_allows(current_tier: Tier | str, required_tier: Tier | str) -> bool:
    current = Tier(current_tier)
    required = Tier(required_tier)
    return _TIER_RANK[current] >= _TIER_RANK[required]


def is_path_allowed_for_tier(path: str, tier: Tier | str) -> bool:
    if is_license_enforcement_exempt(path):
        return True
    return tier_allows(tier, required_tier_for_path(path))


__all__ = [
    "LICENSE_ENFORCEMENT_ALLOWED_PREFIXES",
    "PATH_PREFIX_MIN_TIER",
    "Tier",
    "is_license_enforcement_exempt",
    "is_path_allowed_for_tier",
    "normalize_path",
    "path_matches_prefix",
    "required_tier_for_path",
    "tier_allows",
]
