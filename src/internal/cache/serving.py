"""The process-level cache the serving path shares.

Off until ``configure_serving_cache`` is called — the web app's lifespan does
that from ``AGENTIC_SEARCH_SEARCH_CACHE_TTL`` and resets on shutdown, so
nothing is cached in a process that never started the web app. Consumers
(``SearchClient.retrieve``, the web providers in ``search_tool``,
``RerankHTTPRankingStage``) build their own tuple keys; anything that narrows
what a caller may read (the serialised access filters) must be in the key.
"""

from __future__ import annotations

from src.internal.cache.ttl_cache import TTLCache

_cache: TTLCache | None = None


def configure_serving_cache(
    ttl_seconds: float, *, max_entries: int = 1024, stale_seconds: float = 0.0
) -> None:
    """Enable caching with *ttl_seconds*; a non-positive TTL disables it.

    *stale_seconds* keeps expired entries that much longer for ``get_stale``,
    which the consumers read only after their live call failed.
    """
    global _cache
    _cache = (
        TTLCache(ttl_seconds, max_entries=max_entries, stale_seconds=stale_seconds)
        if ttl_seconds > 0
        else None
    )


def reset_serving_cache() -> None:
    global _cache
    _cache = None


def serving_cache() -> TTLCache | None:
    return _cache
