"""``GET /ready``: can this process answer a query right now?

``/health`` is liveness and always says ok. Readiness checks the two things a
query cannot do without -- the store and the retrieval server -- concurrently
and uncached. Errors are reported as exception class names only, because the
route is unauthenticated and an exception message can carry a URL or
credentials.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlsplit, urlunsplit

import httpx

from src.internal.configs import get_timeout_policies
from src.internal.db import AgenticSearchStore

Check = dict[str, object]


def retrieval_health_url(search_url: str) -> str:
    """``http://h:8001/retrieve?x=1`` -> ``http://h:8001/health``."""
    parts = urlsplit(search_url)
    return urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))


def _retrieval_client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout)


def _passed() -> Check:
    return {"ok": True, "error": None}


def _failed(error: str) -> Check:
    return {"ok": False, "error": error}


async def _check_store(db: AgenticSearchStore) -> Check:
    timeout = get_timeout_policies().readiness.probe_timeout_seconds
    try:
        # A worker thread, bounded: a request thread may hold the store lock for
        # a long write, and a probe that hangs is worse than one that says no.
        await asyncio.wait_for(asyncio.to_thread(db.ping), timeout)
    except Exception as exc:
        return _failed(type(exc).__name__)
    return _passed()


async def _check_retrieval(search_url: str) -> Check:
    timeout = get_timeout_policies().readiness.probe_timeout_seconds
    try:
        async with _retrieval_client(timeout) as client:
            response = await client.get(retrieval_health_url(search_url))
    except Exception as exc:
        return _failed(type(exc).__name__)
    if not response.is_success:
        return _failed(f"HTTP {response.status_code}")
    return _passed()


async def check_readiness(
    db: AgenticSearchStore, search_url: str
) -> tuple[bool, dict[str, Check]]:
    store, retrieval = await asyncio.gather(
        _check_store(db), _check_retrieval(search_url)
    )
    checks = {"store": store, "retrieval": retrieval}
    return all(check["ok"] for check in checks.values()), checks
