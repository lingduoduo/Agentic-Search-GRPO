"""Shared HTTP plumbing for the public data-source tools.

One place owns the timeout, the User-Agent, and the error shape, so each theme
module is only about its upstream's response format.

Absolute import of the aiohttp shim: this module sits one package deeper than
``tools.search``, and a four-dot relative import is needlessly hard to read.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from typing import Any, Callable

from src.context.retrieval.client import aiohttp
from src.internal.tools.base import FailureCategory, ToolErrorText, ToolFailure

logger = logging.getLogger(__name__)

# Nominatim's usage policy requires an identifying User-Agent, and a generic
# browser string risks getting the whole project blocked. Individual callers
# may still override it (Yahoo rejects non-browser agents).
USER_AGENT = "AgenticSearch/1.0 (+https://github.com/linghypshen/Agentic-Search)"

DEFAULT_TIMEOUT_SECONDS = 10.0

# Retried because they say "try again", not "no": rate limits and the
# gateway/unavailable family. Any other 4xx is an answer and repeating it only
# wastes the turn. Measured need: web.archive.org answers ~2 of 6 identical
# requests with 503, independent of User-Agent and query parameters.
_RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = (0.4, 0.8)
# No retry *starts* after this much wall time. The per-call timeout is a
# parameter, not a constant -- search_nearby_places passes a much longer
# Overpass budget -- so an attempt cap alone would let one dead host cost three
# full timeouts inside an agent turn.
_RETRY_BUDGET_SECONDS = 15.0

# Upper bound on any single document body handed back to the model. Abstracts
# and article intros are otherwise long enough to crowd out the rollout budget.
#
# This is bounded by the consumer, not just the rollout budget: ToolAgentLoop
# caps a whole tool message at `ToolAgentLoopConfig.max_tool_response_length`
# (src/agents/tool/tool_calling.py). When a JSON array of results exceeds that
# cap, it is trimmed by whole items keeping the leading (best-ranked) ones
# (via _fit_json_array), so raising this number must still fit under that cap.
# Prose-format responses fall back to character slicing with the default
# `tool_response_truncate_side="left"`, which keeps the start.
MAX_CONTENT_CHARS = 400


class PublicDataError(Exception):
    """An upstream call failed. ``guarded`` turns this into {"error": ...}."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        attempts: int = 1,
        retry_after: float | None = None,
        transport: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.attempts = attempts
        self.retry_after = retry_after
        self.transport = transport


def _retry_after(value: str | None) -> float | None:
    """Delta-seconds only; an HTTP-date or garbage is ignored."""
    try:
        seconds = float(value) if value is not None else None
    except ValueError:
        return None
    return seconds if seconds is not None and seconds >= 0 else None


async def _fetch(
    method: str,
    url: str,
    *,
    params: dict | None = None,
    data: Any = None,
    headers: dict | None = None,
    timeout_seconds: float,
    as_json: bool,
) -> Any:
    merged = {"User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    # Only GET. Retrying a write needs an idempotency guarantee this layer
    # cannot make about somebody else's API.
    attempts = _MAX_ATTEMPTS if method.upper() == "GET" else 1
    started = time.monotonic()
    body: str | None = None
    last_error: PublicDataError | None = None

    for attempt in range(attempts):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    method, url, params=params, data=data, headers=merged
                ) as response:
                    if response.status >= 400:
                        raise PublicDataError(
                            f"{url} returned HTTP {response.status}",
                            status=response.status,
                            retry_after=_retry_after(
                                response.headers.get("Retry-After")
                            ),
                        )
                    body = await response.text()
            break
        except PublicDataError as exc:
            if exc.status is not None and exc.status not in _RETRYABLE_STATUSES:
                exc.attempts = attempt + 1
                raise
            last_error = exc
        except Exception as exc:
            logger.debug("public data request to %s failed", url, exc_info=True)
            last_error = PublicDataError(
                f"request to {url} failed: {exc}", transport=True
            )

        if attempt + 1 >= attempts:
            break
        if time.monotonic() - started >= _RETRY_BUDGET_SECONDS:
            break
        await asyncio.sleep(_RETRY_BACKOFF_SECONDS[attempt])

    if body is None:
        if last_error is not None:
            last_error.attempts = attempt + 1
        raise last_error or PublicDataError(f"request to {url} failed")

    if not as_json:
        return body
    # Read text then parse, rather than response.json(): several of these hosts
    # return JSON under a non-JSON content type, which aiohttp rejects.
    try:
        return json.loads(body)
    except ValueError as exc:
        raise PublicDataError(f"{url} returned a non-JSON body") from exc


async def get_json(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    """GET *url* and parse the response as JSON. Raises PublicDataError."""
    return await _fetch(
        "GET",
        url,
        params=params,
        headers=headers,
        timeout_seconds=timeout_seconds,
        as_json=True,
    )


async def get_text(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """GET *url* and return the raw body. Raises PublicDataError."""
    return await _fetch(
        "GET",
        url,
        params=params,
        headers=headers,
        timeout_seconds=timeout_seconds,
        as_json=False,
    )


async def post_json(
    url: str,
    *,
    data: Any,
    headers: dict | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    """POST *data* to *url* and parse the response as JSON."""
    return await _fetch(
        "POST",
        url,
        data=data,
        headers=headers,
        timeout_seconds=timeout_seconds,
        as_json=True,
    )


_TRANSIENT_MESSAGE = "upstream temporarily unavailable"
_PERMANENT_MESSAGE = "upstream refused or could not answer the request"
_UNKNOWN_MESSAGE = "tool failed unexpectedly"


def _classify(exc: PublicDataError) -> ToolFailure:
    transient = exc.transport or exc.status in _RETRYABLE_STATUSES
    return ToolFailure(
        FailureCategory.TRANSIENT if transient else FailureCategory.PERMANENT,
        _TRANSIENT_MESSAGE if transient else _PERMANENT_MESSAGE,
        retry_after=exc.retry_after,
        provider_attempts=exc.attempts,
    )


def guarded(fn: Callable) -> Callable:
    """Adapt a tool coroutine to the tool return contract.

    The wrapped function returns a plain ``dict``/``list``; this serializes it
    and converts any failure into ``{"error": ...}`` so one dead upstream
    degrades a single tool rather than the whole turn.
    """

    @functools.wraps(fn)
    async def _wrapped(**kwargs: Any) -> str:
        try:
            return json.dumps(await fn(**kwargs))
        except PublicDataError as exc:
            return ToolErrorText(json.dumps({"error": str(exc)}), _classify(exc))
        except Exception as exc:  # noqa: BLE001 - a tool must never raise
            logger.debug("tool %s failed", getattr(fn, "__name__", "?"), exc_info=True)
            return ToolErrorText(
                json.dumps({"error": f"{type(exc).__name__}: {exc}"}),
                ToolFailure(FailureCategory.UNKNOWN, _UNKNOWN_MESSAGE),
            )

    return _wrapped
