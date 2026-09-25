"""HTTP client for the repo's search and retrieval servers.

Servers may expose either:
    Request:  {"queries": ["..."], "topk": N}
    Response: {"result": [[item, ...], ...]}  — one inner list per query

or a trainer-friendly single-query shape:
    Request:  {"query": "...", "top_k": N}
    Response: {"query": "...", "results": [item, ...]}

Item shapes differ by server:
    retrieval_server (return_scores=True):  {"document": {...}, "score": float}
    retrieval_server (return_scores=False): document dict directly
    google_search_server / serp_search_server: {"document": {"contents": "..."}}

SearchResult.from_api_item handles all three shapes.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from dataclasses import dataclass, field
from importlib import import_module
from urllib.parse import urlparse, urlunparse

from src.context.search import SearchResult
from src.internal.cache.serving import serving_cache
from src.internal.configs.timeouts import get_timeout_policies
from src.internal.observability.stage_metrics import note_retrieval

logger = logging.getLogger(__name__)


def _client_policy():
    return get_timeout_policies().retrieval.client


class _LazyAiohttp:
    """Import aiohttp on first use while preserving monkeypatchable attributes."""

    def __getattr__(self, name: str):
        module = import_module("aiohttp")
        value = getattr(module, name)
        setattr(self, name, value)
        return value


aiohttp = _LazyAiohttp()


@dataclass(frozen=True)
class SearchClientConfig:
    url: str
    topk: int = 5
    timeout_seconds: float = field(
        default_factory=lambda: _client_policy().timeout_seconds
    )
    max_retries: int = field(default_factory=lambda: _client_policy().max_retries)
    # Explicit /fetch endpoint. When None, derived from url by replacing /retrieve with /fetch.
    fetch_url: str | None = None

    def get_fetch_url(self) -> str:
        if self.fetch_url:
            return self.fetch_url
        parsed = urlparse(self.url)
        path = parsed.path.rstrip("/")
        if path.endswith("/retrieve"):
            path = path[: -len("/retrieve")]
        path = path.rstrip("/") + "/fetch"
        return urlunparse(parsed._replace(path=path, query="", fragment=""))


class SearchClient:
    """Async client for any POST /retrieve endpoint."""

    def __init__(self, config: SearchClientConfig) -> None:
        self.config = config
        self._timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _post_json(self, url: str, payload: dict, action: str) -> dict:
        last_exc: Exception | None = None

        for attempt in range(self.config.max_retries):
            session = await self._get_session()
            try:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as exc:
                if isinstance(exc, aiohttp.ClientResponseError) and exc.status < 500:
                    # 4xx errors are client-side mistakes — retrying won't help.
                    raise
                last_exc = exc
                if self._session is session:
                    await self.aclose()
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(
                        _client_policy().backoff_base_seconds * (2**attempt)
                    )

        raise RuntimeError(
            f"SearchClient.{action} failed after {self.config.max_retries} retries "
            f"against {url}"
        ) from last_exc

    async def retrieve(
        self,
        queries: list[str],
        topk: int | None = None,
        filters: dict | None = None,
    ) -> list[list[SearchResult]]:
        """Return one list of SearchResult per query.

        Raises RuntimeError after max_retries exhausted.
        Uses exponential backoff between retries.

        Repeats within the serving cache's TTL are served per query without a
        request; only the misses are posted. The serialised ``filters`` are
        part of the key, so two callers with different access filters never
        share an entry — and every caller still enforces on what it gets back,
        so a hit is checked exactly as a miss is.
        """
        topk = topk or self.config.topk
        cache = serving_cache()
        filters_key = (
            json.dumps(filters, sort_keys=True, default=str) if filters else ""
        )
        keys = [("retrieve", self.config.url, q, topk, filters_key) for q in queries]
        rows_by_index: dict[int, list[dict]] = {}
        if cache is not None:
            for index, key in enumerate(keys):
                hit = cache.get(key)
                if hit is not None:
                    # A fresh copy per hit: from_api_item shares nested dicts
                    # with the row, and a caller may mutate what it gets.
                    rows_by_index[index] = copy.deepcopy(hit)
        missing = [i for i in range(len(queries)) if i not in rows_by_index]
        elapsed_ms = 0.0
        if missing:
            payload = {
                "queries": [queries[i] for i in missing],
                "topk": topk,
                "return_scores": True,
            }
            if filters:
                payload["filters"] = filters
            started = time.perf_counter()
            try:
                data = await self._post_json(self.config.url, payload, "retrieve")
            except BaseException as exc:
                # A failed retrieval still spent the request's time; file it
                # so an outage shows up in the retrieval bucket, not nowhere.
                note_retrieval(
                    elapsed_ms=(time.perf_counter() - started) * 1000.0, docs=0
                )
                # A failed server is answered from stale cached rows, but only
                # when every missing query has one: never a mix of stale and
                # missing. Cancellation is not a failure and always propagates.
                stale = (
                    [cache.get_stale(keys[i]) for i in missing]
                    if cache is not None and isinstance(exc, Exception)
                    else []
                )
                if not stale or any(row is None for row in stale):
                    raise
                logger.info(
                    "SearchClient.retrieve: %s failed; serving %d stale rows",
                    self.config.url,
                    len(missing),
                )
                for index, row in zip(missing, stale):
                    rows_by_index[index] = copy.deepcopy(row)
                results = [
                    [SearchResult.from_api_item(item) for item in rows_by_index[i]]
                    for i in range(len(queries))
                ]
                for index in missing:
                    for result in results[index]:
                        result.metadata["stale"] = True
                return results
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            rows = data.get("result", data.get("results", []))
            if rows and isinstance(rows[0], dict):
                rows = [rows]
            if len(rows) != len(missing):
                # Never silently drop a query's evidence: queries without a
                # row get an empty list below, and the gap is logged.
                logger.warning(
                    "SearchClient.retrieve: %s returned %d rows for %d queries",
                    self.config.url,
                    len(rows),
                    len(missing),
                )
            for index, row in zip(missing, rows):
                rows_by_index[index] = row
                # Empty rows are cheap to recompute and may be a transient miss.
                if cache is not None and row:
                    cache.set(keys[index], copy.deepcopy(row))
        if queries:
            # The retrieval side of the request's stage metrics: every caller
            # of the ``retrieval`` provider comes through here.
            note_retrieval(
                elapsed_ms=elapsed_ms,
                docs=sum(len(row) for row in rows_by_index.values()),
                cache_hit=not missing,
            )
        return [
            [SearchResult.from_api_item(item) for item in rows_by_index.get(i, [])]
            for i in range(len(queries))
        ]

    async def retrieve_one(
        self,
        query: str,
        topk: int | None = None,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Convenience wrapper for a single query."""
        results = await self.retrieve([query], topk=topk, filters=filters)
        return results[0] if results else []

    async def fetch_urls(
        self,
        urls: list[str],
    ) -> list[SearchResult]:
        """Fetch full-page content for specific URLs from a compatible server."""
        payload = {"urls": urls}
        fetch_url = self.config.get_fetch_url()
        data = await self._post_json(fetch_url, payload, "fetch_urls")
        return [SearchResult.from_api_item(item) for item in data.get("result", [])]
