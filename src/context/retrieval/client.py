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
from dataclasses import dataclass
from importlib import import_module
from urllib.parse import urlparse, urlunparse

from src.context.search import SearchResult
from src.internal.cache.serving import serving_cache

logger = logging.getLogger(__name__)


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
    timeout_seconds: int = 10
    max_retries: int = 3
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
                    await asyncio.sleep(0.5 * (2**attempt))

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
        if missing:
            payload = {
                "queries": [queries[i] for i in missing],
                "topk": topk,
                "return_scores": True,
            }
            if filters:
                payload["filters"] = filters
            data = await self._post_json(self.config.url, payload, "retrieve")
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
