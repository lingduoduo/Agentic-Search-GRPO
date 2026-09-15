"""Async AnySearch REST client shared by search providers and native tools.

Capability tags route structured requests; taxonomy query hints are applied only
by the existing general search tools. Importing this module never reads dotenv
files, changes process configuration, or touches standard streams.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from urllib.parse import urlsplit

from ...context.retrieval.client import aiohttp
from .search_domains import normalize_search_domain

DEFAULT_API_BASE_URL = "https://api.anysearch.com"
CLIENT_HEADER = "agentic-search/0.1.0"


def parse_search_params(value: str | dict[str, Any] | None) -> dict[str, Any] | None:
    """Accept JSON objects or the sample's key=value / {key:value} aliases."""
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("params must be a JSON object or key=value pairs")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        stripped = value.strip()
        braces = stripped.startswith("{") and stripped.endswith("}")
        content = stripped[1:-1] if braces else stripped
        separator = ":" if braces else "="
        parsed = {}
        for pair in content.split(","):
            key, sep, val = pair.partition(separator)
            key = key.strip().strip("\"'")
            if not sep or not key:
                raise ValueError("params must be valid JSON or key=value pairs")
            parsed[key] = val.strip().strip("\"'")
    if not isinstance(parsed, dict):
        raise ValueError("params must be a JSON object")
    return parsed


def normalize_search_item(item: dict[str, Any]) -> dict[str, Any]:
    """Validate single and batch searches identically, before making requests."""
    if not isinstance(item, dict):
        raise ValueError("each query item must be an object")
    query = item.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query is required")
    result: dict[str, Any] = {"query": query}
    tag, alias = item.get("tag"), item.get("sub_domain")
    if tag and alias and tag != alias:
        raise ValueError("tag and sub_domain must match when both are provided")
    tag = tag or alias
    domain = item.get("domain")
    if domain is not None:
        domain = normalize_search_domain(domain)
        if not tag:
            raise ValueError(
                "domain requires a tag or sub_domain; discover capabilities first"
            )
    if tag:
        if not isinstance(tag, str) or "." not in tag:
            raise ValueError(
                "tag must include a domain and capability, such as finance.quote"
            )
        prefix, capability = tag.split(".", 1)
        canonical = normalize_search_domain(prefix)
        if not capability.strip() or tag != tag.strip() or prefix != canonical:
            raise ValueError("tag must use a canonical domain and nonempty capability")
        if domain is not None and canonical != domain:
            raise ValueError("domain must match the tag prefix")
        result["tag"] = tag
    params = item.get("params") if "params" in item else item.get("sub_domain_params")
    parsed = parse_search_params(params)
    if parsed is not None:
        result["params"] = parsed
    zone = item.get("zone")
    if zone is not None:
        if zone not in ("cn", "intl"):
            raise ValueError("zone must be cn or intl")
        result["zone"] = zone
    language = item.get("language")
    if language is not None:
        if not isinstance(language, str) or not language.strip():
            raise ValueError("language must be a nonempty string")
        result["language"] = language
    limit = item.get("max_results")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("max_results must be an integer")
        result["max_results"] = max(1, min(limit, 10))
    return result


class AnySearchError(RuntimeError):
    """Provider failure with HTTP status and correlation ID, without process exits."""

    def __init__(
        self, message: str, *, status: int = 0, request_id: str = "", data: Any = None
    ):
        super().__init__(message)
        self.status = status
        self.request_id = request_id
        self.data = data


class AnySearchClient:
    """A stateless client; each call closes its session, including on failure."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30,
    ) -> None:
        # None uses configuration; an explicit empty key requests anonymous access.
        self.api_key = (
            os.getenv("ANYSEARCH_API_KEY", "") if api_key is None else api_key
        )
        self.base_url = (
            base_url or os.getenv("ANYSEARCH_API_BASE_URL") or DEFAULT_API_BASE_URL
        ).rstrip("/")
        if (
            urlsplit(self.base_url).scheme not in ("http", "https")
            or not urlsplit(self.base_url).netloc
        ):
            raise ValueError("AnySearch base URL must be an HTTP(S) URL")
        self.timeout_seconds = timeout_seconds

    def _redact(self, value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(self.api_key, "[REDACTED]") if self.api_key else value
        if isinstance(value, dict):
            return {key: self._redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        return value

    async def _request(
        self, method: str, path: str, *, payload=None, params=None
    ) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "X-Anysearch-Client": CLIENT_HEADER,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    method,
                    f"{self.base_url}{path}",
                    json=payload,
                    params=params,
                    headers=headers,
                ) as response:
                    status = response.status
                    try:
                        body = await response.json(content_type=None)
                    except ValueError:
                        raise AnySearchError(
                            f"Invalid JSON response (HTTP {status})", status=status
                        ) from None
        except asyncio.TimeoutError:
            raise AnySearchError("AnySearch request timed out") from None
        except aiohttp.ClientError:
            raise AnySearchError("Unable to reach the AnySearch API") from None
        if not isinstance(body, dict):
            raise AnySearchError(f"Invalid API response (HTTP {status})", status=status)
        if status >= 400 or body.get("code", 0) != 0:
            raise AnySearchError(
                self._redact(str(body.get("message") or f"HTTP {status}")),
                status=status,
                request_id=self._redact(str(body.get("request_id") or "")),
                data=self._redact(body.get("data")),
            )
        if not isinstance(body.get("data"), dict):
            raise AnySearchError("API response data must be an object", status=status)
        return body

    async def search(self, query: str, **options: Any) -> dict[str, Any]:
        payload = normalize_search_item({"query": query, **options})
        return await self._request("POST", "/v1/search", payload=payload)

    async def get_sub_domains(self, domains: list[str]) -> dict[str, Any]:
        if not isinstance(domains, list) or not 1 <= len(domains) <= 5:
            raise ValueError("provide one to five domains")
        params = [("domain", normalize_search_domain(domain)) for domain in domains]
        return await self._request("GET", "/v1/sub-domains", params=params)

    async def extract(self, url: str) -> dict[str, Any]:
        if (
            not isinstance(url, str)
            or urlsplit(url).scheme not in ("http", "https")
            or not urlsplit(url).netloc
        ):
            raise ValueError("url must be an HTTP(S) URL")
        return await self._request("POST", "/v1/extract", payload={"url": url})

    async def batch_search(
        self, queries: list[dict[str, Any]], **shared_options: Any
    ) -> list[dict[str, Any] | Exception]:
        """Run one to five requests concurrently and preserve per-item failures/order."""
        if not isinstance(queries, list) or not 1 <= len(queries) <= 5:
            raise ValueError("batch_search supports one to five queries")

        async def run(item):
            if not isinstance(item, dict):
                raise ValueError("each query item must be an object")
            defaults = dict(shared_options)
            # A per-item capability replaces the entire inherited route.
            if item.get("tag") or item.get("sub_domain"):
                for key in ("tag", "domain", "sub_domain"):
                    defaults.pop(key, None)
            if "params" in item or "sub_domain_params" in item:
                defaults.pop("params", None)
                defaults.pop("sub_domain_params", None)
            normalized = normalize_search_item({**defaults, **item})
            return await self.search(**normalized)

        # gather keeps input order, never strands a queue waiter on worker errors,
        # and still propagates cancellation of the parent operation.
        return await asyncio.gather(
            *(run(item) for item in queries), return_exceptions=True
        )
