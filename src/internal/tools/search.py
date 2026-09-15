"""Reusable search helpers for function-calling tools."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import parse_qsl
from urllib.parse import urlencode
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

from ...context.search import SearchResult
from ...context.retrieval.client import SearchClient, SearchClientConfig, aiohttp
from ..cache.serving import serving_cache
from .base import FunctionTool, Tool, ToolEffect, ToolSchema
from .html_text import _html_to_text
from .search_domains import normalize_search_domain
from .search_domains import prepare_domain_query
from .search_domains import search_domain_parameter

logger = logging.getLogger(__name__)

SearchProvider = Literal["retrieval", "google", "serpapi", "serper"]

GOOGLE_SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
SERPAPI_SEARCH_ENDPOINT = "https://serpapi.com/search.json"
SERPER_DEV_ENDPOINT = "https://google.serper.dev/search"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _sanitize_query(query: str) -> str:
    parts = []
    for c in query:
        code = ord(c)
        if code >= 32 and code != 127:
            parts.append(c)
        elif code != 127:
            parts.append(" ")
    sanitized = "".join(parts)
    return " ".join(sanitized.split())


def _normalize_queries_input(raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        raw = [raw]
    elif not isinstance(raw, list):
        return []
    result: list[str] = []
    for q in raw:
        if q is None:
            continue
        sanitized = _sanitize_query(str(q))
        if sanitized:
            result.append(sanitized)
    return result


@dataclass(frozen=True)
class SearchPage:
    title: str = ""
    summary: str = ""
    url: str = ""
    error: str | None = None
    score: float = 0.0
    # Retrieval-side metadata, carried so downstream consumers can enforce the
    # document's ACL. Dropping it here made access control impossible on any
    # path that goes through SearchPage.
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_search_result(cls, result: SearchResult) -> "SearchPage":
        return cls(
            title=result.title or "",
            summary=_compact_contents(result.contents),
            url=result.url or "",
            metadata=dict(result.metadata or {}),
            score=result.score,
        )


async def google_custom_search(
    query: str,
    *,
    page: int = 1,
    page_size: int = 5,
    api_key: str | None = None,
    cse_id: str | None = None,
    timeout_seconds: int = 15,
) -> list[SearchPage]:
    """Search Google Custom Search and return normalized pages."""

    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    cse_id = cse_id or os.getenv("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        return [SearchPage(error="GOOGLE_API_KEY and GOOGLE_CSE_ID are required.")]

    try:
        data = await _get_json(
            GOOGLE_SEARCH_ENDPOINT,
            params={
                "key": api_key,
                "cx": cse_id,
                "q": query,
                "num": page_size,
                "start": (page - 1) * page_size + 1,
            },
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        return [SearchPage(error=_redact_secret_params(str(exc)))]
    return [
        SearchPage(
            title=item.get("title", ""),
            summary=item.get("snippet", ""),
            url=item.get("link", ""),
        )
        for item in data.get("items", [])
    ]


async def serpapi_search(
    query: str,
    *,
    page: int = 1,
    page_size: int = 5,
    api_key: str | None = None,
    timeout_seconds: int = 15,
) -> list[SearchPage]:
    """Search SerpAPI Google results and return normalized pages."""

    api_key = api_key or os.getenv("SERPAPI_API_KEY") or os.getenv("SERP_API_KEY")
    if not api_key:
        return [SearchPage(error="SERPAPI_API_KEY or SERP_API_KEY is required.")]

    try:
        data = await _get_json(
            SERPAPI_SEARCH_ENDPOINT,
            params={
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "num": page_size,
                "start": (page - 1) * page_size,
            },
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        return [SearchPage(error=_redact_secret_params(str(exc)))]

    pages = [
        SearchPage(
            title=item.get("title", ""),
            summary=item.get("snippet", ""),
            url=item.get("link", ""),
        )
        for item in data.get("organic_results", [])
    ]

    answer_box = data.get("answer_box", {})
    answer = answer_box.get("snippet") or answer_box.get("answer")
    if answer:
        pages.insert(
            0,
            SearchPage(
                title=answer_box.get("title", ""),
                summary=answer,
                url=answer_box.get("link", ""),
            ),
        )
    return pages[:page_size]


async def serper_dev_search(
    query: str,
    *,
    page_size: int = 5,
    api_key: str | None = None,
    timeout_seconds: int = 10,
) -> list[SearchPage]:
    """Search via Serper.dev (Google results) and return normalized pages."""

    api_key = api_key or os.getenv("SERPER_API_KEY")
    if not api_key:
        return [SearchPage(error="SERPER_API_KEY is required.")]

    try:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                SERPER_DEV_ENDPOINT,
                json={"q": query, "num": page_size},
                headers=headers,
            ) as response:
                response.raise_for_status()
                data = await response.json()
    except Exception as exc:
        err = str(exc).replace(api_key, "[REDACTED]") if api_key else str(exc)
        return [SearchPage(error=_redact_secret_params(err))]

    results = data.get("organic") or []
    return [
        SearchPage(
            title=item.get("title", ""),
            summary=item.get("snippet", ""),
            url=item.get("link", ""),
        )
        for item in results[:page_size]
    ]


async def retrieval_search(
    query: str,
    *,
    search_url: str,
    page_size: int = 5,
    timeout_seconds: int = 10,
    max_retries: int = 3,
    fetch_url: str | None = None,
    filters: dict | None = None,
) -> list[SearchPage]:
    """Search the repo's /retrieve server and return normalized pages."""

    client = SearchClient(
        SearchClientConfig(
            url=search_url,
            topk=page_size,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            fetch_url=fetch_url,
        )
    )
    try:
        return [
            SearchPage.from_search_result(result)
            for result in await client.retrieve_one(
                query, topk=page_size, filters=filters
            )
        ]
    except Exception as exc:
        return [SearchPage(error=_redact_secret_params(str(exc)))]
    finally:
        await client.aclose()


async def search_tool(
    query: str,
    *,
    provider: SearchProvider = "retrieval",
    page: int = 1,
    page_size: int = 5,
    search_url: str = "http://localhost:8000/retrieve",
    timeout_seconds: int = 15,
    max_retries: int = 3,
    fetch_url: str | None = None,
    filters: dict | None = None,
) -> list[SearchPage]:
    """Route one query to a configured provider.

    ``filters`` (per-user access filters) apply only to the internal
    ``retrieval`` provider; web providers have no ACL metadata and ignore them.
    """

    if provider == "retrieval":
        return await retrieval_search(
            query,
            search_url=search_url,
            page_size=page_size,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            fetch_url=fetch_url,
            filters=filters,
        )
    if provider not in ("google", "serpapi", "serper"):
        raise ValueError(
            "provider must be 'retrieval', 'google', 'serpapi', or 'serper'"
        )
    # Web providers are slow and rate-limited; repeats within the TTL are served
    # from the process-local serving cache. They carry no ACL, so the key is
    # just the lookup itself. Hits are deep-copied so a caller mutating a page
    # (or a nested value in its metadata) cannot poison later hits.
    cache = serving_cache()
    cache_key = ("web", provider, query, page, page_size)
    if cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return copy.deepcopy(hit)
    if provider == "google":
        pages = await google_custom_search(
            query,
            page=page,
            page_size=page_size,
            timeout_seconds=timeout_seconds,
        )
    elif provider == "serpapi":
        pages = await serpapi_search(
            query,
            page=page,
            page_size=page_size,
            timeout_seconds=timeout_seconds,
        )
    else:
        pages = await serper_dev_search(
            query,
            page_size=page_size,
            timeout_seconds=timeout_seconds,
        )
    # Never cache an empty or failed lookup: for a web provider that is usually
    # a transient failure, and pinning it for the TTL would hide the recovery.
    if cache is not None and pages and not any(p.error for p in pages):
        cache.set(cache_key, copy.deepcopy(pages))
    return pages


def _pages_are_usable(pages: list[SearchPage]) -> bool:
    """True when at least one page carries a result and none is an error page."""
    if not pages:
        return False
    return any(p.url for p in pages) and not any(p.error for p in pages)


def make_web_cascade_search(
    *,
    browser_search_url: str | None = None,
    serpapi_fn=serpapi_search,
    browser_fn=search_tool,
):
    """Return a ``search_fn`` that tries SerpAPI, then falls back to the browser
    search server (retrieval-shaped ``/retrieve``). First usable result wins.

    Compatible with ``MultiQueryWebSearchTool(search_fn=...)``.
    """

    async def _cascade(
        query: str,
        *,
        provider: SearchProvider = "serpapi",
        search_url: str = "http://localhost:8000/retrieve",
        page: int = 1,
        page_size: int = 5,
        timeout_seconds: int = 15,
    ) -> list[SearchPage]:
        del provider, search_url  # cascade owns provider selection
        # Why each leg failed, kept so an unusable cascade can say so. Returning
        # [] renders as "No results found.", which is indistinguishable from a
        # working search over a topic with no hits — it hid a missing key, an
        # exhausted quota, and an unconfigured fallback alike.
        failures: list[SearchPage] = []

        serp_pages = await serpapi_fn(
            query, page=page, page_size=page_size, timeout_seconds=timeout_seconds
        )
        if _pages_are_usable(serp_pages):
            return serp_pages
        failures.extend(p for p in serp_pages if p.error)

        if browser_search_url:
            try:
                browser_pages = await browser_fn(
                    query,
                    provider="retrieval",
                    search_url=browser_search_url,
                    page=page,
                    page_size=page_size,
                )
                if _pages_are_usable(browser_pages):
                    return browser_pages
                failures.extend(p for p in browser_pages if p.error)
            except Exception as exc:  # noqa: BLE001
                logger.warning("browser cascade leg failed for %r: %s", query, exc)
                failures.append(SearchPage(error=f"Browser search failed: {exc}"))
        elif failures:
            # Only when a leg actually errored — a genuinely empty result set is
            # not a configuration problem.
            failures.append(
                SearchPage(
                    error=(
                        "No browser fallback is configured; set "
                        "AGENTIC_SEARCH_BROWSER_SEARCH_URL to add one."
                    )
                )
            )

        logger.warning("web cascade produced no usable results for %r", query)
        return failures

    return _cascade


async def search_for_list(
    query: str,
    *,
    provider: SearchProvider = "retrieval",
    search_url: str = "http://localhost:8000/retrieve",
    page: int = 1,
    page_size: int = 5,
) -> list[dict[str, str]]:
    """Return normalized search results as dictionaries."""

    pages = await search_tool(
        query,
        provider=provider,
        search_url=search_url,
        page=page,
        page_size=page_size,
    )
    return [
        {
            "title": page.title,
            "summary": page.summary,
            "url": page.url,
            **({"error": page.error} if page.error else {}),
        }
        for page in pages
    ]


async def search_for_tool_string(
    query: str,
    *,
    provider: SearchProvider = "retrieval",
    search_url: str = "http://localhost:8000/retrieve",
    page: int = 1,
    page_size: int = 5,
) -> str:
    """Return search results formatted for a text-only tool response."""

    pages = await search_tool(
        query,
        provider=provider,
        search_url=search_url,
        page=page,
        page_size=page_size,
    )
    return format_search_pages(pages)


async def fetch_url(
    url: str, *, max_length: int = 2000, timeout_seconds: int = 15
) -> str:
    """Fetch readable webpage text with lightweight HTML extraction."""

    if not url:
        return ""
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        headers = {"User-Agent": DEFAULT_USER_AGENT}
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                html = await response.text()
        return _html_to_text(html)[:max_length]
    except Exception as exc:
        return f"[fetch error] {exc}"


async def fetch_pages_concurrently(
    pages: list[SearchPage],
    *,
    max_chars: int = 2000,
    timeout_seconds: int = 10,
) -> list[SearchPage]:
    """Fetch full page content for each SearchPage that has a URL and no error."""
    fetchable = [p for p in pages if p.url and not p.error]
    results = await asyncio.gather(
        *[
            fetch_url(p.url, max_length=max_chars, timeout_seconds=timeout_seconds)
            for p in fetchable
        ],
        return_exceptions=True,
    )
    url_to_content: dict[str, str] = {}
    for page, content in zip(fetchable, results):
        if isinstance(content, str) and not content.startswith("[fetch error]"):
            url_to_content[page.url] = content

    return [
        SearchPage(
            title=p.title,
            summary=url_to_content.get(p.url, p.summary)
            if (p.url and not p.error)
            else p.summary,
            url=p.url,
            error=p.error,
        )
        for p in pages
    ]


async def search_for_detail(
    query: str,
    *,
    provider: SearchProvider = "retrieval",
    search_url: str = "http://localhost:8000/retrieve",
    page: int = 1,
    page_size: int = 5,
    chunk_size: int = 500,
) -> str:
    """Search and fetch detailed content for each result URL."""

    pages = await search_tool(
        query,
        provider=provider,
        search_url=search_url,
        page=page,
        page_size=page_size,
    )
    contents = await asyncio.gather(
        *[
            fetch_url(page.url, max_length=chunk_size)
            for page in pages
            if not page.error
        ]
    )
    content_iter = iter(contents)

    sections: list[str] = []
    for page in pages:
        if page.error:
            sections.append(f"Error: {page.error}")
            continue
        sections.append(
            f"Title: {page.title}\nURL: {page.url}\nContent: {next(content_iter, '')}"
        )
    return "\n\n".join(sections) if sections else "No results found."


class MultiQueryWebSearchTool(Tool):
    """Tool that accepts multiple queries and runs them in parallel.

    Designed for use with ToolAgentLoop. The LLM passes {"queries": ["q1", "q2"]}
    and all queries execute concurrently, with results deduplicated by URL.
    An optional domain applies the same topic hint to every query in the call.
    """

    def __init__(
        self,
        search_fn: Any = None,
        *,
        provider: SearchProvider = "retrieval",
        search_url: str = "http://localhost:8000/retrieve",
        page_size: int = 5,
        timeout_seconds: int = 15,
    ) -> None:
        self._search_fn = search_fn or search_tool
        self._provider = provider
        self._search_url = search_url
        self._page_size = page_size
        self._timeout_seconds = timeout_seconds
        self._schema = ToolSchema(
            name="web_search",
            description=(
                "Search the web for information. Pass multiple queries to search in parallel."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "One or more search queries to run in parallel.",
                    },
                    "domain": search_domain_parameter(),
                },
                "required": ["queries"],
            },
        )

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    @property
    def citeable(self) -> bool:
        return True

    async def execute(
        self, instance_id: str, arguments: dict[str, Any]
    ) -> tuple[str, Any, Any]:
        del instance_id
        domain = normalize_search_domain(arguments.get("domain", "general"))
        raw_queries = arguments.get("queries", [])
        queries = _normalize_queries_input(raw_queries)

        if not queries:
            return "No results found.", [], {}

        executed_queries = [prepare_domain_query(q, domain) for q in queries]
        results_per_query: list[list[SearchPage]] = await asyncio.gather(
            *[
                self._search_fn(
                    q,
                    provider=self._provider,
                    search_url=self._search_url,
                    page_size=self._page_size,
                    timeout_seconds=self._timeout_seconds,
                )
                for q in executed_queries
            ]
        )

        seen_urls: set[str] = set()
        merged: list[SearchPage] = []
        for pages in results_per_query:
            for page in pages:
                if page.url and page.url in seen_urls:
                    continue
                if page.url:
                    seen_urls.add(page.url)
                merged.append(page)

        metadata: dict[str, Any] = {"queries": queries}
        if domain != "general":
            metadata.update(domain=domain, executed_queries=executed_queries)
        return format_search_pages(merged), merged, metadata


def build_search_tool(
    *,
    provider: SearchProvider = "retrieval",
    search_url: str = "http://localhost:8000/retrieve",
    page_size: int = 5,
) -> FunctionTool:
    """Build a FunctionTool for ToolAgentLoop search usage."""

    async def search(query: str, domain: str = "general") -> str:
        return await search_for_tool_string(
            prepare_domain_query(query, domain),
            provider=provider,
            search_url=search_url,
            page_size=page_size,
        )

    return FunctionTool(
        fn=search,
        name="search",
        description="Search for information on a topic.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "domain": search_domain_parameter(),
            },
            "required": ["query"],
        },
        effect=ToolEffect.READ_ONLY,
        citeable=True,
    )


def format_search_pages(pages: list[SearchPage]) -> str:
    sections: list[str] = []
    for page in pages:
        if page.error:
            sections.append(f"Error: {page.error}")
            continue
        sections.append(
            f"Title: {page.title}\nSummary: {page.summary}\nURL: {page.url}"
        )
    return "\n\n".join(sections) if sections else "No results found."


def _redact_secret_params(text: str) -> str:
    for marker in ("url=URL('", 'url="', "url='"):
        start = text.find(marker)
        if start == -1:
            continue
        url_start = start + len(marker)
        quote = marker[-1]
        url_end = text.find(quote, url_start)
        if url_end == -1:
            continue
        original = text[url_start:url_end]
        text = f"{text[:url_start]}{_redact_url(original)}{text[url_end:]}"
    return text


def _redact_url(url: str) -> str:
    parsed = urlsplit(url)
    redacted_query = urlencode(
        [
            (key, "[REDACTED]" if key.lower() in {"key", "api_key"} else value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        ]
    )
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, redacted_query, parsed.fragment)
    )


async def _get_json(
    url: str,
    *,
    params: dict[str, Any],
    timeout_seconds: int,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, params=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()


def _compact_contents(contents: str, limit: int = 500) -> str:
    lines = [line.strip() for line in contents.splitlines() if line.strip()]
    if len(lines) > 1 and lines[0].startswith('"') and lines[0].endswith('"'):
        text = " ".join(lines[1:])
    else:
        text = " ".join(lines)
    return text[:limit]
