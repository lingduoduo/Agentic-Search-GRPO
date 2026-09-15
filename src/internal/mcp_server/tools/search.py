"""Search tools for the Agentic Search MCP server."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from typing import Any
from typing import TypeVar

from src.internal.tools.search import fetch_url
from src.internal.tools.search import anysearch_search
from src.internal.tools.search import google_custom_search
from src.internal.tools.search import serper_dev_search
from src.internal.tools.search import serpapi_search
from src.internal.tools.search_domains import normalize_search_domain
from src.internal.tools.search_domains import prepare_domain_query
from src.internal.tools.search_domains import search_domain_parameter

from ..api import mcp_server
from ..retrieval_client import AuthenticatedRetrievalError
from ..retrieval_client import authenticated_retrieve

logger = logging.getLogger(__name__)

_SearchFn = TypeVar("_SearchFn", bound=Callable[..., Any])


def _describe_search_domain(fn: _SearchFn) -> _SearchFn:
    """Include the shared taxonomy before MCP captures the tool description."""
    fn.__doc__ = (
        (fn.__doc__ or "") + "\n\n" + str(search_domain_parameter()["description"])
    )
    return fn


def _error_payload(error: str) -> dict[str, Any]:
    return {"error": error, "results": []}


@mcp_server.tool()
async def search_indexed_documents(
    query: str,
    source_types: list[str] | None = None,
    document_set_names: list[str] | None = None,
    time_cutoff: str | None = None,
    skip_query_expansion: bool = False,
) -> dict[str, Any]:
    """
    Search the knowledge base indexed in Agentic Search.
    Use this tool for information specific to your organization, team, or documents
    that have been indexed in the retrieval server.

    `document_set_names` narrows results to documents the caller is authorized to
    access in those sets. `source_types`, `time_cutoff`, and `skip_query_expansion`
    are accepted for API compatibility but are not currently applied.

    Returns ``{"results": [{title, url, content}, ...]}``.

    Example usage:
    ```
    {"query": "dense retrieval with FAISS"}
    ```
    """
    logger.info("MCP Server: document search: query='%s'", query)

    try:
        documents = await authenticated_retrieve(
            query,
            top_k=5,
            document_set_names=document_set_names or None,
        )
    except AuthenticatedRetrievalError as err:
        logger.error("MCP Server: Document search error: %s", err, exc_info=True)
        return _error_payload(f"Document search failed: {str(err)}")

    results = [
        {"title": doc.title, "url": doc.url or "", "content": doc.content}
        for doc in documents
    ]
    logger.info("MCP Server: document search returned %s results", len(results))
    return {"results": results}


@mcp_server.tool()
@_describe_search_domain
async def search_web(
    query: str,
    limit: int = 5,
    domain: str = "general",
) -> dict[str, Any]:
    """
    Search the public internet for general knowledge and current events.
    Use this tool for publicly available information such as news, documentation,
    or general facts.

    `domain` optionally appends a topic hint to the query. It does not filter
    results by category. Omit it or use `general` to preserve the original query.

    Returns ``{"results": [{title, url, snippet}, ...], "query": query}``.
    Non-general calls also return `domain` and `executed_query`, including when
    the provider raises an error. Invalid domains fail before provider dispatch.
    Use ``open_urls`` to fetch full content from returned URLs.

    The search provider is selected via the ``MCP_WEB_SEARCH_PROVIDER`` env var
    (``google``, ``serpapi``, ``serper``, or ``anysearch``; defaults to ``google``).

    Example usage:
    ```
    {"query": "React 19 release notes", "limit": 5}
    ```
    """
    domain = normalize_search_domain(domain)
    executed_query = prepare_domain_query(query, domain)
    domain_metadata = (
        {"domain": domain, "executed_query": executed_query}
        if domain != "general"
        else {}
    )
    logger.info("MCP Server: web search: query='%s', limit=%s", query, limit)

    provider = os.getenv("MCP_WEB_SEARCH_PROVIDER", "google")
    try:
        if provider == "anysearch":
            pages = await anysearch_search(executed_query, page_size=limit)
        elif provider == "serpapi":
            pages = await serpapi_search(executed_query, page_size=limit)
        elif provider == "serper":
            pages = await serper_dev_search(executed_query, page_size=limit)
        else:
            pages = await google_custom_search(executed_query, page_size=limit)
    except Exception as exc:
        logger.error("MCP Server: Web search error: %s", exc, exc_info=True)
        return {
            "error": f"Web search failed: {str(exc)}",
            "results": [],
            "query": query,
            **domain_metadata,
        }

    results = [
        {"title": p.title, "url": p.url, "snippet": p.summary}
        for p in pages
        if not p.error
    ]
    errors = [p.error for p in pages if p.error]
    if errors:
        logger.warning("MCP Server: %d web search result(s) had errors", len(errors))

    return {"results": results, "query": query, **domain_metadata}


@mcp_server.tool()
async def open_urls(
    urls: list[str],
) -> dict[str, Any]:
    """
    Retrieve the complete text content from specific web URLs.
    Use this tool to fetch full page content for URLs returned by ``search_web``.

    Returns ``{"results": [{url, content}, ...]}``.

    Example usage:
    ```
    {"urls": ["https://react.dev/learn/react-compiler"]}
    ```
    """
    logger.info("MCP Server: open_urls: fetching %d URLs", len(urls))

    try:
        contents = await asyncio.gather(*[fetch_url(url) for url in urls])
    except Exception as err:
        logger.error("MCP Server: URL fetch error: %s", err, exc_info=True)
        return _error_payload(f"URL fetch failed: {str(err)}")

    results = [{"url": url, "content": content} for url, content in zip(urls, contents)]
    return {"results": results}
