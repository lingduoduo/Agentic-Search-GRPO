"""Search runners that build normalized context bundles."""

from __future__ import annotations

import logging

from src.context.retrieval.client import SearchClient
from src.context.retrieval.client import SearchClientConfig
from src.context.search import SearchResult
from src.internal.configs.timeouts import get_timeout_policies
from src.internal.tools.search import google_custom_search
from src.internal.tools.search import serpapi_search

from ..enums import SearchType
from ..models import SearchContextBundle
from ..models import SearchFilters
from ..models import SearchRequest
from ..utils import build_context_bundle

logger = logging.getLogger(__name__)


async def run_search(
    request: SearchRequest,
    *,
    search_url: str = "http://localhost:8000/retrieve",
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
    fetch_url: str | None = None,
) -> list[SearchResult]:
    request.validate()
    runner_policy = get_timeout_policies().retrieval.search_runner
    if timeout_seconds is None:
        timeout_seconds = runner_policy.timeout_seconds
    if max_retries is None:
        max_retries = runner_policy.max_retries
    if request.provider == SearchType.RETRIEVAL:
        client = SearchClient(
            SearchClientConfig(
                url=search_url,
                topk=request.top_k,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                fetch_url=fetch_url,
            )
        )
        try:
            if request.filters:
                results = await client.retrieve_one(
                    request.query,
                    topk=request.top_k,
                    filters=request.filters.to_payload(),
                )
                return _apply_filters(results, request.filters)
            return await client.retrieve_one(request.query, topk=request.top_k)
        finally:
            await client.aclose()

    if request.provider == SearchType.GOOGLE:
        pages = await google_custom_search(
            request.query,
            page_size=request.top_k,
            timeout_seconds=timeout_seconds,
        )
    elif request.provider == SearchType.SERPAPI:
        pages = await serpapi_search(
            request.query,
            page_size=request.top_k,
            timeout_seconds=timeout_seconds,
        )
    else:
        raise ValueError(f"Unsupported search provider: {request.provider}")

    results = [
        SearchResult(
            title=page.title,
            contents=f'"{page.title}"\n{page.summary}' if page.title else page.summary,
            url=page.url,
        )
        for page in pages
        if not page.error
    ]
    return _apply_filters(results, request.filters)


async def build_search_context(
    request: SearchRequest,
    *,
    search_url: str = "http://localhost:8000/retrieve",
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
    fetch_url: str | None = None,
):
    results = await run_search(
        request,
        search_url=search_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        fetch_url=fetch_url,
    )
    return build_context_bundle(
        request.query,
        _apply_filters(results, request.filters),
        max_documents=request.top_k,
    )


async def build_search_contexts(
    queries: list[str],
    *,
    top_k: int = 5,
    filters: SearchFilters | None = None,
    search_url: str = "http://localhost:8000/retrieve",
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
) -> list[SearchContextBundle]:
    """Retrieve for several queries in one request; one bundle per query.

    The retrieval API is natively multi-query ({"queries": [...]}), so N
    independent queries cost one round trip on one session rather than N of
    each. Bundles come back in input order and are built exactly as
    `build_search_context` builds a single one, so this is a transport change
    only.

    Retrieval provider only: the multi-query request shape is specific to
    /retrieve.
    """
    if not queries:
        return []
    runner_policy = get_timeout_policies().retrieval.search_runner
    if timeout_seconds is None:
        timeout_seconds = runner_policy.timeout_seconds
    if max_retries is None:
        max_retries = runner_policy.max_retries
    client = SearchClient(
        SearchClientConfig(
            url=search_url,
            topk=top_k,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    )
    try:
        rows = await client.retrieve(
            queries,
            topk=top_k,
            filters=filters.to_payload() if filters is not None else None,
        )
    finally:
        await client.aclose()
    if len(rows) != len(queries):
        # A malformed or short response must never silently drop a query's
        # evidence (e.g. SearchClient.retrieve's single-query response-shape
        # collapse can hand back one row for many queries). Pad/trim to line
        # up 1:1 with `queries` -- missing queries get an empty bundle, same
        # as a query that legitimately found nothing -- and log the gap so
        # it's visible instead of silent.
        logger.warning(
            "build_search_contexts: retrieval at %s returned %d rows for %d "
            "queries; queries without a row get an empty bundle instead of "
            "losing their evidence silently",
            search_url,
            len(rows),
            len(queries),
        )
        rows = (rows + [[]] * len(queries))[: len(queries)]
    # Enforce, don't just forward: a third-party backend need not honour the
    # forwarded filter, and anything returned here reaches a model's context.
    return [
        build_context_bundle(query, _apply_filters(row, filters), max_documents=top_k)
        for query, row in zip(queries, rows)
    ]


def combine_search_results(result_sets: list[list[SearchResult]]) -> list[SearchResult]:
    unique: dict[tuple[str | None, str], SearchResult] = {}
    for result_set in result_sets:
        for result in result_set:
            key = (result.url, result.contents)
            previous = unique.get(key)
            if previous is None or result.score > previous.score:
                unique[key] = result
    return sorted(unique.values(), key=lambda result: result.score, reverse=True)


def _apply_filters(
    results: list[SearchResult],
    filters,
) -> list[SearchResult]:
    if filters is None:
        return results
    return [result for result in results if filters.matches(result.metadata)]
