"""Search API router for Agentic Search.

Provides three endpoints that mirror the search_backend:
  POST /search/search-flow-classification  — keyword vs chat routing
  POST /search/send-search-message         — run a (possibly expanded) search
  GET  /search/search-history              — past sessions for the caller

The previous versions depended on SQLAlchemy, Celery, Redis, and EE models.
This implementation uses the repo's own types:
  - classify_is_search_flow  (src/secondary_llm_flows/)
  - run_expanded_search      (src/search/)
  - AgenticSearchStore       (src/db/)
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import StreamingResponse

from src.context.models import SearchFilters
from src.context.preprocessing.access_filters import build_user_only_filters
from src.internal.auth import AuthenticatedUser
from src.internal.db import AgenticSearchStore
from src.internal.search.process_search_query import SearchQueryResult
from src.internal.search.process_search_query import run_expanded_search
from src.internal.servers.secondary_llm_flows import classify_is_search_flow
from src.internal.servers.query_and_chat.models import SearchDocWithContent
from src.internal.servers.query_and_chat.models import SearchFlowClassificationRequest
from src.internal.servers.query_and_chat.models import SearchFlowClassificationResponse
from src.internal.servers.query_and_chat.models import SearchFullResponse
from src.internal.servers.query_and_chat.models import SearchHistoryResponse
from src.internal.servers.query_and_chat.models import SearchQueryResponse
from src.internal.servers.query_and_chat.models import SendSearchQueryRequest
from src.internal.servers.query_and_chat.streaming_models import SearchDocsPacket
from src.internal.servers.query_and_chat.streaming_models import SearchErrorPacket
from src.internal.servers.query_and_chat.streaming_models import SearchQueriesPacket
from src.internal.servers.users.api import resolve_active_user

logger = logging.getLogger(__name__)

_MAX_QUERY_FOR_FLOW_CLASSIFICATION = 200


def _authenticated_search_filters(
    request: Request,
    requested: SearchFilters | None,
    store: AgenticSearchStore,
) -> SearchFilters:
    user = resolve_active_user(request, store)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    acl = build_user_only_filters(
        user.id,
        email=user.email,
        group_ids=store.list_group_ids_for_user(user.id),
    )
    return SearchFilters(
        source_types=requested.source_types if requested else None,
        document_sets=requested.document_sets if requested else None,
        tags=requested.tags if requested else None,
        access_acl=acl.access_acl,
        time_cutoff=requested.time_cutoff if requested else None,
    )


def _build_flow_classifier_llm():
    """Build the env-configured LLM for search-flow classification, or None.

    Returns ``None`` when no LLM is configured so the endpoint can safely default
    to chat instead of guessing. (The prior implementation hardcoded a stub that
    always returned ``"search"``, making the endpoint constant.)
    """
    api_key = os.environ.get("GEN_AI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    from src.internal.llm.interfaces import LLMConfig
    from src.internal.llm.providers import OpenAICompatibleLLM

    return OpenAICompatibleLLM(
        LLMConfig(
            model_provider=os.environ.get("GEN_AI_MODEL_PROVIDER", "openai"),
            model_name=os.environ.get("GEN_AI_MODEL_VERSION", "gpt-4o-mini"),
            api_key=api_key,
            api_base=os.environ.get("GEN_AI_API_BASE"),
            max_input_tokens=int(os.environ.get("GEN_AI_MAX_INPUT_TOKENS", "8192")),
        )
    )


def create_search_router(
    store: AgenticSearchStore,
    *,
    search_url: str = "http://localhost:8000/retrieve",
) -> APIRouter:
    """Return an APIRouter for search endpoints bound to *store*."""

    router = APIRouter(prefix="/search", tags=["search"])

    def _get_user(request: Request) -> AuthenticatedUser | None:
        return resolve_active_user(request, store)

    @router.post("/search-flow-classification")
    def search_flow_classification(
        body: SearchFlowClassificationRequest,
        http_request: Request,
    ) -> SearchFlowClassificationResponse:
        """Return whether the query is better served by document search or chat.

        Queries longer than 200 characters are always classified as chat
        (the user is composing prose, not searching for a document).
        """
        query = body.user_query
        if len(query) > _MAX_QUERY_FOR_FLOW_CLASSIFICATION:
            return SearchFlowClassificationResponse(is_search_flow=False)

        llm = _build_flow_classifier_llm()
        if llm is None:
            # No LLM configured → cannot classify by content; default to chat.
            return SearchFlowClassificationResponse(is_search_flow=False)

        try:
            is_search = classify_is_search_flow(query, llm)
        except Exception:
            logger.exception(
                "Search flow classification failed; defaulting to chat flow"
            )
            is_search = False

        return SearchFlowClassificationResponse(is_search_flow=is_search)

    @router.post("/send-search-message", response_model=None)
    async def send_search_message(
        body: SendSearchQueryRequest,
        http_request: Request,
    ) -> StreamingResponse | SearchFullResponse:
        """Execute a search with optional query expansion.

        Returns JSON if ``stream=False``, or newline-delimited JSON if
        ``stream=True``.

        NDJSON is **not** SSE: these frames carry no ``data:`` prefix and no
        blank-line terminator, so an SSE reader parses nothing out of this
        response. The sibling streams -- ``/api/agent/stream``,
        ``/chat/send-chat-message`` and ``/tool/send-tool-message`` -- are SSE;
        this one is deliberately not, and the two are not interchangeable.

        No shipped caller sets ``stream=True``: the web app and the MCP
        retrieval client both hardcode ``stream: false``. The branch is kept
        and tested rather than removed, pending a decision on whether the
        search surface should stream at all.
        """
        filters = _authenticated_search_filters(http_request, body.filters, store)

        async def _run() -> SearchQueryResult:
            return await run_expanded_search(
                body.search_query,
                search_url=search_url,
                top_k=body.num_hits,
                filters=filters,
                expand=body.run_query_expansion,
            )

        if not body.stream:
            try:
                result = await _run()
                return SearchFullResponse(
                    all_executed_queries=result.executed_queries,
                    search_docs=[
                        SearchDocWithContent.from_search_result(r)
                        for r in result.results
                    ],
                )
            except Exception as exc:
                logger.exception("Search failed for query: %r", body.search_query)
                return SearchFullResponse(
                    all_executed_queries=[body.search_query],
                    search_docs=[],
                    error=str(exc),
                )

        async def _stream_generator() -> AsyncGenerator[str, None]:
            try:
                # Yield queries packet immediately so the client knows what ran.
                queries_packet = SearchQueriesPacket(
                    all_executed_queries=[body.search_query]
                )
                yield queries_packet.model_dump_json() + "\n"

                result = await _run()

                # Update with actual executed queries after expansion.
                if result.executed_queries != [body.search_query]:
                    updated = SearchQueriesPacket(
                        all_executed_queries=result.executed_queries
                    )
                    yield updated.model_dump_json() + "\n"

                docs_packet = SearchDocsPacket(
                    search_docs=[
                        SearchDocWithContent.from_search_result(r)
                        for r in result.results
                    ]
                )
                yield docs_packet.model_dump_json() + "\n"
            except Exception as exc:
                logger.exception(
                    "Streaming search failed for query: %r", body.search_query
                )
                error_packet = SearchErrorPacket(error=str(exc))
                yield error_packet.model_dump_json() + "\n"

        return StreamingResponse(_stream_generator(), media_type="application/x-ndjson")

    @router.get("/search-history")
    def get_search_history(
        limit: int = 100,
        filter_days: int | None = None,
        http_request: Request = None,
    ) -> SearchHistoryResponse:
        """Return past sessions for the authenticated caller.

        Chat sessions are used as a proxy for search history since the repo
        does not maintain a separate search-query log.
        """
        if limit <= 0 or limit > 1000:
            raise HTTPException(
                status_code=400,
                detail="limit must be between 1 and 1000",
            )
        if filter_days is not None and filter_days <= 0:
            raise HTTPException(
                status_code=400,
                detail="filter_days must be greater than 0",
            )

        user = _get_user(http_request) if http_request else None
        if user is None or user.is_anonymous:
            return SearchHistoryResponse(search_queries=[])

        sessions = store.list_sessions_for_user(
            user.id,
            limit=limit,
            filter_days=filter_days,
        )
        return SearchHistoryResponse(
            search_queries=[
                SearchQueryResponse(
                    query=session.title or session.id,
                    query_expansions=None,
                    created_at=session.created_at,  # type: ignore[arg-type]
                )
                for session in sessions
                if session.created_at
            ]
        )

    return router


__all__ = ["create_search_router"]
