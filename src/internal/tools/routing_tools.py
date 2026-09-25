"""FunctionTool wrappers for search and RAG, used as ToolAgentLoop routing tools."""

from __future__ import annotations

import json

from .base import (
    FailureCategory,
    FunctionTool,
    ToolEffect,
    ToolErrorText,
    ToolFailure,
    ResultKind,
)
from .search import search_tool

# Mirrors search_tool's own default max_retries (src/internal/tools/search.py),
# so the failure we report reflects the attempts it actually made.
_SEARCH_ATTEMPTS = 3

_SEARCH_TOOL_PARAMS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1, "description": "The search query."},
    },
    "required": ["query"],
    "additionalProperties": False,
}

_RAG_TOOL_PARAMS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "description": "The question to answer using retrieval.",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


def build_search_routing_tool(
    *, search_url: str, top_k: int, name: str = "search", filters=None
) -> FunctionTool:
    """FunctionTool that retrieves documents from the corpus.

    Named ``search`` by default: it *is* the corpus search a model should reach
    for, and an opaque name costs tool-selection accuracy on small models.

    ``filters`` are both sent to retrieval and enforced on what comes back.
    Sending alone is not enforcement: ``demo.py`` and ``hybrid.py`` accept the
    field and ignore it, so an unpaired send would hand the model documents the
    caller may not read. ``None`` keeps the tool unfiltered for callers that
    have no identity (training scripts, evals).
    """

    async def _execute(query: str) -> str:
        pages = await search_tool(
            query,
            provider="retrieval",
            search_url=search_url,
            page_size=top_k,
            filters=filters.to_payload() if filters is not None else None,
        )
        if filters is not None:
            pages = [p for p in pages if p.error or filters.matches(p.metadata or {})]
        results = [
            {"title": p.title or "", "content": p.summary or "", "url": p.url}
            for p in pages
            if not p.error
        ]
        if not results and any(p.error for p in pages):
            errors = [p.error for p in pages if p.error]
            return ToolErrorText(
                json.dumps({"error": errors[0]}),
                ToolFailure(
                    FailureCategory.TRANSIENT,
                    "search backend unavailable",
                    provider_attempts=_SEARCH_ATTEMPTS,
                ),
            )
        return json.dumps(results)

    return FunctionTool(
        fn=_execute,
        name=name,
        description="Retrieve relevant documents from the corpus given a search query.",
        parameters=_SEARCH_TOOL_PARAMS,
        effect=ToolEffect.READ_ONLY,
        citeable=True,
        result_kind=ResultKind.DOCUMENTS,
        retries_internally=True,
    )


def build_rag_routing_tool(
    *,
    llm,
    search_url: str,
    top_k: int,
    filters=None,
) -> FunctionTool:
    """FunctionTool that generates a RAG answer."""

    async def _execute(query: str) -> str:
        from src.context import answer_with_retrieval

        result = await answer_with_retrieval(
            query,
            llm=llm,
            search_url=search_url,
            top_k=top_k,
            filters=filters,
        )
        return json.dumps({"answer": result.answer, "citations": result.citations})

    return FunctionTool(
        fn=_execute,
        name="rag_routing_tool",
        description="Answer a question using retrieval-augmented generation.",
        parameters=_RAG_TOOL_PARAMS,
        effect=ToolEffect.READ_ONLY,
        result_kind=ResultKind.JSON,
    )
