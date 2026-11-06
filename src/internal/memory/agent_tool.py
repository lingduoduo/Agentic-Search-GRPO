"""Request-bound ``search_memory`` tool for the SEARCH/TOOL agent loops.

Built per request around one ``user_id`` and never registered globally, so a
call can only ever read the signed-in caller's own memories.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.internal.memory.categories import CATEGORIES
from src.internal.memory.search import hybrid_search
from src.internal.tools.base import (
    FailureCategory,
    ResultKind,
    Tool,
    ToolEffect,
    ToolErrorText,
    ToolFailure,
    ToolSchema,
)

NO_MATCH_TEXT = "no matching memories"


class _SearchMemoryTool(Tool):
    def __init__(self, store, user_id: str, encoder=None) -> None:
        self._store, self._user_id, self._encoder = store, user_id, encoder

    @property
    def name(self) -> str:
        return "search_memory"

    @property
    def effect(self) -> ToolEffect:
        return ToolEffect.READ_ONLY

    @property
    def citeable(self) -> bool:
        return False

    @property
    def result_kind(self) -> ResultKind:
        return ResultKind.TEXT

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="search_memory",
            description=(
                "Search what is remembered about the current user "
                "(preferences, constraints, profile, projects, decisions)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )

    async def execute(self, instance_id: str, arguments: dict[str, Any]):
        del instance_id
        query = str(arguments.get("query", "")).strip()
        if not query:
            text = "query is required"
            failure = ToolFailure(FailureCategory.INVALID_INPUT, text)
            return ToolErrorText(text, failure), None, {"failure": failure}
        try:
            max_results = max(1, min(10, int(arguments.get("max_results") or 5)))
        except (TypeError, ValueError):
            max_results = 5
        category = arguments.get("category")
        if category not in CATEGORIES:
            category = None
        # The store is synchronous SQLite; keep it off the event loop.
        hits = await asyncio.to_thread(
            hybrid_search,
            self._store,
            self._user_id,
            query,
            max_results=max_results,
            encoder=self._encoder,
            category=category,
        )
        if not hits:
            return NO_MATCH_TEXT, None, {}
        return "\n".join(f"[{r.category}] {r.memory_text}" for r, _ in hits), None, {}


def build_search_memory_tool(store, user_id: str, encoder=None) -> Tool:
    return _SearchMemoryTool(store, user_id, encoder)
