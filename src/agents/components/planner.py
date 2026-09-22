"""Planner: turn the policy LM's tagged output into the search loop's next step.

Parsing is deliberately lenient. A turn may carry several tags, in any order,
with attributes (``retriever`` on a search tag, ``rerank="true"``), and an
unparseable generation yields no actions rather than raising -- the loop's
format-recovery path handles that case.

Everything here is a pure function of the generation text plus, for
``partition_search_requests``, the queries already run.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..core.state import AgentState, Retriever


def _normalize_query(query: str) -> str:
    """Whitespace- and case-insensitive key for duplicate detection."""
    return " ".join(query.split()).casefold()


class Planner:
    """Parse one generation step into the search loop's next actions."""

    @staticmethod
    def parse_actions(text: str, action_tags: Sequence[str]) -> list[tuple[str, str]]:
        tags = "|".join(re.escape(tag) for tag in action_tags)
        action_re = re.compile(
            rf"<({tags})(?:\s+[^>]*)?>(.*?)</\1>",
            re.DOTALL,
        )
        return [
            (match.group(1), match.group(2).strip())
            for match in action_re.finditer(text)
        ]

    @staticmethod
    def round_retriever(text: str, search_tags: Sequence[str]) -> Retriever:
        tags = "|".join(re.escape(tag) for tag in search_tags)
        match = re.search(
            rf'<(?:{tags})\s+[^>]*retriever="(?P<retriever>\w+)"',
            text,
            re.IGNORECASE,
        )
        if match and match.group("retriever").lower() == "web":
            return Retriever.WEB
        return Retriever.VECTOR_DB

    @staticmethod
    def round_rerank(text: str, search_tags: Sequence[str]) -> bool:
        tags = "|".join(re.escape(tag) for tag in search_tags)
        match = re.search(
            rf'<(?:{tags})\s+[^>]*rerank="(?P<rerank>\w+)"',
            text,
            re.IGNORECASE,
        )
        return bool(match and match.group("rerank").lower() == "true")

    @staticmethod
    def partition_search_requests(
        query_specs: list[tuple[str | None, str]],
        state: AgentState,
        effective_limit: int,
    ) -> tuple[list[tuple[str | None, str]], list[str], list[str]]:
        seen = {_normalize_query(query) for query in state.previous_queries}
        at_limit = effective_limit > 0 and state.search_rounds >= effective_limit
        allowed: list[tuple[str | None, str]] = []
        repeated: list[str] = []
        overflow: list[str] = []
        for task_id, query in query_specs:
            if _normalize_query(query) in seen:
                repeated.append(query)
            elif at_limit:
                overflow.append(query)
            else:
                allowed.append((task_id, query))
        return allowed, repeated, overflow
