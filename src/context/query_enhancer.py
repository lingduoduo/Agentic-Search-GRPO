"""Query enhancement for best-in-class RAG: decomposition and HyDE."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

from .models import ChatMessage, LLMClient

logger = logging.getLogger(__name__)

_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+[.)]\s*|[-*•]\s*)")
_ARTIFACT_RE = re.compile(r'[\[\]"\'`]')

_DECOMPOSE_PROMPT = """Break the question into 2-4 focused, non-overlapping sub-questions that together cover its full intent.
Each sub-question must be self-contained and keyword-rich so it retrieves well on its own.
Return one sub-question per line. Do not number them. Do not repeat the original question.
If the question is already atomic, return it unchanged on a single line.

Question: {query}""".strip()

_HYDE_PROMPT = """Write a single concise paragraph that would be the ideal answer to the following question.
Write as if you are an expert. Be factual and specific. Do not say "I" or ask follow-up questions.

Question: {query}
Answer:""".strip()

_STEP_BACK_PROMPT = """Write one broader background question whose answer provides the concepts needed to answer the original.
Keep the same domain and key entities; generalise only the specifics.
Return only the rewritten question, no explanation.

Question: {query}
Broader question:""".strip()

_REWRITE_PROMPT = """Rewrite the following search query into one clear, canonical question.
Fix typos and remove filler, but preserve the original meaning and all key terms.
Return only the rewritten query, no explanation.

Query: {query}
Rewritten query:""".strip()


def _clean_line(line: str) -> str:
    return _ARTIFACT_RE.sub("", _LIST_MARKER_RE.sub("", line)).strip()


def _llm_text(response: object) -> str:
    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "content"):
        return response.content
    return str(response)


@dataclass
class QueryBundle:
    original: str
    sub_queries: list[str] = field(default_factory=list)
    hyde_text: str | None = None
    step_back_query: str | None = None

    def all_queries(self) -> list[str]:
        """Keep original intent first, then novel, nonblank enhancement variants."""
        seen: set[str] = set()
        result: list[str] = []
        for q in [
            self.original,
            *self.sub_queries,
            self.hyde_text,
            self.step_back_query,
        ]:
            if not q:
                continue
            key = " ".join(q.lower().split())
            if key and key not in seen:
                seen.add(key)
                result.append(q)
        return result


class QueryEnhancer:
    """Enhance a query with LLM-driven decomposition and HyDE.

    Both strategies degrade gracefully: when no LLM is provided, or on any
    LLM failure, the original query is returned unchanged so the retrieval
    pipeline is never blocked.
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm

    def decompose(self, query: str) -> list[str]:
        """Break query into focused sub-queries.  Falls back to [query] on failure."""
        if self.llm is None:
            return [query]
        try:
            raw = _llm_text(
                self.llm.complete(
                    [
                        ChatMessage(
                            role="user", content=_DECOMPOSE_PROMPT.format(query=query)
                        )
                    ]
                )
            ).strip()
        except Exception as exc:
            logger.warning("Query decomposition failed: %s", exc)
            return [query]

        if not raw:
            return [query]

        original_lower = query.lower()
        seen: set[str] = {original_lower}
        result: list[str] = []
        for line in raw.splitlines():
            cleaned = _clean_line(line)
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                result.append(cleaned)

        return result if result else [query]

    def hyde(self, query: str) -> str | None:
        """Generate a hypothetical answer for HyDE dense retrieval.  Returns None on failure."""
        if self.llm is None:
            return None
        try:
            raw = _llm_text(
                self.llm.complete(
                    [ChatMessage(role="user", content=_HYDE_PROMPT.format(query=query))]
                )
            ).strip()
            return raw or None
        except Exception as exc:
            logger.warning("HyDE generation failed: %s", exc)
            return None

    def step_back(self, query: str) -> str | None:
        """Generate a broader, more abstract version of the query for wider retrieval coverage.
        Returns None on failure or when no LLM is configured."""
        if self.llm is None:
            return None
        try:
            raw = _llm_text(
                self.llm.complete(
                    [
                        ChatMessage(
                            role="user", content=_STEP_BACK_PROMPT.format(query=query)
                        )
                    ]
                )
            ).strip()
            return raw or None
        except Exception as exc:
            logger.warning("Step-back generation failed: %s", exc)
            return None

    def rewrite(self, query: str) -> str | None:
        """Return a cleaned, canonical rewrite of the query. None on failure/no LLM."""
        if self.llm is None:
            return None
        try:
            raw = _llm_text(
                self.llm.complete(
                    [
                        ChatMessage(
                            role="user", content=_REWRITE_PROMPT.format(query=query)
                        )
                    ]
                )
            ).strip()
            return _clean_line(raw) or None
        except Exception as exc:
            logger.warning("Query rewrite failed: %s", exc)
            return None

    def enhance(self, query: str) -> QueryBundle:
        """Run decompose + HyDE + step-back and return a QueryBundle."""
        return QueryBundle(
            original=query,
            sub_queries=self.decompose(query),
            hyde_text=self.hyde(query),
            step_back_query=self.step_back(query),
        )

    async def enhance_async(self, query: str) -> QueryBundle:
        """Async `enhance`: the three strategies run off-loop and concurrently.

        `llm.complete` is a blocking `requests` call, so calling `enhance` from
        a coroutine froze the event loop for all three round trips -- every
        other in-flight session stalled behind one user's query preparation.
        The strategies are independent and each already falls back on its own
        exception, so nothing here can raise and the round costs one round trip
        rather than three.
        """
        sub_queries, hyde_text, step_back_query = await asyncio.gather(
            asyncio.to_thread(self.decompose, query),
            asyncio.to_thread(self.hyde, query),
            asyncio.to_thread(self.step_back, query),
        )
        return QueryBundle(
            original=query,
            sub_queries=sub_queries,
            hyde_text=hyde_text,
            step_back_query=step_back_query,
        )
