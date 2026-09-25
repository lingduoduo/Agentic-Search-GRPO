"""Agentic RAG loop: iterative hybrid retrieval + LLM-driven sufficiency assessment."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.core.control_flow_trace import ControlFlowRecorder

from src.context.models import (
    AnswerGenerationRequest,
    ChatMessage,
    ContextDocument,
    LLMClient,
    SearchContextBundle,
    SearchFilters,
)
from src.context.pipeline import generate_answer, retrieve_contexts
from src.context.query_enhancer import QueryEnhancer
from src.internal.configs.timeouts import get_timeout_policies

logger = logging.getLogger(__name__)

_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+[.)]\s*|[-*•]\s*)")
_ARTIFACT_RE = re.compile(r'[\[\]"\'`]')

_SUFFICIENCY_PROMPT = """You are evaluating whether retrieved documents are sufficient to fully answer a question.
Respond with exactly "yes" or "no".

Question: {question}

Retrieved context (first 1500 chars):
{context}""".strip()

_GAP_ANALYSIS_PROMPT = """You are analyzing whether retrieved documents fully answer a question.

Question: {question}

Retrieved context (first 1000 chars):
{context}

Step 1 — List the specific pieces of information the question requires but the context does NOT provide.
         Write each gap as a short phrase (e.g. "training cost of GPT-4").
         If nothing is missing, write "none".

Step 2 — For each gap, write one focused search query that would retrieve the missing information.
         Format: one query per line, no numbering, no extra text.
         Queries only — do not repeat the gap phrases.

Output format:
GAPS:
<gap 1>
<gap 2>

QUERIES:
<query 1>
<query 2>""".strip()


def _llm_text(response: object) -> str:
    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "content"):
        return response.content
    return str(response)


def _clean_line(line: str) -> str:
    return _ARTIFACT_RE.sub("", _LIST_MARKER_RE.sub("", line)).strip()


def _norm_query(q: str) -> str:
    return " ".join(q.lower().split())


def _dedupe_novel(queries: list[str], seen: set[str]) -> list[str]:
    """Return queries whose normalized form is new; record each into `seen`.

    Dedupes both within this batch and against earlier rounds. Returned
    strings are the original queries — retrieval must use the raw text.
    """
    novel: list[str] = []
    for q in queries:
        norm = _norm_query(q)
        if norm not in seen:
            seen.add(norm)
            novel.append(q)
    return novel


def _doc_key(doc: ContextDocument) -> str:
    if doc.url:
        return doc.url.strip().lower()
    return hashlib.sha256(doc.content.encode("utf-8")).hexdigest()


def _parse_gap_queries(raw: str) -> list[str]:
    """Extract the QUERIES section from a structured gap-analysis response.

    Falls back to treating every non-empty line as a query when the
    structured format is absent (e.g. legacy LLM response).
    """
    if "QUERIES:" in raw:
        queries_section = raw.split("QUERIES:", 1)[1]
    elif "GAPS:" in raw:
        return []
    else:
        queries_section = raw
    queries: list[str] = []
    for line in queries_section.splitlines():
        cleaned = _clean_line(line)
        if cleaned:
            queries.append(cleaned)
    return queries


@dataclass(frozen=True)
class AgenticRAGConfig:
    max_rounds: int = 3
    topk: int = 5
    retrieval_url: str = "http://localhost:8001/retrieve"
    max_followups_per_round: int = 5
    sufficiency_timeout_s: float = field(
        default_factory=lambda: get_timeout_policies().llm.sufficiency_timeout_seconds
    )
    filters: SearchFilters | None = None


@dataclass
class AgenticRAGResult:
    answer: str
    citations: list[str]
    rounds_used: int
    context: SearchContextBundle
    # True when a sufficiency check failed open (LLM error or timeout) rather
    # than returning a verdict. The loop then stops early for a reason that has
    # nothing to do with the evidence, so `rounds_used == 1` does not mean the
    # first round was enough. Callers that report confidence should say so.
    sufficiency_degraded: bool = False


class AgenticRAGLoop:
    """Iterative RAG loop with query enhancement and evidence sufficiency gating.

    Flow per run():
      1. Enhance query (decompose + HyDE via QueryEnhancer)
      2. Retrieve for every current query; accumulate unique docs by id
      3. Ask LLM: is evidence sufficient? → if yes, break; if no, generate follow-ups
      4. Repeat up to max_rounds
      5. Synthesize grounded answer from all accumulated evidence
    """

    def __init__(self, config: AgenticRAGConfig, llm: LLMClient | None = None) -> None:
        self.config = config
        self.llm = llm
        self._enhancer = QueryEnhancer(llm)

    def _record_search_stage(
        self,
        query: str,
        top_k: int,
        round_index: int,
        documents: list[ContextDocument],
    ) -> None:
        # Deferred import: src.internal.servers.web's package __init__ imports
        # agent-loop modules (via app.py), so a top-level import here would be
        # circular.
        from src.internal.servers.web import request_capture as _capture

        if _capture.active() is None:
            return

        _capture.record_stage(
            "search",
            "retrieve",
            {
                "query": query,
                "top_k": top_k,
                "round": round_index,
                "documents": [
                    {
                        "id": d.id,
                        "title": getattr(d, "title", ""),
                        "text": getattr(d, "text", d.content),
                        "score": getattr(d, "score", None),
                        "source": getattr(d, "source", d.url),
                    }
                    for d in documents
                ],
            },
        )

    async def run(
        self,
        question: str,
        *,
        retrieval_query: str | None = None,
        chat_history: list[ChatMessage] | None = None,
        recorder: "ControlFlowRecorder | None" = None,
        user_memory: str | None = None,
        on_claim: Callable[[str], None] | None = None,
    ) -> AgenticRAGResult:
        def _emit(
            component: str,
            action: str,
            status: str,
            duration_ms: int | None = None,
            **details: object,
        ) -> None:
            if recorder is not None:
                recorder.record(
                    turn=max(rounds_used, 1),
                    component=component,
                    action=action,
                    status=status,
                    duration_ms=duration_ms,
                    details=details,
                )

        # Retrieval (enhancement, sufficiency, gap queries) runs on the resolved
        # standalone query; synthesis still answers the question as asked.
        search_question = retrieval_query or question

        accumulated: dict[str, ContextDocument] = {}
        seen_queries: set[str] = set()
        rounds_used = 0
        sufficiency_degraded = False

        t0 = time.perf_counter()
        bundle = await self._enhancer.enhance_async(search_question)
        current_queries = bundle.all_queries()
        # Key by content fingerprint so docs from different retrieve_context() calls
        # (which all produce ephemeral D1-D5 IDs) are deduplicated correctly.
        _emit(
            "query_enhancer",
            "enhance",
            "completed",
            duration_ms=round((time.perf_counter() - t0) * 1000),
            query_count=len(current_queries),
        )

        merged = SearchContextBundle(query=question, documents=[])

        for round_idx in range(self.config.max_rounds):
            rounds_used += 1
            # current_queries is already deduped+recorded into seen_queries when
            # it originates from the follow-up branch below; only the initial
            # enhance() batch (round_idx == 0) still needs deduping here.
            # INVARIANT: every in-loop assignment to current_queries must route
            # through _dedupe_novel (see the follow-up branch), or round 2+ would
            # retrieve un-deduped queries.
            if round_idx == 0:
                novel_queries = _dedupe_novel(current_queries, seen_queries)
            else:
                novel_queries = current_queries
            if not novel_queries:
                break

            t_retr = time.perf_counter()
            # One request for the whole round: the retrieval API is natively
            # multi-query, so N queries cost one round trip on one session
            # instead of N concurrent ones. A transport failure fails the
            # round's queries together, which the per-query handler below
            # reports exactly as it reported an individual failure.
            try:
                contexts: list[object] = list(
                    await retrieve_contexts(
                        novel_queries,
                        search_url=self.config.retrieval_url,
                        top_k=self.config.topk,
                        filters=self.config.filters,
                    )
                )
            except Exception as exc:  # noqa: BLE001 -- degrade, as before
                contexts = [exc] * len(novel_queries)
            for q, ctx in zip(novel_queries, contexts):
                try:
                    if isinstance(ctx, Exception):
                        raise ctx
                    for doc in ctx.documents:
                        key = _doc_key(doc)
                        if key not in accumulated:
                            accumulated[key] = doc
                    self._record_search_stage(
                        q, self.config.topk, rounds_used, ctx.documents
                    )
                except Exception as exc:
                    logger.warning("Retrieval failed for query %r: %s", q, exc)
            retr_ms = round((time.perf_counter() - t_retr) * 1000)

            # Re-assign stable D1..DN IDs so citations are consistent across rounds.
            stable_docs = [
                ContextDocument(
                    id=f"D{i}",
                    title=doc.title,
                    content=doc.content,
                    url=doc.url,
                    score=doc.score,
                    metadata=doc.metadata,
                )
                for i, doc in enumerate(accumulated.values(), 1)
            ]
            merged = SearchContextBundle(query=question, documents=stable_docs)
            _emit(
                "search_tool",
                "vector_db_search",
                "completed",
                duration_ms=retr_ms,
                query_count=len(novel_queries),
                document_count=len(stable_docs),
                search_round=rounds_used,
            )

            # On the last round always proceed to synthesis; otherwise check sufficiency.
            is_last = round_idx == self.config.max_rounds - 1
            if not is_last:
                t_suff = time.perf_counter()
                sufficient, degraded = await self._is_sufficient(
                    search_question, merged
                )
                sufficiency_degraded = sufficiency_degraded or degraded
                # A failed-open check is reported as a failure that fell back,
                # not as a verdict: "sufficient" is this loop's stop condition, so
                # a silent fail-open is indistinguishable from real sufficiency.
                _emit(
                    "evidence_judge",
                    "sufficiency_check",
                    "failed" if degraded else "decided",
                    duration_ms=round((time.perf_counter() - t_suff) * 1000),
                    sufficient=sufficient,
                    fallback=degraded,
                    search_round=rounds_used,
                )
                if sufficient:
                    break
                follow_ups = await self._generate_followup(search_question, merged)
                novel_follow_ups = _dedupe_novel(follow_ups, seen_queries)
                if not novel_follow_ups:
                    break
                current_queries = novel_follow_ups[
                    : self.config.max_followups_per_round
                ]

        t_syn = time.perf_counter()
        # Offloaded, like the sufficiency and gap-analysis calls below it.
        # `generate_answer` is synchronous and calls `llm.complete`, a blocking
        # `requests` call -- awaiting it inline froze the whole event loop for
        # the full completion, stalling every other in-flight session behind one
        # user's answer. This is the most expensive call in the loop, so it was
        # the worst one to leave unprotected.
        gen_kwargs: dict[str, object] = {"llm": self.llm}
        if on_claim is not None:
            gen_kwargs["on_claim"] = on_claim
        gen_result = await asyncio.to_thread(
            generate_answer,
            AnswerGenerationRequest(
                question=question,
                context=merged,
                chat_history=chat_history or [],
                user_memory=user_memory,
            ),
            **gen_kwargs,
        )
        _emit(
            "answer_generator",
            "synthesize",
            "completed",
            duration_ms=round((time.perf_counter() - t_syn) * 1000),
            citation_count=len(gen_result.citations),
            document_count=len(merged.documents),
            llm_first_token_ms=(
                round(gen_result.timings.llm_first_token_ms, 1)
                if gen_result.timings is not None
                and gen_result.timings.llm_first_token_ms is not None
                else None
            ),
            time_to_first_claim_ms=(
                round(gen_result.timings.time_to_first_claim_ms, 1)
                if gen_result.timings is not None
                and gen_result.timings.time_to_first_claim_ms is not None
                else None
            ),
        )
        return AgenticRAGResult(
            answer=gen_result.answer,
            citations=gen_result.citations,
            rounds_used=rounds_used,
            context=merged,
            sufficiency_degraded=sufficiency_degraded,
        )

    async def _is_sufficient(
        self, question: str, context: SearchContextBundle
    ) -> tuple[bool, bool]:
        """Return ``(sufficient, degraded)`` for the current evidence.

        ``degraded`` is True only when the check could not produce a verdict at
        all (LLM error or timeout) and failed open. It is not an error the loop
        recovers from -- it changes what ``sufficient=True`` means -- so it
        travels with the verdict instead of being swallowed in a log line.
        """
        if not context.documents:
            return False, False
        if self.llm is None:
            return True, False
        prompt = _SUFFICIENCY_PROMPT.format(
            question=question,
            context=context.to_context_text()[:1500],
        )
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.llm.complete,
                    [ChatMessage(role="user", content=prompt)],
                ),
                timeout=self.config.sufficiency_timeout_s,
            )
            return _llm_text(response).strip().lower().startswith("yes"), False
        except Exception as exc:  # includes asyncio.TimeoutError
            logger.warning("Sufficiency check failed or timed out: %s", exc)
            return True, True  # fail-open → stop looping on error/timeout

    async def _generate_followup(
        self, question: str, context: SearchContextBundle
    ) -> list[str]:
        if self.llm is None:
            return []
        prompt = _GAP_ANALYSIS_PROMPT.format(
            question=question,
            context=context.to_context_text()[:1000],
        )
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.llm.complete,
                    [ChatMessage(role="user", content=prompt)],
                ),
                timeout=self.config.sufficiency_timeout_s,
            )
            return _parse_gap_queries(_llm_text(response).strip())
        except Exception as exc:  # includes asyncio.TimeoutError
            logger.warning("Gap analysis failed or timed out: %s", exc)
            return []
