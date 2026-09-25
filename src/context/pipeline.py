"""High-level retrieval, prompt, and answer-generation pipeline."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from src.internal.configs.timeouts import get_timeout_policies
from src.internal.observability.stage_metrics import mark_answer_generation

from .models import AnswerGenerationRequest
from .models import AnswerGenerationResult
from .models import GenerationTimings
from .models import AnswerClaim
from .models import ChatMessage
from .models import ContextDocument
from .models import ContextSection
from .models import EvidenceSnippet
from .models import EvidenceSource
from .models import LLMClient
from .models import LLMResponse
from .models import LLMTimeoutError
from .models import PromptBundle
from .models import SearchContextBundle
from .models import SearchFilters
from .models import SearchRequest
from .models import VerificationStatus
from .models import VerificationResult
from .models import GroundedGenerationConfig
from .prompts import build_corrective_answer_prompt
from .prompts import build_chat_prompt
from .prompts import build_structured_answer_prompt
from .retrieval.search_runner import build_search_context
from .retrieval.search_runner import build_search_contexts
from .streaming_draft import IncrementalDraftReader
from .tool_evidence import ToolRegistry
from .tool_evidence import ToolSelector
from .utils import extract_citations
from .structured_output import SchemaUnsupportedError
from .structured_output import StructuredOutputCapability
from .structured_output import StructuredOutputRequest
from .structured_output import answer_draft_json_schema


async def retrieve_context(
    question: str,
    *,
    search_url: str = "http://localhost:8000/retrieve",
    top_k: int = 5,
    filters: SearchFilters | None = None,
) -> SearchContextBundle:
    return await build_search_context(
        SearchRequest(query=question, top_k=top_k, filters=filters),
        search_url=search_url,
    )


async def retrieve_contexts(
    questions: list[str],
    *,
    search_url: str = "http://localhost:8000/retrieve",
    top_k: int = 5,
    filters: SearchFilters | None = None,
) -> list[SearchContextBundle]:
    """Batched `retrieve_context`: one bundle per question, in input order."""
    return await build_search_contexts(
        questions, top_k=top_k, filters=filters, search_url=search_url
    )


@mark_answer_generation
def generate_answer(
    request: AnswerGenerationRequest,
    *,
    llm: LLMClient | None = None,
    on_claim: Callable[[str], None] | None = None,
) -> AnswerGenerationResult:
    legacy_prompt = build_chat_prompt(
        request.question,
        request.context,
        history=request.chat_history,
        config=request.behavior,
        user_memory=request.user_memory,
    )
    config = request.grounded_generation
    evidence = request.evidence
    if evidence is None:
        from .safety import evidence_from_context

        evidence = evidence_from_context(request.context)
    tool_evidence = [item for item in evidence if item.provenance == "tool"]
    confidence: float | None = None
    verification_status: VerificationStatus | None = None
    abstained = False
    retry_count = 0
    structured_output_requested = False
    structured_output_applied = False
    structured_output_downgraded = False
    structured_output_category = None
    generation_timings = None

    if llm is None:
        answer, verification = _generate_extractive_answer(
            request.question,
            evidence,
            evidence_sufficiency=request.evidence_sufficiency,
            overlap_threshold=config.overlap_threshold,
        )
        confidence = verification.confidence
        verification_status = verification.status
        abstained = verification.status is VerificationStatus.ABSTAINED
        prompt = legacy_prompt
    elif not config.enabled:
        raw = llm.complete(legacy_prompt.messages)
        answer = raw.text if isinstance(raw, LLMResponse) else str(raw)
        prompt = legacy_prompt
    elif not evidence:
        answer = _canonical_abstention()
        confidence = 0.0
        verification_status = VerificationStatus.ABSTAINED
        abstained = True
        prompt = build_structured_answer_prompt(
            request.question,
            request.context,
            request.behavior,
            history=request.chat_history,
            evidence=evidence,
            user_memory=request.user_memory,
        )
    else:
        prompt = build_structured_answer_prompt(
            request.question,
            request.context,
            request.behavior,
            history=request.chat_history,
            evidence=evidence,
            user_memory=request.user_memory,
        )
        (
            answer,
            confidence,
            verification_status,
            retry_count,
            structured_output_requested,
            structured_output_applied,
            structured_output_downgraded,
            structured_output_category,
            generation_timings,
        ) = _generate_guarded_answer(request, llm, prompt, evidence, on_claim)
        abstained = verification_status is VerificationStatus.ABSTAINED

    grounding_report = None
    if request.verify_grounding:
        from .grounding import GroundingVerifier

        report = GroundingVerifier().verify(
            answer, request.context, tool_evidence=tool_evidence
        )
        answer = report.answer_clean
        grounding_report = report

    return AnswerGenerationResult(
        answer=answer,
        citations=extract_citations(answer),
        context=request.context,
        prompt=prompt,
        grounding_report=grounding_report,
        confidence=confidence,
        verification_status=verification_status,
        abstained=abstained,
        tool_evidence=tool_evidence,
        retry_count=retry_count,
        structured_output_requested=structured_output_requested,
        structured_output_applied=structured_output_applied,
        structured_output_downgraded=structured_output_downgraded,
        structured_output_category=structured_output_category,
        timings=generation_timings,
    )


def _generate_guarded_answer(
    request: AnswerGenerationRequest,
    llm: LLMClient,
    prompt: PromptBundle,
    evidence: list[EvidenceSource],
    on_claim: Callable[[str], None] | None = None,
) -> tuple[
    str, float, VerificationStatus, int, bool, bool, bool, str | None, GenerationTimings
]:
    from .safety import (
        TIMEOUT_DEGRADED_ANSWER,
        parse_answer_draft,
        render_claim,
        render_claims,
        render_verified_answer,
        verify_answer_draft,
        verify_claim,
    )

    cap = get_timeout_policies().llm.grounded_max_retries
    max_attempts = 1 + min(max(request.grounded_generation.max_retries, 0), cap)
    raw_text = ""
    feedback = ""
    result = None
    capability = getattr(
        llm, "structured_output_capability", StructuredOutputCapability.PROMPT_ONLY
    )
    schema_request: StructuredOutputRequest | None = None
    if capability is StructuredOutputCapability.JSON_SCHEMA:
        schema_request = StructuredOutputRequest(
            name="answer_draft", schema=answer_draft_json_schema()
        )
    requested = schema_request is not None
    downgraded = False
    applied = False
    category = None

    evidence_by_id = {item.id: item for item in evidence}
    committed: list[AnswerClaim] = []
    committed_keys: set[tuple[str, frozenset[str]]] = set()
    stream_fn = getattr(llm, "stream_complete", None) if on_claim else None

    def _commit(claim: AnswerClaim) -> None:
        """Emit a supported claim once. Emitted claims are permanent."""
        nonlocal first_claim_ms
        key = (claim.text, frozenset(claim.evidence_ids))
        if key in committed_keys:
            return
        committed_keys.add(key)
        committed.append(claim)
        on_claim(render_claim(claim))
        if first_claim_ms is None:
            first_claim_ms = (time.perf_counter() - t_gen) * 1000

    def _timings() -> GenerationTimings:
        return GenerationTimings(
            llm_first_token_ms=first_token_ms,
            time_to_first_claim_ms=first_claim_ms,
        )

    def _committed_answer(
        category: str | None,
    ) -> tuple[
        str,
        float,
        VerificationStatus,
        int,
        bool,
        bool,
        bool,
        str | None,
        GenerationTimings,
    ]:
        """Build the return tuple from `committed`, not from `result`.

        The invariant: the answer is exactly what the user was shown. Every
        early exit from the attempt loop below routes through this one
        function when `committed` is non-empty, so there is a single place
        that enforces "emitted claims are never dropped" rather than several
        copies that can drift out of sync.
        """
        status = (
            VerificationStatus.VERIFIED
            if result is not None and not result.unsupported_claims
            else VerificationStatus.PARTIAL
        )
        return (
            render_claims(committed),
            result.confidence if result is not None else 0.0,
            status,
            attempt,
            requested,
            applied,
            downgraded,
            category,
            _timings(),
        )

    first_token_ms: float | None = None
    first_claim_ms: float | None = None
    t_gen = time.perf_counter()

    for attempt in range(max_attempts):
        active_prompt = prompt
        if attempt:
            active_prompt = build_corrective_answer_prompt(
                request.question,
                request.context,
                original_draft=raw_text,
                verifier_feedback=feedback,
                config=request.behavior,
                history=request.chat_history,
                evidence=evidence,
                user_memory=request.user_memory,
            )
        try:
            if stream_fn is not None:
                reader = IncrementalDraftReader(set(evidence_by_id))
                parts: list[str] = []
                for delta in stream_fn(
                    active_prompt.messages,
                    **({"structured_output": schema_request} if schema_request else {}),
                ):
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - t_gen) * 1000
                    parts.append(delta)
                    for claim in reader.feed(delta):
                        verdict = verify_claim(
                            claim,
                            evidence_by_id,
                            overlap_threshold=request.grounded_generation.overlap_threshold,
                        )
                        if verdict.supported:
                            _commit(claim)
                raw = "".join(parts)
            else:
                raw = llm.complete(
                    active_prompt.messages,
                    **({"structured_output": schema_request} if schema_request else {}),
                )
        except LLMTimeoutError:
            if committed:
                return _committed_answer("timeout")
            return (
                TIMEOUT_DEGRADED_ANSWER,
                0.0,
                VerificationStatus.ABSTAINED,
                attempt,
                requested,
                applied,
                downgraded,
                "timeout",
                _timings(),
            )
        except SchemaUnsupportedError:
            if schema_request is None:
                raise
            downgraded = True
            schema_request = None
            raw = llm.complete(active_prompt.messages)
        if isinstance(raw, LLMResponse):
            applied = applied or raw.structured.applied
            if raw.structured.refused:
                if committed:
                    return _committed_answer("refused")
                return (
                    _canonical_abstention(),
                    0.0,
                    VerificationStatus.ABSTAINED,
                    attempt,
                    requested,
                    applied,
                    downgraded,
                    "refused",
                    _timings(),
                )
            if raw.structured.incomplete_reason is not None:
                category = "incomplete"
                feedback = "Provider returned incomplete structured output."
                continue
        raw_text = raw.text if isinstance(raw, LLMResponse) else str(raw)
        try:
            draft = parse_answer_draft(raw_text, evidence)
        except ValueError as exc:
            feedback = str(exc)
            continue
        result = verify_answer_draft(
            draft,
            evidence,
            overlap_threshold=request.grounded_generation.overlap_threshold,
            evidence_sufficiency=request.evidence_sufficiency,
            retry_occurred=bool(attempt),
        )
        if on_claim is not None:
            for claim in result.supported_claims:
                _commit(claim)
        if draft.abstain or not result.unsupported_claims:
            break
        feedback = _verifier_feedback(result)

    if committed:
        return _committed_answer(category)

    if result is None:
        return (
            _canonical_abstention(),
            0.0,
            VerificationStatus.ABSTAINED,
            max_attempts - 1,
            requested,
            applied,
            downgraded,
            category,
            _timings(),
        )
    return (
        render_verified_answer(result),
        result.confidence,
        result.status,
        int(result.retry_occurred),
        requested,
        applied,
        downgraded,
        category,
        _timings(),
    )


def _verifier_feedback(result: VerificationResult) -> str:
    return "\n".join(
        f"Unsupported claim: {verdict.claim.text} ({verdict.reason})"
        for verdict in result.verdicts
        if not verdict.supported
    )


def _canonical_abstention() -> str:
    from .safety import CANONICAL_ABSTENTION

    return CANONICAL_ABSTENTION


async def answer_with_retrieval(
    question: str,
    *,
    llm: LLMClient | None = None,
    chat_history: list[ChatMessage] | None = None,
    search_url: str = "http://localhost:8000/retrieve",
    top_k: int = 5,
    filters: SearchFilters | None = None,
    tool_registry: ToolRegistry | None = None,
    tool_selector: ToolSelector | None = None,
    max_tool_calls: int = 2,
    max_tool_result_chars: int = 8192,
    tool_timeout_seconds: float = 5.0,
    grounded_generation: GroundedGenerationConfig | None = None,
    evidence_sufficiency: float | None = None,
    user_memory: str | None = None,
    retrieval_query: str | None = None,
) -> AnswerGenerationResult:
    from .safety import evidence_from_context
    from .tool_evidence import collect_tool_evidence
    from src.internal.observability.tracer import get_tracer

    tracer = get_tracer()
    tool_statuses: list[tuple[str, str]] = []
    with tracer.span("rag.query", top_k=top_k):
        with tracer.span("rag.retrieve"):
            context = await retrieve_context(
                retrieval_query or question,
                search_url=search_url,
                top_k=top_k,
                filters=filters,
            )
        evidence = evidence_from_context(context)
        if tool_registry is not None and tool_selector is not None:
            tool_evidence = await collect_tool_evidence(
                question,
                tool_registry,
                tool_selector,
                max_calls=max_tool_calls,
                timeout_seconds=tool_timeout_seconds,
                max_result_chars=max_tool_result_chars,
                status_callback=lambda name, status: tool_statuses.append(
                    (name, status)
                ),
            )
            evidence.extend(tool_evidence)
        with tracer.span(
            "rag.generate",
            num_docs=len(context.documents),
            has_llm=llm is not None,
        ):
            # `generate_answer` is synchronous and calls `llm.complete`, a
            # blocking `requests` call. Awaiting it inline from this async
            # handler froze the event loop for the full completion -- seconds,
            # not milliseconds -- so every other concurrent session stalled
            # behind one user's answer.
            result = await asyncio.to_thread(
                generate_answer,
                AnswerGenerationRequest(
                    question=question,
                    context=context,
                    chat_history=chat_history or [],
                    evidence=evidence,
                    grounded_generation=grounded_generation
                    or GroundedGenerationConfig(),
                    evidence_sufficiency=evidence_sufficiency,
                    user_memory=user_memory,
                ),
                llm=llm,
            )
        evidence_types = sorted({item.provenance for item in evidence})
        verification_status = (
            result.verification_status.value
            if result.verification_status
            else "unverified"
        )
        with tracer.span(
            "rag.summary",
            evidence_count=len(evidence),
            evidence_types=",".join(evidence_types),
            tool_names=",".join(name for name, _ in tool_statuses),
            tool_statuses=",".join(status for _, status in tool_statuses),
            retry_count=result.retry_count,
            verification_status=verification_status,
            confidence=result.confidence,
            abstained=result.abstained,
            structured_output_requested=result.structured_output_requested,
            structured_output_applied=result.structured_output_applied,
            structured_output_downgraded=result.structured_output_downgraded,
            structured_output_category=result.structured_output_category,
        ):
            pass
    return result


def synthesize_answer_from_context(question: str, context: SearchContextBundle) -> str:
    """Extractive fallback answer when no LLM is available.

    Scores every sentence in the retrieved documents by keyword overlap with
    the question, then assembles the top sentences into a grounded answer with
    inline citations.  This is intentionally conservative — it never fabricates
    information not present in the retrieved context.
    """
    if not context.documents:
        return f"I could not find retrieved context to answer: {question}"

    snippets = rank_evidence_snippets(question, context, max_snippets=3)
    if not snippets:
        return _canonical_abstention()

    parts = [f"{snippet.text} {snippet.citation}" for snippet in snippets]
    return " ".join(parts)


def _generate_extractive_answer(
    question: str,
    evidence: list[EvidenceSource],
    *,
    evidence_sufficiency: float | None,
    overlap_threshold: float,
) -> tuple[str, VerificationResult]:
    """Select and verify extractive claims from the normalized evidence bundle."""
    from .models import AnswerDraft
    from .safety import render_verified_answer, verify_answer_draft

    claims = _rank_normalized_evidence(question, evidence, max_snippets=3)
    draft = AnswerDraft(claims=claims, abstain=not claims)
    verification = verify_answer_draft(
        draft,
        evidence,
        overlap_threshold=overlap_threshold,
        evidence_sufficiency=evidence_sufficiency,
    )
    return render_verified_answer(verification), verification


def _rank_normalized_evidence(
    question: str,
    evidence: list[EvidenceSource],
    *,
    max_snippets: int,
) -> list[AnswerClaim]:
    """Return relevant verbatim claims while retaining their stable source IDs."""
    if max_snippets < 1:
        return []
    question_tokens = _tokenize(question)
    if not question_tokens:
        return []

    scored: list[tuple[float, int, int, str, str]] = []
    for evidence_index, source in enumerate(evidence):
        for sentence_index, sentence in enumerate(_split_sentences(source.text)):
            overlap = _overlap_score(question_tokens, _tokenize(sentence))
            if overlap <= 0:
                continue
            title_overlap = _overlap_score(question_tokens, _tokenize(source.title))
            score = (
                overlap * 10.0
                + title_overlap * 2.0
                + 0.2 / (evidence_index + 1)
                + 0.1 / (sentence_index + 1)
            )
            scored.append((score, evidence_index, sentence_index, sentence, source.id))

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected: list[AnswerClaim] = []
    selected_ids: set[str] = set()
    for _, _, _, sentence, evidence_id in scored:
        if evidence_id in selected_ids:
            continue
        selected.append(AnswerClaim(text=sentence, evidence_ids=[evidence_id]))
        selected_ids.add(evidence_id)
        if len(selected) >= max_snippets:
            break
    return selected


def rank_evidence_snippets(
    question: str,
    context: SearchContextBundle,
    *,
    max_snippets: int = 3,
) -> list[EvidenceSnippet]:
    """Return ranked, citation-ready evidence snippets for grounded synthesis."""
    if max_snippets < 1:
        return []

    question_tokens = _tokenize(question)
    if not question_tokens:
        doc = context.documents[0] if context.documents else None
        if doc is None:
            return []
        section = _section_for_document(doc, context.sections)
        return [
            EvidenceSnippet(
                citation=doc.citation,
                title=doc.title,
                text=_first_sentence(_contextualized_content(doc, context.sections)),
                score=float(doc.score),
                document=doc,
                section=section,
            )
        ]

    scored: list[EvidenceSnippet] = []
    for doc_index, doc in enumerate(context.documents):
        section = _section_for_document(doc, context.sections)
        content = _contextualized_content(doc, context.sections)
        for sentence_index, sentence in enumerate(_split_sentences(content)):
            score = _evidence_score(
                question_tokens,
                sentence,
                doc,
                doc_index=doc_index,
                sentence_index=sentence_index,
            )
            if score > 0:
                scored.append(
                    EvidenceSnippet(
                        citation=doc.citation,
                        title=doc.title,
                        text=sentence,
                        score=score,
                        document=doc,
                        section=section,
                    )
                )

    if not scored:
        return []

    scored.sort(key=lambda snippet: snippet.score, reverse=True)

    seen_citations: set[str] = set()
    selected: list[EvidenceSnippet] = []
    for snippet in scored:
        if snippet.citation not in seen_citations:
            selected.append(snippet)
            seen_citations.add(snippet.citation)
        if len(selected) >= max_snippets:
            break
    return selected


# ---------------------------------------------------------------------------
# Extractive helpers
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can need dare ought used "
    "to of in on at for by with from about into through during before "
    "after above below between among and or but nor so yet both either "
    "neither not only also just more most some any such other each every "
    "both few more most other some such no nor not only own same so than "
    "too very i me my we our you your he she it its they them their "
    "what which who whom this that these those i s t don doesn won couldn "
    "how when where why".split()
)


def _tokenize(text: str) -> set[str]:
    import re

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _overlap_score(query_tokens: set[str], sentence_tokens: set[str]) -> float:
    if not query_tokens or not sentence_tokens:
        return 0.0
    shared = query_tokens & sentence_tokens
    # Jaccard-style but biased toward query recall
    return len(shared) / len(query_tokens)


def _evidence_score(
    query_tokens: set[str],
    sentence: str,
    document: ContextDocument,
    *,
    doc_index: int,
    sentence_index: int,
) -> float:
    sentence_tokens = _tokenize(sentence)
    overlap = _overlap_score(query_tokens, sentence_tokens)
    if overlap <= 0:
        return 0.0
    title_overlap = _overlap_score(query_tokens, _tokenize(document.title))
    retrieval_score = max(float(document.score or 0.0), 0.0)
    rank_boost = 1.0 / (doc_index + 1)
    lead_boost = 1.0 / (sentence_index + 1)
    return (
        overlap * 10.0
        + title_overlap * 2.0
        + retrieval_score
        + rank_boost * 0.2
        + lead_boost * 0.1
    )


def _split_sentences(text: str) -> list[str]:
    import re

    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 20]


def _first_sentence(text: str) -> str:
    sentences = _split_sentences(text)
    return sentences[0] if sentences else text[:200].strip()


def _section_for_document(
    document: ContextDocument,
    sections: list[ContextSection],
) -> ContextSection | None:
    for section in sections:
        if any(candidate.id == document.id for candidate in section.documents):
            return section
    return None


def _contextualized_content(
    document: ContextDocument,
    sections: list[ContextSection],
) -> str:
    section = _section_for_document(document, sections)
    return section.combined_content if section else document.content
