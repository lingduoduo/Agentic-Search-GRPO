"""Serving-time intent recognition cascade and telemetry assembly."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from src.internal.configs import load_app_settings
from src.internal.servers.web import request_capture as _capture

from . import similarity
from .rules import _regex_route, _rule_based_route_or_none
from .types import (
    Clarification,
    ClarificationOption,
    RouteDecision,
    RouteStrategy,
)

if TYPE_CHECKING:
    from src.context.models import LLMClient
    from src.internal.configs import AppSettings

logger = logging.getLogger(__name__)

_CLARIFICATION = Clarification(
    question="I can take this a few different ways — which would you like?",
    options=(
        ClarificationOption("chat", "Explain or summarize it"),
        ClarificationOption("search", "Find the document or facts"),
        ClarificationOption("tool", "Take an action on it"),
    ),
)
_ROUTE_PROMPT = (
    "Classify how to best answer the user's request. Reply with exactly one "
    "label and nothing else:\n"
    "- chat: a descriptive or conversational question best answered from the "
    "knowledge base with synthesis, or a self-contained generative request "
    "(e.g. summaries, comparisons, how-tos, write a poem, translate this)\n"
    "- search: look up facts about a specific entity/term or current "
    "information — including a bare keyword or product/library name "
    "(e.g. 'FAISS', 'vector database benchmarks', find/look up X)\n"
    "- tool: take an action via a tool or API "
    "(e.g. send, create, schedule, call an API)\n\n"
    "Request: {user_query}\n"
    "Label:"
)
_LABEL_BY_VALUE = {strategy.value: strategy for strategy in RouteStrategy}


def classify_route(query: str, llm: "LLMClient") -> tuple[RouteStrategy | None, dict]:
    """Classify with an LLM while retaining only a safe label sentinel."""
    from src.context.models import ChatMessage

    prompt = _ROUTE_PROMPT.format(user_query=query)
    from src.internal.configs.timeouts import get_timeout_policies

    # A one-word label: a slow classifier must hand over to the rules router
    # quickly rather than hold the request for the provider's 30s default.
    response = llm.complete(
        [ChatMessage(role="user", content=prompt)],
        temperature=0.0,
        timeout_override=get_timeout_policies().llm.route_classifier_timeout_seconds,
    )
    content = (
        (response if isinstance(response, str) else response.content).strip().lower()
    )
    if not content:
        logger.warning("Route classification empty; defaulting to chat.")
        return None, {"raw_label": "empty"}

    labels = {value for value in _LABEL_BY_VALUE if re.search(rf"\b{value}\b", content)}
    if len(labels) == 1:
        captured_label = labels.pop()
        strategy = _LABEL_BY_VALUE[captured_label]
    else:
        captured_label = "unexpected"
        strategy = None
        logger.warning("Route classification response invalid; defaulting to chat.")
    return strategy, {"raw_label": captured_label}


def _record_intent(mechanism: str, strategy: RouteStrategy, detail: dict) -> None:
    _capture.record_stage(
        "intent",
        mechanism,
        {"mechanism": mechanism, "strategy": strategy.value, **detail},
    )


def recognize_intent(
    query: str,
    *,
    llm: "LLMClient | None",
    explicit_source: bool,
    settings: "AppSettings | None" = None,
) -> RouteDecision:
    """Recognize an ideal route and return clarification plus diagnostics."""
    resolved_settings = settings or load_app_settings()
    metadata: dict = {}

    def decided(mechanism: str, strategy: RouteStrategy, detail: dict) -> RouteDecision:
        _record_intent(mechanism, strategy, detail)
        metadata["route_mechanism"] = mechanism
        return RouteDecision(strategy, metadata=dict(metadata))

    def guessed(strategy: RouteStrategy, detail: dict) -> RouteDecision:
        if not resolved_settings.route_clarification:
            return decided("heuristic_default", strategy, detail)
        _record_intent("clarify", strategy, detail)
        metadata["route_mechanism"] = "clarify"
        return RouteDecision(strategy, _CLARIFICATION, dict(metadata))

    if explicit_source:
        return decided("explicit_source", RouteStrategy.SEARCH, {})
    regex_choice = _regex_route(query)
    if regex_choice is not None:
        return decided("rules", regex_choice, {})

    fallback_detail: dict = {}
    model_choice = similarity.predict_route(query, settings=resolved_settings)
    if model_choice is not None:
        fallback_reason = model_choice.abstain_reason
        abstained = fallback_reason is not None
        model_detail = {
            "predicted_intent": model_choice.strategy.value,
            "confidence": model_choice.confidence,
            "margin": model_choice.margin,
            "abstained": abstained,
            "fallback_reason": fallback_reason,
            "latency_ms": model_choice.latency_ms,
            "modules": list(model_choice.modules),
            "composite": model_choice.composite,
        }
        _capture.record_stage("intent_model", "evaluation", model_detail)
        metadata.update(
            route_predicted_intent=model_choice.strategy.value,
            route_confidence=model_choice.confidence,
            route_abstained=abstained,
            route_model_latency_ms=model_choice.latency_ms,
            route_modules=list(model_choice.modules),
            route_composite=model_choice.composite,
        )
        if resolved_settings.intent_shadow_mode:
            metadata.update(
                route_shadow_intent=model_choice.strategy.value,
                route_shadow_abstained=abstained,
                route_shadow_fallback_reason=fallback_reason,
            )
            fallback_detail = {"fallback_reason": "shadow_mode"}
        elif not abstained:
            return decided("model", model_choice.strategy, model_detail)
        else:
            fallback_detail = {"fallback_reason": fallback_reason}

    if fallback_detail:
        metadata["route_fallback_reason"] = fallback_detail["fallback_reason"]
    if llm is not None:
        try:
            strategy, detail = classify_route(query, llm)
            merged = {**detail, **fallback_detail}
            if strategy is None:
                return guessed(RouteStrategy.CHAT, merged)
            return decided("classifier", strategy, merged)
        except Exception:  # noqa: BLE001
            logger.warning("Route classifier failed, using rule-based.")
    heuristic = _rule_based_route_or_none(query)
    if heuristic is not None:
        return decided("heuristic_default", heuristic, fallback_detail)
    return guessed(RouteStrategy.CHAT, fallback_detail)
