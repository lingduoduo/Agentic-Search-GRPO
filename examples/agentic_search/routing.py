"""Request routing for ``examples.run_agentic_search``.

Two opt-in behaviours, both off unless a flag turns them on:

  --intent_index    route the question against a canonical-example index and,
                    when the index does not abstain, adjust the search settings
  --model_routing   pick the generation model for this one request

Neither is part of the three modes the CLI demonstrates, which is why they live
beside it rather than in it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IntentPrediction:
    """One accepted route for the request, with its cosine similarity.

    ``confidence`` is the configured top-k mean cosine to the winning route's canonical
    examples — not a softmax probability. Thresholds compared against it must
    be tuned on that scale.
    """

    intent: str
    confidence: float


def _load_intent_prediction(index_dir: str, question: str) -> IntentPrediction | None:
    """Route *question* against a canonical-example index, or return None.

    None means the winning route did not clear the configured margin over
    the runner-up. This is not a signal worth switching a generation model on.
    """
    from src.internal.configs import load_app_settings

    # Imported from the defining submodule, not the package: the package
    # __init__ re-exports bind their own reference at import time, so patching
    # `intent.model.encode_texts` would not reach a package-level alias.
    from src.model.pre_training.intents.model import (
        DEFAULT_ENCODER,
        INDEX_FILENAME,
        IntentIndex,
        encode_texts,
    )

    settings = load_app_settings()
    index = IntentIndex.load(Path(index_dir) / INDEX_FILENAME)
    if index.encoder != DEFAULT_ENCODER:
        # Both all-MiniLM-L6-v2 and e5-small-v2 are 384-dimensional, so a
        # mismatched encoder has no other symptom: no shape error, no
        # exception, just a confident, meaningless number driving
        # resolve_search_settings silently. Match the same guard
        # run_index_evaluation and intent.similarity.load_intent_index apply.
        raise ValueError(
            f"--intent_index at {index_dir} was built with encoder "
            f"{index.encoder!r}, but this CLI encodes queries with "
            f"{DEFAULT_ENCODER!r}. Rebuild the index with the current "
            "encoder (`python -m src.model.pre_training.intents.cli build`) before "
            "using --intent_index."
        )
    decision = index.decide(
        encode_texts([question])[0],
        min_margin=settings.intent_min_route_margin,
        min_module_score=settings.intent_min_module_score,
        top_k=settings.intent_top_k,
    )
    if decision.abstained:
        return None
    return IntentPrediction(intent=decision.route, confidence=decision.confidence)


def resolve_search_settings(
    prediction: IntentPrediction,
    *,
    topk: int,
    max_search_limit: int,
    require_evidence: bool,
    allow_internal_knowledge: bool,
) -> tuple[int, int, bool, bool, dict[str, Any]]:
    """Apply the per-intent search policy for one CLI request.

    There is no confidence comparison here, and there is no longer one
    anywhere: ``IntentIndex.decide`` returns an abstention on a low *margin*
    only, and ``_load_intent_prediction`` turns that into ``None``. So any
    prediction reaching this function was served rather than abstained.
    """

    meta: dict[str, Any] = {
        "intent_routing_used": True,
        "predicted_intent": prediction.intent,
        "intent_confidence": prediction.confidence,
        "intent_policy_applied": True,
    }
    policy: dict[str, tuple[int, int, bool, bool]] = {
        "chat": (topk, max_search_limit, require_evidence, allow_internal_knowledge),
        "search": (max(topk, 8), max(max_search_limit, 3), True, False),
        "tool": (topk, max_search_limit, require_evidence, allow_internal_knowledge),
    }
    t, s, r, a = policy.get(
        prediction.intent,
        (topk, max_search_limit, require_evidence, allow_internal_knowledge),
    )
    return t, s, r, a, meta


@dataclass(frozen=True)
class ModelRouteDecision:
    """Selected generation model for one CLI request."""

    model: str
    route: str
    reason: str
    metadata: dict[str, Any]


def _resolve_model_route(
    args: argparse.Namespace,
    intent_prediction: IntentPrediction | None = None,
) -> ModelRouteDecision:
    """Choose a request-level generation model without touching agent loops.

    The selected model is still passed through the existing tokenizer and
    server-manager path.  This is deliberately request-level routing; per-turn
    model routing would require a multi-backend server manager.
    """

    metadata: dict[str, Any] = {
        "model_routing": args.model_routing,
        "base_model": args.model,
    }
    if args.model_routing == "off":
        return ModelRouteDecision(
            model=args.model,
            route="base",
            reason="model routing disabled",
            metadata=metadata,
        )

    if intent_prediction is None:
        metadata["model_routing_applied"] = False
        return ModelRouteDecision(
            model=args.model,
            route="base",
            reason="no intent prediction available",
            metadata=metadata,
        )

    metadata.update(
        {
            "predicted_intent": intent_prediction.intent,
            "intent_confidence": intent_prediction.confidence,
        }
    )
    if intent_prediction.confidence < args.model_routing_min_confidence:
        metadata["model_routing_applied"] = False
        return ModelRouteDecision(
            model=args.model,
            route="base",
            reason="intent confidence below routing threshold",
            metadata=metadata,
        )

    route_by_intent = {
        "search": "fast",
        "chat": "balanced",
        "tool": "reasoning",
    }
    route = route_by_intent.get(intent_prediction.intent, "base")
    model_by_route = {
        "base": args.model,
        "fast": args.fast_model or args.model,
        "balanced": args.balanced_model or args.model,
        "reasoning": args.reasoning_model or args.balanced_model or args.model,
    }
    model = model_by_route[route]
    metadata.update(
        {
            "model_routing_applied": model != args.model,
            "selected_route": route,
            "selected_model": model,
        }
    )
    return ModelRouteDecision(
        model=model,
        route=route,
        reason=f"intent={intent_prediction.intent}",
        metadata=metadata,
    )
