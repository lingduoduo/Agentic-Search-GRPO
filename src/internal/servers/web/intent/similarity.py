"""Lazy canonical-index adapter for serving-time intent recognition."""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from time import perf_counter

from src.internal.configs import AppSettings, load_app_settings

from .types import IntentModelDecision, RouteStrategy

logger = logging.getLogger(__name__)

_INTENT_INDEXES: dict[Path, object | None] = {}
# recognize_intent runs in worker threads (#657): one cold load per path, and a
# failure is decided once rather than by whichever racing thread wrote last.
_INTENT_INDEXES_LOCK = threading.Lock()
_ROUTE_VALUES = {strategy.value for strategy in RouteStrategy}


def encode_texts(texts: list[str]):
    """Load the optional encoder implementation only when prediction needs it."""
    from src.model.pre_training.intents.model import encode_texts as encode

    return encode(texts)


def load_intent_index(settings: AppSettings | None = None) -> object | None:
    """Load and cache the configured canonical index by resolved path."""
    resolved = settings or load_app_settings()
    configured = resolved.intent_index_path
    if configured is None:
        return None
    directory = configured.resolve()
    if directory in _INTENT_INDEXES:
        return _INTENT_INDEXES[directory]
    with _INTENT_INDEXES_LOCK:
        if directory in _INTENT_INDEXES:
            return _INTENT_INDEXES[directory]
        try:
            from src.model.pre_training.intents.model import (
                DEFAULT_ENCODER,
                INDEX_FILENAME,
                IntentIndex,
            )

            index = IntentIndex.load(directory / INDEX_FILENAME)
            if index.encoder != DEFAULT_ENCODER:
                raise ValueError(
                    f"intent-index built with encoder {index.encoder!r}, "
                    f"serving uses {DEFAULT_ENCODER!r}"
                )
        except Exception:
            logger.exception("intent-index: load failed — similarity routing disabled")
            _INTENT_INDEXES[directory] = None
        else:
            low_support = index.low_support_modules()
            if low_support:
                logger.warning(
                    "intent-index: modules below support, not emitted: %s",
                    ", ".join(low_support),
                )
            _INTENT_INDEXES[directory] = index
        return _INTENT_INDEXES[directory]


def predict_route(
    query: str, *, settings: AppSettings | None = None
) -> IntentModelDecision | None:
    """Return a typed similarity decision, including any margin abstention."""
    resolved = settings or load_app_settings()
    index = load_intent_index(resolved)
    if index is None:
        return None
    start = perf_counter()
    try:
        vector = encode_texts([query])[0]
        decision = index.decide(
            vector,
            min_margin=resolved.intent_min_route_margin,
            min_module_score=resolved.intent_min_module_score,
            top_k=resolved.intent_top_k,
        )
    except Exception:
        logger.exception("intent-index: predict failed — deferring")
        return None
    latency_ms = (perf_counter() - start) * 1_000

    if decision.route not in _ROUTE_VALUES:
        logger.warning("intent-index: unsupported route %r — deferring", decision.route)
        return None
    confidence = float(decision.confidence)
    if not math.isfinite(confidence) or not -1.0 <= confidence <= 1.0:
        logger.warning(
            "intent-index: non-finite or out-of-range cosine confidence — deferring"
        )
        return None

    return IntentModelDecision(
        strategy=RouteStrategy(decision.route),
        confidence=confidence,
        latency_ms=latency_ms,
        modules=decision.modules,
        composite=decision.composite,
        margin=float(decision.margin),
        abstain_reason=(
            "margin_below_threshold"
            if decision.abstain_reason == "margin_below_threshold"
            else None
        ),
    )
