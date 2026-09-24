"""Lazy e5 embedder for the SEARCH direct-gate semantic tier.

Loads ``intfloat/e5-base-v2`` on first use and caches it as a module-level
singleton. Returns ``None`` when the semantic tier is disabled via
``AGENTIC_SEARCH_SEARCH_DIRECT_SEMANTIC=0`` or when the model cannot be loaded,
so callers degrade gracefully instead of crashing the hot path.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def search_direct_cos_min() -> float:
    return float(os.environ.get("AGENTIC_SEARCH_SEARCH_DIRECT_COS_MIN", "0.8"))


# Chosen on the multi-turn eval's dev half (2026-09-24, τ grid 0.70–0.95). No
# lower τ beat the semantic rule being off: e5 scores several topic switches as
# high as the one follow-up that reaches the rule, so ties resolve to the top.
DEFAULT_FOLLOW_UP_COS_MIN = 0.95


def follow_up_cos_min() -> float:
    return float(
        os.environ.get(
            "AGENTIC_SEARCH_FOLLOW_UP_COS_MIN", str(DEFAULT_FOLLOW_UP_COS_MIN)
        )
    )


def make_cosine_fn(embedder):
    """Return (query, passage) -> cosine|None using e5 prefixes; None if no model."""
    if embedder is None:
        return lambda _query, _passage: None

    def _cosine(query: str, passage: str):
        try:
            vecs = embedder([f"query: {query}", f"passage: {passage}"])
        except Exception:
            return None
        qv = np.asarray(vecs[0], dtype=np.float32)
        pv = np.asarray(vecs[1], dtype=np.float32)
        qn = float(np.linalg.norm(qv))
        pn = float(np.linalg.norm(pv))
        if qn == 0.0 or pn == 0.0:
            return None
        return float(np.dot(qv, pv) / (qn * pn))

    return _cosine


_GATE_EMBEDDER: object | None = None  # None=unset, False=failed, callable=loaded


def gate_embedder():
    """Lazy singleton e5 embedder for the semantic tier; None when unavailable."""
    global _GATE_EMBEDDER

    if os.environ.get("AGENTIC_SEARCH_SEARCH_DIRECT_SEMANTIC", "1") == "0":
        return None
    if _GATE_EMBEDDER is not None:
        return _GATE_EMBEDDER or None
    try:
        from sentence_transformers import SentenceTransformer

        name = os.environ.get(
            "AGENTIC_SEARCH_SEARCH_DIRECT_MODEL", "intfloat/e5-base-v2"
        )
        model = SentenceTransformer(name)

        def _fn(texts):
            return model.encode(texts, normalize_embeddings=True)

        _GATE_EMBEDDER = _fn
    except Exception:
        logger.exception(
            "direct-gate: embedding model unavailable — semantic tier disabled"
        )
        _GATE_EMBEDDER = False
        return None
    return _GATE_EMBEDDER
