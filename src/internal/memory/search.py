"""Memory grep: BM25 lexical leg + cached dense (e5) leg, fused with RRF."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Callable

from src.internal.db.models import UserMemoryRecord
from src.internal.retrieval.fusion import rrf_rank

logger = logging.getLogger(__name__)

Encoder = Callable[[list[str]], Any]  # list[str] -> np.ndarray

_QUERY_STOPWORDS = frozenset(
    "a an and are as at be but by can do for from has have how i if in is it "
    "its me my of on or our so that the their them they this to was we what "
    "when where which who will with would you your".split()
)


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^0-9a-z]+", text.lower()) if t]


def bm25_scores(
    query_tokens: list[str],
    docs_tokens: list[list[str]],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[float]:
    """Okapi BM25 of each document against the query; 0.0 where no term matches."""
    n = len(docs_tokens)
    if n == 0:
        return []
    avgdl = sum(len(d) for d in docs_tokens) / n or 1.0
    counts = [Counter(d) for d in docs_tokens]
    scores = [0.0] * n
    for term in set(query_tokens):
        df = sum(1 for c in counts if term in c)
        if not df:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, c in enumerate(counts):
            tf = c.get(term, 0)
            if tf:
                norm = tf + k1 * (1 - b + b * len(docs_tokens[i]) / avgdl)
                scores[i] += idf * tf * (k1 + 1) / norm
    return scores


def encoder_identity(encoder) -> str:
    model_id = getattr(encoder, "model_id", None)
    if model_id:
        return str(model_id)
    return f"{getattr(encoder, '__module__', '')}.{getattr(encoder, '__qualname__', type(encoder).__qualname__)}"


def ensure_embeddings(store, user_id: str, records, encoder) -> dict[str, Any]:
    """id -> vector for *records*, embedding only missing or stale ones in one batch."""
    import numpy as np

    identity = encoder_identity(encoder)
    cached = store.get_memory_embeddings(user_id)
    vectors: dict[str, Any] = {}
    todo: list[UserMemoryRecord] = []
    for r in records:
        hit = cached.get(r.id)
        if hit is not None and hit[1] == identity:
            vectors[r.id] = np.frombuffer(hit[0], dtype=np.float32)
        else:
            todo.append(r)
    if todo:
        matrix = np.asarray(encoder([f"passage: {r.memory_text}" for r in todo]))
        rows = []
        for r, v in zip(todo, matrix):
            vec = np.asarray(v, dtype=np.float32)
            vectors[r.id] = vec
            rows.append((r.id, vec.tobytes(), identity))
        store.set_memory_embeddings(user_id, rows)
    return vectors


def _dense_leg(store, user_id, records, query, encoder, limit):
    import numpy as np

    vectors = ensure_embeddings(store, user_id, records, encoder)
    qvec = np.asarray(encoder([f"query: {query}"]), dtype=np.float32)[0]
    scored = [(r, float(np.dot(qvec, vectors[r.id]))) for r in records]
    scored = [x for x in scored if x[1] > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored[:limit]]


def hybrid_search(
    store,
    user_id: str,
    query: str,
    *,
    max_results: int = 5,
    encoder: Encoder | None = None,
    category: str | None = None,
) -> list[tuple[UserMemoryRecord, float]]:
    records = store.get_user_memory_records(user_id, category=category)
    if not records or not query.strip():
        return []
    all_tokens = _tokenize(query)
    q_tokens = [t for t in all_tokens if t not in _QUERY_STOPWORDS] or all_tokens
    bm25 = bm25_scores(q_tokens, [_tokenize(r.memory_text) for r in records])
    lexical = sorted(
        ((r, s) for r, s in zip(records, bm25) if s > 0),
        key=lambda x: x[1],
        reverse=True,
    )
    dense: list[UserMemoryRecord] = []
    if encoder is not None:
        try:
            dense = _dense_leg(
                store, user_id, records, query, encoder, max(20, 4 * max_results)
            )
        except Exception as exc:  # noqa: BLE001 — memory search is best-effort
            logger.warning("memory dense search failed, lexical only: %s", exc)
    if not dense:
        return lexical[:max_results]
    by_id = {r.id: r for r in records}
    fused = rrf_rank([[r for r, _ in lexical], dense], key_fn=lambda r: r.id)
    return [(by_id[rid], score) for rid, score in fused[:max_results]]
