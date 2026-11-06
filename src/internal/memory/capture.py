"""Per-turn automatic memory capture.

A cheap lexical gate decides whether a turn is worth an LLM call; the LLM then
extracts durable facts, and ``reconcile`` adds new ones or refreshes
near-duplicates. Capture never deletes. Every write goes through store methods,
which redact.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from src.internal.memory.categories import CATEGORIES, normalize_category
from src.internal.memory.search import _QUERY_STOPWORDS, _tokenize

logger = logging.getLogger(__name__)

# Token Jaccard on content tokens, or encoder cosine, at/above which two
# memories are the same fact.
NEAR_DUP_MIN_SCORE = 0.6
NEAR_DUP_MIN_COSINE = 0.9

_SIGNAL = re.compile(
    r"\b(i'm|i am|i've|i have|i prefer|i like|i love|i hate|i dislike|i avoid|"
    r"i need|i want|i always|i never|i work|i live|my \w+ is|allergic|allergy|"
    r"vegetarian|vegan|we decided|i decided|we chose|we're using|we use|"
    r"remember|from now on|always|never|don't like|do not like|prefer)\b",
    re.IGNORECASE,
)

_EXTRACT_SYSTEM = (
    "You extract durable long-term memories about the user from one conversation "
    "turn. Only record lasting facts, preferences, constraints, or decisions, not "
    "one-off questions. Keep each memory a single contextual sentence. Do NOT "
    "store secrets (passwords, PINs, full SSNs, or full card/account numbers). "
    "Categories: {categories}. Reply with JSON only: "
    '{{"memories": [{{"text": "...", "category": "..."}}]}}, or '
    '{{"memories": []}} when there is nothing durable.'
)


def has_durable_signal(user_message: str) -> bool:
    return bool(_SIGNAL.search(user_message or ""))


def _content_tokens(text: str) -> set[str]:
    return {t for t in _tokenize(text) if t not in _QUERY_STOPWORDS}


def is_near_duplicate(text_a: str, text_b: str, *, cosine: float | None = None) -> bool:
    if cosine is not None and cosine >= NEAR_DUP_MIN_COSINE:
        return True
    a, b = _content_tokens(text_a), _content_tokens(text_b)
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= NEAR_DUP_MIN_SCORE


def extract_memories(llm, user_message: str, answer: str) -> list[tuple[str, str]]:
    prompt = (
        _EXTRACT_SYSTEM.format(categories=", ".join(CATEGORIES))
        + f"\n\nUSER: {user_message}\nASSISTANT: {answer}\n"
    )
    raw = llm.complete(prompt, max_tokens=400, temperature=0.0)
    if not isinstance(raw, str):
        return []
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        items = json.loads(raw[start : end + 1]).get("memories")
    except (ValueError, AttributeError):
        return []
    out: list[tuple[str, str]] = []
    for item in items if isinstance(items, list) else []:
        text = item.get("text") if isinstance(item, dict) else None
        if isinstance(text, str) and text.strip():
            out.append((text.strip(), normalize_category(item.get("category"))))
    return out


def _cosines(encoder, text: str, others: list[str]) -> list[float | None]:
    """Cosine of *text* against each of *others*; None for all on any failure."""
    try:
        import numpy as np

        vecs = np.asarray(encoder([text, *others]), dtype=float)
        norms = np.linalg.norm(vecs, axis=1)
        norms[norms == 0] = 1.0
        vecs = vecs / norms[:, None]
        return [float(v) for v in vecs[1:] @ vecs[0]]
    except Exception:  # noqa: BLE001 - fall back to lexical only
        logger.warning("capture cosine failed; using Jaccard only", exc_info=True)
        return [None] * len(others)


def reconcile(store, user_id: str, items, *, encoder=None) -> dict[str, int]:
    counts = {"added": 0, "updated": 0}
    records = list(store.get_user_memory_records(user_id))
    for text, category in items:
        cosines = (
            _cosines(encoder, text, [r.memory_text for r in records])
            if encoder is not None and records
            else [None] * len(records)
        )
        match = next(
            (
                r
                for r, c in zip(records, cosines)
                if is_near_duplicate(text, r.memory_text, cosine=c)
            ),
            None,
        )
        if match is None:
            records.append(store.add_user_memory(user_id, text, category=category))
            counts["added"] += 1
        elif match.memory_text.strip() != text:
            updated = store.update_user_memory(
                user_id, match.id, text, category=category
            )
            records[records.index(match)] = updated or match
            counts["updated"] += 1
    return counts


async def capture_turn(
    store, user_id, session_id, llm, user_message, answer, *, encoder=None
) -> dict | None:
    from src.internal.memory.service import _readable

    if not user_id or llm is None:
        return None
    session = await asyncio.to_thread(store.get_chat_session, session_id)
    if not _readable(session, user_id):
        return None
    if not has_durable_signal(user_message):
        return None
    items = await asyncio.to_thread(extract_memories, llm, user_message, answer)
    if not items:
        return None
    counts = await asyncio.to_thread(reconcile, store, user_id, items, encoder=encoder)
    await asyncio.to_thread(
        store.add_memory_trajectory,
        user_id,
        session_id=session_id,
        model=getattr(getattr(llm, "config", None), "model_name", ""),
        trajectory={
            "source": "auto_turn",
            "memories": [{"text": t, "category": c} for t, c in items],
            "counts": counts,
        },
    )
    return counts
