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

# First-person statements only. Bare adverbs (always/never/remember) and
# "i want/need/am" fire on ordinary questions, so they are not triggers.
_SIGNAL = re.compile(
    r"\b(i prefer|i like|i love|i hate|i dislike|i avoid|i always|i never|"
    r"i work|i live|i('m| am) (a|an) |i('m| am) allergic|allergic to|"
    r"i('m| am) (vegetarian|vegan)|i have an? allergy|my \w+ is|"
    r"we decided|i decided|we chose|we('re| are) using|remember that|"
    r"from now on|don't like|do not like)\b",
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
    """Jaccard must match; a cosine, when given, can only veto (never match)."""
    a, b = _content_tokens(text_a), _content_tokens(text_b)
    if not a or not b or len(a & b) / len(a | b) < NEAR_DUP_MIN_SCORE:
        return False
    return cosine is None or cosine >= NEAR_DUP_MIN_COSINE


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

        vecs = np.asarray(
            encoder([f"passage: {t}" for t in (text, *others)]), dtype=float
        )
        norms = np.linalg.norm(vecs, axis=1)
        norms[norms == 0] = 1.0
        vecs = vecs / norms[:, None]
        return [float(v) for v in vecs[1:] @ vecs[0]]
    except Exception:  # noqa: BLE001 - fall back to lexical only
        logger.warning("capture cosine failed; using Jaccard only", exc_info=True)
        return [None] * len(others)


def _norm(text: str) -> str:
    return " ".join(text.lower().split()).rstrip(".!? ")


def reconcile(store, user_id: str, items, *, encoder=None) -> dict:
    """Add or refresh memories; never deletes and never rewrites a constraint.

    ``stored`` lists the texts as persisted (redacted), for the trajectory.
    """
    counts: dict = {"added": 0, "updated": 0, "stored": []}
    records = list(store.get_user_memory_records(user_id))
    for text, category in items:
        if category == "constraints":
            if any(
                r.category == "constraints" and _norm(r.memory_text) == _norm(text)
                for r in records
            ):
                continue
            rec = store.add_user_memory(user_id, text, category=category)
            if rec is not None:
                records.append(rec)
                counts["added"] += 1
                counts["stored"].append(rec.memory_text)
            continue
        pool = [r for r in records if r.category != "constraints"]
        cosines = (
            _cosines(encoder, text, [r.memory_text for r in pool])
            if encoder is not None and pool
            else [None] * len(pool)
        )
        match = next(
            (
                r
                for r, c in zip(pool, cosines)
                if is_near_duplicate(text, r.memory_text, cosine=c)
            ),
            None,
        )
        if match is None:
            rec = store.add_user_memory(user_id, text, category=category)
            if rec is not None:
                records.append(rec)
                counts["added"] += 1
                counts["stored"].append(rec.memory_text)
        elif _norm(match.memory_text) != _norm(text):
            rec = store.update_user_memory(user_id, match.id, text, category=category)
            if rec is not None:
                records[records.index(match)] = rec
                counts["updated"] += 1
                counts["stored"].append(rec.memory_text)
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
            # Stored (redacted) texts, never the raw LLM strings.
            "memories": counts["stored"],
            "counts": {k: v for k, v in counts.items() if k != "stored"},
        },
    )
    return {k: v for k, v in counts.items() if k != "stored"}
