"""Memory consolidation and the per-category index summaries.

Order: exact dedup -> near-duplicate merge -> contradictions -> reclassify
``general`` -> summaries. Constraints are protected (``is_protected``: by
label or by text, so an LLM mislabel cannot unprotect one): only exact
duplicates of one collapse; nothing else here deletes, merges or moves one.
LLM output is untrusted: only ids from the batch that was sent are honored,
and unparseable replies change nothing.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from typing import Any

from src.internal.db.models import UserMemoryRecord
from src.internal.memory.capture import is_near_duplicate
from src.internal.memory.categories import (
    CATEGORIES,
    ROOT_NODE,
    is_protected,
    normalize_category,
)
from src.internal.memory.search import _QUERY_STOPWORDS, _tokenize, ensure_embeddings
from src.internal.memory.service import _attribute, _count_line

logger = logging.getLogger(__name__)

_CONSOLIDATE_EVERY_DEFAULT = 10


def _parse_consolidate_every(raw: str) -> int:
    """Clamp to >= 1; a non-integer logs a warning and uses the default."""
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "AGENTIC_SEARCH_MEMORY_CONSOLIDATE_EVERY=%r is not an integer; using %d",
            raw,
            _CONSOLIDATE_EVERY_DEFAULT,
        )
        return _CONSOLIDATE_EVERY_DEFAULT


CONSOLIDATE_EVERY = _parse_consolidate_every(
    os.getenv(
        "AGENTIC_SEARCH_MEMORY_CONSOLIDATE_EVERY", str(_CONSOLIDATE_EVERY_DEFAULT)
    )
)
# Index node whose memory_count is the active count at the last consolidation.
CONSOLIDATED_NODE = "__consolidated__"
LLM_BATCH_MAX = 30
_SUMMARY_MAX_CHARS = 200
_PROTECTED = "constraints"


def _protected(r: UserMemoryRecord) -> bool:
    return is_protected(r.memory_text, normalize_category(r.category))


_CONTRADICTION_PROMPT = (
    "These are long-term memories about one user, category '{category}', oldest "
    "first. List the ids of memories that a NEWER memory in this list contradicts "
    "or makes obsolete. Do not list a memory that is merely related. Reply with "
    'JSON only: {{"superseded": ["<id>", ...]}}.\n\n{lines}\n'
)
_MERGE_PROMPT = (
    "These memories about one user say the same thing. Rewrite them as ONE "
    "single contextual sentence that keeps every detail. Reply with JSON only: "
    '{{"text": "..."}}.\n\n{lines}\n'
)
_RECLASSIFY_PROMPT = (
    "Assign each memory about a user to one category from: {categories}. Use "
    "general when none fits. Reply with JSON only: "
    '{{"categories": {{"<id>": "<category>", ...}}}}.\n\n{lines}\n'
)
_SUMMARY_PROMPT = (
    "Summarize these memories about a user (category '{category}') in one short "
    "sentence of at most 25 words. Reply with the sentence only.\n\n{lines}\n"
)


def _newest_first(records: list[UserMemoryRecord]) -> list[UserMemoryRecord]:
    return sorted(
        records,
        key=lambda r: (r.updated_at or "", r.created_at or "", r.id),
        reverse=True,
    )


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _ask_json(llm, prompt: str) -> dict:
    """``llm.complete`` parsed leniently ({...} span); {} on any failure."""
    try:
        raw = llm.complete(prompt, max_tokens=400, temperature=0.0)
    except Exception:  # noqa: BLE001 - consolidation degrades, never raises
        logger.warning("memory consolidation LLM call failed", exc_info=True)
        return {}
    text = raw if isinstance(raw, str) else getattr(raw, "text", "")
    if not isinstance(text, str):
        return {}
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _lines(records: list[UserMemoryRecord]) -> str:
    return "\n".join(f"{r.id}: {r.memory_text}" for r in records)


def _vectors(store, user_id: str, records, encoder) -> dict[str, Any]:
    if encoder is None or not records:
        return {}
    try:
        return ensure_embeddings(store, user_id, records, encoder)
    except Exception:  # noqa: BLE001 - lexical only
        logger.warning("consolidation embeddings failed; Jaccard only", exc_info=True)
        return {}


def _cosine(vectors: dict[str, Any], a: str, b: str) -> float | None:
    va, vb = vectors.get(a), vectors.get(b)
    if va is None or vb is None:
        return None
    import numpy as np

    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb)) / denom if denom else None


def _dedup(store, user_id, records, report) -> list[UserMemoryRecord]:
    seen: set[tuple[str, str]] = set()
    keep: list[UserMemoryRecord] = []
    for r in _newest_first(records):
        key = (normalize_category(r.category), _norm(r.memory_text))
        if key in seen:
            store.delete_user_memory(user_id, r.id)
            report["duplicates_removed"] += 1
        else:
            seen.add(key)
            keep.append(r)
    return keep


def _merge(store, user_id, records, report, llm, encoder) -> list[UserMemoryRecord]:
    keep: list[UserMemoryRecord] = [r for r in records if _protected(r)]
    mergeable = [r for r in records if not _protected(r)]
    vectors = _vectors(store, user_id, mergeable, encoder)
    for category in CATEGORIES:
        if category == _PROTECTED:
            continue
        group = [r for r in mergeable if normalize_category(r.category) == category]
        clusters: list[list[UserMemoryRecord]] = []
        for r in _newest_first(group):  # each cluster's seed is its newest
            for cluster in clusters:
                seed = cluster[0]
                cos = _cosine(vectors, seed.id, r.id)
                if is_near_duplicate(seed.memory_text, r.memory_text, cosine=cos):
                    cluster.append(r)
                    break
            else:
                clusters.append([r])
        for cluster in clusters:
            newest = cluster[0]
            if len(cluster) > 1:
                if llm is not None:
                    text = _ask_json(llm, _MERGE_PROMPT.format(lines=_lines(cluster)))
                    merged = text.get("text")
                    if isinstance(merged, str) and merged.strip():
                        newest = (
                            store.update_user_memory(user_id, newest.id, merged.strip())
                            or newest
                        )
                for r in cluster[1:]:
                    store.delete_user_memory(user_id, r.id)
                report["merged"] += len(cluster) - 1
            keep.append(newest)
    return keep


def _supersede(store, user_id, loser, winner) -> None:
    store.update_user_memory(
        user_id, loser.id, loser.memory_text, {"superseded_by": winner.id}
    )
    store.delete_user_memory(user_id, loser.id)


def _contradictions(store, user_id, records, report, llm) -> list[UserMemoryRecord]:
    removed: set[str] = set()
    if llm is None:
        by_attr: dict[str, list[UserMemoryRecord]] = {}
        for r in records:
            attr = _attribute(r)
            if attr is not None and not _protected(r):
                by_attr.setdefault(attr, []).append(r)
        for attr, group in by_attr.items():
            ordered = _newest_first(group)
            if len(ordered) < 2:
                continue
            for r in ordered[1:]:
                _supersede(store, user_id, r, ordered[0])
                removed.add(r.id)
            report["conflicts_resolved"].append(
                {
                    "attribute": attr,
                    "kept": ordered[0].memory_text,
                    "superseded": [r.memory_text for r in ordered[1:]],
                }
            )
        return [r for r in records if r.id not in removed]

    for category in CATEGORIES:
        if category == _PROTECTED:
            continue
        batch = _newest_first(
            [
                r
                for r in records
                if normalize_category(r.category) == category and not _protected(r)
            ]
        )[:LLM_BATCH_MAX]
        if len(batch) < 2:
            continue
        newest = batch[0]
        chronological = list(reversed(batch))
        reply = _ask_json(
            llm,
            _CONTRADICTION_PROMPT.format(
                category=category, lines=_lines(chronological)
            ),
        )
        ids = reply.get("superseded")
        wanted = (
            {i for i in ids if isinstance(i, str)} if isinstance(ids, list) else set()
        )
        # Only this batch's ids, and never its newest: nothing newer supersedes it.
        losers = [r for r in batch[1:] if r.id in wanted]
        for r in losers:
            _supersede(store, user_id, r, newest)
            removed.add(r.id)
        if losers:
            report["conflicts_resolved"].append(
                {
                    "category": category,
                    "kept": newest.memory_text,
                    "superseded": [r.memory_text for r in losers],
                }
            )
    return [r for r in records if r.id not in removed]


def _reclassify(store, user_id, records, report, llm) -> None:
    if llm is None:
        return
    batch = _newest_first(
        [
            r
            for r in records
            if normalize_category(r.category) == "general" and not _protected(r)
        ]
    )[:LLM_BATCH_MAX]
    if not batch:
        return
    reply = _ask_json(
        llm,
        _RECLASSIFY_PROMPT.format(
            categories=", ".join(CATEGORIES), lines=_lines(batch)
        ),
    )
    assigned = reply.get("categories")
    if not isinstance(assigned, dict):
        return
    for r in batch:  # only unprotected general rows were sent
        new = normalize_category(assigned.get(r.id))
        if new != "general" and store.set_user_memory_category(user_id, r.id, new):
            report["reclassified"] += 1


def consolidate_memories(
    store,
    user_id: str,
    resolve_conflicts: bool = True,
    *,
    llm=None,
    encoder=None,
) -> dict[str, Any]:
    records = store.get_user_memory_records(user_id)
    report: dict[str, Any] = {
        "initial": len(records),
        "duplicates_removed": 0,
        "conflicts_resolved": [],
        "merged": 0,
        "reclassified": 0,
        "summaries_rebuilt": 0,
        "final": 0,
    }
    keep = _dedup(store, user_id, records, report)
    keep = _merge(store, user_id, keep, report, llm, encoder)
    if resolve_conflicts:
        keep = _contradictions(store, user_id, keep, report, llm)
    _reclassify(store, user_id, keep, report, llm)
    report["summaries_rebuilt"] = rebuild_index_summaries(store, user_id, llm=llm)
    report["final"] = len(store.get_user_memory_records(user_id))
    store.upsert_memory_index_node(
        user_id, CONSOLIDATED_NODE, summary="", memory_count=report["final"]
    )
    return report


def _deterministic_summary(group: list[UserMemoryRecord]) -> str:
    tokens = Counter(
        t
        for r in group
        for t in _tokenize(r.memory_text)
        if t not in _QUERY_STOPWORDS and len(t) >= 3
    )
    terms = ", ".join(t for t, _ in tokens.most_common(5))
    latest = " | ".join(
        r.memory_text if len(r.memory_text) <= 80 else r.memory_text[:79] + "…"
        for r in _newest_first(group)[:2]
    )
    return f"{len(group)} memories; key terms: {terms}; latest: {latest}"


def _summary(llm, category: str, group: list[UserMemoryRecord]) -> str:
    if llm is not None:
        try:
            raw = llm.complete(
                _SUMMARY_PROMPT.format(
                    category=category,
                    lines="\n".join(
                        f"- {r.memory_text}"
                        for r in _newest_first(group)[:LLM_BATCH_MAX]
                    ),
                ),
                max_tokens=80,
                temperature=0.0,
            )
            text = raw if isinstance(raw, str) else getattr(raw, "text", "")
            text = " ".join(text.split()) if isinstance(text, str) else ""
            if text:
                return text[:_SUMMARY_MAX_CHARS]
        except Exception:  # noqa: BLE001 - fall back to the deterministic summary
            logger.warning("memory summary LLM call failed", exc_info=True)
    return _deterministic_summary(group)


def rebuild_index_summaries(
    store, user_id: str, *, llm=None, only_dirty: bool = True
) -> int:
    """Re-summarize category nodes (dirty ones by default) and the root.

    The root summary is always the deterministic category-count line. Returns
    the number of nodes written.
    """
    nodes = {n.node: n for n in store.get_memory_index_nodes(user_id)}
    dirty = {name for name, n in nodes.items() if n.dirty}
    records = store.get_user_memory_records(user_id)
    written = 0
    for category in CATEGORIES:
        if only_dirty and category not in dirty:
            continue
        group = [r for r in records if normalize_category(r.category) == category]
        if not group and category not in nodes:
            continue
        store.upsert_memory_index_node(
            user_id,
            category,
            summary=_summary(llm, category, group) if group else "",
            memory_count=len(group),
        )
        written += 1
    if written or not only_dirty or ROOT_NODE in dirty:
        store.upsert_memory_index_node(
            user_id, ROOT_NODE, summary=_count_line(records), memory_count=len(records)
        )
        written += 1
    return written


def run_due(store, user_id: str, *, llm=None, encoder=None) -> str:
    """Consolidate when memory grew by CONSOLIDATE_EVERY, else refresh dirty summaries.

    A missing marker (first turn after upgrade) is seeded at the current active
    count without consolidating, so only later growth counts toward the trigger.
    Returns ``"consolidated"``, ``"summaries"`` or ``"idle"``.
    """
    active = len(store.get_user_memory_records(user_id))
    nodes = {n.node: n for n in store.get_memory_index_nodes(user_id)}
    marker = nodes.pop(CONSOLIDATED_NODE, None)
    if marker is None:
        store.upsert_memory_index_node(
            user_id, CONSOLIDATED_NODE, summary="", memory_count=active
        )
    elif active - marker.memory_count >= CONSOLIDATE_EVERY:
        consolidate_memories(store, user_id, llm=llm, encoder=encoder)
        return "consolidated"
    if any(n.dirty for n in nodes.values()):
        rebuild_index_summaries(store, user_id, llm=llm)
        return "summaries"
    return "idle"
