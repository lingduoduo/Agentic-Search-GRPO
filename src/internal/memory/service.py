"""Conversation-memory service: MCP-free logic over AgenticSearchStore."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Callable

from src.internal.db.models import (
    ChatMessageRecord,
    UserMemoryRecord,
    UserProfileEntryRecord,
)
from src.internal.memory.tools import build_memory_registry

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_USER_ID = "default_user"
MAX_CURATION_TURNS = 6
MEMORY_GATHER_CHAR_BUDGET = 12000
MEMORY_INJECTION_MAX = 20

# Above the cap, this many preamble slots go to relevance hits for the
# current query; the rest stay the most recent memories.
MEMORY_RELEVANT_SLOTS = MEMORY_INJECTION_MAX // 2

Encoder = Callable[[list[str]], Any]  # list[str] -> np.ndarray

_MEMORY_PREAMBLE_HEADER = (
    "\n\nWhat you know about this user (remembered from earlier conversations). "
    "Use these facts when relevant and apply them proactively — honor stated "
    "preferences, and warn about allergies or constraints:\n"
)


def memory_preamble(
    store,
    user_id: str,
    *,
    max_items: int = MEMORY_INJECTION_MAX,
    query: str | None = None,
    encoder: Encoder | None = None,
) -> str:
    """Format the user's active memories as a system-prompt preamble.

    Returns an instructional block (leading blank line included) listing up to
    *max_items* memories, or ``""`` when the user has none. The instructional
    wording is what drives proactive use (e.g. warn about a stored allergy).

    At or below the cap every memory is listed. Above it, and given a
    *query*, half the slots go to the memories most relevant to that query
    (``search_memories``) and the rest to the most recent ones, so an old but
    relevant fact still reaches the model while recent facts stay always
    present. Without a query the most recent *max_items* are listed.
    """
    memories = store.get_user_memories(user_id)
    if not memories:
        return ""
    if len(memories) <= max_items or not (query or "").strip():
        chosen = memories[-max_items:]
    else:
        chosen = _select_relevant(store, user_id, query, memories, max_items, encoder)
    return _MEMORY_PREAMBLE_HEADER + "\n".join(f"- {m}" for m in chosen)


def _select_relevant(
    store,
    user_id: str,
    query: str,
    memories: list[str],
    max_items: int,
    encoder: Encoder | None,
) -> list[str]:
    """Relevance hits first, then newest-first fill, returned in chronological order."""
    try:
        hits = search_memories(
            store,
            user_id,
            query,
            max_results=min(MEMORY_RELEVANT_SLOTS, max_items),
            encoder=encoder,
        )
    except Exception as exc:  # noqa: BLE001 — recall is best-effort
        logger.warning("relevant memory recall failed for %s: %s", user_id, exc)
        return memories[-max_items:]
    selected: list[str] = []
    for record, _score in hits:
        if record.memory_text not in selected:
            selected.append(record.memory_text)
    for text in reversed(memories):
        if len(selected) >= max_items:
            break
        if text not in selected:
            selected.append(text)
    keep = set(selected)
    ordered: list[str] = []
    for text in memories:
        if text in keep:
            ordered.append(text)
            keep.discard(text)
    return ordered


def maybe_build_encoder() -> Encoder | None:
    """Build the e5 memory encoder when AGENTIC_SEARCH_MEMORY_SEMANTIC is set,
    else return None so callers use the lexical search fallback."""
    if os.getenv("AGENTIC_SEARCH_MEMORY_SEMANTIC", "").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return None
    try:
        from src.internal.servers.retrieval.hybrid import build_e5_encoder

        return build_e5_encoder(
            device=os.getenv("AGENTIC_SEARCH_MEMORY_EMBED_DEVICE", "cpu")
        )
    except Exception as exc:  # noqa: BLE001 — fall back to lexical
        logger.warning("Memory e5 encoder unavailable, using lexical search: %s", exc)
        return None


def save_memory(store, user_id: str, text: str) -> str | None:
    record = store.add_user_memory(user_id, text)
    return record.id if record is not None else None


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^0-9a-z]+", text.lower()) if t]


def search_memories(
    store,
    user_id: str,
    query: str,
    max_results: int = 5,
    encoder: Encoder | None = None,
) -> list[tuple[UserMemoryRecord, float]]:
    records = store.get_user_memory_records(user_id)
    if not records or not query.strip():
        return []
    if encoder is not None:
        import numpy as np

        matrix = encoder([f"passage: {r.memory_text}" for r in records])
        qvec = encoder([f"query: {query}"])
        sims = (np.asarray(qvec) @ np.asarray(matrix).T)[0]
        ranked = sorted(zip(records, sims), key=lambda x: float(x[1]), reverse=True)
        return [(r, float(s)) for r, s in ranked[:max_results]]
    # Lexical fallback: token-overlap, normalized by query length.
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return []
    scored: list[tuple[UserMemoryRecord, float]] = []
    for r in records:
        overlap = len(q_tokens & set(_tokenize(r.memory_text)))
        if overlap:
            scored.append((r, overlap / len(q_tokens)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max_results]


def _attribute(record: UserMemoryRecord) -> str | None:
    tags = record.metadata.get("tags") if isinstance(record.metadata, dict) else None
    if isinstance(tags, list) and tags and isinstance(tags[0], str) and tags[0].strip():
        return tags[0].strip()
    return None


def consolidate_memories(
    store, user_id: str, resolve_conflicts: bool = True
) -> dict[str, Any]:
    records = store.get_user_memory_records(user_id)
    report: dict[str, Any] = {
        "initial": len(records),
        "duplicates_removed": 0,
        "conflicts_resolved": [],
        "final": 0,
    }

    def newest_first(rs: list[UserMemoryRecord]) -> list[UserMemoryRecord]:
        return sorted(rs, key=lambda r: (r.updated_at or "", r.id), reverse=True)

    # Exact-content dedup, keeping the newest.
    seen: set[str] = set()
    keep: list[UserMemoryRecord] = []
    remove: list[UserMemoryRecord] = []
    for r in newest_first(records):
        key = r.memory_text.strip().lower()
        if key in seen:
            remove.append(r)
            report["duplicates_removed"] += 1
            continue
        seen.add(key)
        keep.append(r)

    if resolve_conflicts:
        by_attr: dict[str, list[UserMemoryRecord]] = {}
        untagged: list[UserMemoryRecord] = []
        for r in keep:
            attr = _attribute(r)
            (untagged if attr is None else by_attr.setdefault(attr, [])).append(r)
        resolved = list(untagged)
        for attr, group in by_attr.items():
            ordered = newest_first(group)
            resolved.append(ordered[0])
            if len(ordered) > 1:
                remove.extend(ordered[1:])
                report["conflicts_resolved"].append(
                    {
                        "attribute": attr,
                        "kept": ordered[0].memory_text,
                        "superseded": [r.memory_text for r in ordered[1:]],
                    }
                )
        keep = resolved

    for r in remove:
        store.delete_user_memory(user_id, r.id)
    report["final"] = len(keep)
    return report


_CURATION_SYSTEM = (
    "You maintain a user's long-term memory. Given a recent conversation and the "
    "user's current memories, reconcile them by calling add_memory, update_memory, "
    "and delete_memory. Add new durable facts/preferences, update changed ones, and "
    "delete outdated or contradicted ones. Keep each memory a single contextual "
    "sentence. Do NOT store secrets (passwords, PINs, full SSNs, or full card/account "
    "numbers). When there is nothing left to change, reply with STOP and no tool calls."
)

_CURATION_USER = (
    "Recent conversation:\n{conversation}\n\n"
    "Current memories (id: text):\n{memories}\n\n"
    "Update the memory set now using the tools."
)


def _readable(session, user_id: str) -> bool:
    """Whether *user_id* may read *session*. Ownership is strict.

    An ownerless session used to be readable by anyone holding the id, on the
    "declares no ACL means public" rule this codebase applies to documents. A
    session is not a document: an ownerless one is still somebody's actual
    conversation, recorded before they signed in, so the analogy does not carry.

    The cost is real and deliberate: anonymous callers' sessions are stored with
    a NULL ``user_id``, so ``curate --session-id`` no longer works for them at
    all. ``curate`` says so rather than reporting a bare "empty".
    """
    return session is not None and session.user_id == user_id


def _gather_sources(store, user_id: str, session_id: str | None) -> str:
    # The by-id branch is the only way into a session the caller did not ask for
    # by identity -- `list_sessions_for_user` is scoped by `WHERE user_id = ?`.
    # Unchecked, naming someone else's session put their transcript in the
    # prompt and filed the result under the caller's memories.
    sessions = (
        [s for s in [store.get_chat_session(session_id)] if _readable(s, user_id)]
        if session_id
        else store.list_sessions_for_user(user_id)
    )
    lines: list[str] = []
    for sess in sessions:
        if sess is None:
            continue
        for msg in store.list_chat_messages(sess.id):
            lines.append(f"{msg.role.upper()}: {msg.content}")
    text = "\n".join(lines)
    return text[-MEMORY_GATHER_CHAR_BUDGET:]


def _format_memories(store, user_id: str) -> str:
    records = store.get_user_memory_records(user_id)
    if not records:
        return "(none)"
    return "\n".join(f"{r.id}: {r.memory_text}" for r in records)


def _stream_turn(
    llm, messages: list[dict], schemas: list[dict]
) -> tuple[str, list[dict]]:
    content_parts: list[str] = []
    acc: dict[int, dict[str, str]] = {}
    for chunk in llm.stream(messages, tools=schemas, max_tokens=1024):
        delta = chunk.choice.delta
        if delta.content:
            content_parts.append(delta.content)
        for tcd in delta.tool_calls:
            slot = acc.setdefault(tcd.index, {"id": "", "name": "", "arguments": ""})
            if tcd.id:
                slot["id"] = tcd.id
            if tcd.function is not None:
                if tcd.function.name:
                    slot["name"] = tcd.function.name
                if tcd.function.arguments:
                    slot["arguments"] += tcd.function.arguments
    tool_calls = [acc[i] for i in sorted(acc)]
    return "".join(content_parts), tool_calls


async def curate_from_conversation(
    store,
    user_id: str,
    llm,
    session_id: str | None = None,
    max_turns: int = MAX_CURATION_TURNS,
    *,
    conversation: str | None = None,
    source: str = "manual",
) -> dict[str, Any]:
    # A caller that already holds the text (the compression task, which
    # curates exactly the span it summarized) hands it over; everyone else
    # reads the user's sessions as before.
    sources = (
        conversation
        if conversation is not None
        else _gather_sources(store, user_id, session_id)
    )
    if not sources.strip():
        # A named session that yielded nothing is reported distinctly, so
        # losing access does not read as "nothing to do". One message covers
        # both causes -- not yours, and does not exist -- so it still confirms
        # nothing about anyone else's session.
        return {
            "status": "empty",
            "message": (
                "session not found, or not readable by you"
                if session_id
                else "no conversations or notes yet"
            ),
            "counts": {},
        }

    before = [r.memory_text for r in store.get_user_memory_records(user_id)]
    registry, counts, schemas = build_memory_registry(store, user_id)
    messages: list[dict] = [
        {"role": "system", "content": _CURATION_SYSTEM},
        {
            "role": "user",
            "content": _CURATION_USER.format(
                conversation=sources, memories=_format_memories(store, user_id)
            ),
        },
    ]
    tool_call_log: list[dict] = []
    for _ in range(max_turns):
        content, tool_calls = await asyncio.to_thread(
            _stream_turn, llm, messages, schemas
        )
        assistant: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls
            ]
        messages.append(assistant)
        if not tool_calls:
            break
        for tc in tool_calls:
            try:
                args = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                result = f"error: invalid JSON arguments: {exc}"
                tool_call_log.append(
                    {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                        "result": result,
                    }
                )
            else:
                response, _raw, errors = await registry.invoke(tc["name"], args)
                result = response or ("; ".join(errors) if errors else "ok")
                tool_call_log.append(
                    {"name": tc["name"], "arguments": args, "result": result}
                )
            messages.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": result}
            )

    after = [r.memory_text for r in store.get_user_memory_records(user_id)]
    trajectory = {
        "memory_before": before,
        "tool_calls": tool_call_log,
        "memory_after": after,
        "counts": dict(counts),
        "source": source,
    }
    record = store.add_memory_trajectory(
        user_id,
        session_id=session_id,
        model=getattr(getattr(llm, "config", None), "model_name", ""),
        trajectory=trajectory,
    )
    return {
        "status": "ok",
        "trajectory_id": record.id,
        "counts": dict(counts),
        "memory_count": len(after),
    }


def _format_span(records: list[ChatMessageRecord]) -> str:
    text = "\n".join(f"{r.role.upper()}: {r.content}" for r in records)
    return text[-MEMORY_GATHER_CHAR_BUDGET:]


async def curate_span(
    store, user_id: str, llm, session_id: str, records: list[ChatMessageRecord]
) -> bool:
    """Curate one span of a session into *user_id*'s memories.

    The compression task calls this over the turns it just summarized. True
    means curation ran; False means there was nothing to curate or it did
    not complete. Formats the span exactly as ``_gather_sources`` formats
    whole sessions so the curation prompt sees the same shape either way.
    """
    if not records:
        return False
    # Same rule as the manual path: only a session this user owns may feed
    # this user's memories. The surfaces let a signed-in caller continue an
    # ownerless (pre-login) session, so without this an anonymous transcript
    # would be filed under whoever picked the session up.
    if not _readable(store.get_chat_session(session_id), user_id):
        return False
    result = await curate_from_conversation(
        store,
        user_id,
        llm,
        session_id=session_id,
        conversation=_format_span(records),
        source="auto",
    )
    return result.get("status") == "ok"


_PROFILE_SYSTEM = (
    "You build a concise structured profile of a user from their memories. "
    "Return ONLY a JSON array of objects with keys 'topic', 'subtopic', and "
    "'content'. Group related facts under a shared topic. No prose outside the array."
)

_PROFILE_USER = "User memories:\n{memories}\n\nReturn the JSON profile array now."


def _parse_profile_json(text: str) -> list[dict[str, str]]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    out: list[dict[str, str]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and (item.get("topic") or item.get("content")):
                out.append(
                    {
                        "topic": str(item.get("topic", "")),
                        "subtopic": str(item.get("subtopic", "")),
                        "content": str(item.get("content", "")),
                    }
                )
    return out


def generate_user_profile(store, user_id: str, llm) -> list[UserProfileEntryRecord]:
    memories = [r.memory_text for r in store.get_user_memory_records(user_id)]
    if not memories:
        return store.replace_user_profile(user_id, [])
    prompt = [
        {"role": "system", "content": _PROFILE_SYSTEM},
        {
            "role": "user",
            "content": _PROFILE_USER.format(
                memories="\n".join(f"- {m}" for m in memories)
            ),
        },
    ]
    raw = llm.complete(prompt, max_tokens=800, temperature=0.0)
    text = raw if isinstance(raw, str) else getattr(raw, "text", "")
    return store.replace_user_profile(user_id, _parse_profile_json(text))


def get_user_profile(store, user_id: str) -> list[UserProfileEntryRecord]:
    return store.get_user_profile(user_id)
