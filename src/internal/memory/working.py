"""Working memory: the session tail plus a compressed summary of what fell off it.

Every conversational surface used to keep the last N messages and forget the
rest. This module keeps the same tail and, when turns fall outside it, hands
them to a background summarizer whose output is prepended to the next turn as
one system message. Per-session state (the summary and the id of the last
message it covers) lives in the process's ``CacheBackend`` -- in-memory by
default, Redis when ``CACHE_BACKEND=redis`` -- so no new dependency is added.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

from src.context.models import ChatMessage
from src.internal.cache.interface import CacheBackend
from src.internal.db.models import ChatMessageRecord

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 40
SUMMARY_PREFIX = "Earlier in this conversation: "

_STATE_KEY = "session_memory:{session_id}"


@dataclass(frozen=True)
class SessionMemoryState:
    """What the cache remembers about one session's working memory."""

    summary: str = ""
    # Id of the last message the summary covers. None until the first compress.
    summarized_through: str | None = None
    # Reserved for auto-curation (next PR); never written here.
    curated_through: str | None = None


def load_state(cache: CacheBackend, session_id: str) -> SessionMemoryState:
    """Read the session's state; the default on a missing key, bad JSON, or a
    failing cache. A cache problem must never fail the request that reads it."""
    key = _STATE_KEY.format(session_id=session_id)
    try:
        raw = cache.get(key)
    except Exception as exc:  # noqa: BLE001 - degrade to no summary
        logger.warning("session memory read failed for %s: %s", session_id, exc)
        return SessionMemoryState()
    if raw is None:
        return SessionMemoryState()
    try:
        data = json.loads(raw)
        return SessionMemoryState(
            summary=str(data.get("summary", "")),
            summarized_through=data.get("summarized_through"),
            curated_through=data.get("curated_through"),
        )
    except (ValueError, AttributeError):
        return SessionMemoryState()


def save_state(cache: CacheBackend, session_id: str, state: SessionMemoryState) -> None:
    cache.set(_STATE_KEY.format(session_id=session_id), json.dumps(asdict(state)))


@dataclass(frozen=True)
class WorkingMemory:
    """What a surface hands to its loop, plus what still needs summarizing."""

    messages: list[ChatMessage]
    summary: str
    pending: list[ChatMessageRecord]


def load_working_memory(
    store,
    session_id: str,
    *,
    keep_last: int = MAX_HISTORY_MESSAGES,
    cache: CacheBackend | None = None,
) -> WorkingMemory:
    """Return the last ``keep_last`` messages, prefixed by the stored summary
    when one covers the dropped prefix.

    ``cache=None`` (the flag off) skips state entirely: the result is exactly
    the tail-slice every surface used before, and ``pending`` is empty so no
    compression is ever scheduled.
    """
    records = store.list_chat_messages(session_id)
    tail = records[-keep_last:]
    dropped = records[:-keep_last]
    messages = [ChatMessage(role=r.role, content=r.content) for r in tail]
    if cache is None or not dropped:
        return WorkingMemory(messages=messages, summary="", pending=[])

    state = load_state(cache, session_id)
    dropped_ids = [r.id for r in dropped]
    if state.summary and state.summarized_through in dropped_ids:
        cut = dropped_ids.index(state.summarized_through) + 1
        summary_message = ChatMessage(
            role="system", content=SUMMARY_PREFIX + state.summary
        )
        return WorkingMemory(
            messages=[summary_message, *messages],
            summary=state.summary,
            pending=dropped[cut:],
        )
    # No summary, or its cursor is not in this session's dropped prefix (state
    # lost, or the cap grew so the covered turns are back in the tail): start over.
    return WorkingMemory(messages=messages, summary="", pending=list(dropped))
