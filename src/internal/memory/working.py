"""Working memory: the session tail plus a compressed summary of what fell off it.

Every conversational surface used to keep the last N messages and forget the
rest. This module keeps the same tail and, when turns fall outside it, hands
them to a background summarizer whose output is prepended to the next turn as
one system message. Per-session state (the summary and the id of the last
message it covers) lives in the process's ``CacheBackend`` -- in-memory by
default, Redis when ``CACHE_BACKEND=redis`` -- so no new dependency is added.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass

from src.context.models import ChatMessage
from src.internal.cache.interface import CacheBackend, get_cache_backend
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


_LOCK_KEY = "session_memory:{session_id}:compress"
_SUMMARY_MAX_TOKENS = 400
_SUMMARY_SYSTEM = (
    "You compress a conversation. Rewrite the prior summary and the new turns "
    "into one concise summary that keeps facts, decisions, user preferences, "
    "and open questions. Output the summary only."
)

# Sessions being summarized in this process. `_InMemoryCacheLock` always
# acquires, so on the default backend this set is the real mutual exclusion;
# the cache lock below covers a Redis deployment with several processes.
_inflight: set[str] = set()
# Strong references to scheduled tasks, so the event loop cannot drop them.
_tasks: set[asyncio.Task] = set()


def _summary_prompt(prior: str, pending: list[ChatMessageRecord]) -> list[dict]:
    turns = "\n".join(f"{r.role.upper()}: {r.content}" for r in pending)
    return [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {
            "role": "user",
            "content": f"Prior summary:\n{prior or '(none)'}\n\nNew turns:\n{turns}",
        },
    ]


def _complete(llm, prompt: list[dict]) -> str:
    raw = llm.complete(prompt, max_tokens=_SUMMARY_MAX_TOKENS, temperature=0.0)
    text = raw if isinstance(raw, str) else getattr(raw, "text", "")
    return (text or "").strip()


async def compress_session(
    session_id: str,
    llm,
    *,
    pending: list[ChatMessageRecord],
    cache: CacheBackend | None = None,
) -> bool:
    """Summarize ``pending`` into the session's stored summary.

    Returns True when the state advanced. False means nothing to do, another
    task owns this span, or the summarizer failed -- in which case the state
    is untouched and the next turn retries the same span.
    """
    if not pending or llm is None or session_id in _inflight:
        return False
    cache = cache if cache is not None else get_cache_backend()
    lock = cache.lock(_LOCK_KEY.format(session_id=session_id), timeout=120)
    if not lock.acquire(blocking=False):
        return False
    _inflight.add(session_id)
    try:
        state = load_state(cache, session_id)
        last_id = pending[-1].id
        if state.summarized_through == last_id:
            return False
        text = await asyncio.to_thread(
            _complete, llm, _summary_prompt(state.summary, pending)
        )
        if not text:
            return False
        save_state(
            cache,
            session_id,
            SessionMemoryState(
                summary=text,
                summarized_through=last_id,
                curated_through=state.curated_through,
            ),
        )
        return True
    except Exception as exc:  # noqa: BLE001 - compression is a delivery detail
        logger.warning("session memory compression failed for %s: %s", session_id, exc)
        return False
    finally:
        _inflight.discard(session_id)
        lock.release()


def schedule_compression(
    wm: WorkingMemory,
    *,
    session_id: str,
    llm,
    enabled: bool,
    cache: CacheBackend | None = None,
) -> asyncio.Task | None:
    """Fire-and-forget ``compress_session`` when there is something to compress.

    Must be called from a running event loop. Returns the task (tests await
    it) or None when nothing was scheduled.
    """
    if not enabled or llm is None or not wm.pending:
        return None
    task = asyncio.create_task(
        compress_session(session_id, llm, pending=wm.pending, cache=cache)
    )
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task
