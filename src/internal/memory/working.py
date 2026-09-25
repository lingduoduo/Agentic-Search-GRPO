"""Working memory: the session tail plus a compressed summary of what fell off it.

Every surface sends the newest messages that fit both a message cap and an
estimated-token budget (``AGENTIC_SEARCH_MEMORY_HISTORY_TOKENS``). When turns
fall outside that tail they go to a background summarizer, one bounded chunk
per call, whose output is prepended to the next turn as one system message.
Summarization is on by default for ``/api/agent`` only; see
``AGENTIC_SEARCH_MEMORY_COMPRESSION``. Per-session state (the summary and the
id of the last message it covers) lives in the process's ``CacheBackend`` --
in-memory by default, Redis when ``CACHE_BACKEND=redis`` -- so no new
dependency is added.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, replace

from src.context.models import ChatMessage
from src.internal.cache.interface import CacheBackend, get_cache_backend
from src.internal.db.models import ChatMessageRecord
from src.internal.memory.service import curate_span

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 40
# Default for AGENTIC_SEARCH_MEMORY_HISTORY_TOKENS: leaves room in the local
# loops' 4096-token prompt for the system prompt, the summary and the new turn.
DEFAULT_HISTORY_TOKENS = 2500
_MESSAGE_OVERHEAD_TOKENS = 4
SUMMARY_PREFIX = "Earlier in this conversation: "
SESSION_MEMORY_TTL_SECONDS = 30 * 24 * 3600

_STATE_KEY = "session_memory:{session_id}"


@dataclass(frozen=True)
class SessionMemoryState:
    """What the cache remembers about one session's working memory."""

    summary: str = ""
    # Id of the last message the summary covers. None until the first compress.
    summarized_through: str | None = None
    # Id of the last message curated into the user's long-term memories.
    # None until the first successful curation.
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
    cache.set(
        _STATE_KEY.format(session_id=session_id),
        json.dumps(asdict(state)),
        ex=SESSION_MEMORY_TTL_SECONDS,
    )


@dataclass(frozen=True)
class WorkingMemory:
    """What a surface hands to its loop, plus what still needs summarizing."""

    messages: list[ChatMessage]
    summary: str
    pending: list[ChatMessageRecord]


def estimate_tokens(text: str) -> int:
    """A rough token count, four characters per token; no tokenizer needed."""
    return math.ceil(len(text) / 4)


def _record_tokens(record: ChatMessageRecord) -> int:
    return estimate_tokens(record.content) + _MESSAGE_OVERHEAD_TOKENS


def _fit_budget(
    records: list[ChatMessageRecord], token_budget: int
) -> list[ChatMessageRecord]:
    """The newest ``records`` whose estimates fit ``token_budget``. The newest
    one is always kept, even alone over budget: a turn with no history beats
    a turn that silently loses the message it is answering."""
    used = 0
    kept = 0
    for record in reversed(records):
        used += _record_tokens(record)
        if kept and used > token_budget:
            break
        kept += 1
    return records[len(records) - kept :]


def load_working_memory(
    store,
    session_id: str,
    *,
    keep_last: int = MAX_HISTORY_MESSAGES,
    token_budget: int | None = None,
    cache: CacheBackend | None = None,
) -> WorkingMemory:
    """Return the newest messages -- at most ``keep_last`` and, with
    ``token_budget`` set, only as many as fit its estimate (the newest always)
    -- prefixed by the stored summary when one covers the dropped prefix.

    ``token_budget=None`` is the pure message count. ``cache=None`` (the flag
    off) skips state entirely: the result is exactly the tail, and ``pending``
    is empty so no compression is ever scheduled.
    """
    records = store.list_chat_messages(session_id)
    tail = records[-keep_last:]
    if token_budget is not None:
        tail = _fit_budget(tail, token_budget)
    dropped = records[: len(records) - len(tail)]
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
# Estimated tokens of turns one summarizer call may see. A backlog larger than
# this (the first compression of a long session) drains one chunk per turn.
_SUMMARY_INPUT_TOKENS = 3000
_CLIP_MARKER = " [...truncated]"
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

# Curates one span of records into a user's long-term memories. Built by
# ``schedule_compression`` from the store, user and llm; ``compress_session``
# only ever sees the callable.
CurateFn = Callable[[list[ChatMessageRecord]], Awaitable[bool]]


def _bounded_span(pending: list[ChatMessageRecord]) -> list[ChatMessageRecord]:
    """The oldest records of ``pending`` that fit ``_SUMMARY_INPUT_TOKENS``,
    always at least one so the cursor can advance."""
    used = 0
    span: list[ChatMessageRecord] = []
    for record in pending:
        used += _record_tokens(record)
        if span and used > _SUMMARY_INPUT_TOKENS:
            break
        span.append(record)
    return span


def _clip(text: str) -> str:
    """Cap one record at the input budget, for the prompt only."""
    limit = _SUMMARY_INPUT_TOKENS * 4
    return text if len(text) <= limit else text[:limit] + _CLIP_MARKER


def _summary_prompt(prior: str, pending: list[ChatMessageRecord]) -> list[dict]:
    turns = "\n".join(f"{r.role.upper()}: {_clip(r.content)}" for r in pending)
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


async def _curate_after_summary(
    cache: CacheBackend,
    session_id: str,
    curate: CurateFn,
    pending: list[ChatMessageRecord],
    last_id: str,
) -> None:
    """Best-effort: curate the span the summary just covered, then move the
    curated cursor. A failure is logged and the span is not retried -- the
    summary is the record, long-term memory is a bonus, and a six-turn
    tool-calling loop is not worth re-running on every transient error."""
    try:
        curated = await curate(pending)
    except Exception as exc:  # noqa: BLE001 - curation is best-effort
        logger.warning("memory auto-curation failed for %s: %s", session_id, exc)
        return
    if not curated:
        return
    try:
        current = load_state(cache, session_id)
        if current.summarized_through is None and not current.summary:
            # load_state degrades a read failure, bad JSON, or an expired key
            # to a blank state. Merging onto that would write summary="" over
            # the summary this task saved a moment ago; leaving the cursor
            # unmoved is the same accepted outcome as a failed cursor write.
            logger.warning(
                "memory auto-curation cursor merge skipped for %s: state re-read "
                "came back blank",
                session_id,
            )
            return
        save_state(cache, session_id, replace(current, curated_through=last_id))
    except Exception as exc:  # noqa: BLE001 - the summary is already saved
        logger.warning(
            "memory auto-curation cursor write failed for %s: %s", session_id, exc
        )


async def compress_session(
    session_id: str,
    llm,
    *,
    pending: list[ChatMessageRecord],
    cache: CacheBackend | None = None,
    curate: CurateFn | None = None,
) -> bool:
    """Summarize the oldest chunk of ``pending`` (up to
    ``_SUMMARY_INPUT_TOKENS``) into the session's stored summary; the rest
    drains on later turns.

    Returns True when the state advanced. False means nothing to do, another
    task owns this span or advanced past it while this one ran, or the
    summarizer failed -- in every case this task wrote nothing, and the next
    turn recomputes ``pending`` from whatever cursor is stored.

    When ``curate`` is given it runs after the summary is saved, over the
    same span; see ``_curate_after_summary``.
    """
    if not pending or llm is None or session_id in _inflight:
        return False
    cache = cache if cache is not None else get_cache_backend()
    # None until acquired: ``cache.lock(...)`` and ``lock.acquire(...)`` are
    # I/O on real backends (Redis, Postgres) and can raise, so they must sit
    # inside the try below; `lock` only becomes non-None once acquired, which
    # is also what tells `finally` whether a release is owed.
    lock = None
    try:
        candidate = cache.lock(_LOCK_KEY.format(session_id=session_id), timeout=120)
        if not candidate.acquire(blocking=False):
            return False
        lock = candidate
        _inflight.add(session_id)
        state = load_state(cache, session_id)
        # Two overlapping compression tasks can run their tasks out of order;
        # drop anything up to and including the (possibly newer) stored
        # cursor so the later cursor is never overwritten by a stale span and
        # already-covered turns never re-enter the prompt.
        pending_ids = [r.id for r in pending]
        if state.summarized_through in pending_ids:
            pending = pending[pending_ids.index(state.summarized_through) + 1 :]
            if not pending:
                return False
        pending = _bounded_span(pending)
        last_id = pending[-1].id
        text = await asyncio.to_thread(
            _complete, llm, _summary_prompt(state.summary, pending)
        )
        if not text:
            return False
        # The summarizer call can outlive the 120s lock lease. Re-read before
        # writing: if the cursor moved meanwhile, another process summarized
        # a later span and this text is stale -- writing it would roll the
        # session back and re-cover (and re-curate) those turns. A blank
        # re-read on a session that had a cursor counts as moved: better to
        # leave state alone than to write over a value the read did not see.
        current = load_state(cache, session_id)
        if current.summarized_through != state.summarized_through:
            logger.warning(
                "session memory compression skipped for %s: cursor moved during "
                "summarization",
                session_id,
            )
            return False
        save_state(
            cache,
            session_id,
            replace(current, summary=text, summarized_through=last_id),
        )
        if curate is not None:
            await _curate_after_summary(cache, session_id, curate, pending, last_id)
        return True
    except Exception as exc:  # noqa: BLE001 - compression is a delivery detail
        logger.warning("session memory compression failed for %s: %s", session_id, exc)
        return False
    finally:
        # Safe on the lock-not-acquired path too: the pre-`try` guard already
        # returned for any same-process caller that saw this id in the set.
        _inflight.discard(session_id)
        if lock is not None:
            try:
                lock.release()
            except Exception as exc:  # noqa: BLE001 - release is best-effort
                logger.warning(
                    "session memory lock release failed for %s: %s", session_id, exc
                )


def schedule_compression(
    wm: WorkingMemory,
    *,
    session_id: str,
    llm,
    enabled: bool,
    cache: CacheBackend | None = None,
    store=None,
    user_id: str | None = None,
    auto_curate: bool = False,
) -> asyncio.Task | None:
    """Fire-and-forget ``compress_session`` when there is something to compress.

    Must be called from a running event loop. Returns the task (tests await
    it) or None when nothing was scheduled.

    Tasks are not drained at shutdown; a cancelled task releases its lock and
    in-flight entry in ``compress_session``'s ``finally``, and the interrupted
    span is re-summarized on the next turn.

    With ``auto_curate`` on, a ``store`` and a ``user_id``, the same task also
    curates the summarized span into that user's memories. An anonymous
    session (``user_id`` is None) is never curated: auto-curation is silent,
    and silently pooling anonymous transcripts into a shared bucket is the
    leak ``AGENTIC_SEARCH_MEMORY_REQUIRE_AUTH`` exists to prevent.
    """
    if not enabled or llm is None or not wm.pending:
        return None
    curate: CurateFn | None = None
    if auto_curate and store is not None and user_id is not None:
        curate = functools.partial(curate_span, store, user_id, llm, session_id)
    task = asyncio.create_task(
        compress_session(
            session_id, llm, pending=wm.pending, cache=cache, curate=curate
        )
    )
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task
