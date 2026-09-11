# Working-memory compression

## Origin

Every conversational surface loads a session's history the same way: read all
messages, keep the last N. Assist (`app.py` `_trim_history`, 40 messages;
search mode 6), Chat (`chat_backend.py`, 40) and Tools (`tool_backend.py`, 40)
each carry their own copy of that slice. Turns that fall outside the tail are
gone from the model's context. Nothing summarizes them, and the long-term memory
store (`user_memories`) only fills when someone calls `curate` by hand.

This is the first of three PRs that borrow the three-tier shape of a reference
`AgentMemory` class (working memory with compression, a long-term store, a KV
store for structured state) without adding a dependency:

1. **This PR** — working-memory compression, with state in the existing
   `CacheBackend`.
2. Auto-curation on the compression event, scoped to the dropped turns.
3. Query-relevant recall in the memory preamble when a user exceeds the cap.

## Scope

- One helper, `load_working_memory`, replaces the three tail-slices.
- When turns fall outside the tail, a background task summarizes them and the
  next turn sees `[summary] + tail` instead of `tail`.
- Per-session state (summary and cursors) lives in `CacheBackend`, so it is
  in-memory by default and Redis when `CACHE_BACKEND=redis`.
- Everything ships behind `AGENTIC_SEARCH_MEMORY_COMPRESSION` (default off).

## Non-goals

- Curating the dropped turns into `user_memories` (PR 2). The
  `curated_through` cursor is defined here so PR 2 does not change the state
  shape, but nothing writes it.
- Token-count triggers. The trigger is the existing message-count cap. Token
  budgeting still happens in the loop's `_crop_prompt_ids`.
- Compression inside the agent loops themselves. The CLI and the trainers keep
  their current behavior.
- A hard Redis dependency. The local 3-process stack runs unchanged.

## Design

### State

`src/internal/memory/working.py` defines:

```python
@dataclass(frozen=True)
class SessionMemoryState:
    summary: str = ""
    summarized_through: str | None = None   # id of the last message the summary covers
    curated_through: str | None = None      # reserved for PR 2; never written here
```

Stored as JSON under `session_memory:<session_id>` via `get_cache_backend()`.
`load_state(cache, session_id) -> SessionMemoryState` returns the default on a
missing key or on malformed JSON. `save_state(cache, session_id, state)` writes
it with no expiry. Both are pure with respect to the store.

### Loading working memory

```python
@dataclass(frozen=True)
class WorkingMemory:
    messages: list[ChatMessage]          # what the surface hands to the loop
    summary: str                         # the applied summary, "" when none
    pending: list[ChatMessageRecord]     # dropped turns not yet summarized

def load_working_memory(store, session_id, *, keep_last, cache=None) -> WorkingMemory
```

1. `records = store.list_chat_messages(session_id)`; `state = load_state(...)`.
2. `tail = records[-keep_last:]`; `dropped = records[:-keep_last]` (empty when
   the session fits).
3. `pending` is the suffix of `dropped` after `summarized_through`. If the
   cursor is unknown to this session (state was lost, or the id is not in
   `dropped`), `pending = dropped` and the stale summary is ignored.
4. `messages` is `tail` mapped to `ChatMessage`, prefixed by one
   `ChatMessage(role="system", content=SUMMARY_PREFIX + state.summary)` when
   `state.summary` is non-empty and the cursor is known.

`SUMMARY_PREFIX` is `"Earlier in this conversation: "`. The summary is a system
message so every loop treats it as context rather than as a user turn. Each
loop prepends its own system prompt ahead of the buffer, and `_crop_prompt_ids`
protects only that first system message. The summary sits in the croppable
region like the rest of the history, which is right: under a token overflow it
is the oldest content and the first to go. `summary` is also returned on its
own so a surface that re-caps the messages can re-apply it.

### Call sites

| Surface | Today | After |
|---|---|---|
| Assist, `app.py` | `_trim_history(list_chat_messages(...))`, `MAX_HISTORY_MESSAGES = 40` | `load_working_memory(db, session_id, keep_last=MAX_HISTORY_MESSAGES)` |
| Search mode, `app.py` | `_trim_history(history, 6)` inside `_build_search_agent_messages` | `_build_search_agent_messages` keeps a leading `role == "system"` message (the summary) ahead of the tail it caps to 6. No new argument |
| Chat, `chat_backend.py` | `[-_MAX_HISTORY_MESSAGES:]` | `load_working_memory(store, session_id, keep_last=40)` |
| Tools, `tool_backend.py` | `_history()` slice | same |

`_trim_history` stays as the tail primitive the helper uses; the two private
`_MAX_HISTORY_MESSAGES` constants are removed. `MAX_HISTORY_MESSAGES` moves to
`working.py` as the default `keep_last`, and `app.py` re-exports it.

`create_chat_router(db)` becomes `create_chat_router(db, *, llm=None,
memory_compression=False)` and `create_tool_router(...)` gains the same two
keyword arguments. A bool rather than the settings object, because the
settings class lives in `app.py`, which imports both routers. Chat and Tool
schedule the task right after persisting the user message: their answer comes
from the local model, so the remote `llm` contends with nothing. Assist
schedules in its `finally`, after the reply, because there the same `llm`
produces the answer.

### Compression

```python
async def compress_session(session_id, llm, *, pending, cache=None) -> bool
```

Scheduled by each surface with `asyncio.create_task` **after** the assistant
reply is persisted, only when `pending` is non-empty, the flag is on, and `llm`
is not `None`. The surface keeps the task in a module-level set (the
`add`/`discard` idiom) so it is not garbage-collected.

The task:

1. Takes `cache.lock(f"session_memory:{session_id}:compress", timeout=120)`
   non-blocking. If not acquired, returns `False`: another turn is already
   summarizing this span.
   The task also keeps a module-level `_inflight` set of session ids being
   summarized in this process. `_InMemoryCacheLock` always acquires, so on the
   default backend this set is the real guard; the cache lock covers
   cross-process deployments.
2. Re-reads state. If `summarized_through` already equals the last pending id,
   returns `False` (the other task finished first).
3. Builds one prompt: system `"You compress a conversation. Rewrite the prior
   summary and the new turns into one concise summary that keeps facts,
   decisions, user preferences, and open questions. Output the summary only."`,
   user `"Prior summary:\n{summary or '(none)'}\n\nNew turns:\n{ROLE: content
   lines}"`.
4. Calls `llm.complete(messages, max_tokens=400, temperature=0.0)` through
   `asyncio.to_thread`. `complete` returns `LLMResponse | str`; take `.text`
   when present.
5. On a non-empty result, `save_state` with the new summary and
   `summarized_through = pending[-1].id`. Returns `True`.

`pending` is passed in rather than recomputed so the task summarizes exactly
the span the request saw; a message appended in between is picked up next turn.

### Flag

`SearchExperienceSettings` gains `memory_compression: bool = False`, populated
by `_flag("AGENTIC_SEARCH_MEMORY_COMPRESSION")` in `from_app_settings`, and
documented in `docs/configuration.md`. The Chat and Tool routers receive
`memory_compression` as a bool (see Call sites).

Flag off: `load_working_memory` is still the loader (the consolidation is
unconditional) but the caller passes `cache=None`, which skips state entirely,
so `messages` is exactly today's tail and `pending` is empty. No task is ever
scheduled. Behavior is byte-for-byte unchanged.

### Failure

- `llm` is `None` (the local-model-only stack): no task, plain tail.
- Cache read or write raises: logged at warning, treated as default state.
  The request proceeds with the plain tail.
- `llm.complete` raises or returns empty: logged, state untouched, lock
  released. The next turn recomputes the same `pending` and retries.
- Malformed JSON in the cache: default state, then overwritten on the next
  successful compress.

None of these paths can fail a request. Compression is a delivery detail, not
a run detail, the same stance `_guarded_token_callback` takes.

## Testing

`tests/unit/test_working_memory.py`, using `AgenticSearchStore(":memory:")` and
`InMemoryCache`, with a fake `llm` whose `complete` records its prompt and
returns a canned string:

- Below the cap: `messages` equals the full history, `pending` empty, no
  summary.
- Above the cap with no state: `messages` is the tail, `pending` is the
  dropped prefix.
- Above the cap with a summary covering the dropped prefix: `messages` starts
  with the summary system message, `pending` empty.
- Summary exists but its cursor is not in the session: summary ignored,
  `pending` is the whole dropped prefix.
- `cache=None`: identical to today's slice.
- `compress_session` advances `summarized_through` to the last pending id and
  stores the model's text.
- `compress_session` on an llm that raises: state unchanged, lock released
  (a second call with a working llm succeeds).
- Two concurrent `compress_session` calls on one session: exactly one calls
  the llm.
- The prompt handed to the llm contains the prior summary and every pending
  turn.

Surface tests: `test_search_agent_history.py` keeps passing unchanged, plus
one case showing `_build_search_agent_messages(query, history, summary=...)`
puts the summary before the capped tail. Chat and Tool backend tests assert
that with the flag off the history handed to the runner equals the last 40
messages, and with the flag on and a stored summary it is prefixed.

Each new test gets a mutation check: delete the behavior it guards and confirm
it goes red before the implementation lands.

## Follow-ups (not in this PR)

- PR 2 writes `curated_through` and calls `curate_from_conversation` on the
  same pending span, in the same task, after the summary is saved.
- PR 3 makes `memory_preamble` relevance-aware above the 20-memory cap.
