# Auto-curation on the compression event

## Origin

PR #578 added working-memory compression: when a session's history overflows
the tail, the dropped turns are summarized in a background task and the next
turn sees the summary. It reserved a `curated_through` cursor in
`SessionMemoryState` and never wrote it.

Long-term memory (`user_memories`, curated by `curate_from_conversation`) still
fills only when someone calls `curate` by hand through the CLI, the `/api/memory`
router, or the MCP tool. This PR, the second of three, makes the compression
event feed it: the same background task that summarizes a span also curates
that span into the user's memories. One overflow event updates both tiers,
which is the shape of the reference `AgentMemory._compress` this work borrows
from.

PR 3 (relevance-aware recall above the 20-memory cap) is separate.

## Scope

- `compress_session` runs an optional curation step after the summary is
  saved, over exactly the span it summarized, and advances `curated_through`.
- `curate_from_conversation` can be handed the conversation text directly
  instead of reading whole sessions from the store.
- `schedule_compression` learns who the user is and whether curation is on;
  the three surfaces pass their already-resolved user id.
- Behind `AGENTIC_SEARCH_MEMORY_AUTO_CURATE` (default off), effective only
  when `AGENTIC_SEARCH_MEMORY_COMPRESSION` is also on.

## Non-goals

- Curating anonymous sessions. See Whose memory.
- Retrying a failed curation. See Failure.
- Any change to what curation stores, the curation prompt, consolidation,
  or profiles. The auto path sees only the dropped span, so a fact restated
  in a still-retained turn is invisible to it; accepted.
- Any change inside `src/agents/`.

## Design

### Whose memory

Curation writes to one user's bucket, so the task needs a user id. The three
surfaces already resolve it before they schedule compression: Assist has
`capabilities.user_id`, Chat has its `user_id` local, Tools has
`capabilities.user_id`. Each passes it as `user_id=` to `schedule_compression`.

An anonymous caller's session has no owner. The manual curation paths pool
anonymous callers into the shared `default_user` bucket, documented as a
local-research convenience, and `AGENTIC_SEARCH_MEMORY_REQUIRE_AUTH` exists to
turn that off. Auto-curation is silent, so silently filing anonymous
transcripts into a shared bucket is the cross-caller leak the require-auth
work closed. Anonymous sessions therefore skip curation entirely, regardless
of the require-auth flag. `user_id is None` means no curation.

`curate_span` also applies the manual path's ownership rule (`_readable`):
only a session owned by `user_id` may feed that user's memories. A session
started signed-out is ownerless forever, so its turns are never auto-curated
even after the caller signs in; that is the same cost the manual path already
accepts, and it stops an ownerless transcript being filed under whoever
continues it.

### Scoping curation to the span

`src/internal/memory/service.py`:

```python
async def curate_from_conversation(
    store, user_id, llm, session_id=None, max_turns=MAX_CURATION_TURNS,
    *, conversation: str | None = None,
) -> dict[str, Any]
```

When `conversation` is given, it is used as the source text and
`_gather_sources` is not called; otherwise behavior is unchanged. The
trajectory record still carries `session_id` when the caller passes one.

A new helper formats a span the way `_gather_sources` formats sessions, so
the curation prompt sees identical text either way:

```python
async def curate_span(store, user_id: str, llm, session_id: str,
                      records: list[ChatMessageRecord]) -> bool
```

It joins `f"{r.role.upper()}: {r.content}"` lines, keeps the last
`MEMORY_GATHER_CHAR_BUDGET` characters, calls `curate_from_conversation(...,
session_id=session_id, conversation=text)`, and returns `True` when the
result's `status` is `"ok"`. An empty span returns `False` without calling the
LLM.

### The compression task

`src/internal/memory/working.py`:

```python
CurateFn = Callable[[list[ChatMessageRecord]], Awaitable[bool]]

async def compress_session(session_id, llm, *, pending, cache=None,
                           curate: CurateFn | None = None) -> bool
```

After `save_state` succeeds for the summary, and only then:

1. If `curate` is `None`, return `True` as today.
2. Otherwise `await curate(pending)` inside its own `try`. On `True`,
   re-read the state and write it back with only `curated_through` set to
   `last_id`, so a summary or cursor another process advanced while curation
   ran is preserved (curation can outlive the 120 s lock lease). On `False`
   or an exception, log at warning and leave `curated_through` as it was.
   Either way return `True`: the summary advanced, which is what the return
   value reports.

`pending` here is the span after the stale-cursor trim, so curation and the
summary cover the same messages. The lock and in-flight guard already wrap
the whole task, so curation runs under them too.

```python
def schedule_compression(wm, *, session_id, llm, enabled, cache=None,
                         store=None, user_id: str | None = None,
                         auto_curate: bool = False) -> asyncio.Task | None
```

Builds `curate = functools.partial(curate_span, store, user_id, llm,
session_id)` when `auto_curate and store is not None and user_id is not None`,
else `None`, and passes it to `compress_session`. The existing early return
(`not enabled or llm is None or not wm.pending`) is unchanged, so curation
can never run without compression.

`working.py` imports `curate_span` from `service.py`. `service.py` does not
import `working.py`, so there is no cycle.

### Call sites

| Surface | Change |
|---|---|
| Assist, `app.py` `_run_agent_impl` `finally` | add `store=db, user_id=user_id, auto_curate=settings.memory_auto_curate` |
| Chat, `chat_backend.py` | add `store=store, user_id=user_id, auto_curate=memory_auto_curate`; `create_chat_router` gains `memory_auto_curate: bool = False` |
| Tools, `tool_backend.py` | add `store=store, user_id=capabilities.user_id, auto_curate=memory_auto_curate`; `create_tool_router` gains `memory_auto_curate: bool = False` |
| `_register_routers` | gains `memory_auto_curate: bool = False`, passes it to both routers; `create_web_app` passes `settings.memory_auto_curate` |

### Flag

`SearchExperienceSettings.memory_auto_curate: bool = False`, populated by
`_flag("AGENTIC_SEARCH_MEMORY_AUTO_CURATE")`, documented in
`docs/configuration.md` directly after the `AGENTIC_SEARCH_MEMORY_COMPRESSION`
row, including that it does nothing unless compression is on, that anonymous
sessions are never curated, and that each overflow costs one curation loop
(up to `MAX_CURATION_TURNS` tool-calling LLM turns) against the configured
remote LLM client.

### Failure

Curation is best-effort on top of a correct summary:

- `curate_span` raising or returning `False`: logged; `curated_through` does
  not move; the span is not retried. The next overflow curates its own span.
  Retrying a six-turn tool-calling loop against the same span would double
  the cost of every transient failure, and long-term memory is a bonus, not
  the record.
- A cache write failure after curation: caught by `compress_session`'s
  existing boundary; the summary write already landed, so the only loss is
  the curated cursor, which means the span may then be curated a second
  time; nothing runs `consolidate_memories` automatically, so such
  duplicates persist until a human runs it.
- Nothing in this path can fail a request or raise out of the task; the
  existing never-raise boundary covers the new code.
- Two sessions of one user overflowing together run two curation loops
  against the same memory bucket. The memory tools are id-scoped and degrade
  to "memory not found", so there is no corruption or cross-user write, but
  lost updates and duplicate adds are possible; accepted for a best-effort
  path. A process restart mid-curation leaves that span's memory writes
  partially applied with `curated_through` unmoved; also accepted.

## Testing

`tests/unit/memory/test_working_memory.py` (fake `curate` coroutines recording
their argument):

- success: `curated_through == summarized_through == pending[-1].id`; the
  curate fake received exactly the post-trim `pending`.
- curate returns `False`: summary saved, `curated_through` unchanged,
  `compress_session` returned `True`.
- curate raises: same as `False`, and the lock and in-flight entry are
  released (a second call on the session succeeds).
- `curate=None`: state identical to today (`curated_through` preserved).
- `schedule_compression`: `auto_curate=True` with `user_id=None` schedules
  compression with no curation (cursor never moves); with a user id and store
  it curates; `auto_curate=False` never curates; `enabled=False` schedules
  nothing even with `auto_curate=True`.

`tests/unit/memory/test_curation.py`:

- `curate_from_conversation(conversation=...)` uses the given text (it
  appears in the LLM prompt) and does not call `list_sessions_for_user` or
  `get_chat_session` (a store double that raises on those).
- `curate_span` formats `ROLE: content` lines, returns `True` on
  `status == "ok"`, `False` on an empty span without touching the LLM.

Surface tests: each of the three flag-on scheduler-recorder tests additionally
asserts `kw["user_id"]`, `kw["store"]`, and `kw["auto_curate"]`; the
`_register_routers` test asserts the new kwarg reaches both routers. A
settings test covers the new flag's default and env read.

Every new test gets a mutation check.
