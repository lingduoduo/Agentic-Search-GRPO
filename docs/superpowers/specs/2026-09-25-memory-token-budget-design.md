# Working memory: a token-budgeted tail, summarized by default: design

## Problem

Summarization is never forced (fallback investigation, 2026-09-25):

- **Messages are counted, not tokens.** `load_working_memory`
  (`src/internal/memory/working.py`) keeps the last `MAX_HISTORY_MESSAGES =
  40` messages. Forty long messages overflow the local loops' 4096-token
  prompt. `_fit_messages_to_budget` then silently drops whole old messages,
  and the remote-LLM paths have no check at all, so a long prompt ends in
  "context too long", which becomes a 502.
- **Summarization is off by default.** It only runs with
  `AGENTIC_SEARCH_MEMORY_COMPRESSION=1`. Without it, anything outside the
  tail is simply forgotten.
- **The summarizer's input is unbounded.** On the first compression, or when
  the flag is turned on for an existing session, `pending` is the entire
  dropped prefix. It goes out in one prompt with no chunking. If that call
  fails, nothing is written, `pending` keeps growing, and the summarizer can
  fail on every turn from then on. The only signal is a WARNING log.
- **Dead config and docs suggest otherwise.** `COMPRESSION_TRIGGER_RATIO`
  (`src/internal/configs/chat_configs.py`, `default_config.py`) and
  `src/internal/chat/COMPRESSION.md` describe token-ratio compression that
  does not exist.

## Decision (approved by the user)

### 1. A token-budgeted tail

`load_working_memory(store, session_id, *, keep_last=MAX_HISTORY_MESSAGES,
token_budget=None, cache=None)`:

- **Estimate.** `estimate_tokens(text) = ceil(len(text) / 4)`, plus a
  per-message overhead of 4. It is a module-level function, with no new
  dependency.
- **Walking the history.** Start from the newest record and keep records
  while the running estimate stays within `token_budget`, keeping at most
  `keep_last` records. **The newest record is always kept**, even if it alone
  exceeds the budget. Everything older is the dropped prefix, and it feeds
  `pending` exactly as the message-count cut does today.
- **No budget.** `token_budget=None` keeps today's behavior exactly (a pure
  message count), for callers that do not opt in.
- **The budget value.** It comes from `AGENTIC_SEARCH_MEMORY_HISTORY_TOKENS`,
  default **2500**. That leaves room in the local loops' 4096-token prompt
  for the system prompt, the summary and the new message. It is read into
  `SearchExperienceSettings.memory_history_tokens`, and every surface that
  calls `load_working_memory` passes it: `/api/agent`,
  `/chat/send-chat-message` and `/tool/send-tool-message`.
- **Bad values.** A non-positive or non-integer value raises at settings load,
  failing fast, in the same way the existing settings do.

### 2. Summarization on by default, only where it adds no new data flow

- **Where the defaults come from.** `AGENTIC_SEARCH_MEMORY_COMPRESSION` goes
  from a boolean flag to a tri-state:
  - **unset (default)**: on for `/api/agent` only;
  - **truthy** (`1/true/yes`): on for every surface;
  - **falsy** (`0/false/no`): off for every surface.
- **Why `/api/agent` only.** `/api/agent` already sends the conversation
  history to the remote LLM, so the summarizer, which uses that same `llm`,
  adds no new flow. `/chat` and `/tool` answer with the **local** model, so
  turning summarization on there would send their older turns to the remote
  LLM for the first time.
- **Settings.** `SearchExperienceSettings` gets two fields,
  `memory_compression` (`/api/agent`) and `memory_compression_direct`
  (`/chat` and `/tool`). The direct routers receive the latter.
- **No LLM configured.** Summarization cannot run without a remote `llm`,
  which is already handled by `schedule_compression`. The tail is still
  token-budgeted.
- **`AGENTIC_SEARCH_MEMORY_AUTO_CURATE`** is unchanged and still defaults off.

### 3. Bounded summarizer input

`compress_session` summarizes only the **oldest** part of `pending`, whatever
fits in `_SUMMARY_INPUT_TOKENS = 3000` estimated tokens:

- It always takes at least one record. A single record over the limit has
  its content clipped to the limit, with a marker, for the prompt only; the
  stored message is not changed.
- `last_id` is the last record included, so the cursor advances through the
  backlog one bounded step per turn and the backlog drains over successive
  turns.
- The existing lock, re-read, stale-cursor and curation logic is unchanged.
  Curation covers the same bounded span.

### 4. Remove the dead config

- Delete `COMPRESSION_TRIGGER_RATIO` from `chat_configs.py` and from
  `default_config.py`, including any test or doc reference.
- Delete `src/internal/chat/COMPRESSION.md`, whose functions do not exist.
- Grep first. If anything reads them, stop and record it in the plan instead
  of deleting.

### Documentation

- `docs/configuration.md` gets the new env var and the tri-state semantics
  (the documented-env-vars test enforces this).
- The docstring of `working.py` is updated to match.

## Out of scope

- Synchronous (blocking) summarization.
- A tokenizer-accurate count.
- Token checks on the remote LLM answer prompt, beyond what the budgeted
  history now provides.
- The search agent's separate 6-message cap.
- Auto-curation defaults.

## Testing

- **Tail.**
  - Budget respected.
  - The newest record is kept even when it is over budget.
  - `keep_last` is still a cap.
  - `token_budget=None` matches today's slices exactly.
  - Dropped records become `pending`.
  - An existing summary cursor inside the dropped prefix still gives the
    summary plus the pending span.
- **Settings.**
  - Unset: `memory_compression` on, direct off.
  - `1`: both on.
  - `0`: both off.
  - Budget env var parsed, and a bad value rejected.
- **Surfaces.**
  - `/api/agent` with default settings schedules compression once history
    exceeds the budget, using a fake llm and the in-memory cache.
  - `/chat` and `/tool` with default settings do not.
  - With `=1` they do.
- **Bounded input.**
  - A 20-record backlog of 1000-token messages is summarized one bounded
    chunk per call, and the cursor advances to the last included record.
  - Repeated calls drain the backlog.
  - An oversized single record is clipped in the prompt only.
- **Regression.** The existing working-memory, loop-runner, chat/tool backend
  and web tests pass. Tests that pinned default-off are updated deliberately
  and noted in the plan.
- **Mutation checks.**
  - Drop the budget check and watch the tail test go red.
  - Remove the input cap and watch the bounded-input test go red.
