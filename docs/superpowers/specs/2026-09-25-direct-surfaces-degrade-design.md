# /chat and /tool degrade when the model is unavailable: design

## Problem

#653 made `/api/agent` degrade to a search-only answer when the model is
unavailable (`ModelUnavailableError` in `src/context/models.py`). The direct
surfaces still fail with an empty answer:

- `/chat/send-chat-message`
  (`src/internal/servers/query_and_chat/chat_backend.py`):
  `ChatMessageResponse(answer="", error=str(exc))`, or an SSE `error` event.
- `/tool/send-tool-message` (`tool_backend.py`):
  `ToolAgentMessageResponse(answer="", error=str(exc))`, or an SSE `error`
  event.

Both answer with the **local** model, through `OpenAIServerManager`, which
now raises `ModelUnavailableError` on a connect error, a timeout or an open
circuit. Or they use `LocalServerManager`, which does not raise it.

## Decision (approved by the user)

### `/tool/send-tool-message`: corpus-only, search-only answer

- **The fallback.** The surface catches `ModelUnavailableError` and runs the
  same search-only fallback `/api/agent` uses: `_auto_search_pipeline` in
  `src/internal/servers/web/app.py`. It passes:
  - `llm=None`;
  - `source_provider="retrieval"`, which is corpus-only, so an outage never
    sends the query to SerpAPI or the browser;
  - the surface's `search_url`;
  - `filters=SearchFilters(access_acl=capabilities.access_acl)`, the
    filters the tool run itself uses;
  - `history`, `top_k` = the surface's existing default,
    `extra={"route_degraded": "model_unavailable"}`.
- **Importing it.** Import it lazily, as the module already does for
  `tool_agent_runner`, to avoid the import cycle. If the import cycle makes
  that impossible, factor the smallest shared helper out of `app.py` and
  record it in the plan.
- **The response.** `ToolAgentMessageResponse` gains `degraded: str | None =
  None` (additive), set to `"model_unavailable"`. `answer` is the search-only
  answer, `error` is None, `tool_calls=[]`, and `num_turns=0`. The assistant
  turn **is persisted**, because it is real content.
- **Streaming.** Emit `{"type":"answer","text":...}` and then
  `{"type":"done", ..., "degraded":"model_unavailable"}` in place of the
  `error` event.
- **The answer text (ruling made during review).** The shared search-only
  text reports a count and cites `[Dn]` for a Sources panel. `/tool` has
  neither the panel nor a documents field, and the answer is saved to
  history. So on `/tool` the degraded answer lists the documents found,
  each with its title, URL and a snippet of at most 200 characters, under
  one line saying the tool model is unavailable. When nothing is found it
  says so.
- **If the fallback itself raises,** the surface keeps today's error
  behavior: `answer=""` plus `error`, or an SSE error.

### `/chat/send-chat-message`: a clear unavailable message

- **Why a message.** Plain chat does no retrieval, so it gets no search
  answer. Catching `ModelUnavailableError` returns `answer =
  CHAT_MODEL_UNAVAILABLE_MESSAGE`, a module constant: "The chat model is
  temporarily unavailable. Please try again in a moment."
- **The response.** `ChatMessageResponse` gains `degraded: str | None = None`,
  set to `"model_unavailable"`, with `error` None.
- **Not persisted.** The message is **not persisted** as an assistant turn:
  it would pollute later context and summaries. The user message is already
  stored, exactly as today's error path leaves it.
- **Streaming.** Emit `answer` with the message, then `done` with `degraded`,
  in place of `error`. Tokens already streamed before the failure are
  followed by the message. This is acceptable, because a connect error or an
  open circuit fails before any token.

### Unchanged

- Any other exception keeps `answer=""` plus `error`, or an SSE `error`.
- The 400 for "no local model configured" stays, including its existing
  wording.
- The `/api/agent` behavior is unchanged.

## Out of scope

- Degrading on `LocalServerManager` failures, which never raise this error.
- Extractive answers.
- Changing the no-local-model 400.

## Testing

**`/tool`**, with a manager whose `generate` raises `ModelUnavailableError`,
or by patching `_run_tool_agent` to raise it:
- non-stream: 200, the search-only answer, `degraded == "model_unavailable"`,
  `error` None, and the assistant turn persisted;
- `_auto_search_pipeline` is called with `source_provider="retrieval"`,
  `llm=None`, and the capabilities ACL;
- streaming: `answer` and `done` events carry `degraded`, with no `error`
  event;
- a `ValueError` keeps `answer=""` plus `error`;
- a failing fallback keeps `answer=""` plus `error`;
- ACL: a private document is filtered through the real fallback, with the
  corpus rows stubbed via `SearchClient`.

**`/chat`**:
- non-stream: the unavailable message, `degraded`, `error` None, and **no**
  assistant turn persisted, while the user turn is;
- streaming: `answer` and `done` events with `degraded`;
- a `ValueError` keeps `answer=""` plus `error`.

**Mutation checks:**
- Remove each new `except` arm and watch its tests go red.
- Persist the chat message and watch the no-persist test go red.
