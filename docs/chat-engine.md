# Chat engine

[← Back to README](../README.md)

This guide covers the chat agent: conversational, retrieval-grounded answering.
For the authoritative deep dives, see [API request routing](request-routing.md)
and [Frontend development](frontend.md).

## Capabilities

- **Grounded conversation** — retrieval-grounded synthesis with citations, over
  either a single answer call (`chat_once`) or the iterative `AgenticRAGLoop`
  (`chat_loop`: query decomposition, HyDE, iterative retrieval, then synthesis).
- **Multi-turn memory** — each turn carries the newest session messages that
  fit an estimated-token budget (`AGENTIC_SEARCH_MEMORY_HISTORY_TOKENS`, default
  `2500`, still capped at 40 messages). Turns that fall off that tail are
  summarized in the background by the configured LLM client and prepended to
  the next turn as one system message. Summarization is on by default for
  `/api/agent` only; `/chat` and `/tool` answer with the local model and need
  `AGENTIC_SEARCH_MEMORY_COMPRESSION=1`. Deleting a chat session also drops its
  summary. See [Configuration](configuration.md#application-and-authentication).
- **React chat UI** — streaming responses, source inspection, and observability
  surfaces in the development frontend.

## Routing into chat

With `mode` omitted, `/api/agent` classifies each request as `chat`, `search`, or
`tool`. Conversational or generative requests route to `chat`, which runs the
grounded `AgenticRAGLoop`; `chat` is also the fallback when a `tool` route has no
usable result. Chat requires an LLM client for synthesis; if that model is
unavailable at runtime (connection error or HTTP 5xx/429), the request degrades
to a search-only answer marked `route_degraded: "model_unavailable"` instead of
failing with 502. See
[API request routing](request-routing.md) for the full decision order, explicit
`chat_once` / `chat_loop` modes, and response metadata.

## Direct chat surface (`/chat/send-chat-message`)

Beyond the `/api/agent` auto-router, chat has a direct endpoint that calls the
local model with no retrieval and no tools:

- `POST /chat/send-chat-message` — runs `PlainGenerationLoop` over the session
  history + the new message, streaming SSE `token` chunks, then the full
  `answer`, then `done` (`error` on failure). Requires a local model
  (`SEARCH_AGENT_MODEL` / `SEARCH_AGENT_SERVER_URL`); returns **400** otherwise.
  `stream:false` returns one JSON `{ session_id, answer }`. The runner is
  `src/internal/servers/web/plain_chat_runner.py`.
- If the local model is unavailable mid-request (connection error, timeout, or
  an open circuit), the answer is the fixed notice "The chat model is
  temporarily unavailable. Please try again in a moment." and the response (or
  the `done` event) carries `degraded: "model_unavailable"`. The notice is not
  saved to the session; the user turn is.

In the web UI, the **Chat** tab drives this endpoint and renders a running
transcript of the session's turns (accumulated client-side as you chat). A
degraded turn shows `⚠ Model unavailable — degraded answer` under the answer.
