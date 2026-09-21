# Make the SSE surface consistent

## Goal

Three endpoints stream Server-Sent Events and a fourth streams something else.
Make the three agree on how a response is built, make the fourth stop claiming
to be one of them, and put the agreement somewhere a future endpoint inherits
it rather than has to remember it.

## The surface as found

| Endpoint | Media type | Anti-buffer headers | Driven as a stream |
|---|---|---|---|
| `POST /api/agent/stream` | `text/event-stream` | both | yes |
| `POST /chat/send-chat-message` | `text/event-stream` | **neither** | yes |
| `POST /tool/send-tool-message` | `text/event-stream` | **neither** | yes |
| `POST /search/send-search-message` | `application/x-ndjson` | n/a | **no** |

## 1. Two of three SSE endpoints could not stream through a proxy

`/api/agent/stream` returned `Cache-Control: no-cache` and
`X-Accel-Buffering: no`. The other two returned a bare
`StreamingResponse(_gen(), media_type="text/event-stream")`.

nginx buffers proxied responses by default; `X-Accel-Buffering: no` is what
turns that off. Without it, the whole stream is delivered the moment the
generator finishes — the response is still correct, just not streamed, which is
the entire feature.

What makes this worth guarding rather than just fixing: **no local environment
reproduces it.** Vite's dev proxy does not buffer, and FastAPI's `TestClient`
does not proxy at all, so both endpoints looked correct in dev and in CI. The
only signal available was that a third endpoint, written by someone who had
thought about it, disagreed with them.

## 2. The search stream was neither SSE nor reachable

Its docstring read "a newline-delimited JSON stream (SSE-compatible)". NDJSON
is not SSE-compatible: no `data:` prefix, no blank-line terminator. An SSE
reader pointed at it parses nothing. With three genuine SSE siblings next door,
that is worse than a merely inaccurate comment.

The branch is also unreachable in practice. `stream: bool = False`; the web app
hardcodes `stream: false`; the MCP retrieval client hardcodes `"stream": False`.
No caller anywhere sets it true, and all four existing tests passed
`stream=False`, so nothing executed the generator.

**It is kept, not deleted.** Whether the search surface should stream is a
product decision. This only makes the current state honest and covered, so that
decision can be taken on evidence later.

## 3. The frame encoder existed three times on each side

`_sse` was a two-line closure nested inside each of the three handlers. On the
frontend, the same read/decode/split/`data:`-strip loop appeared three times in
`api.ts`, byte-identical apart from the error string and the TypeScript event
type.

That duplication is what allowed (1): with three independent copies of "return
an SSE response", two could drift from the third and nothing noticed.

## Architecture

`src/internal/servers/sse.py` owns framing and response construction:
`SSE_HEADERS`, `sse_frame(data)` and `sse_response(generator)`. It sits beside
`_auth.py` because `app.py` is not under `query_and_chat/` and both need it.

`web/src/api.ts` gains `readSSE<T>(body)`, one generic async generator the three
streaming functions delegate to with `yield*`.

The invariant is then structural, and stronger than the defect that prompted it:
**no module other than `sse.py` may construct a `text/event-stream`
`StreamingResponse`.** Forgetting the headers is no longer possible, because
constructing the response at all is no longer possible outside the helper.

## Tests

- `test_sse_response_contract.py` walks `src/` with `ast` and fails if any file
  other than `sse.py` builds an event-stream response. It carries two
  guard-the-guard tests: one that the helper is still found (an invariant over
  an empty set passes for free) and one that all three endpoint modules still
  call `sse_response`, so the rule cannot be satisfied by not streaming.
  It also asserts a real constructed response carries the headers — declaring
  them in a constant is not the same as wiring them.
- Each endpoint keeps a test asserting both headers on a genuinely streamed
  response, so the guarantee is checked at the wire and not only in the AST.
- Four new tests drive `search`'s `stream=True`: queries-then-docs, the second
  queries packet after expansion, the error packet, and one pinning NDJSON as
  *not* SSE — the mistake the docstring made.

Mutation-checked: removing the headers from `chat_backend` turns both the
structural test and that endpoint's own header test red; making the search
generator skip its docs packet turns two of the new stream tests red.

## Deliberately not done

**No heartbeat.** None of the three streams emits a keepalive, so an idle
connection can be dropped by an intermediary. Adding one changes wire behaviour
and needs an interval and a client that ignores comment frames; `sse.py` is now
the one place it would go. **No `id:`/`Last-Event-ID`**, so no resumption.
**No `event:` field** — every event is `data:` with a `type` key in the JSON,
which the frontend switches on.

`EventSource` remains unusable here and that is not a defect: these endpoints
are POST and `EventSource` is GET-only. The manual reader is the correct
approach, which is worth stating so it is not "fixed" later.
