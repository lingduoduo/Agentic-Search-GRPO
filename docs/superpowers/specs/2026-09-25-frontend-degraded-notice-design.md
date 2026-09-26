# Frontend: show degraded answers on /chat and /tools: design

## Problem

#654 made `/chat/send-chat-message` and `/tool/send-tool-message` degrade when
the model is unavailable. Both return `degraded: "model_unavailable"` in the
JSON response and in the SSE `done` event. The frontend ignores the field,
so a degraded answer looks exactly like a normal one:

- in `/tools`, a search-only list of documents;
- in `/chat`, the "model temporarily unavailable" notice.

AssistPage already shows `route_degraded` with the `route-pill--degraded`
styling.

## Decision (approved by the user)

- **Types.** In `web/src/types.ts`, the chat and tool message response types,
  and their stream `done` event types, gain `degraded?: string | null`.
- **Views.** `ChatView` and `ToolAgentView` (`web/src/components/`) capture
  `degraded` from:
  - the non-stream response;
  - the stream's `done` event.

  Each view attaches it to the assistant message it renders.
- **The notice.** A message whose `degraded` is set renders a small notice
  next to the answer: `⚠ Model unavailable — degraded answer`, with the
  `title` attribute set to the raw reason, for example `model_unavailable`.
  - It re-uses the existing degraded styling (the `route-pill--degraded`
    colours). If that class does not fit the message layout, a minimal new
    class goes next to it in the existing stylesheet.
  - It has an accessible role (`role="status"`) or equivalent text.
- **Unchanged.** Rendering is unchanged when `degraded` is absent or null. No
  backend change.

## Testing

Use the existing frontend unit tests (`npm run test:unit`, the same runner
and patterns as the existing component tests):

- **ChatView:**
  - a non-stream response with `degraded` renders the notice;
  - without `degraded`, no notice;
  - a stream whose `done` carries `degraded` renders the notice.
- **ToolAgentView:** the same three cases.
- **Type check:** `npm run typecheck` passes.
- **Mutation check:** remove the notice render and watch the tests go red.
