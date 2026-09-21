# SSE Streaming Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the three SSE endpoints build their responses the same way, stop the fourth stream from claiming to be SSE, and move the agreement into one place a future endpoint inherits.

**Architecture:** A shared `src/internal/servers/sse.py` owning `SSE_HEADERS`, `sse_frame` and `sse_response`; a matching `readSSE<T>` in `web/src/api.ts`; and a structural test asserting nothing else may construct an event-stream response.

**Tech Stack:** Python 3, FastAPI, pytest, `ast`, TypeScript, Vitest.

**Spec:** `docs/superpowers/specs/2026-09-21-sse-streaming-consistency-design.md`

## Global Constraints

- **Do not delete the search NDJSON branch.** It is unreachable, but whether the search surface should stream is a product decision. Make it honest and covered instead.
- **Do not add a heartbeat, `id:`, or `event:`.** Each changes wire behaviour and needs a client-side counterpart. `sse.py` is where they would go later.
- **Do not replace the manual reader with `EventSource`.** These endpoints are POST; `EventSource` is GET-only. The manual reader is correct.
- Remove only the imports this work orphans (`_json`, `StreamingResponse` in the two backends), not pre-existing dead code.
- Run `ruff check . --fix && ruff format .`, plus `npm run typecheck` and `npm test`, before each commit.

---

### Task 1: Stop proxy buffering on chat and tool

**Files:**
- Modify: `src/internal/servers/query_and_chat/chat_backend.py`, `tool_backend.py`
- Test: `tests/unit/test_chat_backend.py`, `tests/unit/test_tool_backend.py`, and new `tests/unit/servers/web/test_sse_response_contract.py`

- [x] **Step 1: Add the two headers**

Give both `StreamingResponse` calls the `headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}` that `/api/agent/stream` already had.

- [x] **Step 2: Assert them on a real streamed response**

One test per endpoint, reusing each file's existing `_make_app(with_model=True)` helper and stubbing the runner, asserting both headers inside the `client.stream(...)` block.

- [x] **Step 3: Guard future endpoints structurally**

An `ast` sweep of `src/` for `StreamingResponse(media_type="text/event-stream")`, asserting each carries both headers — plus a guard-the-guard test that the sweep still finds the known endpoints, since an invariant over an empty set passes.

**Verify:** mutation — remove the headers from `chat_backend`; both that endpoint's test and the sweep go red.

---

### Task 2: Make the search stream honest and covered

**Files:**
- Modify: `src/internal/servers/query_and_chat/search_backend.py`
- Test: `tests/unit/servers/query_and_chat/test_search_backend.py`

- [x] **Step 1: Correct the docstring**

It claimed the NDJSON was "SSE-compatible". Say plainly that it is not (no `data:` prefix, no blank-line terminator), that the three siblings are SSE and these are not interchangeable, and that no shipped caller sets `stream=True`.

- [x] **Step 2: Cover the branch**

Add a `_stream_lines` helper that drives `stream=True` and returns decoded frames, then four tests: queries-then-docs, the second queries packet after expansion, the error packet, and one asserting no frame starts with `data:` — pinning the distinction the docstring got wrong.

**Verify:** mutation — make the generator skip its docs packet; two of the four go red.

---

### Task 3: One frame encoder per side

**Files:**
- Create: `src/internal/servers/sse.py`
- Modify: `src/internal/servers/web/app.py`, `chat_backend.py`, `tool_backend.py`, `web/src/api.ts`
- Test: rewrite `tests/unit/servers/web/test_sse_response_contract.py`

- [x] **Step 1: The shared module**

`SSE_HEADERS`, `sse_frame(data)` and `sse_response(generator)`, beside `_auth.py` — `app.py` is not under `query_and_chat/` and both need it.

- [x] **Step 2: Repoint all three endpoints**

Delete each local `_sse`, rename calls to `sse_frame`, return `sse_response(...)`. Then drop the imports this orphans: `_json` in all three, and `StreamingResponse` in the two backends — `app.py` still needs it for a return annotation.

- [x] **Step 3: `readSSE<T>` on the frontend**

One generic async generator; the three streaming functions become `yield* readSSE<T>(response.body)`. Document that the server never emits multi-line `data:`, `event:`, `id:` or `retry:`, so the reader does not handle them.

- [x] **Step 4: Strengthen the invariant**

The rule is no longer "every call site remembered the headers" but "only `sse.py` builds an event-stream response". Add a test that all three endpoint modules still call `sse_response`, so the rule cannot be satisfied by not streaming, and one that a constructed response really carries the headers.

**Verify:** `pytest -q` → 4325 passed, 1 skipped; `npm run typecheck` clean; `npm test` → 231 passed.
