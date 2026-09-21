# A WebSocket control channel for agent sessions

## Goal

Add a bidirectional WebSocket control channel for agent sessions, reusing the
existing single-use WS token authentication and extending the existing approval
workflow onto that channel, without changing agent execution semantics and
without adding audio.

This is a transport, not a product. It is explicitly *not* a Realtime API.

## Why this is small

An investigation into what it would take to support a Realtime-API-shaped
protocol found the repo already holds two thirds of this feature, unused:

**Unused WebSocket authentication.** `src/internal/servers/redis/redis_pool.py`
implements `store_ws_token` and `retrieve_ws_token_data` — short-lived
(`WS_TOKEN_TTL_SECONDS = 60`) single-use tokens, consumed atomically with
`GETDEL`, rate-limited per user (`WS_TOKEN_RATE_LIMIT_MAX = 10` per 60s window).
It has **zero callers** in `src/` or `tests/`. `auth_check.py:192` likewise
carries a `WebSocketRoute` branch that can never fire.

**A working approval/resume state machine.** `ToolApprovalBroker.request()`
parks an `asyncio` future and calls an `on_registered` hook; `decide()` resolves
it, with a complete error vocabulary already mapped to HTTP
(`ApprovalNotFound` → 404, `ApprovalForbidden` → 403, `ApprovalConflict` → 409,
`ApprovalExpired` → 410). Both methods are already transport-agnostic: `decide`
takes `(approval_id, owner_user_id, decision)` and knows nothing about HTTP.

**And the event production is already transport-neutral.** This is the finding
that makes the work contained. In `/api/agent/stream`, the callbacks
(`on_turn`, `on_claim`, `on_trace`, `on_approval`) push **plain dicts** into a
bounded `asyncio.Queue(maxsize=100)`; `sse_frame()` is applied only at the point
of yield. Nothing upstream of that line knows it is producing SSE.

So the work is not "build a streaming protocol". It is: lift the run + queue +
callbacks out of the endpoint closure into a transport-neutral driver, then
give it a second consumer.

## Event vocabulary (v1)

**client → server:** `session.start`, `approval.submit`, `session.cancel`,
`ping`

**server → client:** `session.started`, `progress`, `claim`, `trace`,
`approval_required`, `answer`, `error`, `done`, `pong`

Seven of the nine already exist verbatim as SSE `type` values; only
`session.started` and `pong` are new, and `token` is deliberately excluded.

**Corrected during implementation: `tool_call` is not one of them.** This spec
first listed it, and the agent stream never emits such an event -- tool calls
are reported inside `done.tool_calls`. A separate `tool_call` event exists only
on `/tool/send-tool-message`, a different surface. Listing it would have been
the exact failure this document warns about two paragraphs below, committed in
the act of warning about it.

`ws_channel.SERVER_EVENTS` now holds the list, and a test asserts it equals
what the three modules actually emit -- in both directions, so neither an
undocumented event nor a documented phantom survives.

**No `token` event.** The grounded agent's meaningful streaming unit is the
claim — `/api/agent/stream` emits `claim` because the answer is the join of
verified claims. Adding `token` to resemble a realtime protocol would advertise
a granularity this path does not have. (`token` does exist, but only on
`/chat/send-chat-message`, which is the ungrounded local-model path. It stays
there.)

**No `session.update`.** There is no mutable mid-flight setting to change:
configuration arrives as `AgentExperienceRequest` and is fixed when the run
starts. Adding the event before the setting exists would be an advertised
capability with nothing behind it — a pattern this repo has been burned by
three times (`OpenAIEmbedder`, the search NDJSON branch, and the very WS
plumbing this spec revives).

## The socket does not own the run

Each run gets a `run_id`. The socket is its transport and control channel, not
its owner.

```
  session.start ──► RunDriver(run_id)  ──► _run_agent_impl task
                         │
                    asyncio.Queue           (already exists, bounded at 100)
                         │
  socket  ◄──────────────┘ send_json        SSE ◄── sse_frame  (same queue)
```

Keeping `socket lifetime == model execution lifetime` apart matters immediately
for cancellation, disconnect handling, and observability, and it is what lets
SSE and WebSocket share one driver instead of two copies of the run loop.

A v1 socket carries exactly one run. That is a restriction on the transport,
not on the model: `run_id` is on every event from day one, so multiplexing later
does not change the wire format.

## Authentication

```
POST /api/agent/ws-token        (authenticated, existing HTTP auth)
   └─► store_ws_token(token, user_id)      60s TTL, 10/min per user

WS /api/agent/ws?token=...
   └─► retrieve_ws_token_data(token)       GETDEL → single-use
       └─► accept, bind user_id to the connection
```

This gives the dead Redis code its first caller and makes replay protection
explicit rather than theoretical.

### Two consequences that must be handled, not assumed

**1. The route auditor will reject the socket.** `check_router_auth` *raises
`RuntimeError`* on any unguarded route, and computes:

```python
guarded = isinstance(route, APIRoute) and (
    _has_auth_dependency(route.dependant) or _has_inline_guard(route.endpoint)
)
```

A `WebSocketRoute` is never an `APIRoute`, so a WS route is classified unguarded
and **the app fails to start**. Declaring it in `PUBLIC_ENDPOINT_SPECS` would
make the audit pass by lying — the endpoint is token-authenticated, not public.

The honest fix is to teach the auditor that a `WebSocketRoute` can be guarded
inline. `_has_inline_guard` already works on any callable; only the `isinstance`
gate excludes it. That is what actually makes the dormant `WebSocketRoute`
branch reachable *and* enforcing, rather than merely reachable.

**2. Redis becomes a dependency of this transport.** `app.py` touches Redis
nowhere today, and the documented three-process dev stack does not run it. So
`/api/agent/ws-token` must return **503** when Redis is unavailable, and the
socket must refuse the connection — never fall back to an in-process token
store, which would break the single-use guarantee across workers and re-create
the plumbing this spec exists to give a caller.

SSE stays Redis-free and unaffected. The WebSocket transport is opt-in, and
"unavailable" is a correct state for it.

## Approvals: one broker, two transports

```
today                              added
─────                              ─────
SSE ── approval_required ─► client  WS ── approval_required ─► client
HTTP POST /approvals/{id} ─┐        WS ◄── approval.submit ────┘
                           ▼                   │
                    ToolApprovalBroker  ◄───────┘
                           ▼
                     agent resumes
```

The WS dispatcher calls `broker.decide(approval_id, user_id, decision)` — the
same method the HTTP endpoint calls. No second state machine, and the
403/404/409/410 vocabulary is preserved by mapping those same four exceptions
onto `error` events with a `code` field, so the two transports cannot drift into
subtly different behaviour.

`on_approval` is currently passed only when `auth_user is not None`. The socket
is always authenticated, so approvals are always available on it.

## Non-goals

Audio, STT, TTS, WebRTC; `session.update` or any config mutation; reconnection
and event replay; multiple runs per socket; and **replacing the SSE endpoints**.

SSE keeps working unchanged. WebSocket proves itself as an additional transport
first; whether carrying both is justified is a later decision made on evidence.

## Acceptance criteria

1. `POST /api/agent/ws-token` mints a token for an authenticated caller and
   401s for an anonymous one.
2. A token authenticates exactly once; the second use is rejected (`GETDEL`).
3. A token past `WS_TOKEN_TTL_SECONDS` is rejected.
4. Exceeding `WS_TOKEN_RATE_LIMIT_MAX` in the window returns 429.
5. A socket with a missing or invalid token is closed without starting a run.
6. `session.start` starts a run and replies `session.started` carrying `run_id`.
7. `progress`, `claim`, `trace`, `tool_call`, `answer`, `done` all cross the
   socket with the same payload shape they have on SSE.
8. `approval_required` → `approval.submit` resolves **the same broker future**,
   and the run resumes.
9. Approval errors map to `error` events preserving the 403/404/409/410
   distinctions.
10. `session.cancel` cancels the run and emits a terminal event.
11. Terminal `done`/`error` cleans up the run and its queue.
12. A mid-run disconnect leaks no tasks or futures (assert via
    `asyncio.all_tasks()`, as `test_sse_streaming.py` already does).
13. `check_router_auth` passes with the socket registered, and still raises for
    a genuinely unguarded WebSocket route.
14. `/api/agent/ws-token` returns 503 when Redis is unavailable; SSE is
    unaffected.
15. Existing SSE and HTTP-approval behaviour is unchanged — their tests pass
    untouched.

## Risks

**The driver extraction is the risky part, not the socket.** Lifting the run
loop out of `stream_agent`'s closure touches a working, well-tested path
(13 tests in `test_sse_streaming.py`, including a thread-safety hop and
backpressure drop-counting). It must be behaviour-preserving: SSE tests change
not at all, and are the guard.

**Claims arrive from a worker thread.** `on_claim` hops via
`loop.call_soon_threadsafe` because `AgenticRAGLoop.run` offloads generation
with `asyncio.to_thread` (#547). Any transport consuming the queue inherits that
requirement.

**Backpressure semantics must carry over.** The queue is bounded at 100 and
drops trace and claim events with counters rather than blocking generation. The
socket must not quietly reintroduce unbounded buffering.
