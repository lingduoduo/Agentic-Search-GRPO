# WebSocket Control Channel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A bidirectional WebSocket control channel for agent sessions, reusing the unused single-use WS token auth and the existing `ToolApprovalBroker`, without changing agent execution semantics and without audio.

**Architecture:** Extract the existing run + bounded queue + callbacks from `stream_agent`'s closure into a transport-neutral `RunDriver`; give it a second consumer. SSE and WebSocket then share one run loop and one approval broker.

**Tech Stack:** Python 3.10+, FastAPI 0.136 / Starlette 1.0 (`WebSocket`, `WebSocketRoute`), `asyncio`, Redis (WS transport only), pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-websocket-control-channel-design.md`

## Global Constraints

- **Do not change SSE behaviour.** `tests/unit/servers/web/test_sse_streaming.py` (13 tests) must pass **unmodified** at every step. If a step requires editing an SSE test, the extraction was not behaviour-preserving — stop and reconsider.
- **One approval state machine.** The WS dispatcher calls the same `broker.decide(approval_id, owner_user_id, decision)`. Never add a parallel path, and preserve the 403/404/409/410 distinctions.
- **The socket does not own the run.** Every run has a `run_id`; every event carries it. One run per socket in v1, but the wire format must not assume it.
- **No `token` event, no `session.update`.** See the spec's reasoning — both would advertise a capability that does not exist.
- **No in-process fallback for WS tokens.** Redis unavailable means the transport is unavailable (503), not that single-use is quietly abandoned.
- **Do not touch audio, WebRTC, reconnection, replay, multiplexing, or the SSE endpoints.**
- Run `ruff check . --fix && ruff format .` before each commit. Verify on Python 3.10 as well as 3.12 (see `test_python_version_floor.py`).

---

### Task 1: Extract the run driver (no new behaviour)

**Files:**
- Create: `src/internal/servers/web/run_driver.py`
- Modify: `src/internal/servers/web/app.py`

**Interfaces:**
- Produces: a driver owning `run_id`, the bounded `asyncio.Queue(maxsize=100)`, the `on_turn`/`on_claim`/`on_trace`/`on_approval` callbacks, the drop counters, and the `_run_agent_impl` task. It yields **event dicts**; it must not know about SSE or WebSocket.

- [x] **Step 1: Lift the closure verbatim**

Move the queue, callbacks and task construction out of `stream_agent`. Keep `loop.call_soon_threadsafe` in `on_claim` — claims arrive from the `asyncio.to_thread` worker (#547) — and keep `put_nowait` + drop counters for trace/claim, which is the backpressure contract.

- [x] **Step 2: Repoint the SSE endpoint at it**

`stream_agent` becomes: build a driver, drain its events, `sse_frame` each one. The terminal `answer`/`done`/`error` framing stays in the endpoint for now.

**Verify:** `pytest tests/unit/servers/web/test_sse_streaming.py -q` → 13 passed, **file unmodified**. Then mutation-check the extraction: break the drop-counting and confirm a test reddens.

---

### Task 2: Mint WS tokens over HTTP

**Files:**
- Modify: `src/internal/servers/web/app.py`
- Test: new `tests/unit/servers/web/test_ws_token.py`

**Interfaces:**
- Produces: `POST /api/agent/ws-token` → `{"token": ...}`, authenticated with the same helper the approval endpoint uses (`_require_auth`).

- [x] **Step 1: The endpoint**

Generate a token (`secrets.token_urlsafe`), call `store_ws_token(token, user.id)`, return it. Map `WsTokenRateLimitExceeded` → **429**, and a Redis connection failure → **503**.

- [x] **Step 2: Tests**

401 anonymous; 200 authenticated; 429 past `WS_TOKEN_RATE_LIMIT_MAX`; 503 with Redis down. Use a fake Redis rather than requiring a live one — the unit suite must not gain a service dependency.

**Verify:** the four cases pass; `pytest -q` overall still green.

---

### Task 3: Teach the route auditor about guarded sockets

**Files:**
- Modify: `src/internal/servers/web/auth_check.py`
- Test: `tests/unit/servers/web/test_route_auth_enforcement.py`

- [x] **Step 1: Stop rejecting every WebSocket route**

`guarded` is `isinstance(route, APIRoute) and (...)`, so a `WebSocketRoute` is always unguarded and `check_router_auth` **raises**, meaning the app will not start. Allow a `WebSocketRoute` to be guarded by `_has_inline_guard(route.endpoint)`, which already works on any callable.

Do **not** add the socket to `PUBLIC_ENDPOINT_SPECS`. It is token-authenticated, not public; the allowlist would make the audit pass by lying.

- [x] **Step 2: Prove both directions**

A WebSocket route with an inline token guard passes. A WebSocket route **without** one still raises — otherwise this step has traded one dormant branch for a silent hole.

**Verify:** both tests pass; the existing unguarded-route test is untouched and still red-on-violation.

---

### Task 4: The socket and its event protocol

**Files:**
- Create: `src/internal/servers/web/ws_channel.py`
- Modify: `src/internal/servers/web/app.py`
- Test: new `tests/unit/servers/web/test_ws_channel.py`

**Interfaces:**
- Produces: `WS /api/agent/ws`; client events `session.start`, `approval.submit`, `session.cancel`, `ping`; server events `session.started`, `progress`, `claim`, `trace`, `tool_call`, `approval_required`, `answer`, `error`, `done`, `pong`.

- [x] **Step 1: Authenticate and accept**

Read `token` from the query string, `retrieve_ws_token_data` (GETDEL — single use), bind `user_id` to the connection, else close without starting anything. Reject a second use of the same token.

- [x] **Step 2: Dispatch client events**

A small table, not a chain of `if`s. Unknown `type` → `error` with a code, never a disconnect. Malformed JSON → `error`, not a traceback.

- [x] **Step 3: `session.start` → drive a run**

Build the Task-1 driver, reply `session.started` with `run_id`, then pump its events with `send_json`. Do not re-implement the run loop.

- [x] **Step 4: Approvals across the socket**

`approval_required` goes out from the same `on_approval` hook. `approval.submit` calls `broker.decide(...)`. Map `ApprovalNotFound`/`Forbidden`/`Conflict`/`Expired` onto `error` events carrying 404/403/409/410 as a `code`, so the two transports cannot drift.

- [x] **Step 5: Cancel, terminal cleanup, disconnect**

`session.cancel` cancels the run task. Terminal `done`/`error` drops the run and its queue. A mid-run disconnect cancels the run and leaves no orphan tasks or futures.

**Verify:** acceptance criteria 5–12 from the spec. Assert no task leak with `asyncio.all_tasks()`, the way `test_sse_streaming.py` already does.

---

### Task 5: Prove the two transports agree

**Files:**
- Test: `tests/unit/servers/web/test_ws_channel.py`

- [x] **Step 1: Same events, same shapes**

Run the same stubbed agent through SSE and through the socket; assert the event sequence and payload shapes match, modulo framing. This is the guard against the two transports drifting.

- [x] **Step 2: Same approval semantics**

Assert the four approval error cases produce the same distinctions on both transports.

**Verify:** `pytest -q` on 3.12 and on 3.10; `npm run typecheck` unaffected (no frontend change in this PR).

---

### Deliberately deferred

A frontend client for the socket. The backend must prove itself first, and the existing UI keeps working on SSE throughout — which is the whole point of not replacing it yet.


---

## What changed during implementation

Two deviations from this plan, both recorded because they are the kind a
reviewer should see rather than infer from the diff.

**Task 1 Step 2 said the terminal events stay in the endpoint.** They moved to
module-level `_terminal_events` / `_error_event` instead. Leaving them behind
would have forced the socket to rebuild the `answer`/`done` payload, which is
precisely the drift the spec exists to prevent; the parity test in Task 5 is
only meaningful because both transports go through one builder.

**Task 4 Step 1 put authentication inside `serve()`.** It moved into the route
body. `_has_inline_guard` scans an endpoint's own source, so a guard hidden
behind a dispatcher is invisible to the audit -- the app failed to start with
exactly the `RuntimeError` Task 3 predicted, which is the plan catching its own
mistake. Authenticating where the route can be read to do so is also plainer.

**One thing the spec missed entirely.** `_run_agent_impl` derives identity from
request headers, but the socket authenticates by token, so a WS run would have
executed anonymously -- wrong attribution, and wrong ACL, since `group_ids`
drives it. `_run_agent_impl` now takes an optional `caller`, and `_ws_caller`
rebuilds the identity from the store rather than the token, which carries only
a subject. Reading groups at connect time is also strictly better than trusting
the mint: it keeps a socket's access equal to the same user's over SSE after
their groups change.

**A false completion, caught late.** Every checkbox here was ticked with one
blanket replace, which marked Task 4 Step 4 and Task 5 Step 2 -- approvals over
the socket -- as done when they had code but no tests. They have tests now, and
both redden when the broker call is bypassed. Tick boxes one at a time; a
blanket replace cannot distinguish written from verified.

**Verification.** 4347 passed on 3.12 (4327 before, +20). The 3.10 interpreter
available here cannot run the web suite -- `boto3` is missing from that env,
pre-existing and unrelated -- so the new modules were checked with `py_compile`
on 3.10 and by the version-floor guard instead. That is weaker than running
them, and worth saying plainly.
