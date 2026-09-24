# Tool-call fallback: retry, degrade, escalate — design

## Problem

A tool call that fails today has no coordinated recovery (investigation:
`.planning/2026-09-24-tool-failure-investigation/findings.md`, verified against
the code):

- **Failures look like successes.** `ToolRegistry.invoke` drops the execution
  metadata and returns no errors after executing, and `ToolAgentLoop._call_tool`
  marks every such return `COMPLETED`. Public-data `guarded` turns every exception
  into a `{"error": ...}` JSON string, and MCP `_result_text` flattens `isError`
  into `"Error: ..."` text, so both are recorded as successful calls.
- **Retry is provider-specific.** Public-data GETs retry 429/502/503/504 three
  times inside `_fetch`; nothing else retries, and the loop never fills
  `ToolExecutionResult.retry_count`. Exceptions go back to the model, which may
  retry, loop, switch tools or stop.
- **No degradation contract and no human handoff.** A failed tool leaves the model
  to improvise; `_request_approval` is a pre-execution permission gate, not a
  failure handoff.

This spec adds one recovery policy with three ordered levels — **retry →
degrade → escalate to the user** — owned by the tool loop, so both tool entry
points (`/api/agent`'s TOOL route and `/tool/send-tool-message`) get it through the
shared runner.

**Decisions made with the user:** the human is the **end user in the chat**;
degrading a read-only tool means **continuing without it**; the policy is
**owned by the loop** (not the registry, not the model).

**Out of scope:** operator queues, declared fallback-tool mappings, durable
resumption across requests, idempotency keys or write reconciliation, circuit
breakers, and changes to `SearchAgentLoop`/`AgenticRAGLoop` (they do not call
registry tools).

## 1. Failure classification

**`ToolFailure`** (frozen dataclass, `src/internal/tools/base.py`):

| field | meaning |
|---|---|
| `category` | `transient`, `permanent`, `invalid_input`, `not_found`, `unknown` |
| `message` | short, fixed-vocabulary description safe to show the user |
| `retry_after` | seconds from a provider `Retry-After`, else `None` |
| `provider_attempts` | attempts the provider already made itself (0 = none) |

**`ToolErrorText(str)`** carries a `.failure: ToolFailure`. Adapters return it
where they return error text today, so every other `registry.invoke` caller
(`tool_evidence`, memory service, `/api/tools`, the MCP server export) sees
byte-identical text; only the loop inspects `.failure`.

Adapter classification — nothing ever searches text for "error":

- **Public-data `guarded`.** `PublicDataError` gains optional `status`,
  `attempts` and `retry_after` attributes (its message constructor stays
  compatible; `_fetch` sets them). 429/502/503/504, timeouts and transport
  errors → `transient`; other 4xx/5xx → `permanent`; any other exception →
  `unknown`. `provider_attempts` = the attempts `_fetch` made (3 for a GET that
  exhausted its retries, 1 for a POST).
- **MCP `_result_text`.** `isError` → `unknown` (the protocol does not say why).

**`FunctionTool.execute`** returns `{"failure": result.failure}` as its metadata
when the function returned a `ToolErrorText`.

**`ToolRegistry.invoke_detailed(name, arguments, *, validate=True) ->
ToolInvocation(response, raw, errors, failure)`** shares lookup, validation and
the create/execute/release path with `invoke`. It maps a missing tool to
`not_found`, schema errors to `invalid_input`, the execute metadata's `failure`
through, and a raised exception to a failure (`asyncio.TimeoutError`,
`ConnectionError`, `aiohttp.ClientError` → `transient`; anything else →
`unknown`). `asyncio.CancelledError` propagates. `invoke()` keeps its signature
and 3-tuple.

**`ToolAgentLoop._call_tool`** calls `invoke_detailed`; any failure yields
`TaskStatus.FAILED` with `error_code = failure.category`.

## 2. The recovery policy

`src/agents/tool/recovery.py`:

```python
class Action(Enum): RETRY, UNAVAILABLE, ESCALATE, FEED_BACK

@dataclass(frozen=True)
class Decision:
    action: Action
    delay: float = 0.0

@dataclass(frozen=True)
class RecoveryPolicy:
    max_retries: int = 2          # extra attempts after the first
    backoff: tuple[float, ...] = (0.5, 1.0)
    retry_after_cap: float = 4.0
    retry_budget: float = 10.0    # seconds of retry sleep per run

    def decide(self, failure, effect, retries_so_far, budget_left, rng) -> Decision
```

`decide` is pure (the random source is injected):

| failure | `READ_ONLY` | `SIDE_EFFECTING` / `UNSPECIFIED` |
|---|---|---|
| `invalid_input`, `not_found` | `FEED_BACK` | `FEED_BACK` |
| `transient`, `provider_attempts <= 1`, retries left, budget left | `RETRY` | `ESCALATE` |
| `transient` otherwise (provider already retried, or out of retries/budget) | `UNAVAILABLE` | `ESCALATE` |
| `permanent`, `unknown` | `UNAVAILABLE` | `ESCALATE` |

`RETRY` delay = `max(backoff[retries_so_far] * uniform(0.5, 1.5), min(retry_after, cap))`,
and a retry is only chosen when that delay fits the remaining budget. The
provider-already-retried rule makes the loop the single retry owner above
`_fetch`: attempts never multiply.

`FEED_BACK` keeps today's behaviour: the failure goes back to the model so it can
correct its arguments or tool name.

## 3. Retry and degrade in the loop

`ToolAgentLoop` keeps per-run state: `retry_budget_left`, an `unavailable: set[str]`
and the escalation count. Calls in one batch already run concurrently
(`asyncio.gather`); the state is read and updated only between awaits (budget
reserved before sleeping, escalation count incremented before awaiting the user),
so concurrent calls cannot overspend the budget or exceed the cap. Concurrent
escalations each get their own card.

Approval is unchanged and separate: a denied or expired approval stays `SKIPPED`,
is never retried, degraded or escalated, and a tool is only ever retried after its
approval was granted.

- **Retry.** `_call_tool` loops: invoke → on failure ask the policy → on
  `RETRY` sleep the delay (from the run budget) and invoke again. The final
  `ToolExecutionResult.retry_count` = retries made.
- **Degrade (`UNAVAILABLE`).** The tool joins `unavailable`; the model receives
  `{"status": "unavailable", "tool": ..., "error_code": ..., "note": "This tool is
  unavailable for the rest of this turn. Answer from what you have and say what is
  missing."}`. A later call to an unavailable tool in the same run is not executed:
  it returns `SKIPPED` with the same note. Other tools' results are untouched.
- **Visible outcome.** `AgentLoopOutput` gains `tool_recovery: dict | None`:
  `{"retries": int, "degraded": [tool names], "escalations": [records]}`, `None`
  when nothing needed recovery. The shared runner (`tool_agent_runner`) copies it
  into response metadata and, when `degraded` is non-empty, appends one fixed line
  to the answer: `Note: <tool[, tool]> was unavailable, so this answer may be
  incomplete.`

## 4. Escalation to the user

Triggered by `ESCALATE`: a `SIDE_EFFECTING`/`UNSPECIFIED` call failed with
anything other than `invalid_input`/`not_found`.

**Request.** `ToolEscalationRequest(escalation_id, tool_name, arguments,
category, message, attempts, created_at, expires_at)`, passed to a new loop
callback `on_escalation: Callable[[ToolEscalationRequest],
Awaitable[EscalationDecision]]`, alongside `on_approval`.
`EscalationDecision`: `RETRY`, `SKIP`, `CANCEL`, `EXPIRED`.

| decision | effect |
|---|---|
| `RETRY` | The user authorised a replay: invoke the same call once more. A new failure escalates again. |
| `SKIP` | Degrade: the tool joins `unavailable`, the model gets the unavailable note, the run continues. |
| `CANCEL` | Stop the run. Answer: `Stopped: <tool> failed (<category>); nothing further was attempted.` |
| `EXPIRED` | Timeout (120 s) or disconnect: stop the run as `unresolved`. |

At most **3 escalations per run**; a failure that would be the 4th stops the run
as `unresolved`. With **no `on_escalation` callback** (plain JSON `/api/agent`,
JSON `/tool/send-tool-message`, CLI), an escalation immediately stops the run as
`unresolved`.

**`unresolved` is never success.** Answer: `<tool> failed and was not retried;
the action may not have completed.` Metadata: `tool_recovery.needs_user: true` and
the escalation record (tool, sanitized arguments, category, attempts, decision).
The runner's existing empty-answer and TOOL→CHAT fallbacks must not replace a
`CANCEL`/`unresolved` answer.

**Transport.**

- **Broker.** The generic mechanics of `ToolApprovalBroker`
  (`src/internal/servers/web/tool_approval.py`) — pending map, owner check,
  timeout, counters, argument sanitization — move into a base class
  `DecisionBroker`. `ToolApprovalBroker` keeps its behaviour and API;
  `ToolEscalationBroker` is the second subclass. The app holds two separate
  instances; approval and escalation share mechanics, never state.
- **Events and endpoints.** SSE/WebSocket event `{"type": "escalation_required",
  "escalation": {...view...}}`; `POST /api/agent/escalations/{id}` with
  `{"decision": "retry" | "skip" | "cancel"}` (owner-checked like approvals: 403
  other owner, 404 unknown, 409 already decided); WebSocket message
  `submit_escalation`. `/tool/send-tool-message` SSE wires the same callback.
- **Frontend.** `ToolEscalationCard` (next to `ToolApprovalCard`) on the Assist and
  Tools pages: tool, sanitized arguments, category message, attempts, a warning that
  the action may or may not have happened, and Retry / Skip / Cancel buttons.

## Testing

- **Classification:** public-data 503 after retries (`transient`,
  `provider_attempts=3`), 404 (`permanent`), timeout (`transient`), non-HTTP
  exception (`unknown`); MCP `isError` (`unknown`); `ToolErrorText` equals its text;
  `invoke()` unchanged; `invoke_detailed` maps not-found, invalid args, raised
  exceptions; `CancelledError` propagates.
- **Policy:** every table row; budget and `retry_after` cap; provider-already-retried
  → no outer retry.
- **Loop:** transient read-only succeeds on retry (`retry_count=1`); exhausted →
  unavailable note and a second call short-circuited without executing; sibling
  results kept; degrade note appended; each escalation decision including expiry,
  no callback, and the 3-escalation cap; `CANCEL`/`unresolved` answers survive the
  runner's fallbacks.
- **Transport:** broker base keeps approval behaviour (existing approval tests pass
  unchanged); escalation endpoint ownership/404/409; SSE event emitted; WebSocket
  `submit_escalation`.
- **Frontend:** `ToolEscalationCard` renders the record and posts each decision.
- Every test mutation-checked; the unit suite passes with torch unimportable.
