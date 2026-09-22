# Remove the orchestration vocabulary no loop maintains

## Goal

`src/agents/core/state.py` described a planning, routing and task-scheduling
layer that this repo does not have. Cut it back to the state the loops actually
keep, and stop `from src import RouteDecision` resolving to the dead half of a
name collision.

## What was there

The module exported sixteen types. A reachability pass over every real importer
(AST, resolving through the three `__init__.py` re-export chains rather than
grepping names, because several of these names exist twice in the tree) splits
them cleanly:

| Reachability | Names |
|---|---|
| Live | `AgentState`, `TaskStatus`, `UserRequest`, `ToolExecutionResult`, `PerformanceMetrics`, `Retriever`, `Citation` |
| **Re-export only** | `Plan`, `PlanStep`, `TaskNode`, `TaskType`, `ToolType`, `RouteDecision`, `RetrievedDocument`, `ToolCall`, `ToolResult` |

Nine types with no importer outside `__init__.py`. Four of them survived earlier
audits because a same-named class *is* live elsewhere and a grep cannot tell
them apart:

- `RouteDecision` — the live one is `src/internal/routing/route.py`
- `RetrievedDocument` — the live one is `grpo/generation`
- `ToolCall` — the live one is `src/internal/llm/models`
- `ToolResult` — the live one is the integration-test model

Alongside them, `AgentState` carried eight fields nothing ever wrote — `route`,
`retrieved_user_docs`, `retrieved_policy_docs`, `plan`, `tool_results`,
`draft_response`, `final_response`, `trace` — and two methods, `record_trace`
and `add_tool_result`, whose only callers in the entire tree were two lines of
`tests/unit/test_state_models.py`. `ToolExecutionResult.to_tool_result` was
test-only for the same reason: it existed to feed `add_tool_result`.

`TaskStatus` declared six members. Three are assigned anywhere: `COMPLETED`,
`FAILED`, `SKIPPED`. `PENDING` existed solely as `TaskNode`'s default field
value; `RUNNING` and `RETRYING` were never assigned at all, and together with
`ToolExecutionResult.retry_count` they advertised a scheduler and a retry path
that do not exist. Nothing in this repo queues or retries a tool —
`ToolRegistry.invoke` calls it inline and reports how it finished.

## Why it matters more than its size

This is not merely unused code. It is an API that documents behaviour the
system does not have. A reader who finds `AgentState.plan` and
`state.record_trace(...)` reasonably concludes there is a planner writing to it
and a trace being collected. Both are false, and nothing fails to reveal that,
because no production path ever touches either.

The `RouteDecision` collision is worse than unused: `from src import
RouteDecision` **returned the dead class**. Code that did this and expected the
router's decision type would typecheck, import cleanly, and be wrong.

## The change

`state.py` keeps the seven live types and drops the other nine, the eight
unwritten fields, `record_trace`, `add_tool_result` and `to_tool_result`.
`TaskStatus` keeps its three terminal outcomes. 269 → 150 lines.

The nine names are removed from all three re-export chains
(`src/__init__.py`, `src/agents/__init__.py`, `src/agents/core/__init__.py`).
`src.__all__` is computed from `globals()`, so it follows automatically.

**The `RouteDecision` export is removed, not re-pointed.** Re-pointing it at
`src/internal/routing/route.py` would keep `from src import RouteDecision`
working and return the right class — but this PR's whole thesis is that the
package should not advertise surface it does not maintain, and promoting an
internal routing type to the top-level API contradicts that. Removal makes the
import fail loudly instead, and callers name the module they mean.

### Kept deliberately

`ToolExecutionResult.success` has no caller in `src/` once `to_tool_result` is
gone. It stays. The cut here is "types and fields implying an orchestration
layer", not "every accessor without a current caller" — `success` is a trivial
derived property on a live class and implies nothing that is not true. Drawing
the line further would make this a churn PR.

`retry_count` stays on `ToolExecutionResult` for the same reason, though it is
the weaker case: it is a plain int field, not a state machine.

## Testing

Deletions do not get a RED phase, so the guards are pinned by mutation instead.
Four tests in `test_agent_state.py` assert the removals hold: the nine names
stay gone from both `state.py` and `src`, `TaskStatus` has exactly the three
terminal outcomes, `AgentState` has exactly the eight written fields, and
`record_trace`/`add_tool_result` stay absent. `from src import RouteDecision`
is asserted to fail while the live class is asserted to live in
`src.internal.routing.route`.

Verified by re-adding `ToolType`, `TaskStatus.RETRYING` and `AgentState.trace`
in one mutation: three of the four guards failed. A guard that asserts absence
is the easiest kind to write false-green, which is why it was checked rather
than assumed.

`test_state_models.py` loses the test that only exercised the dead surface; its
duplicate per-field assertions were already covered by `test_agent_state.py`.
What survives there is the part that module uniquely covers — that `AgentState`
is slotted and `asdict`-able.
