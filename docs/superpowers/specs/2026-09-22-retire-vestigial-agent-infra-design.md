# Retire the second agent framework and the connector-checkpoint surface

## Goal

Stop maintaining a complete streaming-agent framework that has no agents, and a
connector-checkpoint type that has no connectors. Both were kept alive solely by
their own tests.

## Part 1: the second agent framework

This repo has no external agent framework — no LangGraph, LangChain, AutoGen,
CrewAI, llama-index or semantic-kernel anywhere in the dependency tree. It has
**two** of its own.

**The live one** is `AgentLoopBase` plus the loop registry: four registered
loops (`plain_generation`, `single_turn_agent`, `search_agent`, `tool_agent`),
the `components/` collaborators, and streaming through `on_token`/`on_turn`
callbacks.

**The other** is `BaseAgent` in `src/agents/core/graph_base.py` — a Pydantic
model with `AgentConfig`, a `TypedDict` state, a background thread per run, and
a Redis-backed `AgentQueueManager` delivering a `QueueEvent` stream to
`invoke()`/`stream()`. Its own docstring records that it "replaces the previous
LangGraph-backed implementation."

It has **no subclass anywhere in `src/`**. Its consumers are its own unit test
and two docstring mentions in `bamboogle.py`. `AgentQueueManager` is in the same
position: used by `graph_base` and by `test_queue_manager.py`, and by nothing
else. `InvokeFrom` exists only to key that queue's Redis channel.

The cost is not just the line count. The module forces a name collision:
`graph_base.AgentState` (a `TypedDict`) against `state.AgentState` (a
dataclass), which is why `src/agents/core/__init__.py` carries a paragraph
explaining that it deliberately does not re-export one of them.

### Removed

| File | Lines |
|---|---|
| `src/agents/core/graph_base.py` | 207 |
| `src/internal/chat/queue_manager.py` | ~190 |
| `tests/unit/test_graph_base.py` | 223 |
| `tests/unit/test_queue_manager.py` | 193 |

Plus `InvokeFrom` from `configs/constants.py`, orphaned once the queue is gone.

`TenantRedisClient` **stays** — the `redis_*` modules under
`src/internal/servers/redis/` use it independently.

`bamboogle.py` is unaffected in behaviour: `evaluate_bamboogle(agent: Any, ...)`
is duck-typed on `invoke(state) -> Any`. Only its docstrings named the deleted
class, and they now describe the duck-typed contract instead.

## Part 2: connector checkpointing

`ConnectorCheckpoint` describes incremental connector sync. It is reachable only
through the `src/internal/connectors/__init__.py` and `src/__init__.py`
re-export chains; no connector uses it, and the async worker fleet it belonged
to was already removed with the rest of the Onyx heritage.

Removed from `models.py` along with both re-exports. `Document`, `SlimDocument`,
`HierarchyNode` and `ConnectorFailure` are live and untouched.

## Part 3: the broken indexing tests

The audit found one integration test calling an undefined
`MockConnectorCheckpoint`. There were **four**. Every one would `NameError` on
execution; they are never run, because integration tests need a live
Postgres/Redis stack and a mock connector server.

One of them carried `# noqa: F821,F841` — the undefined name had been seen and
silenced rather than fixed.

Two were repaired and two deleted, on whether the repair could be *verified*:

- **Repaired** (`test_repeated_error_state.py`, `test_initial_permission_sync.py`):
  the expression was `MockConnectorCheckpoint(has_more=False).model_dump(mode="json")`
  — a JSON payload posted to a mock server. Substituting the literal
  `{"has_more": False}` is mechanical and obviously equivalent.
- **Deleted** (`test_checkpointing.py`, `test_polling.py`): `test_checkpointing.py`
  is entirely about the removed surface. `test_polling.py` turned out to
  reference *two further* undefined names, `expected_first_start` and
  `time_before_first_attempt`, which the blanket `noqa` had been hiding.
  Reconstructing them means writing new assertions about indexing behaviour
  that cannot be executed here — inventing coverage rather than restoring it.

The repairs are mechanical, not verified: no live stack is available, so these
files are still unrunnable in this environment. They are simply no longer
*guaranteed* to fail.

## Testing

A guard module, `tests/unit/test_retired_agent_infrastructure.py`, asserts that
`graph_base` and `queue_manager` no longer import, that `ConnectorCheckpoint` is
absent from all three levels, that the live registry still resolves its four
loops, and that the live connector models survive.

"Nothing imports it" is precisely the condition that let both of these drift out
of reality unnoticed, so the absence is pinned rather than assumed.
