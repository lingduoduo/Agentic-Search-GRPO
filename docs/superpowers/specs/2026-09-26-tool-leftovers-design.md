# /tool disconnect cancellation and tool-contract leftovers

1. **Bug:** `/tool/send-tool-message` streaming did not cancel its run when the client disconnected (noted as deferred in #643). The disconnect arrives as CancelledError/GeneratorExit (BaseException), which the generator's `except Exception` handlers skip, so the agent kept executing tool calls, side-effecting ones included, for nobody. Fix: a `finally` that cancels and awaits the task, the same pattern as `AgentRunDriver`.
2. The `registry.py` usage example raised ValueError on the strict registry: it declared neither `effect` nor `result_kind`.
3. `web_search`'s description did not state the 5-query cap its schema enforces.
4. Docs: MCP tools can never fail the contract, so the "logs and skips" path is defensive; and `hook_metadata.route` is absent on a `model_unavailable` degrade (known since #653).

Verified and unchanged: the MCP `batch_search` per-query error claim holds.
