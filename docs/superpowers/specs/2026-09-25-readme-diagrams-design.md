# README diagrams: architecture, tool-calling sequence, tool-call states: design

## Problem

The README's only diagram was `agentic-search-grpo-architecture.png`, which
linked to an interactive `.html`. It was last redrawn on 2026-07-11, and 55
commits have touched `src/agents` or `app.py` since then.

**It shows components that no longer exist**, all verified absent:
- `src/backend/`, `src/training/` and `src/retrieval/`;
- the background-worker fleet and the async `indexing_pipeline`;
- `retrieval_rerank.py` and `hybrid_rerank.py`;
- `DeepResearch` and `custom.py`;
- `src/model/generation.py` and `compress_chat_history`.

**It omits what the system does now:**
- intent routing;
- tool-call recovery;
- the degradation paths (model unavailable, stale cache, circuit breakers);
- token-budgeted memory;
- `/metrics`, `/ready` and alerts.

**Two diagrams are missing entirely.** There was no tool-calling sequence
diagram and no state-transition diagram, in the README or in the docs.

## Decision (approved by the user)

The README's Architecture section gets three **Mermaid** diagrams. GitHub
renders them inline, and because they are text they are diffable and easy to
keep in sync. They are:

1. **Architecture flowchart**, covering:
   - the clients;
   - the FastAPI backend (routes, `recognize_intent`, the four loops,
     `SearchPipeline`, the tool registry, resilience, and the SQLite store);
   - the services (retrieval :8000, reranker :8002, browser :8003, SerpAPI,
     and the LLM);
   - the offline indexing and post-training;
   - the Prometheus scrape.
2. **Tool-calling sequence**, from `ToolAgentLoop.run` and `_call_tool` in
   `src/agents/tool/tool_calling.py`:
   1. generate, which is one decision round;
   2. parse tool calls, at most 4 per turn;
   3. the approval gate, which applies only to non-READ_ONLY tools, sends the
      SSE event `approval_required`, and expires after 60 s;
   4. parallel `invoke_detailed`;
   5. on failure, `RecoveryPolicy.decide`, which may escalate via
      `escalation_required` (expires after 120 s);
   6. the SSE events `progress` during the run, then `tool_call`, `answer`
      and `done`;
   7. the loop's stop conditions, with at most 10 assistant turns.
3. **Tool-call state diagram.** It covers one call's path through
   `ApprovalGate`, `Executing` and `Deciding`, then one of `Backoff`,
   `FedBack`, `Unavailable`, `Escalated` or `RunStopped`, ending in the
   recorded `TaskStatus`: `COMPLETED`, `FAILED` or `SKIPPED`. The transitions
   come from `RecoveryPolicy.decide` (`src/agents/tool/recovery.py`), the
   escalation outcomes (retry, skip, cancel, expired, no callback, cap) and
   the approval outcomes (approve, deny, expired).

**Housekeeping:**
- The stale PNG and HTML are removed, since the README was their only
  reference.
- `docs/tool-engine.md` links to the two tool diagrams.

## Verification

- **Checked against the code.** Every node and transition was checked against
  the code named above, including:
  - the timeouts from `timeouts.toml`;
  - the SSE event order from `tool_backend.py` and `app.py`;
  - the `_skipped_tool_result` status (`SKIPPED`).
- **Rendered.** All three render with `@mermaid-js/mermaid-cli` 11, and the
  images were inspected. The state diagram was reworked to put each
  `TaskStatus` inside its state box, because floating notes collided with the
  transition labels.
- **Tests.** The full unit suite and the doc/link tests pass.

## Out of scope

- A circuit-breaker or routing state diagram. The routing text flows in
  `docs/architecture.md` and `docs/request-routing.md` stay.
- Auto-generating diagrams from code.
