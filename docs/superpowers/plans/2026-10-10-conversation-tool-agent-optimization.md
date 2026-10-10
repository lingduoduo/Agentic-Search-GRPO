# Conversation and Tool Agent Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route conversation accurately, bound working memory, and reject invalid tool dispatch with corrective feedback.

**Architecture:** Tighten existing fallback routing, working-memory assembly, and argument decoding at the tool-loop boundary. Retain existing learned routing and registry lifecycle.

**Tech Stack:** Python, asyncio, dataclasses, pytest, in-memory database/cache fixtures.

**Spec:** `docs/superpowers/specs/2026-10-10-conversation-tool-agent-optimization-design.md`

## Global Constraints

- Python >=3.10; no new dependencies.
- Preserve HTTP response shapes and user-selected routes.
- Preserve approval, cancellation, recovery, and registry lifecycle behavior for valid calls.
- Preserve newest-message retention and per-session memory isolation.

## Review Focus

- Tool verbs embedded in explanations must not dispatch actions (Task 1).
- Direct action requests phrased as questions must remain tool candidates (Task 1).
- Long summaries must not silently exceed the history budget (Task 2).
- A summary cursor inside recent history must not duplicate that history (Task 2).
- Invalid calls mixed with valid side-effecting calls must not bypass approval (Task 3).

### Task 1: Conversation-aware fallback routing

**Files:** `src/internal/servers/web/intent/rules.py`, new `tests/unit/test_conversation_routing.py`.
**Interface:** existing `_rule_based_route_or_none(query) -> RouteStrategy | None`.

- [ ] Write parameterized tests for explanation, factual question, drafting, explicit search, direct action, and question-form action requests. Example:

```python
@pytest.mark.parametrize("query", ["Explain how to send an email", "Draft an email", "What is required to create a ticket?"])
def test_conversation_tool_subject_is_chat(query):
    assert _rule_based_route_or_none(query) is RouteStrategy.CHAT
```

- [ ] Run `pytest tests/unit/test_conversation_routing.py -q`; observe explanation/drafting failures.
- [ ] Before the broad `_TOOL_RE` match, check anchored search lookup prefixes, then anchored conversational/generative prefixes. Return SEARCH and CHAT respectively. Leave direct action cues and learned/current-information routing untouched.
- [ ] Run `pytest tests/unit/test_conversation_routing.py tests/unit/test_intent_routing.py -q`; commit `fix(agents): distinguish conversation from tool actions`.

### Task 2: Budget summary and recent history together

**Files:** `src/internal/memory/working.py`, `tests/unit/memory/test_working_memory.py`.
**Interface:** existing `load_working_memory(..., keep_last=40, token_budget=None, cache=None) -> WorkingMemory`.

- [ ] Test limits: negative keep_last and nonpositive explicit token budget raise `ValueError`; keep_last zero yields `WorkingMemory(messages=[], summary="", pending=[])`.
- [ ] Seed ten 400-character messages; store a summary through message three; budget 312. Assert total `sum(estimate_tokens(m.content) + 4 for m in wm.messages) <= 312`, latest message retained, and pending starts after the stored cursor. Add long-summary, oversized-newest, tiny-budget, empty-history, no-budget, and cursor-overlap cases.
- [ ] Run the working-memory suite and observe summary-budget and limit failures.
- [ ] Validate limits before store reads. Explicitly handle zero. Preserve initial tail selection and summary eligibility. Reserve the newest message cost, clip summary to remaining tokens including prefix and overhead, then refit the tail with remaining budget. Recompute dropped/pending from the final tail. Do not mutate stored state.
- [ ] Update the existing budget-plus-summary expectation to account for summary cost. Run `pytest tests/unit/memory/test_working_memory.py -q`; commit `fix(memory): include summaries in conversation budget`.

### Task 3: Reject invalid structured arguments before dispatch

**Files:** `src/internal/tools/parsers.py`, `src/agents/tool/tool_calling.py`, `tests/unit/test_tool_parsers.py`, `tests/unit/test_tool_arg_validation.py`.
**Interfaces:** `FunctionCall.parsed_arguments() -> dict[str, Any]` raises `ValueError` on invalid JSON/nonobjects; loop `_invalid_arguments_result(tool_call) -> ToolExecutionResult | None` supplies typed feedback.

- [ ] Add parser tests for malformed JSON, arrays, strings, numbers, null, booleans, empty raw arguments, and valid objects.
- [ ] Add loop tests with zero-argument tools to prove malformed arguments cannot execute defaults. Invalid side-effecting calls must never reach approval callbacks; mixed valid and invalid calls preserve order and valid approval. A corrective next model turn must execute a valid object call after failed feedback.
- [ ] Run parser and argument-validation suites and observe failures before changing decoding.
- [ ] Make argument decoding strict with explicit JSON/object errors. Implement `_invalid_arguments_result` using `_result` with FAILED, `invalid_arguments`, and empty metadata. Call it before `_call_tool` execution and preflight each capped batch before requesting approval. Skip approval for invalid entries and return their preflight results in place while executing valid entries through existing dispatch. Preserve cancellation propagation and existing valid-call retry/approval behavior.
- [ ] Run parser, validation, approval, recovery, and tool-loop suites; commit `fix(tools): reject malformed structured dispatch arguments`.

### Task 4: Review and deliver

- [ ] Run all changed-area suites together, Ruff lint/format checks, and `git diff --check`.
- [ ] Request one independent whole-branch review against the spec, plan, and Review Focus. Fix material findings with reproducing tests.
- [ ] Mark completed steps and record actual verification results. Push the feature branch and create one PR against main containing the spec, plan, implementation, and tests. Do not merge.
