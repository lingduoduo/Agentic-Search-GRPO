# Routing Off The Event Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A slow routing LLM no longer freezes the event loop, and routing
gives up on it after 3 s.

**Architecture:** `asyncio.to_thread` around `recognize_intent` in
`_run_auto_routed`, plus a `route_classifier_timeout_seconds` policy passed as
`timeout_override` in `classify_route`.

**Tech Stack:** Python 3.10+, asyncio, pytest-asyncio, the `timeouts.toml`
loader.

**Spec:** `docs/superpowers/specs/2026-09-25-routing-off-event-loop-design.md`

## Global Constraints

- No routing-logic change. The order, metadata and decisions are identical.
- The timeout value lives in `timeouts.toml`, not in code; the drift guard
  forbids literals.
- The `LLMClient` protocol stays `complete(messages, **kwargs)`.

## Review Focus

- Request capture and stage metrics still attach intent stages recorded
  inside the thread. Covered by the existing `test_stage_emits_intent`,
  `test_request_capture*` and `test_tool_trace` tests.
- Tests that patch `app.recognize_intent` still take effect through
  `to_thread`. The name is resolved at call time.
- Python 3.10: `asyncio.to_thread` exists from 3.9.

---

### Task 1: Timeout policy and the classifier kwarg

**Files:**
- Modify `src/internal/configs/timeouts.py` (`LLMPolicy`) and `timeouts.toml`
  (`[llm]`)
- Modify `docs/configuration/timeouts.md`
- Modify `src/internal/servers/web/intent/recognizer.py` (`classify_route`)
- Test: `tests/unit/test_intent_routing.py`, `tests/unit/test_timeout_policies.py`

- [ ] **Step 1: Failing tests.**

```python
def test_classify_route_bounds_the_llm_call_by_policy():
    seen = []

    class _LLM:
        def complete(self, messages, **kwargs):
            seen.append(kwargs.get("timeout_override"))
            return "search"

    classify_route("where is the reranker timeout", _LLM())
    policies = load_timeout_policies(
        {}, overrides={"llm": {"route_classifier_timeout_seconds": 1.5}}
    )
    with use_timeout_policies(policies):
        classify_route("where is the reranker timeout", _LLM())
    assert seen == [3.0, 1.5]


def test_classifier_timeout_falls_back_to_rules(monkeypatch):
    monkeypatch.setattr(similarity, "predict_route", lambda q, settings=None: None)

    class _Slow:
        def complete(self, messages, **kwargs):
            raise LLMTimeoutError("LLM request timed out")

    decision = recognize_intent("search for FAISS benchmarks", llm=_Slow())
    assert decision.metadata.get("route_mechanism") in {"heuristic_default", "clarify"}
```

  Also widen the two strict fakes in `test_intent_routing.py` to
  `complete(self, messages, **kwargs)`. Add
  `"route_classifier_timeout_seconds": 3.0` to the expected-defaults dict in
  `test_timeout_policies.py`.

- [ ] **Step 2: Run the tests and expect them to fail.** The `LLMPolicy`
  field is missing, and no `timeout_override` is passed.
- [ ] **Step 3: Implement.**
  - Add `route_classifier_timeout_seconds: float` to `LLMPolicy`.
  - Add this line to `[llm]` in `timeouts.toml`:
    `route_classifier_timeout_seconds = 3.0     # /api/agent routing LLM
    classifier; then the rules router`.
  - Add a docs row.
  - In `classify_route`, call
    `llm.complete([...], temperature=0.0,
    timeout_override=get_timeout_policies().llm.route_classifier_timeout_seconds)`.
- [ ] **Step 4: Pass, then commit.**

### Task 2: Off the event loop

**Files:** Modify `src/internal/servers/web/app.py` (`_run_auto_routed`).
Test: `tests/unit/servers/web/test_agent_router.py`.

- [ ] **Step 1: Failing test.** Patch `app.recognize_intent` with a function
  that waits on a `threading.Event`, bounded at 5 s. Start `_run_auto_routed`
  as a task. On the same loop, await `asyncio.sleep(0.05)` and check that it
  returns promptly. Then set the event and await the task. On the current
  code the patched function blocks the loop, so the sleeping coroutine can't
  finish before the event is released, and the test fails.
- [ ] **Step 2: Run the test and expect it to fail.**
- [ ] **Step 3: Implement.** Change the call to `decision = await
  asyncio.to_thread(recognize_intent, query, llm=llm,
  explicit_source=explicit_source, settings=app_settings)`.
- [ ] **Step 4: Pass.** Run the routing, capture, stage and tool-trace
  suites, then commit.

### Task 3: Verify

- [ ] **Mutation checks.** Revert the `to_thread` change and expect the
  concurrency test to go red. Drop the `timeout_override` kwarg and expect the
  policy test to go red. Restore both and delete `__pycache__`.
- [ ] **Full verification.** Run the full unit suite, then ruff, then
  `git diff --check`. Get a fresh review, then open the PR.
