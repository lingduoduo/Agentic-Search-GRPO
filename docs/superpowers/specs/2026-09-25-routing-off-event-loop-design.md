# Routing classifier: off the event loop, with a short timeout: design

## Problem

`_run_auto_routed` (`src/internal/servers/web/app.py`) is `async`, but it
calls `recognize_intent` (`src/internal/servers/web/intent/recognizer.py`)
**synchronously**. When the regex rules and the kNN model do not decide the
route, `recognize_intent` calls `classify_route`, and that calls
`llm.complete(...)`. This is a blocking `requests` call **on the event loop**,
with no `timeout_override`, so the provider default of **30 s** applies.

As a result, a slow or down routing LLM freezes every in-flight request
served by the process, streams included, for up to 30 s per routed request.
Routing reaches its rules fallback only after that wait.

## Decision (approved by the user)

1. **Off the event loop.** `_run_auto_routed` calls `await
   asyncio.to_thread(recognize_intent, query, llm=..., explicit_source=...,
   settings=...)`.
   - Only the routed request waits for the classifier. Other requests keep
     being served.
   - `asyncio.to_thread` runs in a copy of the caller's context, so the
     request capture and the stage metrics stay attached. They mutate
     per-request objects rather than setting `ContextVar`s inside the
     classifier. The implementer verifies this, and records any `ContextVar`
     that is set inside and would be lost.
2. **A bounded classifier call.**
   - A new key, `[llm] route_classifier_timeout_seconds = 3.0`, goes in the
     bundled `timeouts.toml`. It gets a `LLMPolicy.route_classifier_timeout_seconds`
     field, is validated by the existing loader, can be overridden through
     `AGENTIC_SEARCH_TIMEOUTS_PATH`, and is documented in
     `docs/configuration/timeouts.md`.
   - `classify_route` passes
     `timeout_override=get_timeout_policies().llm.route_classifier_timeout_seconds`
     to `llm.complete`. This is within the `LLMClient` protocol
     (`complete(messages, **kwargs)`), and `OpenAICompatibleLLM.complete`
     already honours it.
   - On timeout the provider raises `LLMTimeoutError`. The existing
     `except Exception` in `recognize_intent` then falls through to the rules
     router, as today, but after 3 s rather than 30 s.
3. **No change to routing logic.** The order (explicit source, then regex
   rules, then the kNN model, then the LLM classifier, then the rules
   fallback, then clarification), the metadata, and the decisions are all
   unchanged.

## Out of scope

- Making `recognize_intent` async end to end.
- Cancelling the worker thread after a timeout. The HTTP timeout already
  bounds the thread.
- The other synchronous LLM calls in auxiliary steps.

## Testing

- **Concurrency.**
  - Drive `_run_auto_routed` with a patched `recognize_intent` that blocks on
    a `threading.Event`, next to a second coroutine on the same loop.
  - The second coroutine completes while routing is still blocked. Then
    release the event and check that routing finishes normally.
  - Mutation: call `recognize_intent` synchronously, and the test goes red
    (it times out or detects the blocked loop).
- **Timeout plumbing.**
  - `classify_route` passes `timeout_override == 3.0` under the bundled
    policies.
  - An override (`use_timeout_policies`) changes the value.
  - Mutation: drop the kwarg, and the test goes red.
- **Timeout fallback.** A fake `llm.complete` raising `LLMTimeoutError` makes
  `recognize_intent` return the rules-based decision.
- **Strict fakes.** Test fakes with a strict `complete(messages,
  temperature=0.0)` signature break the protocol. They are widened to
  `**kwargs`.
- **Regression.** The timeout drift and site guards and all routing, capture
  and stage tests pass.
