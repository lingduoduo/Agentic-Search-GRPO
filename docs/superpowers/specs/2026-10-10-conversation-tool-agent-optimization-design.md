# Conversation and tool-agent optimization

## Intent and scope

Improve existing conversation and tool-agent behavior through accurate fallback routing, bounded working memory, and structured tool dispatch. The user approved these three changes and requested execution using the established Superpowers spec, plan, implementation, and PR workflow.

## Approach

Retain the current intent recognizer, memory cache, parser formats, approval policy, registry lifecycle, and agent loop. Target the existing decision boundaries rather than adding a second router or dispatcher. A new orchestration subsystem would duplicate those paths; prompt-only changes would leave budgeting and malformed-argument bugs intact.

## Routing contract

In `_rule_based_route_or_none`, clear conversational and drafting prefixes take precedence over tool verbs embedded in the subject of discussion. Examples: `Explain how to send an email`, `What is required to create a ticket?`, and `Draft an email` route to CHAT. Direct action requests such as `Send an email` and `Create a ticket` still route to TOOL. Explicit lookup prefixes such as `Find documentation on creating tickets` remain SEARCH. Existing high-precision rules, current-information handling, user route overrides, and learned routing remain unchanged. Do not turn every question ending with a question mark into CHAT in this fallback; action requests can be questions.

## Working-memory contract

`keep_last` must be nonnegative; zero explicitly disables history, summary injection, and compression scheduling for that load. A supplied `token_budget` must be positive. Without a token budget, preserve current message-count behavior.

When a stored summary covers a dropped prefix, allocate the supplied budget across its system message and recent history using the existing four-character token estimate and four-token message overhead. Preserve the newest message even if it alone exceeds the budget; in that exception omit summary. Otherwise clip summary text to the remaining budget and refit the recent tail after reserving its cost. Retain the summary prefix intact or omit the summary when there is insufficient space for prefix plus content. Return `WorkingMemory.summary` equal to the actual injected summary text. Never inject a summary whose cursor is already in the initially selected tail or absent from the session. Preserve the uncovered dropped-prefix list and stored summary/cursor state; clipping affects this prompt only.

## Structured dispatch contract

`FunctionCall.parsed_arguments()` returns an object or raises `ValueError` with actionable text. Empty raw argument text retains the existing zero-argument `{}` convention. Invalid JSON and valid JSON arrays, scalars, booleans, or null must not become executable empty objects. Preserve valid object and JSON-object-string parser formats.

The tool loop preflights argument decoding before approval. Invalid calls do not invoke approval callbacks, registry lifecycle, retries, or tool bodies. They produce a FAILED `ToolExecutionResult` with `error_code="invalid_arguments"`, serializable empty argument metadata, and corrective error text. Feed that result back to the model alongside valid calls in mixed batches, preserving result order and existing parallel-call caps. Direct `_call_tool` use receives the same failure behavior. Valid side-effecting calls continue to require approval. This change does not relax schema validation or approve any operation.

## Constraints and validation

- Python >=3.10; no new dependencies.
- Preserve HTTP response shapes and user-selected routes.
- Preserve approval, cancellation, recovery, and registry lifecycle behavior for valid calls.
- Preserve newest-message retention and per-session memory isolation.

Use deterministic routing examples, in-memory session/cache fixtures, parser-format fixtures, and mocked model/tool bodies. Test negative/zero limits, summary budgeting, tiny budgets, cursor overlap, invalid argument types, malformed JSON, mixed batches, and corrective next turns. Run relevant routing, memory, parser, approval, validation, recovery, and tool-loop suites. No live-provider quality or latency benchmark is claimed. Deliver one focused PR separate from the RAG optimization PR.
