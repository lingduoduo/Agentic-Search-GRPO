# Consolidate serving-time intent recognition

## Problem and scope

The web router in `intent_routing.py` and canonical-index adapter in
`ml_intent.py` import each other. Route types live beside the cascade, and two
entry points expose different portions of the result: `route_query` silently
drops clarification, while `route_request` mutates a caller-owned telemetry
dictionary. This makes consumers and tests depend on routing internals.

The investigation also reproduced two routing defects: `hi` becomes a bare
lookup and routes to search; a classifier completion `not chat; search` picks
chat because the parser searches labels in enum order. Current documentation
still describes a removed confidence gate and incorrectly says provider `auto`
forces search.

The user requested the complete Superpowers spec, plan, code, and PR workflow.
This design implements the serving consolidation proposed in the conversation.

## Architecture

Create `src/internal/servers/web/intent/` with these responsibilities:

| File | Responsibility |
| --- | --- |
| `types.py` | RouteStrategy, clarification types, RouteDecision, IntentModelDecision |
| `rules.py` | Deterministic and last-resort heuristic cues, including bare lookups |
| `similarity.py` | Lazy index loading, cached encoder adaptation, typed predictions |
| `recognizer.py` | LLM classification, cascade, clarification, capture and metadata assembly |
| `__init__.py` | Public recognition function and consumer-facing decision types |

The public call is
`recognize_intent(query, *, llm, explicit_source, settings=None) -> RouteDecision`.
`RouteDecision` retains `strategy` and optional `clarification`, adding a fresh
`metadata` dictionary containing the existing production `route_*` fields.
The recognizer assembles those fields once; the dispatcher copies them into its
response metadata. There is no caller-owned telemetry argument. Default fields
preserve simple construction of RouteDecision in dispatcher test doubles.

Types have no dependency on the cascade, adapter, or web app. Rules import
only types and the standard library. Similarity imports types and the existing
model/index implementation. The recognizer imports those components and the
existing request-capture facility. Package imports must not load an encoder or
require torch/sentence-transformers. Offline data, index building, evaluation,
and CLI stay under `src/model/pre_training/intents/`.

All repository callers and tests migrate to the new package. Delete the two
old modules and the strategy-only shim; these are internal Python interfaces,
not external API contracts. Move post-execution first-tool intent inference
beside its sole production consumer in `tool_agent_runner.py`; it describes
what ran and is not a second pre-execution recognizer.

## Behavior and contracts

- Preserve explicit modes and user-selected routes in the dispatcher.
- Preserve cascade order: explicit provider, deterministic rules, optional
  similarity, LLM, heuristic/default clarification.
- Every provider other than `auto` forces search; `auto` permits recognition.
- Preserve routes `chat`, `search`, `tool` and surfaced `clarify`, dispatch
  degradation, prompt, temperature 0, source policies and authorization.
- Resolve omitted AppSettings once at recognition entry so shadow mode,
  clarification and similarity all use the same configuration.
- Preserve all existing production metadata keys and capture payloads, with
  exactly one deciding `intent` stage and at most one `intent_model` evaluation
  stage. Never record arbitrary classifier text or the query in these stages.
- Preserve E5 encoding and canonical index format. Defaults remain top_k=8,
  min_route_margin=0.010, min_module_score=0.8215. Margin is the only model
  abstention gate. Modules and composite remain diagnostics only.
- Standalone greetings/thanks (`hi`, `hello`, `hi there`, `thanks`, `thank you`,
  with surrounding whitespace, ordinary terminal punctuation and case changes)
  deterministically select chat. Match the whole utterance, so `hello world
  tutorial` and `hi, find the report` retain their non-greeting treatment instead
  of silently becoming greeting-only chat. Include greetings in the bare-lookup exclusions.
- The classifier accepts exactly one distinct supported whole-word label in
  its completion. Reject multiple distinct labels as `unexpected`; retain
  compatibility with explanatory completions containing only one label and
  retain empty/unexpected sentinels and redaction. This is ambiguity detection,
  not natural-language negation interpretation.

## Alternatives

Moving only RouteStrategy would remove the circular import but leave duplicate
entry points and mutable side-channel metadata. A multi-intent planner would
change execution semantics and requires its own design. This package plus one
result object consolidates the existing serving boundary at a smaller scope.

## Validation

First add regressions reproducing greetings and ambiguous classifier responses
against the old implementation, and observe failures. Then migrate existing
routing, similarity, capture and dispatcher tests to the new interface without
weakening their behavioral assertions. Add subprocess import-order checks for
the public package and similarity adapter with encoder dependencies blocked;
this catches the circular dependency without relying on warmed test imports.
Add coverage for omitted-settings shadow/clarification behavior and verify
returned metadata agrees with recorded stages. Run the intent unit tests,
web-server tests and relevant execution-fallback tests, plus Ruff and diff
checks. Heavy encoder accuracy evaluations are not claimed unless actually run;
no encoder, canonical data, scoring or serving-threshold changes are planned.

## Documentation and delivery

Update `docs/request-routing.md`, current configuration/training guidance, `.env.example`, and
live source references. Retain historical benchmark numbers with explicit
historical context, rather than presenting them as fresh measurements. Commit
spec and plan before implementation, review the implementation independently,
then open a PR against main with validation and any environmental limitations.
No merge or deployment is part of this request.
