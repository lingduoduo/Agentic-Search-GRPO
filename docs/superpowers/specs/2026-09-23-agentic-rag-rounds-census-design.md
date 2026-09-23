# AgenticRAG rounds census — design

## Question

The CHAT path runs `AgenticRAGLoop`: enhance → up to 3 × (retrieve + sufficiency
judge + gap analysis) → synthesize. Every request pays at least three serial LLM
calls, and the judge sits between retrieval and the first answer token. The task
is read-only and retryable, so a single retrieve-then-answer pass would suffice
*if* most questions stop after one round. This census measures whether they do.

## Why `rounds_used` alone is not enough

`rounds_used` counts rounds, not reasons. A run can stop at round 1 because the
judge failed open (timeout/error → treated as sufficient), and can stop at round 2
because gap analysis returned nothing novel (its errors are swallowed into `[]`).
Neither is evidence that one round was enough. And a run that uses all 3 rounds
may have retrieved nothing new after round 1.

## What is measured, per run

Derived from the `ControlFlowRecorder` trace the loop already emits — no change to
the loop:

- `rounds_used`
- `stop_reason`: `sufficient` | `judge_failed_open` | `no_new_followups` |
  `max_rounds` | `no_queries`
- `docs_after_round_1`, `docs_final` → `docs_added_after_round_1`
- `post_round_1_ms`: wall time from the end of round-1 retrieval to the start of
  synthesis — the latency a one-pass default would remove

Traces written before the `fallback` detail existed mark a failed-open judge only
by `status == "failed"`; both are handled.

## Sources

1. `--history DB`: stored CHAT answers in the web store (`metadata_json` carries
   `rounds_used` + `control_flow_trace`). No LLM needed.
2. `--questions FILE [--label chat]`: replay questions through a live
   `AgenticRAGLoop` against a retrieval server and an OpenAI-compatible LLM,
   configured as `_run_agentic_rag` configures it (`max_rounds=3`).

## Decision rule reported

`one_round_share` = share of runs whose stop reason is `sufficient` at round 1,
with judge-failed-open runs excluded from the denominator. Also reported: the
share of multi-round runs that added zero documents after round 1 (rounds that
cost latency and bought nothing).

## Out of scope

Changing the loop. Answer-quality comparison of one-pass vs multi-round (a
follow-up if the census says the extra rounds are doing work).
