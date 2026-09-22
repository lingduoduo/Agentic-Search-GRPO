# Bound the tool loop's context, and settle two unreachable code paths

## Goal

Close five defects found while mapping how this repo implements the
Think → Act → Observe cycle. Four are in the agent loops' own bookkeeping —
context budget, prompt cropping, an unused parser, a discarded stop reason — and
one is a fail-open that makes a degraded run indistinguishable from a good one.

None of them change what the loops decide. They change what the loops are
*allowed to do to themselves* while deciding it.

## Background: three loops, one base

`AgentLoopBase` is I/O plumbing — tokenize, generate, decode — not a loop. Each
concrete loop writes its own iteration body, and the two multi-turn ones manage
context in opposite ways:

| | `SearchAgentLoop` | `ToolAgentLoop` |
|---|---|---|
| Buffer | `list[dict]` messages | accumulated token ids |
| Per turn | re-tokenises the whole buffer | appends ids incrementally |
| Over budget | `_crop_prompt_ids` (keeps system prefix) | bare tail slice, once |

The incremental design is the faster one. It is also the one that had no bound.

## The problems

### 1. `ToolAgentLoop` context grows without limit

`run()` builds `prompt_ids` once, then only ever grows it — `prompt_ids +
response_ids` after each generation, `prompt_ids + tool_response_ids` after each
tool batch. Nothing re-checks `prompt_length`. The only bounds in the loop are on
`response_mask` length and turn counts.

Measured with a 600-token budget and a tool returning 400 tokens per call, the
prompt handed to the backend grew monotonically across 43 turns:

```
618, 706, 794, 882, 970, ... 4226, 4314
```

Seven times the configured budget, with the backend left to cope.

### 2. Its one crop discards the system prompt first

`_build_prompt_ids_with_tools_sync` ends in `_as_token_ids(ids)[-self.prompt_length:]`.
A tail slice drops what is at the *front*: the system prompt and the injected
tool schemas — precisely the tokens a tool-calling model needs in order to keep
emitting well-formed calls. `AgentLoopBase._crop_prompt_ids` exists to preserve
that prefix and was not used here.

### 3. `Planner.decide` is a second parser that nothing calls

`planner.py` carries two parsing contracts. `SearchAgentLoop` uses the static
helpers (`parse_actions`, `round_retriever`, `round_rerank`,
`partition_search_requests`). It never calls `decide()`, which returns a typed
`SearchAction | RerankAction | AnswerAction`.

The two disagree on semantics: `decide()` returns **one** action per turn with a
documented precedence (search > rerank > answer), while the loop processes
**every** tag in a turn. `decide()` also falls back to searching the first
non-empty line of any unparseable text — behaviour the loop's format-recovery
path deliberately does not have. Its only consumers were eleven tests.

### 4. `StopReason.BUDGET_EXHAUSTED` is computed and discarded

`LoopController.should_continue_searching` returns it; `_run_search_stage`
branches only on `PLATEAU`. The budget is really enforced upstream, in
`Planner.partition_search_requests`, which moves over-budget queries to
`overflow`. Two mechanisms, one rule — and the second one's return value ignored.

**This one is not what it looks like.** The budget arm is checked *first*, so at
the budget it short-circuits the plateau arm. Because the caller ignores
`BUDGET_EXHAUSTED`, the round proceeds normally and its evidence is injected.
Delete the arm and an at-limit round falls through to the plateau check, which
`_run_search_stage` *does* honour: it appends the search-limit notice and
`continue`s, skipping the round's `<information>`. Verified by mutation — with
the arm removed, the model's transcript contains only Round 1's evidence:

```
information blocks seen: ["... <information>\nRound 1\nQuery 1: first query\n
[R1Q1D1] (Title: Doc A) Alpha body\n</information>"]
```

The final round's documents never reach the model that has to answer from them.
The masking is load-bearing. It was simply undocumented and untested.

### 5. `AgenticRAGLoop` fails open invisibly

`_is_sufficient` returns `True` on any exception, including timeout — a
deliberate choice, since the alternative is looping on a broken LLM. But
`sufficient=True` is this loop's stop condition, so a degraded run is byte-for-byte
indistinguishable from a genuinely sufficient one: `rounds_used == 1` either way.
Under a slow LLM the loop silently collapses to single-round RAG and the caller
reports the answer with full confidence.

## The changes

**1 — stop, don't crop.** Add a prompt-budget guard beside the existing
response-budget guard:

```python
if len(prompt_ids) + len(tool_response_ids) > self.prompt_length:
    break
```

Cropping mid-run is not available: the teardown recovers the prompt/response
split from `len(prompt_ids) - len(response_mask)`, so dropping tokens the mask
already covers would desync the two. This is the only path back to `generate()`,
so guarding it bounds the whole run. The tool results are already in
`working_messages`, so the transcript keeps them.

**2 — crop like the base loop.** Replace the tail slice with `_crop_prompt_ids(...,
self._encode_system_prefix(messages), self.prompt_length)`.

**3 — delete `decide()`** and its three action dataclasses, three regexes, and
two module constants (149 → 80 LOC). `_normalize_query` stays: it backs
`partition_search_requests`. The one deleted test with live coverage
(whitespace/case-insensitive duplicate detection) is re-pointed at
`partition_search_requests` rather than dropped.

**4 — keep `BUDGET_EXHAUSTED`, document and pin it.** A docstring on
`should_continue_searching` explaining the precedence, where enforcement really
lives, and that removing the arm is not the cleanup it resembles. Plus two
characterization tests: an at-budget round still gets its evidence injected, and
the contrast case (a plateau *below* budget) still early-stops.

**5 — report the degradation.** `_is_sufficient` returns `(sufficient,
degraded)`; `run()` carries `sufficiency_degraded` onto `AgenticRAGResult` and
emits it on the control-flow trace.

The trace uses the existing closed vocabulary — status `"failed"` plus the
already-allowlisted `fallback` detail key — rather than widening
`ALLOWED_STATUSES`. `fallback` was allowlisted and unused; this is what it is
for. No frontend change needed.

## Testing

Fixes 1, 2 and 5 are test-first: each test was watched failing for the right
reason before the change (fix 1's failure is the `618 … 4314` growth above).

Fix 4 changes no behaviour, so its tests are characterization tests — they pass
on day one. They are verified by mutation instead: deleting the budget arm turns
both red, which is the evidence quoted in §4.

Fix 3 is a deletion; its guard is the rest of the suite staying green.

## Scope

Not included: `LoopSnapshot.model_emitted_answer` is set and documented as "not
yet read". It is a live loose thread, but it belongs to the Phase 2 state machine
rather than to any of these five, and inventing a use for it here would be
scope creep.
