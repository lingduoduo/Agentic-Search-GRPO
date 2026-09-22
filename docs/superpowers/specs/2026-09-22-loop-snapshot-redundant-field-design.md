# Remove the snapshot field no decision can read

## Goal

Answer the open question on `LoopSnapshot.model_emitted_answer` — should it
participate in control flow? — and act on the answer.

## The question

The field carried the comment `# reserved for Phase 2 state machine; not yet
read`. That framing suggests a field waiting for a consumer. It is not.

`LoopController` exposes two decisions, and the flag's two call sites map onto
them one-to-one:

| Call site | Flag value | Method called next |
|---|---|---|
| `_apply_answer_gate` (search.py:1281) | `True` | `final_answer_decision` |
| `_run_search_stage` (search.py:1578) | `False` | `should_continue_searching` |

`final_answer_decision` is reachable only from the answer gate, which runs only
when the model emitted an answer. `should_continue_searching` is reachable only
from the search stage, which runs only when it did not. So **each method only
ever observes one possible value of the flag.**

A future "Phase 2 state machine" reading it would learn nothing: it would be
reading back which method it already is. The field cannot disambiguate anything,
now or later, while the two decisions stay separate methods.

## The answer

It should not participate, because there is nothing for it to participate in.
Removed from `LoopSnapshot`, from both construction sites, and from the test
helper.

`LoopSnapshot` gains a docstring stating the invariant that made this worth
removing rather than leaving: every field on it must be one a decision actually
reads. A field on a snapshot is a claim that some decision depends on it, and an
unread field is a false claim — the same class of defect as the nine re-export-only
types in #617 and the `AgentLoopOutput.truncated` flag that #620 declined to
extend.

## What this is not

This is not a suggestion that the two decisions should be merged into one
state-machine method. If they ever were, a caller would need to say which
transition it is asking about — and that argument would be reintroduced then, by
the design that needs it, rather than sitting unread until then.

## Testing

A removal, so there is no meaningful RED phase for the behaviour; the guard test
asserts the exact field set of `LoopSnapshot` and was watched failing against the
field's presence. Mutation-verified: re-adding the field fails it.

A second test confirms both controller decisions still resolve correctly without
it, so the removal is not merely absent-checked.
