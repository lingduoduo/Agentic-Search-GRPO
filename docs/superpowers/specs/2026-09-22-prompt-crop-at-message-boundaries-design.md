# Crop the prompt at message boundaries, and say what was dropped

## Goal

Stop the context-budget crop handing the model a malformed prompt, and stop it
losing turns without any signal.

## The problem, demonstrated

`_crop_prompt_ids` slices a *rendered chat template* at an arbitrary token
offset. The surviving tail therefore begins mid-message. Probed with a ChatML
template, six turns, and a 300-token budget:

```
<|im_start|>system
You are a search agent. Always cite evidence.<|im_end|>
ng 4</think><search>query 4</search><|im_end|>
<|im_start|>user
question number 5 about retrieval and ranking<|im_end|>
...
```

Two defects in three lines:

1. **A headless fragment is glued to the system message.** `ng 4</think>...` is
   the tail of "reasoning 4", cut mid-word. To the model it reads as a
   continuation of its own system instructions.
2. **An orphaned `<|im_end|>` closes a block that was never opened.** The marker
   counts balance only by coincidence, against the unclosed generation cue.

So the crop does not merely lose context quietly — it corrupts what survives.
That is the more serious half, and it was invisible because nothing reported the
crop at all.

## Why the reporting could not just be a flag

`AgentLoopOutput.truncated` already exists for generation truncation. It is set
by **one** of the four loops (`ToolAgentLoop`) and read by **nothing**. Adding a
second flag beside it would reproduce the advertised-capability pattern this
repo keeps generating — a signal that exists and informs no one.

The channels actually consumed are `output.metrics` (read by
`post_training/reward.py`) and the control-flow trace (rendered by the Dev
Console). Metrics is the right home: it is where the rest of a run's numbers
already travel, and a reward function can price context loss.

The control-flow trace is deliberately left alone. Its `ALLOWED_DETAIL_KEYS` is a
closed vocabulary and no existing key fits token accounting; widening it was
argued against in #617 and the argument still holds.

## The change

**`_fit_messages_to_budget`** drops whole oldest messages until the rendered
prompt fits. A leading system message and the newest message are always kept, so
every surviving turn has both of its role markers and the prompt is well-formed
by construction.

**`_render_prompt_ids`** is split out of the build so it can be overridden.
`ToolAgentLoop` overrides it to inject the tool schemas, which is what lets the
tool loop share the boundary fitting instead of keeping its own tail slice —
schemas are part of what must fit, and slicing dropped them first.

**`_crop_prompt_ids` survives as the documented last resort** for the one case
boundary cropping cannot handle: a single message that alone exceeds the budget.
That case is flagged separately, because it is the only remaining path that can
still malform a prompt.

**Reporting** is `prompt_messages_dropped` and `prompt_hard_truncated`, written
by `record_prompt_budget_metrics` into both multi-turn loops' metrics — always
both keys, including zeros, so a consumer can tell "nothing dropped" from "this
loop does not report". Plus a `logger.warning` on each event.

After the change, the same probe yields:

```
<|im_start|>system
You are a search agent. Always cite evidence.<|im_end|>
<|im_start|>user
question number 5 about retrieval and ranking<|im_end|>
<|im_start|>assistant
<think>reasoning 5</think><search>query 5</search><|im_end|>
<|im_start|>assistant
```

253 tokens against the 300 budget, `dropped=8`, `hard=False`, and a log line.

## Consequences to weigh

**Over-budget rollout prompts change.** This is the intent, but RL runs that
straddle the change are not comparable. Under-budget prompts are byte-identical,
so only runs that actually overflowed are affected.

**Fitting costs a constant three renders, not one per dropped message.** The
first implementation here dropped one message and re-rendered, which I described
as "cheap in absolute terms at these buffer sizes". That was wrong, and wrong in
exactly the case the code runs — over budget. Measured with
`examples/benchmark_prompt_fitting.py --target fit`:

| turns | budget | drop-and-rerender | estimate-then-render | raw slice | renders |
|---|---|---|---|---|---|
| 5 | 4096 | 6.9ms | 6.3ms | 3.6ms | 1 |
| 20 | 4096 | **233.5ms** | 25.8ms | 14.5ms | 3 |
| 40 | 4096 | **1077.3ms** | 42.3ms | 28.7ms | 3 |
| 40 | 1024 | **1147.2ms** | 33.0ms | 28.4ms | 3 |

Each render is O(buffer), so a render per dropped message made fitting O(n²) per
turn and O(n³) per run — 1.1 seconds for a single prompt build at 40 turns.

The kept suffix is now chosen from per-message token estimates
(`_estimate_message_tokens`: one `encode` of the content plus a per-message
template overhead measured once and cached), then rendered once. The estimate is
deliberately conservative — the overhead is measured with the generation cue
included — so it over-estimates, keeps one message fewer than strictly
necessary, and the corrective re-render almost never fires. Renders stay at three
regardless of how many messages are dropped, and the cost of a well-formed prompt
is ~1.2–1.5× the malformed slice it replaces.

## Testing

Test-first. The two discriminating tests — no headless fragment after the system
block, and balanced role markers — were watched failing against the real defect
before any code changed.

Sixteen tests total: prompt well-formedness, oldest-first whole-message
dropping, the reported counts, the under-budget no-op, the hard-truncation
fallback, and the metrics reaching both multi-turn loops' output.

Mutation-verified: reverting to the token slice fails **6** tests across both
files. The render bound has its own failing-first test (78 renders before, 3
after), so the cost regression cannot silently return.

One pre-existing test changed contract.
`test_build_prompt_ids_falls_back_to_encode` asserted `len(prompt_ids) == 4` —
the budget filled exactly by slicing `"abc\nde"` mid-message. It now asserts the
newest message survives intact at 2 tokens, under budget, with the drop reported.
That test pinned the behaviour being fixed.
