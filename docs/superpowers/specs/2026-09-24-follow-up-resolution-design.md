# Follow-up resolution for retrieval — design

## Problem

History reaches every chat surface but only the answer prompt reads it. The
default retrieval paths search on the latest user message alone, so a follow-up
such as "Which mutation makes HIV resistant to it?" is searched without its
referent. The multi-turn continuity eval (#641,
`docs/superpowers/specs/2026-09-24-multi-turn-continuity-eval-design.md`)
measured this on SciFact: unresolved follow-ups hit 6/12 at TF-IDF top-5, and
prepending the previous user turn (`concat`) recovered 4 of them (+0.33
[+0.08, +0.58]). But `concat` drags the old topic into every topic switch
(carry-over 1.00), and the existing cue-word rewriter (`build_retrieval_context`)
misses half the follow-ups and misfires on 7 of 16 switches.

This spec adds a switch-aware resolver in front of retrieval, and repairs the
eval so its gain can be measured honestly.

**Out of scope:** intent routing (`recognize_intent` keeps the raw message; which
route owns live lookups is a separate decision), SearchAgentLoop and
ToolAgentLoop (their model already sees history and writes its own queries),
memory recall and hooks, and LLM rewriting.

## 1. Resolver

`resolve_follow_up(message, history, *, embedder) -> Resolution` in
`src/internal/search/context.py`.

```python
@dataclass(frozen=True)
class Resolution:
    query: str          # what retrieval searches for
    continuation: bool  # True when the message continues the topic
    reason: str         # no_history | reference | fragment | semantic | switch
```

**Topic.** The most recent earlier user message that the gate itself classified
as standalone (an opening or a switch), looking back at most 3 user turns. Earlier
messages are re-classified in order on each call, so no state is stored and
"weather in Tokyo" → "and in Paris?" → "Berlin?" keeps "weather in Tokyo" as the
topic. Queries never chain. History passes through the existing `_safe_history`
filter; assistant messages are ignored.

**Gate**, in order; the first rule that fires decides:

| rule | fires when | result |
|---|---|---|
| `no_history` | no earlier user message | standalone |
| `reference` | the message contains a whole-word pronoun or demonstrative: it, its, they, them, their, this, that, these, those, one | continuation |
| `fragment` | the message has at most 4 words | continuation |
| `semantic` | an embedder is available and e5 cosine(message, topic) ≥ τ | continuation |
| `switch` | otherwise | standalone |

Cue words ("also", "and", "but", "how about") are deliberately not a rule: alone
they caused the existing rewriter's switch misfires. τ is chosen on the eval's
dev half (section 3) and read from `AGENTIC_SEARCH_FOLLOW_UP_COS_MIN` with that
value as its default.

**Output.** A continuation searches `f"{topic}\n{message}"` — the shape `concat`
measured best. A standalone message searches the message unchanged.

**Embedder.** The one the SEARCH direct gate already warms
(`src/internal/utils/embedding_gate.py`: `gate_embedder()`, e5-base-v2), wrapped
with `make_cosine_fn`. When it is `None` (sentence-transformers missing, or
`AGENTIC_SEARCH_SEARCH_DIRECT_SEMANTIC=0`) the `semantic` rule is skipped and the
gate runs on cues alone; the cosine function already returns `None` on encoder
failure, which is treated the same way.

**Known weak spot.** A standalone message containing "this" or "their"
("Proofread this sentence: …") fires `reference`. The eval measures how often.

**Cost.** One e5 encoding of two short strings per turn with history, no LLM.

## 2. Wiring

`_run_agent_impl` (`src/internal/servers/web/app.py`) calls `resolve_follow_up`
once, after history is loaded, when `AGENTIC_SEARCH_FOLLOW_UP_RESOLUTION` is on.
The call runs in `asyncio.to_thread` (the encoder is synchronous). The resolved
query feeds retrieval only; the answer prompt keeps the raw message.

| consumer | change |
|---|---|
| SEARCH direct gate: `_run_direct_search` and `_direct_gate_decision` | resolved query |
| external web fallback (`_run_direct_search` for serpapi/browser) | resolved query |
| `AgenticRAGLoop.run` (default CHAT route) | new `retrieval_query: str \| None = None` kwarg, used by `enhance_async`, `_is_sufficient` and `_generate_followup`; `question` still drives synthesis |
| `answer_with_retrieval` | new `retrieval_query: str \| None = None` kwarg, used by `retrieve_context`; `question` still drives generation |
| fallback `SearchPipeline` (`src/internal/search/pipeline.py`) | with the flag on, its `retrieval_query` comes from `resolve_follow_up` (with `gate_embedder()`); its bounded history still comes from `build_retrieval_context` |

`build_retrieval_context` itself stays unchanged: it is the eval's frozen `regex`
baseline, and with the flag off every path behaves exactly as today.

**Observability.** The resolution is added to the response metadata as
`follow_up: {"continuation": bool, "reason": str, "query": str}` whenever the flag
is on, so the Dev Console shows what retrieval searched for.

**Flag default.** `AGENTIC_SEARCH_FOLLOW_UP_RESOLUTION` ships off. The same PR
flips the default to on only if every success criterion in section 3 is met; if
any fails, the PR still merges with the flag off and the results state which
criterion failed.

## 3. Eval repairs and measurement

Changes to `examples/measure_multi_turn_continuity.py` and
`data/eval/multi_turn_conversations.jsonl` (edited in place; the #641 results are
the v1 record in git).

**Dataset.**

1. **Realistic SciFact gold rewrites.** A follow-up's `gold_rewrite` becomes the
   follow-up question with its referent substituted and no answer terms ("Which
   mutation makes HIV resistant to AZT?"). Openings stay verbatim BEIR claims.
   `relevant_doc_ids` stay the BEIR qrels. The qrels test keeps checking relevant
   docs for every SciFact turn and text only for openings.
2. **More SciFact.** Grow from 12 to at least 30 SciFact conversations, from more
   same-entity BEIR query pairs. At least 10 get a third turn that switches to an
   unrelated SciFact claim (verbatim, with its qrels), so switch damage is
   measurable where retrieval is hard.
3. Route labels stay as they are (routing is out of scope); the routing table
   carries a note that it is not a finding.

**Harness.**

- **Per-corpus slices**: `<relation>/<corpus>` for retrieval metrics, so SciFact
  is reported on its own (reverses the #641 ruling that kept the split out).
- **Conditions** `gated` (`resolve_follow_up` with `gate_embedder()`) and
  `gated_cues` (`embedder=None`).
- **Gate accuracy**: the share of non-opening turns where the resolution's
  `continuation` equals `relation == "follow_up"`; reported for `gated` and
  `gated_cues`.
- **Latency**: median milliseconds per `resolve_follow_up` call for `gated`.
- **Dev/test split**: each conversation is `dev` or `test` by a stable hash of its
  id (about half each). τ is chosen on dev only, as the value maximising gate
  accuracy over a grid 0.70–0.95 step 0.01; every reported number is test-only.
  `--tau` overrides the choice.

**Success criteria** (test half; all four required to flip the default on):

1. `gated` beats `raw` on SciFact follow-up Hit@5 (TF-IDF), CI lower bound > 0.
2. `gated` topic-switch carry-over ≤ 0.15.
3. `gated` SciFact follow-up Hit@5 (TF-IDF) is at most one turn below `concat`.
4. Median `resolve_follow_up` latency ≤ 30 ms.

## Testing

- **Resolver** (`tests/unit/search/test_follow_up_resolution.py`): each rule
  fires on its own input and not on the others'; cue words alone do not make a
  continuation; the topic skips earlier continuations and looks back at most 3
  user turns; `embedder=None` and a cosine of `None` skip `semantic`; assistant
  and tool-markup messages are ignored.
- **Wiring** (stubbed retrieval and LLM): with the flag on, the resolved query
  reaches `_run_direct_search`, `AgenticRAGLoop.run(retrieval_query=)` and
  `answer_with_retrieval(retrieval_query=)`, and the answer prompt receives the raw
  message; with the flag off, every call receives exactly what it does today and
  no `follow_up` metadata appears.
- **AgenticRAG / answer_with_retrieval**: `retrieval_query` drives enhance,
  sufficiency, gap queries and retrieval; `question` drives synthesis; `None`
  keeps current behaviour.
- **Harness**: gated conditions, gate accuracy, per-corpus slices, dev/test split
  stability, τ chosen on dev only.
- Every test is mutation-checked, and the CI unit job's torch-free import is
  checked.
