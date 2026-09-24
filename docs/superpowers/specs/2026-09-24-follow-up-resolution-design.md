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
    reason: str         # no_history | no_topic | reference | fragment | semantic | switch
```

**Topic.** The most recent earlier user message that the gate itself classified
as standalone (an opening or a switch), looking back at most 3 user turns. Earlier
messages are re-classified in order on each call, so no state is stored and
"weather in Tokyo" → "and in Paris?" → "Berlin?" keeps "weather in Tokyo" as the
topic. Queries never chain. The window's oldest turn is classified against the
turn just before the window and is a topic only if it stands alone; when no turn
in reach stands alone, the message is searched as typed (reason `no_topic`).
History passes through the existing `_safe_history` filter; assistant messages
are ignored.

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

**Cost.** No LLM. A cosine (one e5 encoding of two short strings) is computed
only when the reference and fragment rules do not decide, for the message and for
each earlier window turn being classified: up to 4 cosine calls per resolution.

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
is on and the request takes a path that uses it (auto-routed, `chat_loop`,
`chat_once`), so the Dev Console shows what retrieval searched for. The explicit
`search_tool`, `hybrid_search`, `search_agent` and `tool_agent` modes keep the raw
query and report nothing.

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

## Results (2026-09-24)

Run: `python -m examples.measure_multi_turn_continuity` on dataset v2 (58
conversations). τ chosen on the dev half (27 conversations); every number below
is from the test half (31 conversations, 54 non-opening turns). Brackets are 95%
CIs on the difference from `raw`, resampling conversations. Per-turn rows:
`data/eval/multi_turn_continuity.json`.

**Decision: the flag stays off.** Three of the four success criteria fail.

| criterion | result | met |
|---|---|---|
| 1. `gated` beats `raw` on SciFact follow-up Hit@5 (tfidf), CI low > 0 | +0.20 [+0.00, +0.40] | no |
| 2. topic-switch carry-over ≤ 0.15 | 0.23 (3 of 13) | no |
| 3. SciFact follow-up hits within one of `concat` | 11 vs 13 | no |
| 4. median latency ≤ 30 ms | 16.5 ms | yes |

### τ: the semantic rule never helps

On the dev half the only follow-up that reaches the semantic rule ("Can the virus
infect hematopoietic progenitor cells ex vivo?") scores 0.768 against its topic,
while five topic switches score 0.72–0.83 (figures from a dev-half diagnostic run;
the committed JSON stores test-half rows only) ("Also, can you explain what a
blockchain is?" 0.83). No threshold separates them, every τ from 0.80 up ties at
dev gate accuracy 0.953, and ties resolve to the top of the grid: τ = 0.95, which
switches the rule off in practice. `gated` and `gated_cues` are identical on every
test metric. `DEFAULT_FOLLOW_UP_COS_MIN` is set to 0.95 accordingly. Latency is
the median over resolutions that call the encoder (e5-base-v2, warm, this Mac).

### Gate accuracy: 44 of 54 (0.81)

The rule that decided each test turn:

| gold relation | reference | fragment | switch |
|---|---|---|---|
| follow-up (41) | 27 | 7 | 7 missed |
| topic switch (13) | 1 carried | 2 carried | 10 |

- Missed follow-ups: five- and six-word follow-ups just over the fragment limit ("And
  spinal long term potentiation?", "Should I pack an umbrella?", "Show me a short
  Python example."), a cue-word follow-up the spec deliberately ignores ("What
  about the axonal transport defects?"), and definite descriptions ("Is the
  disease linked to changes in Treg development?", "Which compounds activate the
  dephosphorylated form?", "Which should I use for a small project?").
- Carried switches: four-word standalone questions inside the fragment limit
  ("What is cross-encoder reranking?", "And what is FAISS?") and the predicted
  weak spot ("Proofread this sentence: …" fires `reference`).

The word-count fragment rule is wrong in both directions, and e5 cosine cannot
rescue the cases it misses.

### Retrieval

SciFact follow-ups, Hit@5 (15 test turns):

| retriever | raw | regex | concat | gold_rewrite | gated | gated_cues |
|---|---|---|---|---|---|---|
| tfidf | 8 | 9 | 13 | 11 | 11 | 11 |
| hybrid | 12 | 12 | 13 | 13 | 13 | 13 |

- `gated` matches the realistic standalone rewrite in hit counts (11 and 13),
  though per turn they differ on 2 of the 15 TF-IDF turns (one hit each way), and
  its gain's CI touches zero on TF-IDF (+0.20 [+0.00, +0.40]).
- `concat` beats even `gold_rewrite` on TF-IDF (13 vs 11, +0.33 [+0.13, +0.60]).
  13 of the 18 new SciFact pairs share their relevant document with the opening,
  so carrying the opening claim's own words retrieves that document whether or not
  the follow-up is resolved. On this dataset `concat`'s advantage is partly that
  overlap, not better resolution.
- SciFact topic switches (6 test turns): `concat` loses a hit (4 vs 5 for `raw`
  and `gated`, both retrievers); `gated` carries none of them.

### What this means

- Blind `concat` is the strongest condition on follow-ups and the worst on
  switches (13 of 13 carried, −1 SciFact switch hit); the gate trades most of that
  switch damage away (3 of 13) but also 2 of 15 SciFact follow-up hits.
- The next resolver needs a better standalone signal than word count and e5
  similarity: whether the message names its own subject (a content noun or entity
  not in the topic) would catch both "What is cross-encoder reranking?" (standalone)
  and "And spinal long term potentiation?" (continues), and cue words deserve a
  place as a weak signal combined with that check rather than on their own.
- The eval should add SciFact pairs whose relevant documents differ from the
  opening's, so resolution is measured separately from topic-term overlap.

Caveats: 15 SciFact follow-ups and 13 switches on the test half; one author wrote
all non-BEIR text, rewrites and labels.
