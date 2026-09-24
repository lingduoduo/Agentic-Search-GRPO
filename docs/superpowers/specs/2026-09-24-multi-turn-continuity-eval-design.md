# Multi-turn continuity eval — design

## Question

History reaches every conversational surface (`load_working_memory`, 40
messages) but only the answer prompt reads it. Intent routing
(`recognize_intent`, `src/internal/servers/web/intent/recognizer.py`) and every
default retrieval path (the SEARCH direct gate, AgenticRAG's enhance/retrieve,
`answer_with_retrieval`) see the latest user message alone. A deterministic
follow-up rewriter already exists (`build_retrieval_context`,
`src/internal/search/context.py`) but runs only on the degraded no-LLM paths.
Nothing measures any of this: every eval set in `data/eval/` and `data/intent_*`
is single-turn.

This eval answers: **how much routing and retrieval accuracy do follow-up turns
lose today, how much would cheap resolution recover, and what does a topic switch
cost each approach?** Its numbers decide the fix, which is a separate spec.

## Dataset — `data/eval/multi_turn_conversations.jsonl`

About 40 hand-written conversations, 2–4 user turns each, one JSON object per
conversation. Mixed product traffic:

- **search** over the `demo` corpus (ML topics) and `scifact` (5,183 docs, for
  realistic retrieval difficulty);
- **chat** (explanations, opinions, writing help);
- **tool** follow-ups over the public-data tools (weather, stocks, currency,
  geocoding): "weather in Tokyo" → "and in Paris?".

Per turn:

| field | meaning |
|---|---|
| `text` | the user message as typed |
| `route` | gold `chat` / `search` / `tool` |
| `relation` | `opening`, `follow_up`, or `topic_switch` |
| `kind` | follow-ups only: `pronoun`, `ellipsis`, `comparison`, `elaboration` |
| `gold_rewrite` | the standalone query a perfect resolver would produce (for openings and switches, equal to `text`) |
| `corpus`, `relevant_doc_ids` | search turns only |
| `beir_query_id` | optional; SciFact turns only, the BEIR query the gold labels come from |

Composition rules, so the set does not flatter any condition:

- About half the follow-ups avoid the cue words `_FOLLOW_UP_PREFIX` /
  `_REFERENCE_FOLLOW_UP` match ("its price in 2020?", "compare it with BM25",
  "why?"), since the regex is a condition under test.
- Some topic switches open with a cue word ("also, what's the weather in Oslo?")
  to measure the regex's false-positive continuation.
- Some follow-ups change route (a search answer followed by "convert that to
  euros") so route continuity is not assumed to be route stickiness.
- Conversations store user turns only. The harness interleaves a fixed
  placeholder assistant message, since no condition reads assistant text.

SciFact gold labels are not hand-judged. Each SciFact turn's `gold_rewrite` is a
BEIR SciFact query verbatim (`data/beir/scifact/queries.jsonl`), recorded in an
optional `beir_query_id` field, and its `relevant_doc_ids` are that query's qrels.
A follow-up is authored by pairing two BEIR queries about the same entity and
replacing the entity in the second with a reference.

The dataset is committed before any condition is run. The harness rejects a
`relevant_doc_ids` entry absent from its corpus, and exits with an error naming
the file when a corpus is missing (`corpus_scifact.jsonl` is not tracked) rather
than scoring fewer turns.

## Conditions

Each condition turns a user turn plus its prior turns into one query string.
Routing and retrieval then run on that string.

| condition | query |
|---|---|
| `raw` | `text` — today's behavior on the default paths |
| `regex` | `build_retrieval_context(text, history).retrieval_query` — the existing, unwired rewriter |
| `concat` | previous user message + `"\n"` + `text` |
| `gold_rewrite` | `gold_rewrite` — the ceiling a perfect resolver reaches |

`gold_rewrite` separates two failure causes. If `raw` scores far below it, the
loss is unresolved reference and a rewriter is the fix. If `gold_rewrite` itself
scores low, the router or retriever is the problem and a rewriter cannot help.
It also turns any later rewriter's score into "fraction of the gap closed". It
is only as good as the hand-written rewrites, so it is a practical ceiling, not
a hard one.

## Metrics

- **Route accuracy**, every turn: `recognize_intent(query, llm=None,
  explicit_source=False)` — the production cascade with the LLM classifier
  absent, so the run is deterministic. A clarify decision counts as a miss.
  Broken down by `relation` and by `kind`. Two routers, because the kNN stage is
  off unless `AGENTIC_SEARCH_INTENT_INDEX_PATH` is set:
  - `rules`: regex → heuristic, the default deployment;
  - `knn`: regex → kNN over `data/intent_index` → heuristic. The run fails if the
    index does not load, so `knn` can never silently equal `rules`.
- **Retrieval Hit@5 and MRR@10**, search turns, for both retrievers the default
  search path can be pointed at:
  - `tfidf`: `TfidfRetriever` (`servers/retrieval/demo.py`);
  - `hybrid`: `DenseEmbeddingRetriever` fused with `TfidfRetriever` via
    `_fuse_rows` (`servers/retrieval/hybrid.py`), e5 as in serving.
- **Topic-switch damage**, switch turns: Hit@5 and route accuracy, plus the
  **carry-over rate** — the share of switch turns whose query contains text from
  a prior turn. `concat` is 100% by construction; `regex` is non-zero only on
  false-positive cues.

Every condition-vs-`raw` difference is reported with a paired bootstrap 95% CI,
resampling conversations (turns in one conversation are not independent), via
the unseen-user eval's `cluster_bootstrap_ci`
(`src/model/post_training/eval/stats.py`).

## Harness — `examples/measure_multi_turn_continuity.py`

- Arguments: `--data`, `--routers rules knn`, `--retrievers tfidf hybrid`,
  `--device`, `--resamples`, `--seed`, `--out`.
- Loads each needed corpus once via the corpus registry (`resolve_corpus_docs`).
- Prints one table per metric and writes `data/eval/multi_turn_continuity.json`
  with per-turn rows plus the aggregates, so any number can be traced to turns.
- No LLM. Needs sentence-transformers only for the `knn` router and the `hybrid`
  retriever; `--routers rules --retrievers tfidf` runs without it.

## Testing

`tests/unit/test_measure_multi_turn_continuity.py`, on a 3-conversation fixture
with a tiny in-memory corpus and a stubbed router:

- each condition builds the expected query string (including the regex
  false-positive on a cue-word switch);
- Hit@5 / MRR / carry-over / clarify-as-miss arithmetic;
- bootstrap resamples conversations, not turns;
- dataset validation rejects a search turn without `relevant_doc_ids`, an
  unknown `relation`, or a follow-up as the first turn.

Each test is mutation-checked: break the condition or metric it covers and
confirm it goes red.

## Out of scope

- Any fix: resolver, route continuity, history segmentation. Separate spec, sized
  from these numbers.
- LLM rewriters and end-to-end `/api/agent` answer quality. A small end-to-end
  spot-check comes with the fix.
- The other gaps the investigation found (duplicated user turn on clarify,
  stale `[D1]` labels in history, per-surface sessions). Listed for the fix spec.

## Results (2026-09-24)

Run: `python -m examples.measure_multi_turn_continuity` (both routers, both
retrievers, 2,000 resamples). 40 conversations, 109 user turns: 40 openings, 53
follow-ups, 16 topic switches. Per-turn rows: `data/eval/multi_turn_continuity.json`.
Brackets are 95% CIs on the difference from `raw`, resampling conversations.

### Retrieval: follow-ups lose hits only where retrieval is hard

The demo corpus is saturated: with 20 documents a top-5 hit is nearly free, and
every condition scores 10–11 of 11 demo follow-ups. That was a design flaw in this
spec, and it dilutes the pooled numbers (follow-up Hit@5, TF-IDF: raw 0.74 →
`gold_rewrite` 0.91, +0.17 [+0.00, +0.39]). SciFact (5,183 docs) carries the
signal. Split computed from the per-turn rows:

| SciFact follow-ups, Hit@5 (n=12) | raw | regex | concat | gold_rewrite |
|---|---|---|---|---|
| tfidf | 6 | 8 | 10 | 11 |
| hybrid | 9 | 11 | 11 | 12 |

- On TF-IDF an unresolved follow-up loses about half its hits (6 vs 11 of 12). The
  hybrid retriever's dense leg absorbs part of the loss (9 of 12) but not all.
- `concat` recovers most of the gap (10 of 12); `regex` recovers less (8 of 12)
  because it only fires on cue words, and 6 of the 12 SciFact follow-ups carry none.
- Every raw miss is a follow-up whose text lacks the entity ("which mutation makes
  HIV resistant to **it**?", "does **it** also cause atherosclerotic plaques?").
- n=12 is small: the direction is consistent across both retrievers, but the
  CIs are wide.

### Topic switches: `concat` always carries the old topic

- Carry-over: `concat` 1.00 by construction; `regex` 0.44 [+0.19, +0.69], exactly
  the 7 of 16 switches that open with a cue word ("Also, …", "How about …", "But …").
- The damage this does to retrieval is not measured: all 5 search switches land in
  the saturated demo corpus (every condition 5 of 5).

### Routing: the no-LLM router cannot route standalone queries, so continuity is not its bottleneck

| route accuracy | raw | regex | concat | gold_rewrite |
|---|---|---|---|---|
| openings, `rules` | 0.12 | 0.12 | 0.12 | 0.12 |
| openings, `knn` | 0.17 | 0.17 | 0.17 | 0.17 |
| follow-ups, `rules` | 0.23 | 0.25 | 0.23 | 0.25 (+0.02 [−0.08, +0.14]) |
| follow-ups, `knn` | 0.30 | 0.30 | 0.32 | 0.32 (+0.02 [+0.00, +0.06]) |

- Even the hand-written standalone query is routed correctly at most a third of the
  time, so resolving references cannot move routing: `gold_rewrite` gains +0.02.
  Follow-ups get the same route as their `gold_rewrite` in 28/53 (`rules`) and
  36/53 (`knn`) cases under `raw`, and `regex`/`concat` do not change that.
- Probing `recognize_intent` directly shows why:
  - the rules know action tools (tickets, email, scheduling), not the public-data
    tools: "What's the weather in Tokyo?", "Convert 100 USD to JPY." and "What is
    Tesla's stock price?" all return clarify;
  - any question ending in "?" routes to chat, so "What is BM25?" → chat;
  - the canonical kNN examples label current-info lookups ("what is the queue depth
    right now") `search`, while only the TOOL route runs `ToolAgentLoop`, which is
    where the weather, currency and stock tools live. The taxonomy and the serving
    path disagree about which route owns "look up a live value".
- These numbers are the no-LLM cascade. In production the 19–52 clarify decisions
  per condition would go to the LLM classifier, which this eval deliberately leaves
  out; the routing result is about the deterministic stages only.

### What this decides for the fix spec

1. **Resolve follow-ups before retrieval, and make it switch-aware.** `concat`
   shows most of the retrieval headroom is cheap to win, but it drags the old topic
   into every switch. The fix needs a continuation/switch decision, not blind
   concatenation; `regex`'s cue-word test is too narrow on follow-ups (6 of 12
   SciFact follow-ups uncued) and too broad on switches (7 of 16 cued switches
   misfire).
2. **Routing continuity is not the next routing problem; coverage is.** Carrying
   the previous route would mostly propagate a wrong route. Public-data tool
   queries and "?"-terminated lookups need rules or canonical examples first, and
   the search-vs-tool ownership of live lookups needs a decision.
3. **Before measuring the fix, replace the demo corpus in this eval** (or drop it
   from retrieval scoring) and add SciFact topic switches, so switch damage and a
   rewriter's gain are both measurable.

Caveats: one author wrote every conversation, route label and demo gold rewrite
(SciFact text and qrels are BEIR's). Route accuracy excludes the LLM classifier.
