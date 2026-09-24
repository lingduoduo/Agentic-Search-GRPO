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

The dataset is committed before any condition is run.

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
  explicit_source=False)` — the production cascade (regex → kNN → heuristic) with
  the LLM classifier absent, so the run is deterministic. A clarify decision
  counts as a miss. Broken down by `relation` and by `kind`.
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

- Arguments: `--data`, `--retrievers tfidf hybrid`, `--out`, `--seed`.
- Loads each needed corpus once via the corpus registry (`resolve_corpus_docs`).
- Prints one table per metric and writes `data/eval/multi_turn_continuity.json`
  with per-turn rows plus the aggregates, so any number can be traced to turns.
- No LLM. Needs sentence-transformers for kNN routing and the `hybrid` retriever;
  `--retrievers tfidf` still needs it for routing.

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
