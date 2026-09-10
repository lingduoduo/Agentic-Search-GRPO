# Retrieval vs generation: split the metrics, the feedback, and the telemetry

**Date:** 2026-09-10
**Status:** Implemented on `feat/eval-observability-split`, pending PR review

## Investigation

Three read-only surveys of the tree at `b6fe6835` (evaluation and metrics,
observability, feedback loops). The recurring defect is the same in each
area: a signal about *the documents* and a signal about *the answer* are
recorded, summed, or displayed as one thing, so a regression cannot be
attributed to the retriever or to the generator.

### Evaluation

- `src/internal/servers/web/debug_router.py:102-108` hoists every finite
  numeric top-level key of every `data/eval/*.json` into one flat `metrics`
  dict, and `EvalResultsPanel.tsx:44-57` renders it as one key/value table.
  A Bamboogle summary (`exact_match`, `avg_reward`) and a retrieval baseline
  (`recall@10`, `mrr`) are visually indistinguishable rows. Nested dicts
  (`eval_runner`'s reranked output is `{"retrieval": {...}, "reranked":
  {...}}`) are dropped entirely.
- `src/internal/retrieval/ragas_eval.py:133-140` writes `faithfulness`,
  `answer_relevancy` (generation) and `context_precision`, `context_recall`
  (retrieval) in one flat dict. Worse, `build_ragas_dataset:60-76`
  synthesises the "answer" from the first sentence of the top context, so the
  two generation-named metrics actually measure the retriever.
- `src/model/post_training/reward.py:23-49` — `REWARD_DIMENSIONS` is the only
  place the reward is decomposed, and two of its four buckets are mixed:
  `citation_support` holds `format_reward` (answer formatting) beside
  `citation_support` (fraction of *retrieved* docs cited); `retrieval_quality`
  holds three answer-emission penalties. `reward_components` returns one
  flat dict where `correctness` sits beside `retriever_cost`. `total`
  (`:1160-1169`) is the sum of everything.
- `src/model/post_training/eval/bamboogle.py:144-161` — the summary has
  `exact_match`, `contains_match` (generation) and `avg_reward` (mixed) with no
  retrieval-side number at all; `_to_loop_output:192-216` builds a *stub* loop
  output, so every retrieval reward term is 0 while `avg_reward` is still
  reported as a reward.
- `src/internal/retrieval/eval_runner.py:240-282` has no `--output` flag, but
  `.github/workflows/eval-gate.yml:53-57` invokes it with `--output`. The gate
  is inactive today (its baseline under `data/` is gitignored); the day someone
  commits a baseline it fails on argparse.
- Action-policy (`action_eval.py:75-81`) ANDs `mean_search_rounds` (retrieval)
  with `mean_correctness` (generation) into one `PASS`/`FAIL`; the unseen-user
  harness (`unseen_users.py:305-309`) runs one Benjamini-Hochberg family over
  retrieval-behaviour and generation-format measurements. Both are deliberate
  study designs, documented as such; left alone.

### Observability

- No production-reachable place shows retrieval time or document count
  separately from generation time or token count. The pieces exist in three
  channels: retrieval `duration_ms` + `document_count` only in the
  control-flow trace (`agentic_rag.py:296-303`, `search.py:1002-1013`, dev
  panel only); generation only as TTFT proxies on one loop; token counts only
  as the raw `usage` blob inside the dev-only request capture
  (`providers.py:404`). `_track_llm_cost` is an empty stub and
  `calculate_llm_cost_cents` has no callers.
- The two headline latency numbers blend both stages: `RequestCapture.total_ms`
  and `RouteLatencyStats` for `POST /api/agent`. A p95 regression cannot be
  attributed.
- `capture_stage` (`request_capture.py:159-174`) is the only helper that
  records `duration_ms`, and it has zero call sites; `RequestInspector.tsx:74`
  renders a column that is always empty.
- `RouteLatencyStats` is recorded in every production process
  (`app.py:1396`) but readable only behind `AGENTIC_SEARCH_DEBUG_PANELS`.
- The OTEL tracer spans `rag.retrieve` / `rag.generate` (`pipeline.py:442-497`)
  have exactly the split we want, but `set_tracer` is never called, so the
  singleton `NoOpTracer` swallows them. `setup_phoenix`/`setup_langfuse` have
  no callers.
- `GET /api/debug/workers` (`debug_router.py:65-72`) always returns
  `{"metrics": null}` and has no frontend consumer. Dead.
- The retrieval server's own `latency_ms` (`server.py:77`) is returned over
  the wire and discarded unparsed by `SearchClient`; the structured
  `logger.info(..., extra=...)` fields there and in `control_flow_trace.py`
  are dropped because no JSON log formatter exists.

### Feedback loops

- The human signal is one scalar per session (`retrieval_feedback.signal`)
  with no field for *what* was bad. It reaches training as
  `metadata["human_signal"]` (`data.py:810-853`) and the reward as one
  additive `human_feedback` term (`reward.py:781-794`). There is no way to
  say "the documents were wrong" versus "the answer was wrong given good
  documents".
- No frontend posts either feedback endpoint. `submitFeedback` in
  `web/src/api.ts:357` (per-message like/dislike) is exported and never
  called; `/api/feedback` (the session signal that feeds SFT and GRPO) has no
  caller at all outside tests. The training loop is closed API→DB→trainer
  and open at the UI.
- Consumers with no writer: `dpo/data.py:30` reads a preference JSONL nothing
  produces; `FUSION_WEIGHTS_PATH` (`docs/configuration.md:140`) is documented
  as a startup-loaded file and no Python reads it; `memory_trajectories` is
  written and read by nothing outside tests.
- The BM25/fusion tuners take labelled QA pairs, never user feedback, and
  return JSON without persisting it. The query-router trainer fits 12
  hard-coded rows.

## Goals

1. Every eval result surfaced in the Dev Console is grouped as retrieval,
   generation, reward, latency, or other, by one shared taxonomy.
2. Every `/api/agent` request records retrieval time + document count and
   generation time + token counts as separate numbers, persisted with the
   turn and aggregated into per-stage percentiles that an admin can read in
   production.
3. A user's thumbs can say whether the sources or the answer were at fault,
   the frontend actually posts it, and the summary and the training loader
   carry the distinction.
4. The reward exposes a two-side rollup (retrieval side vs generation side)
   without changing any existing value.
5. Dead and misdocumented pieces found above are removed or corrected.

## Non-goals

- Reclassifying `REWARD_DIMENSIONS` members (changes `dim_*` outputs and the
  golden baseline; the two-side rollup is additive on top).
- Wiring an external tracer (Phoenix/Langfuse) or a JSON log formatter.
- Fixing RAGAS's synthetic "answer", building a DPO exporter, a real
  fusion-weights loader, or feedback-driven ranking. Each is a design of its
  own; they are recorded here as open.
- Per-message like/dislike UI. The session-level `/api/feedback` is the one
  with training consumers; that is the loop worth closing.

## Design

### Metric taxonomy (`src/internal/observability/metric_taxonomy.py`)

Torch-free, stdlib only.

```python
GROUPS = ("retrieval", "generation", "reward", "latency", "other")
def classify_metric(name: str) -> str
def flatten_metrics(data: dict, *, max_depth: int = 2) -> dict[str, float]
def group_metrics(flat: dict[str, float]) -> dict[str, dict[str, float]]
```

`classify_metric` matches the leaf name (case-insensitive, after the last
`.`): retrieval = `recall@k`, `ndcg@k`, `mrr`, `map@k`, `precision@k`,
`hit_rate@k`, `context_precision`, `context_recall`, `routing_accuracy`,
`reranker_improvement_ratio`; generation = `exact_match`, `contains_match`,
`faithfulness`, `answer_relevancy`, `token_f1`, `citation_*`; reward = any
name starting `reward`, `avg_reward`, `dim_`, `side_`; latency = names
containing `latency` or ending `_ms`, and percentile leaves (`p50`, `p95`,
`p99`, `mean`) whose parent path contains `latency`. A parent path segment of
`retrieval`/`reranked` or `generation` decides an otherwise-ambiguous leaf
(`num_queries`, `n`). Everything else is `other`.

`flatten_metrics` keeps finite non-bool numbers, recursing into dicts up to
`max_depth` with dotted keys (`reranked.recall@10`). Top-level numeric keys
keep their bare names, so the existing flat `metrics` field is a superset of
today's.

### Eval results endpoint and panel

`GET /api/debug/eval-results` returns, per file, `metrics` (flattened, as
above) **and** `groups` (`group_metrics(metrics)`, empty groups omitted).
`EvalResultsPanel.tsx` renders one sub-table per non-empty group with the
group as its heading, in `GROUPS` order.

### eval_runner `--output`

`--output PATH` writes the metrics JSON to `PATH` (still printed to stdout).
Fixes the CI gate invocation.

### Reward sides

`reward.py::reward_sides(components) -> {"retrieval": float, "generation":
float}`: retrieval = `retrieval_quality` + `search_efficiency`, generation =
`correctness` + `citation_support`, over `group_reward_components`. Pure,
additive, `sum(sides) == sum(dims)`. `BamboogleSummary` gains
`avg_reward_retrieval` and `avg_reward_generation` (None when no reward
function), computed from each row's components; `__str__` prints them when
present; the summary JSON carries them and the taxonomy files them under
`reward`.

### Per-request stage metrics (`src/internal/observability/stage_metrics.py`)

```python
@dataclass
class RequestStageMetrics:
    retrieval_calls, retrieval_cache_hits, retrieval_ms, retrieval_docs
    generation_calls, generation_ms, prompt_tokens, completion_tokens
    def snapshot() -> {"retrieval": {...}, "generation": {...}}

def start_request() -> Token      # ContextVar, like request_capture
def finish_request(token) -> RequestStageMetrics | None
def note_retrieval(*, elapsed_ms, docs, cache_hit=False)   # no-op outside a request
def note_generation(*, elapsed_ms, prompt_tokens=None, completion_tokens=None)

class StageLatencyStats   # rolling deque per stage, like RouteLatencyStats
    record(snapshot); snapshot() -> {"retrieval": {count, p50_ms, p95_ms, max_ms, avg_docs, cache_hit_rate},
                                     "generation": {count, p50_ms, p95_ms, max_ms, avg_prompt_tokens, avg_completion_tokens}}
STAGE_LATENCY = StageLatencyStats()
```

Hook points (each a few lines, no-op when no request is active):

| Site | Records |
|---|---|
| `SearchClient.retrieve` | elapsed of the post (0 on a full cache hit), rows returned, `cache_hit` when nothing was posted |
| `OpenAICompatibleLLM.complete` (`providers.py`) | elapsed of the HTTP call, `usage.prompt_tokens` / `usage.completion_tokens` |
| `LocalServerManager.generate` (`serving.py`) | elapsed of the executor call, `len(prompt_ids)` / `len(response_ids)` |
| `/api/agent` handler (`app.py`) | `start_request()` beside `start_capture`; in the `finally`, `finish_request()` → `STAGE_LATENCY.record(...)` |
| `_finalize_response` (`app.py`) | the active snapshot is persisted as `metadata["stage_metrics"]` on the assistant turn (a sibling of `pipeline_stages`, whose shape existing tests pin exactly) |

Why these two client-side choke points: every retrieval caller (direct,
hybrid, search agent, agentic RAG, tool agent) goes through
`SearchClient.retrieve`, and every generation goes through one of the two
LLM backends. The control-flow trace stays as it is; this is the always-on,
production layer beneath it.

### Exposure

- `GET /api/debug/latency` gains `"stages": STAGE_LATENCY.snapshot()`;
  `LatencyPanel.tsx` renders a second table, one row per stage.
- New `GET /api/admin/metrics` (`src/internal/servers/web/metrics_router.py`,
  `make_require_admin`, mounted unconditionally): `{"routes": ...,
  "stages": ..., "feedback": db.get_feedback_summary()}`. This is the
  production-reachable read of what the process already records; the debug
  router stays dev-only per `project_tools_surface_consolidation`.

### Feedback target

- `FeedbackRequest.target: Literal["retrieval", "generation", "overall"] =
  "overall"`; `save_retrieval_feedback(target=...)` stores it as
  `metadata_json.target` (no schema change; `json_extract` reads it).
- `get_feedback_summary()` adds `by_target: {target: {"rated": n,
  "thumbs_up_rate": r}}` for the three targets, always present.
  `EvalsSummary` gains `by_target`.
- `load_feedback_examples` adds `metadata["human_signal_target"]` (default
  `"overall"` for legacy rows) so a trainer can weight sides; the reward
  itself is unchanged.
- Frontend: `submitSessionFeedback(sessionId, signal, target)` in `api.ts`
  posting `/api/feedback`. `AnswerPanel` gains optional `sessionId` and
  `onFeedback` props and renders a feedback bar under the answer: 👍 posts
  `thumbs_up`/`overall` at once; 👎 reveals "What was off? Sources · Answer ·
  Both" and posts `thumbs_down` with `retrieval`/`generation`/`overall`. After
  a post the bar shows "Thanks" and disables. `AssistPage` passes the session
  id. The `ConsoleNav` test mock gains the new export.

### Cleanups

- Delete `GET /api/debug/workers` and its two tests.
- `docs/configuration.md`: drop the `FUSION_WEIGHTS_PATH` row (nothing reads
  it); note in `docs/retrieval.md` that the tuning endpoints return weights
  and do not persist or load them.

## Error handling

Stage metrics never raise into a request: every `note_*` is a ContextVar read
plus integer/float adds. A missing `usage` block records the call with no
tokens. `finish_request` outside a request returns `None`. The admin metrics
endpoint returns zeros with an empty store.

## Testing

- Taxonomy: each group's exemplars classify correctly; nested `retrieval.` /
  `reranked.` / `generation.` parents decide ambiguous leaves; `flatten_metrics`
  drops bools/NaN/strings and respects `max_depth`; `group_metrics` omits
  empty groups.
- Eval results endpoint: a Bamboogle summary and a reranked `eval_runner`
  file in a tmp dir come back with `groups` filed correctly and `metrics`
  still containing the old flat keys.
- `eval_runner --output` writes the same JSON it prints.
- `reward_sides`: sums equal the dimension sums for a real components dict.
  Bamboogle summary carries both side averages and prints them.
- Stage metrics: notes outside a request are no-ops; a request accumulates
  two retrievals and one generation into the right buckets;
  `StageLatencyStats.snapshot` percentiles and averages; `SearchClient`
  records elapsed, docs and cache hits; `OpenAICompatibleLLM.complete` records
  tokens from `usage`; `/api/agent` persists `pipeline_stages.timing` and
  records into `STAGE_LATENCY`; `/api/debug/latency` returns `stages`;
  `/api/admin/metrics` requires admin and returns the three sections.
- Feedback: `target` persisted and validated; `by_target` rates; loader
  metadata carries the target; `AnswerPanel` posts the right
  `(signal, target)` pairs and disables after posting; ConsoleNav still
  renders.
- Every new test is mutation-checked by deleting the behaviour it pins.

## Rejected alternatives

- **Use the tracer spans as the collection mechanism.** They only cover the
  `answer_with_retrieval` path; the search agent and agentic RAG loops never
  pass through it. The client-side choke points cover every path.
- **Put the split in the control-flow trace.** It is dev-gated, per-loop, and
  its detail whitelist excludes tokens by design. The trace stays the
  drill-down; this is the always-on layer.
- **Expose stages on the debug router only.** The point of goal 2 is
  production reach; the debug router is dev-only by decision.
- **A `chat_message_feedback` UI instead of `/api/feedback`.** Per-message
  feedback has no training consumer; the session signal does.
