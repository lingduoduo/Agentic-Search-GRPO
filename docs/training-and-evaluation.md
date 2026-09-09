# Training and evaluation

This guide covers dataset preparation, supervised and reinforcement-learning workflows, and benchmark evaluation.

## Serving requests do not train models

`POST /api/agent` and `/api/agent/stream` perform routing, retrieval, tool execution, and model inference only. Even explicit `search_agent` or `tool_agent` mode loads a policy model for generation without updating its weights. A query containing `GRPO` is ordinary search input; it does not invoke the GRPO trainer. SFT, GRPO, and PPO run only through the offline commands in this guide. See [API request routing](request-routing.md) for serving-time dispatch.

Serving still uses indexes built offline by the `index_builder`. Filter-aware and degraded branches can use the composed session-aware retrieval, ranking/reranking, and evidence-grounded inference pipeline; strong unfiltered auto-search retains its direct ranking, sufficiency gate, and provider fallback path. Neither retrieval nor reranking is a training step. No new serving API or request/response schema was introduced; offline trainers may produce model artifacts, but requests only load and infer with them.

## Intent routing by nearest canonical example

Serving enters through `src.internal.servers.web.intent.recognize_intent`, which returns strategy, clarification, and metadata together. The optional similarity adapter lives in `web/intent/similarity.py`; offline build and evaluation remain in `src/model/pre_training/intents/`. Module labels and composite flags are diagnostics, not a multi-step execution plan. See [request routing](request-routing.md#auto-router-decision-order) for the complete cascade.

**There is no intent training run any more.** The optional three-label (`chat`, `search`, `tool`) request router used to be a small MLP trained on generated examples. It is gone — module, checkpoint format, wordpiece bundle and all. What replaced it compares the incoming request against roughly 300 curated canonical examples and takes the route whose nearest examples are closest. The whole offline workflow is three commands, none of which trains anything:

```bash
# 1. Seed a canonical draft from existing labelled examples (optional; run once).
python -m src.model.pre_training.intents.cli seed \
  --examples data/intent_examples.json --output data/intent_canonical.draft.json

# 2. Build the index the router serves from.
python -m src.model.pre_training.intents.cli build \
  --canonical data/intent_canonical.json --output data/intent_index

# 3. Measure it. Never skip this after editing the canonical set.
python -m src.model.pre_training.intents.cli evaluate \
  --index data/intent_index \
  --eval-queries data/intent_eval_queries.json \
  --hard-queries data/intent_eval_hard.json \
  --out-of-scope data/intent_out_of_scope.json \
  --canonical data/intent_canonical.json \
  --output data/intent_index/evaluation_report.json
```

`data/intent_canonical.json` and the evaluation JSON files are tracked (force-added under an otherwise gitignored `data/`). `data/intent_index/` is a regenerable local build artifact and is not.

### How routing works

The route vocabulary and default scoring values live in `src/shared_configs/intent.py`. Existing imports of `model.INTENT_LABELS` and `model.TOP_K` remain supported. Both the web adapter and the example CLI pass the configured `intent_top_k` to the same index scorer; the CLI keeps its own search/model policy.

Each canonical example is encoded once by `intfloat/e5-small-v2` and L2-normalized. At serving time the request is encoded the same way, and each route scores as the **mean of its top-k cosine similarities** to that request. The best route wins; the module labels reported alongside it are diagnostics and can never change the route.

**The prefix contract.** E5 models are trained with instruction prefixes and **degrade silently without them** — no error, no warning, just worse vectors. Every text gets `"query: "`, applied **symmetrically** to canonical anchors and to incoming requests: this is symmetric short-text similarity, not the asymmetric retrieval E5's `"passage: "` prefix is for. The prefix is a property of the encoder, not an argument, so it lives in `MODEL_PREFIXES` in `src/model/pre_training/intents/model.py` and is applied inside `encode_texts` — no call site can forget it. An encoder with no registered prefix raises rather than defaulting to `""`: an unregistered model is far likelier to be one whose prefix nobody looked up than one that genuinely needs none, and guessing wrong is invisible.

**Every index built before this change is invalidated.** Rebuild with command 2 above. What catches a stale index is the **encoder-name check** — `IntentIndex` records the encoder that built it, and both `run_index_evaluation` and `web.intent.similarity.load_intent_index` reject a mismatch by name — **not** a dimension mismatch, because `all-MiniLM-L6-v2` and `e5-small-v2` are **both 384-wide**. A stale index therefore loads, scores, and reports confident, meaningless numbers with no other symptom. That is not hypothetical: it happened once on this branch, and the name check is what turned it into one loud failure naming the rebuild command.

`k` (`TOP_K` in `src/model/pre_training/intents/model.py`) is `8`, re-selected on the wider tuning instrument — see [`top_k` chosen on the split](#top_k-chosen-on-the-split) below, and `AGENTIC_SEARCH_INTENT_TOP_K` in [Configuration](configuration.md) for the env var. It was an unswept `3` from #511 to #521.

**One** threshold gates the answer:

- `AGENTIC_SEARCH_INTENT_MIN_ROUTE_MARGIN` (default `0.010`) — a small **gap** between the best and second-best route means two routes fit equally well. That is an ambiguous request.

It abstains, and abstention defers to the LLM classifier and the clarification path exactly as before.

There was a second gate — an absolute `min_confidence` floor for out-of-scope requests — [removed after measurement](#the-confidence-gate-that-never-fired). This is the concrete reason the softmax head was replaced: probabilities sum to one by construction, so a softmax router cannot express "none of these" — it can only say which of the three is least bad.

**This threshold is a cosine, and its scale moves with the encoder and with `top_k`** — see [the units trap](#the-units-trap). It is re-derived on the tuning slice whenever either changes: at `0.010` the router serves 120 of the 201 test-slice queries (coverage `0.597`) at `0.9667` served accuracy.

### Changing routing behavior

Edit `data/intent_canonical.json` and rebuild. There is no training run, no seed, no schedule, no checkpoint. The canonical examples *are* the model.

Always **append, rebuild, re-measure**, in that order. A badly-phrased canonical example does not fail loudly; it becomes a bad attractor that quietly pulls every nearby query onto its route. The only thing that catches it is the evaluation report, so a canonical edit that has not been re-measured has not been verified. `tests/unit/test_intent_canonical_data.py` additionally guards the set's size band, per-route balance, per-module support, and internal near-duplication; `tests/unit/test_intent_evaluation.py` pins the measured test-slice accuracy, out-of-scope AUC, and latency bars. All of those need `sentence-transformers` and a built index, so they skip in CI — they guard your local rebuild, not the pull request.

### The tuning/test split

Three hyperparameters need values — `top_k`, `min_margin`, `min_module_score` — and **none of them may be chosen on the queries the result is reported from**. All are swept on the tuning slice. The split section of `src/model/pre_training/intents/evaluation.py` enforces the rest:

- **Tuning slice, 70 queries** — all 30 `eval-` legacy queries (already contaminated: they were used as feedback while curating the canonical set, so they are worthless as a gate and free to spend here) plus a route-stratified sample of **40** of the 151 clean `bulk-` queries, drawn with **seed `17`**.
- **Test slice, 111 queries** — every clean query the tuning sample did not take. Untouched by every sweep. Plus `hard-40`, which is likewise never tuned on.

The split is deterministic in the seed alone (input order is normalized first), and the two slices are an exact partition of the eval set. **Every accuracy quoted below as a result comes from queries that no hyperparameter ever saw.** `evaluation_report.json` labels every block with `tuned_on: true|false`; only `tuned_on: false` numbers are results.

**The earlier `0.8287` figure for this encoder was fitted.** It came from exploratory probes where `k` was chosen *after* seeing the results, on the same queries used to report them. The properly-split number below is lower. That is what a fitted number does when you stop fitting it, and it is the entire point of the split.

### What it scores, and why it is still dark

The table records measurements from the current 304-anchor canonical set and earlier evaluation instruments. The current serving settings are `top_k=8`, `min_margin=0.010`, and `min_module_score=0.8215`. Rows naming the 201-query test slice use the wider split (seed 17, tuning 70 / test 201); rows naming clean_151 or the retired 111-query slice are historical and are not directly comparable across instruments. These are recorded benchmark results, not new measurements from the serving-package consolidation.

| Measure | e5-small-v2 (now) | all-MiniLM-L6-v2 (before) |
|---|---|---|
| **Route accuracy, test slice (201 queries; seed 17, tuning 70 / test 201)** | **0.8159** — 95% CI `[0.757, 0.863]` | **0.6667** (on the retired 111-query slice) |
| — the same, on the older clean_151 instrument | `0.7881` | `0.6225` (as published by #511) |
| hard_40 (adversarial, never tuned on) | `0.7000` argmax / `0.8095` served — 95% CI `[0.546, 0.819]` | `0.6250` |
| Out-of-scope **AUC**, test_201 vs. **31 held-out** probes | `0.8578` | `0.8863` (on the retired instrument) |
| — the same, on clean_151 vs. 24 probes | `0.8626` | `0.8681` |
| Out-of-scope Cohen's d, test_201 vs. 31 held-out probes | `1.4746` | `1.6701` (retired instrument) |
| Leave-one-out over the canonical anchors (diagnostic, never a selector) | `0.8191` (249/304) | `0.6382` (194/304) |
| p50 / p95 routing latency, encode + decide | `10.35ms` / `12.20ms` | `5.51ms` / `5.88ms` |
| Out-of-scope raw margin *(encoder-specific — do not compare across this row)* | `0.0234` | `0.1227` |
| Module macro-F1 / joint accuracy (diagnostic; both on the mixed `bulk_181`, like-for-like with the before column — the test slice alone gives `0.6291` / `0.4775`) | `0.6463` / `0.4972` | `0.3471` / `0.2318` |
| Per-route accuracy | see `evaluation_report.json` | chat 24/51, search 25/50, tool 45/50 (on clean_151) |
| Serving hyperparameters | `top_k=8`, `min_margin=0.010`, `min_module_score=0.8215` | `top_k=3`, `min_confidence=0.30`, `min_margin=0.02` |

**What `0.8159` is, precisely.** It is **argmax route accuracy with no abstention**: the fraction of the 201 test queries whose best-scoring route is the right one, counted whether or not the thresholds would have served an answer. It is *not* the accuracy a caller sees. With the shipped `min_margin=0.010` the router **serves 120 of those 201 (coverage `0.597`) at `0.9667` served accuracy** and defers the rest to the LLM classifier.

The `1.0000` served accuracy reported before this instrument change was 58 queries on the old slice; on 120 it is `0.9667` (4 wrong). That is the more believable number, and its arrival is the clearest single argument for having widened the instrument. Argmax is the number the decision rule was written against and the only one comparable to the `0.6667` MiniLM figure, which is argmax too; the served pair is what promotion would actually deliver. Both are reported for every slice in `evaluation_report.json`.

**The earlier margin-gate breakdown is historical.** On the retired 111-query slice, search scored 26/37 argmax and the 58 served queries had no errors. That pattern motivated measuring per-route deferral, but it must not be read as the result on the wider instrument: the 201-query evaluation above served 120 with four wrong routes. See `evaluation_report.json` for the per-route breakdown of a particular rebuild.

For older context, on the retired clean-151 instrument the previous MLP scored `0.4768`, the production regex cascade `0.4238`, and the majority-class floor `0.3377`. Those are different queries under a different encoder — context, not a like-for-like comparison with the column above.

**+0.14 on route accuracy against MiniLM**, measured on the identical 111 queries *and* the identical 304 anchors, each encoder at its own shipped hyperparameters — at roughly 2x the latency and well inside the 25ms ceiling. Keeping both halves of that comparison matched matters: MiniLM's own numbers moved when the anchor set grew (`0.6216` → `0.6667` argmax, leave-one-out `0.6643` → `0.6382`), so a column carried forward unchanged would have silently compared two different models on two different anchor sets.

**The decision rule fixed in advance had three bands: `≥ 0.80` clears the promotion bar, `0.75`–`0.80` is a real improvement, below `0.75` is a hard stop. `0.8108` clears the bar.** It did not at `k=3`, where the same index scored `0.7928` and sat in the middle band; choosing `top_k` on the split (see [below](#top_k-chosen-on-the-split)) moved it to `0.8018`, and the business-vocabulary anchors added after that took it to `0.8108`.

**The confidence interval is now published, and it settles how much that margin is worth.** `0.8159` on 201 queries carries a 95% Wilson interval of **`[0.757, 0.863]`**. The `0.80` bar sits inside it. The point estimate clears the bar; the interval does not clear it, and no amount of further tuning on this instrument will change that — only more queries will.

This is why every accuracy in `evaluation_report.json` now ships with an `accuracy_ci` beside it. Several conclusions in this document's history turned on one- or two-query margins, and a bare point estimate is what made them look decisive.

**The artifact still ships dark.** `AGENTIC_SEARCH_INTENT_INDEX_PATH` remains unset by default and every request falls through the existing LLM/rule cascade. Promotion was always specified as a separate change reviewed on its own terms, and a one-query bar crossing is not a reason to skip that review — if anything it is a reason to hold it more carefully.

One result did **not** improve, and it matters more than the headline:

**Out-of-scope separability is still the one measure MiniLM wins: AUC `0.8720` against `0.8863` on the same 111 queries and the same anchors — `−0.0143`, roughly half the gap #512 measured.** On the older clean_151 slice the same comparison is `0.8626` vs `0.8681` (`−0.0055`), so the regression looks small there and is five times larger on the slice actually reported from; quote the matched pair, never one number from each slice. Either way it misses the `0.90` bar this change set for itself. The `0.927` AUC that made e5 look better at abstaining was measured on **e5-*base*-v2**, a different, larger model that is explicitly out of scope here; the fitted e5-*small* probe scored `0.871`, already under the bar. Abstention is the safety property of this router, so this looked like the single strongest argument against promoting it.

**It does not survive being measured at the operating point** — e5 makes 2 wrong routes against MiniLM's 21 at each model's own tuned threshold, and 7 against 21 at matched coverage. AUC ranks over the whole score range; the margin gate only needs separation at the boundary. See [The out-of-scope regression, measured where it bites](#the-out-of-scope-regression-measured-where-it-bites) below. The `0.90` bar is still missed and the ranking regression is still real — what changed is that neither costs anything at any threshold this router would run at.

**The threshold grid had to be re-derived, because it was in MiniLM's units.** The original grid started at `min_margin=0.02`, which under e5 abstains on more than half the tuning slice; no combination cleared the sweep's `coverage ≥ 0.60` floor, so the first run selected nothing at all. The grid's low end is now derived from the **tuning slice's own margin quantiles** under e5 (min `0.0008`, p25 `0.0116`, median `0.0188`, p75 `0.0274`, max `0.0676`), and at the then-shipped `k=3` it selected `min_margin=0.015`. (Those quantiles are `k=3`'s; the joint `(k, margin)` sweep later moved the shipped pair to `k=15, min_margin=0.010` — see [`top_k` chosen on the split](#top_k-chosen-on-the-split). Margins compress as `k` rises, so a quantile table is specific to its `k` as well as its encoder.) **Derive any future re-tuning from the tuning slice too. Never from the test slice**, whose quantiles (median `0.0129`) are a different distribution and are off-limits: reading them to choose a threshold is test-set fitting even when no code does it.

**Why widening that grid is safe, and would not have been before.** `top_k` is **not** swept for selection — it was pinned at the shipped `3`, and `_select_thresholds` searched only `(min_confidence, min_margin)`. That matters because the reported headline is argmax accuracy, which is abstention-blind: it depends on `top_k` and on nothing else the sweep chooses. A sweep that also chose `k` would couple a tuning-slice search to the held-out number, and widening it after seeing that number could move the number. With `k` pinned, **no threshold this sweep selects can change `test_slice.accuracy` by any amount** — a property, not a promise, and the one `test_the_threshold_sweep_never_chooses_top_k` guards. Choosing a different `k` remains possible, but it is a deliberate, separately-reviewed decision, not a side effect of re-tuning thresholds. `min_module_score` has now had the same treatment — see [the module threshold](#the-module-threshold-derived-and-no-longer-dead).

**Two caveats on the separability numbers, neither fixed here:**

1. **`separability_report`'s Cohen's d is comparable only to other numbers from that function.** Its pooled SD averages the two groups' *population* variances (dividing by `n`) rather than the textbook `(n−1)`-weighted form. The difference is small at these sample sizes but real, so this `1.485` must not be set beside a Cohen's d computed by scipy or any stats package.
2. **The headline AUC and Cohen's d are not fully held out.** The same 24 out-of-scope probes both tie-break the tuning sweep's hyperparameter selection *and* denominate the reported separability. The in-scope side is the untouched test slice, but the out-of-scope side is not held out from everything upstream of it.

Also note the instrument itself: 111 test queries is a small slice, and it is deliberately smaller than the 151 the "before" column used — that is the honest cost of holding 40 queries back for tuning, and it widens the confidence interval on every number in the column.

### The out-of-scope regression, measured where it bites

**Every number in this section was measured at `k=3`**, the setting shipped when #513 ran it. The reasoning is what matters and it generalizes — indeed [`top_k` chosen on the split](#top_k-chosen-on-the-split) reaches the same conclusion by the same route — but do not read these as current serving figures.

The AUC row above says e5 is *worse* at separating out-of-scope requests (`0.8551` against MiniLM's `0.8848` on the same slice), and #512 called that the single strongest argument against promoting it. **Measured at the operating point each encoder would actually run at, the ordering reverses.**

AUC is a threshold-free ranking statistic over the whole score range. Serving does not use the whole range — the margin gate needs separation only near the decision boundary, and abstaining costs an LLM fallback rather than a wrong answer. So the number that matters is not "how well does the score rank in-scope above out-of-scope", but **how much traffic is answered, and how often those answers are wrong.**

Each encoder's `min_margin` is tuned on the **tuning** slice and reported on the **test** slice — a threshold chosen on the reported queries would flatter whichever encoder it was chosen for.

| | all-MiniLM-L6-v2 | e5-small-v2 |
|---|---|---|
| tuned `min_margin` (on the tuning slice) | `0.030` | `0.015` |
| coverage, test slice | `0.6757` (75/111) | `0.4505` (50/111) |
| served accuracy | `0.7200` | **`0.9600`** |
| **wrong routes** | **21** | **2** |

At its own tuned point e5 answers less and is right far more often: **2 wrong routes against 21**. Because the two points serve different volumes, the same comparison at **matched coverage** — e5 loosened to `min_margin 0.008`, answering 73 queries against MiniLM's 75:

| | all-MiniLM-L6-v2 @ `0.030` | e5-small-v2 @ `0.008` |
|---|---|---|
| coverage | `0.6757` (75/111) | `0.6577` (73/111) |
| served accuracy | `0.7200` | `0.9041` |
| **wrong routes** | **21** | **7** |

**A 3× reduction in misroutes at the same answered volume, and 10× at each model's own tuned point.** The AUC regression is real as a ranking property and does not bite at any threshold either model would run at. The reason is that e5 compresses cosine similarities into a narrow high band — which costs global ranking across the full score range while leaving local separation at the boundary cleaner.

Reproduce with:

```bash
python -m examples.measure_intent_operating_point
```

**What this does and does not settle.** It removes the safety argument against promotion: e5 is more accurate *and* misroutes less at every comparable point. What it leaves is a cost question — e5 at its tuned point defers 61 of 111 queries to the LLM classifier against MiniLM's 36, which is more latency and more spend per request. That is a budget decision, not a correctness one.

One limit worth stating: the slice is 111 queries, so `2` versus `21` has a wide interval on the low end — the direction is solid, the ratio is not precise.

**A fitted observation that did not survive checking, recorded because the checking is the point.** (Also `k=3`-era: `0.015` was the shipped margin then, `0.010` is now.) On the test slice, `min_margin 0.012` scored the same 2 wrong routes as the then-shipped `0.015` while covering 6 more queries — apparently strictly better. It is not in the swept grid, which steps `0.010 → 0.015`, so it was never evaluated on tuning data. Checking the tuning curve settles it:

| `min_margin` | coverage | served accuracy | clears the 0.60 floor |
|---|---|---|---|
| `0.010` | `0.7714` | `0.8889` | yes |
| **`0.015`** | `0.6571` | **`0.9130`** | **yes — selected** |
| `0.020` | `0.4714` | `0.9394` | no |

Served accuracy rises monotonically with margin while coverage falls, so the selection rule — highest served accuracy at coverage ≥ `0.60` — takes the last eligible point. `0.012` interpolates between `0.010` and `0.015`: still eligible, but at *lower* served accuracy than `0.015`. **Under the pre-registered rule it loses.** Its advantage exists only on the slice it was read from, which is what a fitted number looks like when you check it. `0.015` stands.

### The confidence gate that never fired

`AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE` was an absolute floor: reject a request whose best route score is too low, on the theory that nothing canonical resembles it. It defaulted to `0.30`, a MiniLM-era value, and was never re-derived when the encoder or `top_k` changed.

**It has been removed.** The sweep was extended to span the range the encoder actually produces, and the pre-registered rule — unchanged from the one that selects the other thresholds — picked `0.79`. That value sits *below* the tuning slice's lowest in-scope score (`0.7905`), which is the definition of inert the plan registered in advance.

The direct measurement is starker than the rule. Comparing every decision at `0.30` against `0.79` across every evaluation set available:

| slice | n | decisions changed |
|---|---|---|
| tuning | 70 | 0 |
| test | 201 | 0 |
| hard_40 | 40 | 1 |
| probes (tuning half) | 29 | 2 |
| probes (reporting half) | 31 | 0 |
| off-domain probes | 45 | 0 |
| **total** | **416** | **3** |

**Three decisions in 416, two of them on the very probes the value was selected from.** On the held-out reporting probes it changed nothing at all.

**Why it earned nothing is structural, not a tuning failure.** Under this encoder, in-scope and out-of-scope scores occupy the same narrow band — the ranges overlap, so no floor separates them. Anything far enough from a single route to fail an absolute floor is, by then, close to two routes and already fails the *margin*. The two gates were not complementary; the margin gate subsumed the confidence gate almost entirely.

Removing it is behaviour-preserving to the digit: test-slice accuracy is `0.8159203980099502` before and after, along with every other reported number. `decide()` now has one route-abstention threshold instead of two, `recognize_intent` derives one abstention instead of two, and the `model_below_threshold` fallback label is gone — every deferral is now `margin_below_threshold`.

**A knob that looks like out-of-scope protection but cannot fire is worse than no knob**, because it invites tuning that does nothing and conceals the absence of the control it appears to offer. A future encoder that separates absolute scores cleanly would need it back; the git history has it.

### The module threshold, derived and no longer dead

`AGENTIC_SEARCH_INTENT_MIN_MODULE_SCORE` sat at `0.45` from the MiniLM era through six PRs. Under e5 that value is below *every* module score the encoder produces — measured over the 111-query test slice at the then-shipped `k=3`, all 1554 module scores fall in `0.7428`–`0.8943`, so the gate **fired for none of them**. It was not mistuned; it could not fire at all.

The consequence was that `_emit_modules` returned every well-supported module of the winning route: recall ≈ `1.0`, precision ≈ `0.2`, and **joint accuracy `0.0`** on all three slices — not one query in 181 got its full module set right.

**The rule was registered before the sweep ran** (`docs/superpowers/plans/2026-08-14-intent-module-threshold.md`, committed in its own commit ahead of any measurement): highest module macro-F1 on the **tuning** slice, ties to the lower threshold. The grid is **computed** from that slice's own module-score quantiles at the `top_k` in force, not written down — see [the grid that must not be a constant](#the-grid-that-must-not-be-a-constant).

The historical sweep at the then-serving `top_k=15` — tuning numbers, abridged (current defaults are `top_k=8` and `min_module_score=0.8215`):

| `min_module_score` | macro-F1 (tuning) | joint (tuning) | mean modules emitted |
|---|---|---|---|
| `0.4500` *(the status quo it replaces)* | 0.3774 | 0.0000 | 4.69 |
| 0.7931 | 0.4751 | 0.0571 | 2.87 |
| 0.8024 | 0.5707 | 0.2143 | 1.99 |
| 0.8117 | 0.5991 | 0.3571 | 1.49 |
| **0.8210 — selected** | **0.6531** | 0.5000 | 1.17 |
| 0.8303 | 0.6169 | 0.4714 | 1.06 |
| 0.8396 | 0.6107 | 0.5000 | 1.00 |

**Why the rule excluded joint accuracy is visible in that table.** Joint accuracy reaches `0.5000` at both `0.8210` and `0.8396` — but at `0.8396` `mean modules emitted` is exactly `1.00`, meaning emission has collapsed to `_emit_modules`'s top-1 fallback for every query. It ties only because most queries carry a single gold module, so "always guess one" wins an exact-set match on them. Macro-F1 separates the two cleanly (`0.6531` against `0.6107`). Had joint accuracy been the selector, the tie-break to the lower threshold would have saved it here by luck — but at `k=3` the same table peaked at the collapsed setting outright, and it would have shipped.

Held out on that historical test slice, at the then-shipped `0.821`:

| | before (`0.45`) | after (`0.821`) |
|---|---|---|
| module macro-F1, test slice | `0.3492` | **`0.6048`** |
| module joint accuracy, test slice | `0.0` | **`0.4595`** |
| **route accuracy, test slice** | **`0.8018`** | **`0.8018`** — unchanged |

The test slice scores slightly *below* the tuning slice that selected the value (`0.6048` against `0.6531`), which is the ordinary direction for a held-out number and small enough to be noise on 111 queries.

### Off-domain coverage, measured and then closed a little

`data/intent_offdomain_probes.json` holds **45 in-scope probes**, 15 per route, in ordinary business/personal English — HR, finance, travel, facilities, scheduling, procurement — carrying none of the IR/ML vocabulary the canonical set was curated from. They were authored and committed in #523 **with no score attached**, deliberately, so they could not be curated against.

Measured at the shipped settings before any anchor was added:

| | value |
|---|---|
| argmax accuracy | `0.8222` (37/45) |
| abstained | 19/45 (42%) |
| wrong best-guess | 8/45 |
| **served wrong** | **2** |

That is *better* than the in-domain test slice scored at the time (`0.8018`), which makes the original "off-domain traffic abstains more often than it should" framing largely obsolete — see [Known limitations](#known-limitations) for why the comparison with #511's `13/16` is not like-for-like.

**The residual weakness was a route, not a vocabulary.** All 6 of the search-route probe failures were `search` misread as `tool` or `chat` — "find the signed lease", "who is listed as the emergency contact", "is the office open on Monday" — while `chat` and `tool` had one each. `search` is also the weak route in-domain (25/37). Adding 24 `search` anchors covering business document and fact lookup moved:

| | before | after |
|---|---|---|
| off-domain argmax | `0.8222` | **`0.8667`** |
| off-domain abstention | 42.2% | **37.8%** |
| test-slice argmax | `0.8018` | **`0.8108`** |
| **test-slice served accuracy** | `0.9825` | **`1.0000`** |
| out-of-scope AUC | `0.8622` | **`0.8720`** |
| hard_40 argmax | `0.7500` | `0.7250` |
| leave-one-out | `0.8393` | `0.8191` |

**Two rows moved the wrong way and neither is what it looks like.** `hard_40` argmax fell by one query, but its *served* accuracy rose (`0.8947` → `0.9048`) on more coverage (19 → 21) — at the operating point the adversarial set improved too, and that is why no floor is pinned to its argmax. Leave-one-out falling is expected and arguably healthy: #518 established it measures the anchor set's self-consistency, and 24 anchors in a vocabulary region the set did not previously cover legitimately lower how well anchors recover their own route from their neighbours.

**These probes are now spent as a clean instrument, and that is recorded rather than hidden.** Their per-item failures were read in order to decide what anchors to write, so they are a development set from here on. The genuinely held-out gates for this change were the ones that had never been looked at for this purpose: test-slice accuracy, hard_40, and out-of-scope AUC. A future off-domain number wanting to be a *result* needs a fresh probe set, authored the same way.

### The instrument, widened — and what that revealed about `top_k`

The router was judged on **111 held-out queries against 24 out-of-scope probes** through #524. Every conclusion in that period turned on one or two of them: the promotion bar cleared by 2 queries, `k=15` beating `k=3` by 1, `hard_40`'s regression 1 of 40.

The instrument is now **201 test queries against 60 probes**, and the probes are **split** — 29 tuning, 31 reporting, stratified by category and disjoint. That closes a caveat carried since #512: the same probes used to both tie-break threshold selection *and* denominate the reported AUC, so the headline separability figure was never fully held out from the thresholds it was measured at.

**Then the wider instrument re-selected `top_k`, and this is the finding that matters most.**

`TOP_K` has now been `3` (never swept, #511–#521), `15` (chosen on the 111-query instrument, #522), and `8` (chosen on this one) — all under the *same* pre-registered rule. Nothing about the rule changed. What changed is the tuning slice, which now samples 40 clean queries from a pool of 241 instead of 151, so it is a different 40.

That instability is itself the result. **A hyperparameter that moves from 15 to 8 when the instrument grows was never really "chosen" at 15** — it was chosen by a slice too small to distinguish the candidates. The tuning curve shows why: `k=8`, `15` and `25` sit within `0.014` of each other on tuning accuracy, and the pre-registered tie-break toward lower `k` decides among them. On a slice of 70, that gap is a couple of queries.

The honest reading is that `k` is under-determined anywhere in `8`–`25`, and the shipped value is the conservative end of a plateau rather than an optimum. That is a better-founded position than "15 is the answer", and it is only visible because the instrument grew.

| | old instrument (111 / 24) | wider instrument (201 / 31 held-out) |
|---|---|---|
| test-slice argmax | `0.8108` | `0.8159` — CI `[0.757, 0.863]` |
| coverage / served accuracy | `0.523` / `1.0000` (58) | `0.597` / `0.9667` (120) |
| out-of-scope AUC | `0.8720` | `0.8578` |
| selected `top_k` | `15` | **`8`** |

**Neither AUC number is a regression against the other** — they are different measurements. The old one was computed against probes that had helped select the thresholds it was reported at; the new one is computed against probes no sweep has seen. The AUC floor was lowered `0.85` → `0.83` for exactly that reason, recorded in the bar comments rather than done silently.

**What is still not fixed.** 201 queries is better, not sufficient: the interval on the headline is still `±0.05`, and `hard_40` remains 40 queries with an interval of roughly `[0.55, 0.82]` — too wide to floor, which is why no bar is pinned to it.

### The grid that must not be a constant

#520 derived `0.84` by hand from the tuning quantiles **at `k=3`**. When `top_k` moved to `15`, that constant excluded **325 of 328** candidate module scores — module emission would have collapsed to the top-1 fallback for nearly every query, silently, with no error.

Module scores fall as `k` rises, because a wider `k` averages in more distant same-module neighbours. So the threshold is a function of *two* things that change: the encoder **and** `top_k`.

This is the third time a grid written in the wrong units has bitten. The margin grid in #512 was in MiniLM's units and selected **nothing at all** — that one failed loudly. The module grid here would have selected something meaningless instead, which is worse. `_module_score_grid` now computes the endpoints from the tuning slice at whatever `top_k` is in force, which removes the whole class rather than fixing this instance of it.

The one grid still written down is `_SWEEP_TOP_K`, and that is deliberate: it is the pre-registration that keeps `k`-selection honest, so it must *not* adapt to data.

**Route accuracy is unchanged to the digit, and that is a property rather than a lucky result.** `_emit_modules` runs *after* `decide()` has taken its argmax over `route_scores`, so no module threshold can move a route. That is why this gate could sit dead for six PRs without a single request being routed wrongly, and why re-deriving it needed no re-litigation of the headline. `test_the_module_sweep_never_changes_the_route` asserts it across the whole grid plus both extremes rather than trusting the argument.

**What it cost.** Emission fell from ~4.7 modules per query to ~1.2, so gold modules are now missed where before every one was emitted, buried among three or four wrong ones. Macro-F1 says the trade is clearly net-positive, but it *is* a trade, and a consumer that needs recall over precision should re-run the sweep under a rule that says so rather than treating `0.821` as settled.

### The ceiling finding, corrected: `TOP_K` was never swept

Under **MiniLM**, leave-one-out accuracy over the canonical set at the shipped `TOP_K=3` measured `0.6750`: scoring each of the 280 anchors against the other 279 with the same top-k-mean rule, a third of them could not recover their own route. (Every figure in this section is MiniLM's and predates both the encoder swap and #516's de-duplication; on today's canonical set the same MiniLM measurement is `0.6643`, and the serving encoder's is `0.7321`. The section is kept for the reasoning, not the values.) An earlier version of this document read that `0.6750` as a representation ceiling — "`all-MiniLM-L6-v2` sentence embeddings with top-3-mean cosine top out near `0.67`–`0.70` no matter how good the examples get" — and named a stronger encoder as the only remaining lever. **That was overstated.** `TOP_K = 3` is an arbitrary constant that was never swept, and sweeping it — same encoder, same 280 anchors — moves both numbers substantially:

| `top_k` | clean_151 accuracy | hard_40 accuracy | out-of-scope separation margin | leave-one-out accuracy |
|---|---|---|---|---|
| **3 (shipped)** | 0.6225 | 0.6250 | **0.1188** | 0.6750 |
| 5 | 0.6358 | 0.6000 | 0.1036 | 0.6893 |
| 8 | 0.6556 | 0.6500 | 0.0918 | 0.7036 |
| **15** | **0.6887** | 0.6500 | 0.0767 | **0.7464** |
| 25 | 0.6755 | 0.6250 | 0.0654 | 0.7643 |

At `k=15`, leave-one-out reaches `0.7464` and clean_151 accuracy reaches `0.6887` — both well above the shipped `k=3` numbers on the same encoder and the same anchors. The `0.6750` figure this document previously called a ceiling reflects `TOP_K`'s arbitrary value at least as much as it reflects the encoder's representation.

**The trade, plainly stated:** out-of-scope separation falls from `0.1188` to `0.0767` as `k` rises from 3 to 15, because averaging more neighbors lifts the confidence floor for every route — including routes the request has nothing to do with — so accuracy and abstention pull against each other. There is no `k` that improves both at once in this table.

**The caveat:** the gap between `k=8` (`0.6556`) and `k=15` (`0.6887`) on clean_151 is about five queries out of 151. That is well within the noise a single held-out slice can produce, so `k` must be chosen on a validation split with the accuracy/abstention trade decided deliberately — not read off this table as if it named a single best value.

**What survives:** the hard stop still stands at every `k` tested. `bulk_181` — the decision-rule input, not clean_151 — lands near `0.72` at `k=15`, still under the `0.75` promotion bar. A stronger encoder and a swept `k` both remain live levers; `k` is the cheaper one to try first, because it costs a config change and a re-run of the evaluation CLI, not a new model.

### `top_k` chosen on the split

`TOP_K` was `3` from #511 to #521 — an arbitrary constant, never swept for selection. It is now `8`, chosen on the tuning slice by a rule registered before the sweep ran: **highest served accuracy at coverage ≥ `0.60`, ties to higher out-of-scope deferral then to lower `k`.** That is the existing `_select_thresholds` rule extended to a second dimension, not a new one invented for the occasion, and `_select_thresholds` now searches `(top_k, min_margin)` jointly.

**Jointly is the important word.** Raising `k` compresses margins, so a `k` evaluated at the old `min_margin=0.015` is evaluated at the wrong threshold for itself. The pair that won on the retired 111-query instrument was `k=15, min_margin=0.010`; the wider instrument subsequently selected `k=8` with the same margin. The following tables describe that earlier selection.

Best eligible row per `k` on the **tuning** slice — tuning numbers, not results:

| `top_k` | served accuracy | coverage | `min_margin` | out-of-scope deferral |
|---|---|---|---|---|
| 3 *(previous)* | 0.9130 | 0.657 | 0.015 | 0.500 |
| 5 | 0.9762 | 0.600 | 0.015 | 0.583 |
| 8 | 0.9583 | 0.686 | 0.010 | 0.583 |
| **15 — selected** | **0.9783** | 0.657 | **0.010** | **0.667** |
| 25 | 0.9778 | 0.643 | 0.010 | 0.667 |

**The accuracy-versus-abstention trade this document has asserted since #511 does not survive being measured at the operating point.** The raw separation margin does fall monotonically with `k` (`0.0416` → `0.0303`), which is what the earlier tables showed and why "no `k` improves both" looked true. But raw margin is a ranking statistic over the whole score range; the gate only needs separation at the boundary. At each `k`'s own tuned threshold, out-of-scope *deferral* **rises** with `k`, from `0.500` at `k=3` to `0.667` at `k=15`.

This is the same lesson as [the out-of-scope regression](#the-out-of-scope-regression-measured-where-it-bites): a scale-free ranking statistic and the behavior at the operating point can point in opposite directions, and only one of them is what serving does.

Every held-out number improved together, which is not what a genuine trade looks like:

| | `k=3` (previous) | `k=15` (selected) |
|---|---|---|
| argmax accuracy, test slice | `0.7928` | **`0.8018`** |
| coverage / served accuracy | `0.4505` / `0.9600` | **`0.5135` / `0.9825`** |
| wrong routes served | 2 | **1** |
| hard_40 | `0.6750` | **`0.7500`** |
| out-of-scope AUC | `0.8551` | **`0.8622`** |
| Cohen's d | `1.4852` | **`1.6030`** |
| leave-one-out | `0.7321` | **`0.8393`** |
| p95 latency | `11.47ms` | `12.20ms` (ceiling 25ms) |

**What giving up the pin cost.** Pinning `k` used to guarantee arithmetically that no threshold the sweep chose could move the abstention-blind argmax headline — which is what let the margin grid be re-derived in #512 *after* the headline was known. Selecting `k` couples them again. Three things replace that guarantee, and all three are load-bearing: `_SWEEP_TOP_K` is **pre-existing** from #511 and was not widened when it became a selection grid; selection still reads the tuning slice only; and the test slice was read once, at the selected pair. Widening that grid in future means re-registering it *before* re-running, never after seeing a number it would influence. `test_the_sweep_searches_top_k_only_over_the_pre_registered_grid` replaces the old pin-guard and asserts exactly that.

`hard_40` is deliberately absent from the per-`k` table: `evaluation_report.json`'s `top_k_sweep` never publishes a per-`k` curve over held-out data, only over the tuning slice and out-of-scope probes.

### Known limitations

- **`top_k` is `8`, and under-determined across `8`–`25`.** It is [chosen on the split](#top_k-chosen-on-the-split) rather than inherited, but it has now been *re-*chosen once: the same pre-registered rule picked `15` on the 111-query instrument and `8` on the 201-query one. That instability is the finding — a value that moves when the instrument grows was never determined at either setting. `_SWEEP_TOP_K` is `{3, 5, 8, 15, 25}` and the candidates sit within `0.014` of each other on tuning accuracy, so the tie-break toward lower `k` decides among them. The grid is deliberately not widened — doing so after seeing the headline it now influences is the fitting the split exists to prevent — so a finer search is possible but must re-register the grid first.
- **Topical concentration is much smaller than #511 measured, and mostly dissolved before it was addressed.** #511 reported 47% IR/ML vocabulary and **13 of 16** off-domain probes abstaining. Measured now against 45 purpose-built probes (`data/intent_offdomain_probes.json`, authored score-free in #523): **19 of 45 abstain (42%)** and off-domain argmax accuracy is `0.8222` — *higher* than the in-domain test slice's `0.8018` at the time. The encoder swap and the `top_k` selection did most of that, not any anchor work.

  Two caveats keep this from being a clean refutation. The instruments are not comparable: different probes, different encoder, different `k`, and #511's 16 probes were not committed, so the `13/16` cannot be re-run. And the 47% figure came from an unrecorded word list — a plausible regex measures the same set at **26.1%**, so those two numbers are not the same measurement either. What is solid is the current number, on a committed instrument anyone can re-run.

  Adding 24 business-vocabulary `search` anchors took off-domain argmax to `0.8667` and abstention to 37.8%, so the residual gap was real but modest.
- **The compose-versus-dispatch boundary.** "write an email to the vendor about the overage" sits at `0.963` cosine (MiniLM-measured) to the canonical "email the vendor about the overage". One verb apart, so it routes to `tool` without abstaining, when composing text is arguably `chat`. Adding more compose anchors did not fix it: the two phrasings are near-identical to the encoder and genuinely ambiguous to a human reader.
- **Route imbalance, and it moved with the encoder.** Under MiniLM the `tool` route scored 45/50 on the clean slice while `search` and `chat` sat near 25/50. Under e5 the imbalance inverts and has held across two instruments: on the 201-query test slice `chat` is 60/67 and `tool` 53/67, while **`search` is the weak route at 51/67** — the same ordering it showed at 33/37 · 30/37 · 25/37 on the retired 111-query slice. Route-level error is therefore a property of the encoder's representation at least as much as of the route, and any conclusion drawn from one encoder's per-route table does not carry to another's.
- **Module emission is derived now, and recall paid for it.** The threshold moved `0.45` → `0.821` (see [the module threshold](#the-module-threshold-derived-and-no-longer-dead)), trading recall for precision: mean modules emitted falls from ~4.7 to ~1.2. Macro-F1 on the test slice roughly doubles (`0.3492` → `0.6048`) and joint accuracy goes `0.0` → `0.4595`, so the trade is clearly net-positive — but it *is* a trade, and a consumer that needs recall over precision should re-run the sweep under a rule that says so rather than treating `0.821` as settled.
- **Module recall is the open question, not observability.** Margin abstentions, module labels and the composite flag now reach production telemetry as `route_fallback_reason="margin_below_threshold"`, `route_modules` and `route_composite` — no debug panel required. What is still unknown is whether the abstention pattern measured on the test slice (`search` serving 11 of 37, `chat` 24 of 37 and `tool` 22 of 37 with no errors at all) holds on real traffic; that needs the telemetry to actually accumulate before it can be answered.
- **The evaluation set is partly contaminated, and 40 more queries are now spent on tuning.** The legacy 30 were used as feedback while the canonical set was being curated, so their score is optimistic and must never be quoted as the router's accuracy; they are spent on the tuning slice for exactly that reason. A further 40 clean queries join them there, which leaves **201** for the honest measurement. That was 111 until the instrument was widened; the 40 spent on tuning are now a much smaller share of the clean pool (40 of 241, against 40 of 151 before), which is why the tuning slice held at 70 while the test slice nearly douburement even where they look comparable.
- **The pinned bars now run in CI, in their own lane.** `tests/unit/test_intent_evaluation.py` and the near-duplicate bar in `tests/unit/test_intent_canonical_data.py` need `sentence-transformers` and a built `data/intent_index/`, so they skip in the fast unit-test job — which deliberately excludes heavy ML packages, and adding them there would reinstate the torch-in-CI collection failures this repo has fixed twice. The `Intent Routing Gate` job in `.github/workflows/eval-gate.yml` installs the encoder, rebuilds the index from the committed canonical set, and runs the bars for real on every pull request. **It fails if any bar reports SKIPPED, and if the bar count drops below a floor** — a bar that skips measured nothing, and a bar that quietly disappears measured nothing either, so a job that let either pass would stay green in exactly the situation it exists to catch. Unlike the retrieval and RAGAS gates, this one needs no committed baseline and no secret: everything it reads is in the repo already, so it is active from its first run.

- **The latency bar stays local-only, and deliberately so.** `test_routing_one_request_stays_under_the_latency_ceiling` is deselected in CI. It measures wall-clock p95 for one encode-and-decide, and the `25ms` ceiling was set against real serving hardware; a shared CI runner is 2–3× slower, so it fails on hardware rather than on anything the gate guards. Its first run in CI measured `29.0ms` against `11.5ms` on a developer machine with no code change between them. It remains pinned and enforced locally, where the number means something. Every bar the gate does run — accuracy, separability, and the canonical-set invariants — is hardware-independent.
- **The near-duplicate pair is fixed, and the ceiling tightened behind it.** "there was a policy about retaining user transcripts" and "what does the retention policy say about transcripts" (both route `search`) scored `0.9340` — the measured maximum over all 39,060 pairs, and a genuine duplicate: two phrasings of one request, on a topic three other anchors already covered. `canon-auth-011` was re-subjected to the failover drill, keeping its vague-recall register and its pure `lookup_document` role. The cleaned maximum is `0.9271` with **no pair at or above `0.93`**, so `_MAX_INTERNAL_COSINE` tightened `0.95` → `0.94`, restoring the rule the constant originally encoded — a hair above a clean set's maximum. The four highest surviving pairs are deliberate contrasts, cross-route or cross-module on a shared subject, not duplicates.

  Worth recording what the fix measured, because it is the argument for doing this kind of cleanup at all: **every held-out accuracy was unchanged** — test-slice accuracy `0.7928`, hard-40 `0.6750`, out-of-scope AUC `0.8551` — while **leave-one-out fell `0.7393` → `0.7321`**. The duplicate had been propping up leave-one-out by making each anchor the other's nearest neighbour, so the set scored better against itself than its content justified. Removing it cost nothing real and made the self-consistency number honest.

  **Two distribution statistics moved with it, and were missed at the time**: Cohen's d `1.4747` → `1.4852` and the out-of-scope raw margin `0.0280` → `0.0278` (MiniLM's equivalents moved the same way, `1.6208` → `1.6365` and leave-one-out `0.6750` → `0.6643`, since the canonical set is shared by both encoders). Neither changes any conclusion — d moved *up*, and raw margin is encoder-specific context that is never a bar — but "unchanged" was too strong a claim: what was invariant was accuracy and the AUC ranking, not every number the report emits. Anything that summarizes the *spread* of the score distribution moves when an anchor moves, and only the tables that quote the accuracy bars are safe to leave alone after a canonical edit.

### Promotion: what would have to be true

The router has shipped dark since #511. `AGENTIC_SEARCH_INTENT_INDEX_PATH` is unset by default and every request falls through the LLM/rule cascade. Seven changes have improved the artifact behind that flag, each ending with some version of "promotion is a separate change, reviewed on its own terms" — without anyone writing down what that review consists of. This is that.

**Nothing here promotes the router**, and none of it argues that it should be promoted.

#### The evidence, stated with its uncertainty

| | value |
|---|---|
| argmax accuracy, test slice (201) | `0.8159`, 95% CI **`[0.757, 0.863]`** |
| coverage | `0.597` — 120 of 201 answered |
| served accuracy | `0.9667`, 95% CI `[0.917, 0.987]` — **4 wrong in 120** |
| hard_40 argmax | `0.7000`, 95% CI `[0.546, 0.819]` |
| out-of-scope AUC (31 held-out probes) | `0.8578` |
| p95 routing latency | `12.20ms`, ceiling `25ms` |

**The `0.80` promotion bar sits inside the accuracy interval.** The point estimate clears it; the interval does not. That is the honest summary, and it does not improve by tuning — only by more queries.

#### The checklist

Pre-registered here so a future decision is taken against stated criteria rather than whichever number is quoted that day. **All five must hold.**

1. **Accuracy interval clears the bar.** Not the point estimate — the *lower bound* of the 95% CI at or above `0.80`. Today that is `0.757`. This is the criterion the current instrument cannot satisfy at any tuning, and closing it means a larger eval set.
2. **Served accuracy at or above `0.95`** with its lower CI bound at or above `0.90`. Today: `0.9667`, lower bound `0.917`. **Met.**
3. **Out-of-scope AUC at or above `0.85`** on probes no sweep has seen. Today: `0.8578`. **Met**, and note this bar is lower than the `0.90` #512 set for itself — that figure came from e5-*base*, a model this work does not use, and was never reachable.
4. **Production shadow data agrees with the eval sets.** Shadow mode must have run long enough to compare the deferral rate and route mix against the test slice's `0.597` coverage. Unmet: no shadow data exists yet.
5. **The cost is accepted explicitly.** At `0.597` coverage the router defers ~40% of auto-routed traffic to the LLM classifier and adds `12.20ms` p95 to every request. That is a budget decision, not a correctness one, and it needs an owner rather than a metric.

**Today: 2 of 5.** The blocking criteria are (1), which needs a larger instrument, and (4), which needs shadow mode to have run.

#### Shadow mode

`AGENTIC_SEARCH_INTENT_SHADOW_MODE=true`, with an index configured, scores requests reaching the similarity step after explicit overrides and deterministic rules, and records what the router *would* have decided — then discards it and falls through to the classifier exactly as if no index were configured.

```
route_shadow_intent            the route it would have returned
route_shadow_abstained         whether it would have deferred
route_shadow_fallback_reason   why, when it would have
```

Deliberately distinct from `route_predicted_intent`, which both modes populate, so a shadow run can never be read back as a served one. This converts promotion from a bet into a measurement: criterion 4 is answerable without any request being routed by the model.

#### Rollback, and the trap in it

Unsetting `AGENTIC_SEARCH_INTENT_INDEX_PATH` disables learned routing. **It does not take effect until the process restarts**, because the loaded index is cached by resolved path and never invalidated.

The same cache makes the *enabling* direction fragile in a way worth stating plainly: **a failed load is cached too.** Starting the web process before the index file exists leaves routing disabled until the next restart, even after the file appears. So:

- **enabling**: build the index, verify the file exists, *then* start the process
- **rolling back**: unset the variable, *then* restart — the variable alone changes nothing in a running process
- **any rollout plan that assumes runtime toggling is wrong** for both directions

Rollback triggers worth pre-committing: served accuracy below `0.90` on production feedback, deferral rate more than 10 points from the shadow-measured `0.597`, or p95 routing latency above the `25ms` ceiling.

#### The decision, for now

**Not yet, on criterion 1.** The accuracy interval contains the bar, so the evidence does not distinguish "clears it" from "does not". Criterion 4 is also unmet and is the cheaper of the two to close — shadow mode exists now, and running it costs nothing because the router stays dark while it gathers.

This is a better-founded "not yet" than the previous seven: it names what would change it.

### Deploying an index

The loaded index is cached by resolved path and is never invalidated, so: **rebuild the index, then restart the web process.** A *failed* load is cached too — starting the web process before the index exists leaves learned routing disabled until the next restart, even after the file appears.

**An index built before the e5 swap must be rebuilt**, or the encoder-name check disables the route on load — deliberately, since both encoders are 384-wide and the alternative is silently meaningless routing.

The e5-small-v2 encoder itself loads lazily on the first request that reaches configured similarity routing, separately from the index, and blocks that request for roughly two seconds while the model loads. This is not the promotion-gate activation checklist above; it is a separate one-time cost the first caller pays. A failing model fetch (missing weights, unreachable HuggingFace) is cached as a failure the same way the index's failed load is: the route disables itself and every later request degrades straight to the LLM classifier instead of retrying the download per request.

### The units trap

`AGENTIC_SEARCH_INTENT_MIN_ROUTE_MARGIN` and `AGENTIC_SEARCH_INTENT_MIN_MODULE_SCORE` are **cosine similarities, not softmax probabilities**, and **its scale has changed again with the encoder**. Three scales in two changes:

| | in-scope confidence, typical | a plausible-looking `0.60` would |
|---|---|---|
| retired MLP (softmax) | above `0.9` | pass almost everything |
| all-MiniLM-L6-v2 (cosine) | mean `0.378` | abstain on almost every request |
| **intfloat/e5-small-v2 (cosine), `k=15`** | **`0.776`–`0.857`, probes `0.767`–`0.829`** | **pass everything, in scope or not** |

These historical distributions were measured at `k=15`; they do not describe the current `k=8` operating point. Under e5, the retired `0.30` confidence floor sat below the distribution and provided almost no additional rejection beyond the margin gate. On the earlier 111-query instrument, margin `0.010` served 57 queries at `0.9825` accuracy, and the module threshold was `0.821`. Current defaults remain margin `0.010`, module score `0.8215`, and `top_k=8`, selected on the wider instrument described above. Confidence has no serving threshold now.

Re-derive on the **tuning slice**, never on the test slice. The old 270-row sweep included an absolute confidence floor; that dimension has been removed. `evaluation_report.json` records the current joint `(top_k, min_margin)` sweep under `threshold_tuning.sweep` and its selected settings under `threshold_tuning.selected`. Module thresholds are derived separately at the selected `top_k`. A future encoder swap requires the same review of score units before examining held-out headline results.

`AGENTIC_SEARCH_INTENT_MODEL_PATH` no longer exists. Serving reads `AGENTIC_SEARCH_INTENT_INDEX_PATH`, a directory holding an `index.npz`. Building or evaluating an index never changes a serving setting.

## Runnable examples

### Agent CLI

| Mode | Loop | Needs retrieval server | Use it for |
|------|------|------------------------|------------|
| `single` | `PlainGenerationLoop` | No | Local generation smoke tests |
| `search` | `SearchAgentLoop` | Yes | Multi-turn RAG, SFT, and RL traces |
| `tool` | `ToolAgentLoop` | Yes | Structured tool-calling experiments |

```bash
# single — no retrieval server needed (plain generation)
# Apple Silicon: use --device mps --allow_unsafe_mps for ~50x faster inference
python3 -m examples.run_agentic_search \
  --mode single --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --device mps --allow_unsafe_mps \
  --allow_remote_model_downloads

# single with retrieval server — small models (≤3B) use --mode single; search/tool require 7B+ to emit structured tags
python3 -m examples.run_agentic_search \
  --mode single --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --device mps --allow_unsafe_mps \
  --search_url http://localhost:8001/retrieve --allow_remote_model_downloads

# search — 3B is the Mac sweet spot (~6 GB unified memory); 7B needs 16 GB+ and will swap
python3 -m examples.run_agentic_search \
  --mode search --question "What is RAG?" \
  --model Qwen/Qwen2.5-3B-Instruct --local --device mps --allow_unsafe_mps \
  --search_url http://localhost:8001/retrieve --allow_remote_model_downloads

# search — server-backed, requires vLLM on :8080 and retrieval on :8001
python3 -m examples.run_agentic_search \
  --mode search --question "Compare dense and sparse retrieval" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_url http://localhost:8080 --search_url http://localhost:8001/retrieve
```

### Intent operating point

Compares the serving encoder against the previous one at the threshold each would actually run at — coverage, served accuracy, and wrong-route count — rather than at the abstention-blind argmax the evaluation report headlines. Tunes each encoder's `min_margin` on the tuning slice and reports on the test slice. This is the evidence behind [The out-of-scope regression, measured where it bites](#the-out-of-scope-regression-measured-where-it-bites).

```bash
python -m examples.measure_intent_operating_point
```

Needs sentence-transformers and both models; MiniLM is pulled only for the comparison. Takes about a minute on CPU.

### Bamboogle evaluation

Always requires the retrieval server on port 8001.

```bash
# Smoke test — local model, 1 example, full trace printed
python3 -m examples.run_bamboogle_eval \
  --model Qwen/Qwen2.5-3B-Instruct --local --device mps --allow_unsafe_mps \
  --search_url http://localhost:8001/retrieve --limit 1 --print_trace \
  --allow_remote_model_downloads

# Full benchmark — Apple Silicon, requires SERP_API_KEY in .env
bin/run_bamboogle_eval.sh --limit 125
```

### PPO/GRPO reward

```bash
python3 -m examples.run_grpo_training_pipeline         # end-to-end reward + GRPO smoke test (no GPU, no model)

# Simulated-judge GRPO — actually updates a policy: bamboogle prompts → generate →
# SimulatedPreferenceJudge → GRPO step. No retrieval server; runs on CPU/MPS.
python3 -m examples.run_bamboogle_grpo_train \
  --model Qwen/Qwen2.5-0.5B-Instruct --device cpu \
  --allow_remote_model_downloads --steps 10
```

### Search pipeline with access filters

Retrieve, enforce access, assemble answer context — four steps over the same
code the web backend calls. No live model, retrieval server or database is
required; the corpus is a literal in the script.

```bash
python3 -m examples.run_search_pipeline
```

Sending filters to a retrieval backend is not enforcement: the backend is free
to ignore them, so the caller checks the documents it gets back. Drop that check
to see what an unchecked backend would hand an anonymous reader:

```bash
python3 -m examples.run_search_pipeline --skip-enforcement
```

`--group search-admins` and `--email owner@example.test` show the two grants the
demo corpus carries.

## Dataset preparation

```bash
# Offline local RAG smoke test (4 examples, existing 30-document demo corpus)
python3 -m examples.prepare_local_rag_smoke_dataset --topk 1 --preview

# Write compact RAG parquet after inspecting the preview
python3 -m examples.prepare_local_rag_smoke_dataset \
  --topk 1 --output_path data/local_rag_smoke.parquet
```

This command requires no retrieval server, network access, FlashRAG dataset, or retrieval caches.

Optional large-dataset workflows:

```bash
# Optional: Search-QA parquet preparation
python3 -m examples.prepare_search_qa_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets --dataset_config nq --local_dir data/nq_search

# Preview before writing
python3 -m examples.prepare_search_qa_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets --dataset_config nq \
  --splits test --max_examples 20 --preview --preview_rows 5

# Optional: full NQ RAG parquet preparation
# Requires an external Wikipedia corpus plus retrieval-cache JSON files keyed
# by NQ question. prepare_search_qa_dataset does not create these inputs.
python3 -m examples.prepare_search_rag_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets --dataset_config nq \
  --corpus_path data/wiki-18.jsonl \
  --train_retrieval_cache data/nq_train_retrieval_cache.json \
  --test_retrieval_cache data/nq_test_retrieval_cache.json \
  --topk 3 --local_dir data/nq_rag
```

## Training

The training pipeline is modular: generate trajectories → score with rewards → compute advantages → optimize.

| Task | Entry point |
|------|-------------|
| QA parquet preparation | `python3 -m examples.prepare_search_qa_dataset` |
| Training data (shell) | `bin/generate_training_data.sh` |
| Reward/GRPO smoke test | `python3 -m examples.run_grpo_training_pipeline` |
| Bamboogle benchmark eval | `python3 -m examples.run_bamboogle_eval` / `bin/run_bamboogle_eval.sh` |
| Unseen-user evaluation (simulated cohort) | `python3 -m examples.run_unseen_user_eval` |
| Unseen-user eval report (simulated cohort) | `docs/benchmarks/unseen-user-evaluation.md` |
| DPO trainer | `src/model/post_training/dpo/trainer.py` |
| DPO preference pairs | `src/model/post_training/dpo/data.py` |
| Reward function | `src/model/post_training/reward.py` |
| Shared response log-probs (DPO + GRPO) | `src/model/post_training/log_probs.py` |
| Rollout trace plots | `python3 -m examples.plot_grpo_rollouts` |
| Optimization benchmarks | `python3 -m examples.benchmark_grpo_optimization` |
| Simulated preference judge | `src/model/post_training/grpo/algorithms.py` |
| GRPO helpers | `src/model/post_training/grpo/algorithms.py` |
| Online GRPO for HF LMs | `src/model/post_training/grpo/training.py` |
| Agent-loop GRPO (full reward) | `src/model/post_training/grpo/training.py` |
| PPO core (clipped surrogate, KL controllers) | `src/model/post_training/ppo/core_algos.py` |
| GRPO/REINFORCE advantages and losses | `src/model/post_training/grpo/algorithms.py` |
| Tabular Q-learning demo | `src/model/post_training/qlearning/` |
| Generation and policy loss | `src/model/post_training/grpo/generation.py` |
| Feedback-driven GRPO | `python3 -m examples.run_feedback_grpo` |
| SFT warm-start + GRPO | `python3 -m examples.run_sft_grpo` |
| Simulated-judge GRPO (policy update) | `python3 -m examples.run_bamboogle_grpo_train` |

The unseen-user evaluation runs against a **simulated cohort**, not real
users: this repository holds two users and zero feedback rows. Its report is
evidence that the pipeline detects an effect of the planted size, on held-out
users, at the reported power -- never that a model converts real users. The
report states its own provenance, planted effect sizes and regeneration
command; do not quote a number from it without them.

### Direct Preference Optimization (DPO)

`src/model/post_training/dpo/` trains a policy directly on preference pairs — no reward
model, no critic, no online sampling. The reference policy plays the role GRPO's
KL penalty plays, anchoring the policy to where it started.

**There is no CLI entry point yet.** DPO is used as a library:

```python
from src.model.post_training.dpo import load_preference_pairs
from src.model.post_training.dpo.trainer import DPOConfig, DPOTrainer

pairs = load_preference_pairs("data/dpo_pairs.jsonl")
trainer = DPOTrainer(
    policy=model,                 # initialized from your SFT checkpoint
    tokenizer=tokenizer,
    optimizer=torch.optim.AdamW(model.parameters(), lr=5e-7),
    config=DPOConfig(beta=0.1, epochs=1),
)
history = trainer.train(pairs)     # list[DPOStep]: loss, margin, accuracy
trainer.save("checkpoints/dpo")
```

The reference defaults to a frozen `deepcopy` of the policy at construction,
which is correct when the policy was initialized from the SFT checkpoint — the
usual DPO setup. Pass `reference_policy=` to point at a different one.

**Input format.** One JSON object per line:

```json
{"prompt": "What is FAISS?", "chosen": "A library for efficient similarity search.", "rejected": "A database from Google."}
```

The loader is strict — malformed JSON, a missing or blank field, an empty file,
or `chosen` equal to `rejected` all raise, naming the offending line. These pairs
*are* the training signal, so a silently skipped row changes what the model
learns with no later symptom. (An identical chosen/rejected pair contributes
exactly `log 2` of constant loss and zero gradient, so it can never teach
anything.)

**Reading the metrics.** `loss` alone cannot distinguish learning from collapse.
Watch `margin` (the implicit reward gap) and `accuracy` (the fraction of pairs
whose gap is positive). Both start at **0** by construction, because the policy
starts equal to the reference — a first-step loss of `log 2` ≈ 0.693 is the
expected, correct starting point, not a bug.

**Where pairs come from.** Today: a file you supply. The `retrieval_feedback`
table records one thumbs-up/down *per session* and has no notion of two
competing answers to a prompt, so it cannot yield pairs directly; and
`SimulatedPreferenceJudge` is a deterministic heuristic placeholder, so
judge-labelled pairs would be exactly as good as that heuristic. `DPOTrainer`
takes `list[PreferenceExample]` and never reads a file, so either source can feed
it later without changing the trainer.

### Fine-tune from user feedback

Train directly on thumbs-up/down sessions collected via `POST /api/feedback` (no GPU required for the smoke path; `--device mps` on Apple Silicon):

```bash
# Feedback-driven GRPO: load rated sessions from the web DB → reward with human_signal → update
python3 -m examples.run_feedback_grpo \
  --db_path data/feedback.sqlite3 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --min_ratings 10 --human_feedback_weight 0.5 \
  --num_rollouts 4 --search_url http://localhost:8001/retrieve --device mps \
  --output_dir data/checkpoints/feedback_grpo/

# SFT warm-start (Phase 1, assistant-token-only CE on thumbs-up traces) then GRPO (Phase 2);
# --sft_epochs 0 skips Phase 1 and runs pure GRPO from the base model
python3 -m examples.run_sft_grpo \
  --db_path data/feedback.sqlite3 --model Qwen/Qwen2.5-1.5B-Instruct \
  --jsonl_path data/sft_pairs.jsonl \
  --sft_epochs 3 --sft_lr 2e-5 --sft_output_dir data/checkpoints/sft_warmstart/ \
  --grpo_output_dir data/checkpoints/sft_grpo/ --device mps
```

`load_feedback_examples` raises if fewer than `--min_ratings` rated sessions exist, so collect feedback first (thumbs in the UI, or `POST /api/feedback`). There is **no HTTP training endpoint** — fine-tuning is offline by design; the only backend endpoint in this loop is `POST /api/feedback` (see [Web Backend API](api-reference.md#web-backend-api)).

### Reward components

`SearchRewardFunction` uses these components:

| Component | Config field | What it measures |
|-----------|-------------|-----------------|
| Correctness | `correctness_weight` | Judge score against gold answer (EM / contains-match) |
| Citation support | `citation_support_weight` | Fraction of retrieved docs cited in the final answer |
| Subquestion coverage | `subquestion_coverage_weight` | Fraction of sub-questions with sufficient evidence |
| Search quality | `search_quality_weight` | Evaluator verdict + per-query search quality |
| Unnecessary search | `unnecessary_search_penalty` | Penalty per search round beyond the first |
| Unnecessary fetch | `unnecessary_fetch_penalty` | Penalty per fetched page not cited in the answer |
| Fetch usefulness | `fetch_usefulness_reward` | Bonus when fetched pages are cited in the final answer |
| Format compliance | `format_reward_weight` | Structural compliance in the final answer |
| Human feedback | `human_feedback_weight` | `human_signal` (±1.0) from thumbs-up/down sessions; `0.0` by default (off) |

Reward preset names: `sparse_final_only` | `simple_sparse_with_search_penalty` | `second_pass` | `third_pass_with_format` | `retriever_aware` (see `SearchRewardConfig` in `src/model/post_training/reward.py`). The Bamboogle eval CLI (`run_bamboogle_eval --reward_preset`) exposes the shorthand `sparse_final_only | simple_sparse | second_pass | third_pass`, which map to the first four config presets; `retriever_aware` is config-only.

**The judge reads the gold answer now.** The `correctness` term is `judge_fn(answer, gold)`, and until recently the GRPO examples passed a judge that ignored the second argument entirely.

`SimulatedPreferenceJudge` scores an answer from its own text — length, lexical diversity, absence of hedging. A policy trained against it optimises *answer shape*, so **a confidently worded wrong answer outscored a correct terse one by construction**. It is still available (`--judge simulated`) for comparison, and it is no longer any script's default.

Two judges now read the reference:

| judge | what it does | when |
|---|---|---|
| `GoldAgreementJudge` | normalised exact match → `1.0`, gold contained → `0.7`, else scaled token-F1 | **default**; no network, fully deterministic |
| `LLMJudge` | LLM-as-judge over `(answer, gold)` behind the existing `GEN_AI_*` config | `--judge llm`; falls back to `GoldAgreementJudge` per item |

Three details that are load-bearing rather than incidental:

- **Partial credit is graded, not binary.** GRPO normalises advantages *within* a rollout group, so if every rollout for a prompt scores `0.0` then every advantage is `0.0` and the prompt contributes no gradient — while still logging as a completed step. Token-F1 partial credit keeps near-misses informative.
- **An unparseable LLM reply raises rather than defaulting.** A judge that quietly returns a middling constant gives every rollout in a group the same score, which is the same silent no-op. `LLMJudge` catches it, falls back for that item only, and increments `parse_failures` so a degraded run is visible.
- **`is_degenerate_group`** names the all-equal-scores condition directly, so it can be asserted on rather than discovered later as "training ran but the model did not move".

Scores are cached by `(answer, gold)` — GRPO scores G rollouts per prompt against one gold, and prompts recur across steps.

There is still no *trained* reward model; that remains a separate design.

**Four reward dimensions** — `reward_components()` also groups every term into four subtotals via `REWARD_DIMENSIONS`, emitted as `dim_correctness`, `dim_citation_support`, `dim_retrieval_quality`, `dim_search_efficiency` (and available directly via `reward_dimensions()` or the pure `group_reward_components(components)`). Pre-scale, so `sum(dims) == terminal_reward + shaping_total == total / reward_scale`. The rollup is purely additive — no weight, preset, or `total` formula changed.

**GRPO** — `score_prompt_group` scores G rollouts for one prompt and normalises within-group advantages. `compute_grpo_outcome_advantage` computes `reward_i - mean(group)` for a flat rewards list. See `src/model/post_training/grpo/algorithms.py`.

**PPO core** — `compute_ppo_policy_loss_core` returns `(pg_loss, pg_clipfrac, ppo_kl, surrogate)` and is the clipped surrogate the GRPO trainers use, with a group-relative advantage in place of GAE. It requires an `eos_mask` tensor. See `src/model/post_training/ppo/core_algos.py` — `ppo/` is a **base algorithm layer, not a training method**: it has no trainer, no critic and no GAE, and `grpo/` depends on it because GRPO *is* this surrogate with a group-relative advantage substituted in.

The PPO-**with-critic** path (`compute_value_loss`, `compute_gae_advantages`) used to live alongside it, exported from three surfaces and called only from tests. It was removed: training here is critic-free GRPO, there is no value model, value head, or critic anywhere in the repo to produce the `values` those helpers consume, and so the path could never be exercised end to end. Exported-but-unreachable code reads as supported API and its test coverage implies a path that is exercised rather than merely arithmetic-checked. If a critic is ever wanted, the git history has both functions.

### Smoke test

End-to-end reward + GRPO, with no GPU:

```bash
python3 -m examples.run_grpo_training_pipeline
```

### XML search protocol

The ReAct-style trace format used by `SearchAgentLoop` uses these model-output tags:

```xml
<think>decide whether to answer or search</think>
<search>one precise query when external evidence is needed</search>
<fetch>comma- or newline-separated URLs when snippets are insufficient</fetch>
<answer>final grounded answer with citation labels</answer>
```

Optional model-output tags for multi-hop tasks:

```xml
<search_decision>answer</search_decision>   <!-- skip search when internal knowledge suffices -->
<subquestions>one research subquestion per line</subquestions>
<searches>parallel independent queries, one per line</searches>
```

Environment-only tags (injected by the loop — never output by the model):

```xml
<information>search results with citation labels</information>
<search_evaluation>sufficiency verdict and weak-query hints</search_evaluation>
<subquestions_feedback>per-subquestion coverage status</subquestions_feedback>
<full_page>fetched page content</full_page>
```

Mask all environment-only tags from policy/SFT action loss.

## Evaluation

### Bamboogle

Bamboogle is a two-hop QA benchmark that requires chaining retrieval across multiple hops — a strong signal for `SearchAgentLoop` quality.

**CLI (local CPU):**

```bash
python3 -m examples.run_bamboogle_eval \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --limit 5 --print_trace
```

**CLI (server-backed):**

```bash
python3 -m examples.run_bamboogle_eval \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_url http://localhost:8080 \
  --search_url http://localhost:8001/retrieve \
  --reward_preset second_pass --limit 125
```

Reward presets: `sparse_final_only` | `simple_sparse` | `second_pass` | `third_pass`

**Apple Silicon shell script** (auto-starts SerpAPI retrieval server, reads `SERP_API_KEY` from `.env`):

```bash
bin/run_bamboogle_eval.sh                              # 5 examples, mps device
bin/run_bamboogle_eval.sh --smoke                      # 1 example, quick sanity check
bin/run_bamboogle_eval.sh --limit 125                  # full benchmark
bin/run_bamboogle_eval.sh --device cpu --limit 10
bin/run_bamboogle_eval.sh --limit 125 --concurrency 8  # ~6-8x faster via parallel SerpAPI calls
bin/run_bamboogle_eval.sh --limit 125 --concurrency 8 --resume  # resume an interrupted run
```

The dataset is cached locally after the first download (`~/.cache/agentic_search/bamboogle_test.jsonl`), so subsequent runs skip the network fetch. `--resume` reads the existing output file and skips already-evaluated questions, appending new results.

**Training data generation:**

```bash
bin/generate_training_data.sh                         # Bamboogle → data/bamboogle_train/
bin/generate_training_data.sh --preview               # print 5 sample rows, no write
bin/generate_training_data.sh --dataset nq            # Natural Questions
bin/generate_training_data.sh --dataset trivia_qa     # TriviaQA
bin/generate_training_data.sh --dataset hotpotqa --max_examples 500
```

Each run writes `data/<dataset>_train/train.parquet` and `data/<dataset>_train/test.parquet` ready for `LLMGRPOTrainer` or SFT.

### Activating the eval gates

The `Eval Gate` CI workflow (`.github/workflows/eval-gate.yml`) has three jobs. The **Intent Routing Gate** is active already — it needs no baseline and no secret, because the canonical set, the evaluation queries and the out-of-scope probes are all committed and the index is regenerable from them, so it rebuilds and runs the pinned routing bars on every pull request (and fails if any of them skips). The other two — retrieval and RAGAS — are **inactive placeholders** until real baselines are committed:

- The retrieval gate reads `data/eval/baseline_metrics.json`, which ships as a zero placeholder, so no regression can trip it. CI emits an `INACTIVE` warning until a real baseline lands.
- The RAGAS gate needs `data/eval/ragas_baseline.json`, which is not committed, so it reports `INACTIVE` and runs nothing.

To enforce them, generate real baselines against your canonical retrieval stack and commit the results:

```bash
# Retrieval baseline (needs a built index / BM25_INDEX_PATH or a running retrieval server)
python -m src.internal.retrieval.eval_runner \
  --dataset data/eval/qa_pairs.jsonl --top_k 10 \
  --output data/eval/baseline_metrics.json

# RAGAS baseline (needs OPENAI_API_KEY + the retrieval stack)
python -m src.internal.retrieval.ragas_eval \
  --dataset data/eval/ragas_qa.jsonl \
  --metrics faithfulness answer_relevancy \
  --output data/eval/ragas_baseline.json
```

Once a non-zero `baseline_metrics.json` (and/or a `ragas_baseline.json`) is committed, the corresponding gate starts enforcing regressions automatically.

[← Back to README](../README.md)
