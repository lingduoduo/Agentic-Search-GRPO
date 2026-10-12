# Automatic category selection v2: multi-label evaluation and better examples

Follow-up to sub-project 3 (#689). It depends on #689's
`src/model/pre_training/categories/` package and `domains: ["auto"]` wiring,
and implementation starts from main after #689 merges.

## Intent

Round one measured opt-in auto selection below its pre-registered bar. On the
48 held-out queries, coverage was 0.729 against 0.80, with 1.00 mean picks. Two
things caused it:

- **Single-category labels:** a correct second pick could never earn credit,
  so tuning drove `select_margin` to 0 and auto effectively picks one
  category.
- **Weak categories:** books, jobs, news and resource were each 1 in 3.

v2 measures selection the way it is used. Each query lists every acceptable
category, the weak categories get more examples, the defaults are re-tuned,
and a **fresh** held-out test set is scored exactly once. The selection method
(e5 nearest examples, top-k mean, margin, at most 3) is unchanged.

## Data and order of work

The order is binding; each step completes before the next begins.

1. **New test set, written blind, then frozen.**
   - **Author:** a fresh subagent writes
     `data/eval/category_selection_test_v2.json`.
   - **Format:** `{"taxonomy": "category_selection_v2", "queries": [{"id",
     "primary", "query", "acceptable": [..]}]}`.
   - **Content:** 60 queries, 3 per `DOMAIN_REGISTRY` category by `primary`.
     `acceptable` lists 1–3 categories and includes `primary`.
   - **Blindness:** the subagent sees only the category names and descriptions.
     It never sees `data/category_examples.json`, the round-one eval queries,
     or any confusion data.
   - **Freezing:** the user spot-checks the file. It is committed only after
   Step 4's leakage check passes, and frozen from then on.
2. **Dev relabel.** A second subagent writes
   `data/eval/category_selection_dev_v2.json` in the same format: the round-one
   64 queries, each given an `acceptable` set from the query text and the
   category descriptions alone. `primary` is the round-one label. The user
   spot-checks it. The round-one label file is not modified.
3. **Example top-up.**
   - **What:** a third subagent appends about 4 examples to each weak category
     in `data/category_examples.json`: books, jobs, news, resource, business,
     events, media, patent and social_media. The existing 160 stay unchanged.
   - **Inputs:** the dev confusions, for example that books queries are picked
     as academic, but **never** the v2 test queries. The confusions come from
     running `evaluate_multilabel` on dev v2 with the round-one examples and
     defaults. This is a dev-only run.
   - **Uniqueness:** the loader's rules still hold, so ids and normalized texts
     stay unique.
4. **Leakage check before any measurement.**
   - **What is compared:** every example and every dev query, against every v2
     test query.
   - **Flagged:** normalized-text equality, or e5 cosine > 0.95.
   - **Fix:** each flagged test query is rewritten by the test-set subagent and
     re-checked until nothing is flagged.
   - **Then:** the test file is committed and frozen.
   - The round-one CI allowlist covers only the round-one pairs and is not
     extended.
5. **Tune on dev v2, then score test v2 once.** The frozen defaults are scored
   exactly once on the 60 test queries, and the result is final.

## Metrics and bar

For a query with acceptable set A and picks P:

- **hit** = 1 if P ∩ A is non-empty;
- **pick precision**, pooled = Σ|P ∩ A| / Σ|P|;
- **mean picks** = Σ|P| / n.

Reported descriptively: top-1 (the first pick is in A), per-primary-category hit
rate, and a confusion table of primary against picks outside A.

**Pre-registered bar:** hit rate ≥ 0.85 **and** pooled pick precision ≥ 0.60
**and** mean picks ≤ 2.0. If the bar is missed, `auto` stays opt-in, and the
docs report v2 beside v1.

## Tuning

The grid is unchanged: `top_k ∈ {1, 2, 3, 5}`, `select_margin ∈ {0.00, 0.01,
0.02, 0.03, 0.05}`, evaluated on dev v2.

1. Keep the grid points with precision ≥ 0.60 and mean picks ≤ 2.0 on dev.
2. Among those, maximize hit rate.
3. Break ties by higher precision, then fewer mean picks, then the smaller
   `top_k`, then the smaller `select_margin`.
4. If no grid point meets the limits, use round one's ordering on hit rate:
   highest hit, then fewest picks, then the smaller `top_k`, then the smaller
   margin.

The winning values replace `DEFAULT_CATEGORY_TOP_K` and
`DEFAULT_CATEGORY_SELECT_MARGIN` in `src/shared_configs/intent.py`. That changes
serving behaviour for `auto` callers only.

## Code

All changes extend `src/model/pre_training/categories/`; round one stays
reproducible.

- **Loader:** `load_multilabel_queries(path)`. It validates the marker, ids,
  `primary ∈ DOMAIN_REGISTRY`, that `acceptable` is a non-empty subset of the
  registry of size ≤ 3 that contains `primary`, and unique normalized queries.
- **Metrics:** `evaluate_multilabel(index, queries, *, top_k, select_margin,
  encode=None)` returns hit, precision, mean_picks, top1, per_primary and
  confusion.
- **Tuning:** `tune_multilabel(index, queries, *, encode=None)` applies the
  objective above and records the full grid.
- **CLI:**
  - `tune-v2 --index --labels data/eval/category_selection_dev_v2.json --report
    data/eval/category_selection_v2_tuning.json`;
  - `evaluate-v2 --index --labels data/eval/category_selection_test_v2.json
    --report data/eval/category_selection_v2.json`. Its report records `bar`,
    `passed`, `split: "test_v2"` and the parameters used.
  - `leakage` gains `--against` for the v2 test file. It checks the examples
    and the dev v2 queries against it, with no allowlist for v2.
- **CI:** the eval-gate step also runs `evaluate-v2` into a temp file, warns if
  the numbers drift from the committed v2 report, and enforces v2 leakage.

## Testing

All tests run offline with stubbed encoders, and every new test is
mutation-checked.

- **Loader:** rejects a bad marker, unknown categories, `acceptable` missing its
  `primary`, more than 3 acceptable categories, and duplicates.
- **Metrics:** hit, pooled precision, mean picks, top-1 and confusion on small
  fixed vectors.
- **Tuning:** limits filter the grid; ties break in the stated order; when no
  point is feasible, the fallback applies.
- **Files:** the v2 test and dev files load; the test set has exactly 3 per
  primary category; no example or dev query normalizes equal to a test query.
- **Leakage:** a new v2 pair fails, and the round-one allowlist does not apply
  to v2.

## Documentation

`docs/search-engine.md` gets a v2 result section beside v1. It covers the hit
rate, pooled precision, mean picks, top-1, the bar and pass/fail, how the labels
and examples were made, and these caveats:

- labels and examples were drafted by LLM subagents;
- acceptable sets are subjective, which is why the user spot-checks them;
- the 60 queries still came from category prompts, so they may be easier than
  real traffic.

`docs/configuration.md` gets the new defaults.

## Out of scope

- Method changes: category descriptions as examples, LLM tie-breakers, or a
  different encoder.
- Making `auto` the default.
- Answer-quality evaluation of auto-picked searches.
- Editing round one's examples, labels or reports.
