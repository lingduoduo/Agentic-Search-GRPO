# Automatic Category Selection v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-measure opt-in auto category selection with multi-category ("acceptable") labels. Strengthen the weak categories' examples, re-tune on the old 64 queries as a dev set, and score a fresh, blind, 60-query test set exactly once against a pre-registered bar.

**Architecture:** extend the existing `src/model/pre_training/categories/` package from #689 with a multi-label loader, metrics, a constrained tuning objective, `tune-v2`/`evaluate-v2`, and leakage `--against`. The data comes from three separate blind subagents, with two user spot-check gates. The selection method is unchanged.

**Tech Stack:** Python 3.10+, numpy, the local e5 encoder (no search calls), pytest, GitHub Actions eval-gate.

**Spec:** `docs/superpowers/specs/2026-10-12-auto-category-selection-v2-design.md`

## Prerequisite

**#689 must be merged first.** Before Task 1, run `git fetch origin && git rebase origin/main` on `feat/auto-category-selection-v2`. That branch holds only docs commits, so the rebase is trivial. Then confirm that `src/model/pre_training/categories/evaluation.py` exists. Never stack this branch on the open #689 branch.

## Global Constraints

- **Order is binding:**
  1. test set written;
  2. dev relabelled;
  3. examples topped up;
  4. leakage check passes;
  5. the test file is committed and frozen;
  6. tune on dev;
  7. score test once.

  Nothing reads the v2 test queries except the leakage check, until Step 7.
- **Blindness:**
  - the test-set author sees only category names and descriptions;
  - the dev relabeller sees only the dev queries and the category descriptions;
  - the example author sees only dev confusions, never the test queries.

  Each is a separate, fresh subagent.
- **v2 label format:**
  - `{"taxonomy": "category_selection_v2", "queries": [{"id", "primary", "query", "acceptable": [...]}]}`;
  - `primary` is in `DOMAIN_REGISTRY` and is included in `acceptable`;
  - `acceptable` has 1 to 3 entries;
  - normalized queries are unique.

  The test set has exactly 3 queries per primary category (60 in total).
- **Metrics:**
  - **hit:** the picks and `acceptable` share any category;
  - **pooled precision:** Σ|P∩A| / Σ|P|;
  - **mean picks:** Σ|P| / n;
  - **top1:** the first pick is in A;
  - **per_primary:** the hit rate for each primary category;
  - **confusion:** primary → picks outside A.
- **Bar:** hit ≥ 0.85 **and** precision ≥ 0.60 **and** mean picks ≤ 2.0. If missed, `auto` stays opt-in.
- **Tuning:** the grid is `top_k ∈ {1,2,3,5}` × `select_margin ∈ {0.00,0.01,0.02,0.03,0.05}`, on dev v2.
  - Feasible points have precision ≥ 0.60 and mean picks ≤ 2.0.
  - Among them, maximize hit, then precision, then fewer picks, then the smaller `top_k`, then the smaller margin.
  - If no point is feasible, maximize hit, then fewer picks, then the smaller `top_k`, then the smaller margin.
  - The winner replaces `DEFAULT_CATEGORY_TOP_K` and `DEFAULT_CATEGORY_SELECT_MARGIN` in `src/shared_configs/intent.py`.
- **Leakage:** normalized-text equality or e5 cosine > 0.95 between any example or dev query and any v2 test query is flagged. Flagged test queries are rewritten before freezing. There is no allowlist for v2; the round-one allowlist stays as it is.
- **Round one stays reproducible:** its examples (the 160 existing ones stay byte-identical, and appending is fine), labels, reports and code paths are not edited.
- **Process:**
  - every new test is mutation-checked;
  - check the branch before each commit and `origin/main` before pushing;
  - open a separate PR and do not merge it;
  - `data/` files need `git add -f`.

## Review Focus

1. **The v2 test set must never influence tuning or examples.** Pinned by the ordering, by a test that the tuning code path loads only the dev file, and by commit history (test file frozen before `tune-v2`).
2. **Ties and boundaries in the feasibility filter** (precision exactly 0.60, mean picks exactly 2.0). These are inclusive. Pinned in Task 1.
3. **An `acceptable` list containing a category outside the registry, or missing its own primary,** is rejected. Pinned in Task 1.
4. **Example top-up must not alter the 160 round-one examples.** Pinned in Task 2 by asserting the first 160 records are unchanged.
5. **CI must not overwrite the committed v2 report,** and must enforce v2 leakage without an allowlist. Pinned in Task 4.

---

### Task 1: Multi-label loader, metrics, tuning and CLI

**Files:** modify `src/model/pre_training/categories/evaluation.py` and `src/model/pre_training/categories/cli.py`. Test: `tests/unit/test_category_evaluation_v2.py` (create).

**Interfaces (produces):**

```python
V2_MARKER = "category_selection_v2"
V2_BAR_HIT, V2_BAR_PRECISION, V2_BAR_MEAN_PICKS = 0.85, 0.60, 2.0

@dataclass(frozen=True)
class MultiLabelQuery: id: str; primary: str; query: str; acceptable: tuple[str, ...]

def load_multilabel_queries(path) -> list[MultiLabelQuery]
def evaluate_multilabel(index, queries, *, top_k, select_margin, encode=None) -> dict
    # {"n","hit","precision","mean_picks","top1","per_primary":{c:{"n","hit"}},"confusion":{c:{picked:count}}}
def tune_multilabel(index, queries, *, encode=None) -> dict
    # {"top_k","select_margin","feasible": bool, "grid":[{"top_k","select_margin","hit","precision","mean_picks"}]}
```

CLI subcommands:
- `tune-v2 --index DIR --labels DEV --report OUT`
- `evaluate-v2 --index DIR --labels TEST --report OUT [--top-k --select-margin]`, whose report adds `split: "test_v2"`, `bar`, `passed` and the parameters
- `leakage ... --against V2FILE`, which checks examples and an optional `--dev V2DEV` against the queries in `V2FILE`, with no allowlist

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_category_evaluation_v2.py`:

```python
"""v2: multi-label loader, hit/precision metrics, constrained tuning."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.model.pre_training.categories.evaluation import (
    V2_BAR_HIT, V2_BAR_MEAN_PICKS, V2_BAR_PRECISION,
    MultiLabelQuery, evaluate_multilabel, load_multilabel_queries, tune_multilabel,
)
from src.model.pre_training.categories.model import CategoryExample, CategoryIndex


def _write(tmp_path, queries, marker="category_selection_v2"):
    p = tmp_path / "v2.json"
    p.write_text(json.dumps({"taxonomy": marker, "queries": queries}))
    return p


def _q(i, primary, text, acceptable):
    return {"id": f"q{i}", "primary": primary, "query": text, "acceptable": acceptable}


def test_loader_reads_valid_records(tmp_path):
    out = load_multilabel_queries(_write(tmp_path, [_q(1, "news", "a", ["news", "financial"])]))
    assert out == [MultiLabelQuery("q1", "news", "a", ("news", "financial"))]


@pytest.mark.parametrize("queries,marker,msg", [
    ([_q(1, "news", "a", ["news"])], "other", "taxonomy"),
    ([_q(1, "nope", "a", ["nope"])], "category_selection_v2", "primary"),
    ([_q(1, "news", "a", ["financial"])], "category_selection_v2", "primary"),
    ([_q(1, "news", "a", ["news", "code", "legal", "books"])], "category_selection_v2", "acceptable"),
    ([_q(1, "news", "a", ["news", "zzz"])], "category_selection_v2", "acceptable"),
    ([_q(1, "news", "Same q", ["news"]), _q(2, "code", "same  q", ["code"])], "category_selection_v2", "query"),
])
def test_loader_rejects(tmp_path, queries, marker, msg):
    with pytest.raises(ValueError, match=msg):
        load_multilabel_queries(_write(tmp_path, queries, marker))


def _index():
    ex = [CategoryExample("n", "news", "n"), CategoryExample("f", "financial", "f"), CategoryExample("c", "code", "c")]
    v = np.eye(3, dtype=np.float32)
    return CategoryIndex(examples=ex, vectors=v, encoder="intfloat/e5-small-v2", fingerprint="f")


def _enc(mapping):
    return lambda texts: np.array([mapping[t] for t in texts], dtype=np.float32)


def _unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()


def test_metrics_hit_pooled_precision_and_confusion():
    qs = [
        MultiLabelQuery("1", "news", "q1", ("news", "financial")),
        MultiLabelQuery("2", "code", "q2", ("code",)),
    ]
    # q1 ties news+financial (both acceptable); q2 picks news (not acceptable)
    enc = _enc({"q1": _unit(1, 1, 0), "q2": _unit(1, 0, 0)})
    r = evaluate_multilabel(_index(), qs, top_k=1, select_margin=0.0, encode=enc)
    assert r["n"] == 2 and r["hit"] == 0.5
    assert r["mean_picks"] == 1.5           # 2 picks + 1 pick
    assert r["precision"] == pytest.approx(2 / 3)  # 2 acceptable of 3 picks
    assert r["confusion"]["code"] == {"news": 1}


def test_tuning_prefers_a_feasible_point_with_max_hit():
    # Index rows are one-hot: news=e1, financial=e2, code=e3; one example each,
    # so top_k does not change scores and the smallest top_k (1) wins ties.
    # q1 = unit(1, 0.95, 0): news 0.7250, financial 0.6887 (gap 0.036):
    #   margins 0.00-0.03 pick [news]; 0.05 picks [news, financial].
    # q2 = e2: always picks [financial].
    # margin < 0.05: hit 0.5, precision 1/2 = 0.50 -> infeasible
    # margin = 0.05: hit 1.0, precision 2/3 ~= 0.667, mean picks 1.5 -> feasible
    qs = [
        MultiLabelQuery("1", "financial", "q1", ("financial",)),
        MultiLabelQuery("2", "financial", "q2", ("financial",)),
    ]
    enc = _enc({"q1": _unit(1, 0.95, 0), "q2": _unit(0, 1, 0)})
    r = tune_multilabel(_index(), qs, encode=enc)
    assert r["feasible"] is True
    assert (r["top_k"], r["select_margin"]) == (1, 0.05)
    row = next(g for g in r["grid"] if (g["top_k"], g["select_margin"]) == (1, 0.05))
    assert row["hit"] == 1.0 and row["precision"] == pytest.approx(2 / 3) and row["mean_picks"] == 1.5


def test_fallback_still_maximizes_hit_when_nothing_is_feasible():
    # q1 alone: margin < 0.05 -> hit 0 (precision 0); margin 0.05 -> hit 1,
    # precision 0.5 (< 0.60). Nothing is feasible; fallback maximizes hit.
    qs = [MultiLabelQuery("1", "financial", "q1", ("financial",))]
    enc = _enc({"q1": _unit(1, 0.95, 0)})
    r = tune_multilabel(_index(), qs, encode=enc)
    assert r["feasible"] is False
    assert (r["top_k"], r["select_margin"]) == (1, 0.05)


def test_tuning_falls_back_when_nothing_is_feasible():
    qs = [MultiLabelQuery("1", "code", "q1", ("code",))]
    enc = _enc({"q1": _unit(1, 0, 0)})  # always picks news: precision 0 everywhere
    r = tune_multilabel(_index(), qs, encode=enc)
    assert r["feasible"] is False
    assert (r["top_k"], r["select_margin"]) == (1, 0.0)


def test_choice_filters_infeasible_points_before_maximizing_hit():
    from src.model.pre_training.categories.evaluation import _choose_tuning
    grid = [
        {"top_k": 1, "select_margin": 0.05, "hit": 1.0, "precision": 0.50, "mean_picks": 2.0},  # infeasible
        {"top_k": 3, "select_margin": 0.00, "hit": 0.9, "precision": 0.90, "mean_picks": 1.0},  # feasible
    ]
    row, feasible = _choose_tuning(grid)
    assert feasible is True and row["hit"] == 0.9
    # With nothing feasible, the highest hit wins.
    row, feasible = _choose_tuning(grid[:1])
    assert feasible is False and row["hit"] == 1.0


def test_feasibility_bounds_are_inclusive():
    from src.model.pre_training.categories.evaluation import _feasible
    assert _feasible({"precision": 0.60, "mean_picks": 2.0})
    assert not _feasible({"precision": 0.5999, "mean_picks": 2.0})
    assert not _feasible({"precision": 0.60, "mean_picks": 2.0001})


def test_bar_constants():
    assert (V2_BAR_HIT, V2_BAR_PRECISION, V2_BAR_MEAN_PICKS) == (0.85, 0.60, 2.0)
```

The expected values in the two tuning tests are worked out in their comments. Re-derive them if you change a fixture.

- [ ] **Step 2: Run to verify they fail.** Run `pytest tests/unit/test_category_evaluation_v2.py -q`. Expect an ImportError.
- [ ] **Step 3: Implement.** In `evaluation.py`:
  - **`load_multilabel_queries`:** validates the marker (message contains "taxonomy"); that `primary` is in `DOMAIN_REGISTRY` and in `acceptable` ("primary"); that `acceptable` has 1–3 entries, each a registry name ("acceptable"); and that normalized queries are unique ("query").
  - **`evaluate_multilabel`:** encodes once, then calls `index.select` per query.
  - **`_feasible(row)`:** `row["precision"] >= V2_BAR_PRECISION and row["mean_picks"] <= V2_BAR_MEAN_PICKS`.
  - **`_choose_tuning(grid) -> (row, feasible)`:** implements the selection rule over grid rows.
  - **`tune_multilabel`:** encodes the dev queries **once** and reuses the vectors across the grid, then calls `_choose_tuning`. This also fixes round one's deferred minor about re-encoding.
    - It evaluates every grid point.
    - If any point is feasible, it picks with the key `(-hit, -precision, mean_picks, top_k, select_margin)` among the feasible points.
    - Otherwise it uses `(-hit, mean_picks, top_k, select_margin)`.
  - **`cli.py`:** adds `tune-v2`, `evaluate-v2`, and `leakage --against [--dev]`, with no allowlist when `--against` is given. The round-one `leakage` behaviour without `--against` is unchanged.
- [ ] **Step 4: Run.** Run the new file plus `tests/unit/test_category_evaluation.py` and `tests/unit/test_category_index.py`. Expect PASS, with the round-one tests unmodified.
- [ ] **Step 5: Mutation-check:**
  - (a) make `_feasible` strict (`>`);
  - (b) drop the feasibility filter in `_choose_tuning` (`test_choice_filters_infeasible_points_before_maximizing_hit` must fail);
  - (c) let `acceptable` omit `primary`;
  - (d) let `leakage --against` apply the round-one allowlist.

  Each must turn red. Restore.
- [ ] **Step 6: Commit.** Message: `feat: multi-label category selection metrics and tuning (v2)`.

### Task 2: Blind data, spot-check gates and leakage

This task is run by the controller, which dispatches three separate fresh subagents.

- [ ] **Step 1: Write the test set.** Dispatch the **test-set author**. It may read only the category names and descriptions from `DOMAIN_REGISTRY`, which the controller pastes into the prompt. It must not open `data/`, `docs/superpowers/` or `.superpowers/`.

  > Write `data/eval/category_selection_test_v2.json` in /Users/linghuang/Git/Agentic-Search. Use the format `{"taxonomy": "category_selection_v2", "queries": [{"id": "t<category>-<n>", "primary": "<category>", "query": "...", "acceptable": ["<primary>", ...]}]}`.
  >
  > For each of the 20 categories below, write exactly 3 realistic search queries whose primary intent fits the category. Phrase them as a person would, with no site names and without the category word itself. Use varied topics.
  >
  > For each query, list in `acceptable` every category (1–3, `primary` first) whose sources would genuinely answer it.
  >
  > Do not open any file under data/, docs/superpowers/ or .superpowers/. Do not commit. Reply with the path and count.
  >
  > Categories: <name: description, for all 20, pasted by the controller>

  Validate with `load_multilabel_queries` (60 queries, exactly 3 per primary). **Do not commit yet.**
- [ ] **Step 2: Relabel the dev set.** Dispatch the **dev relabeller** with the round-one 64 queries and their round-one category (taken from `data/eval/source_category_queries.json`), plus the category descriptions. It writes `data/eval/category_selection_dev_v2.json` with `primary` = the round-one category and an `acceptable` list (1–3 entries, primary first). It must not open the v2 test file or the examples. Validate with the loader.
- [ ] **Step 3: USER GATE, spot-check.** Show the user both files' paths, the distribution of `acceptable` sizes, and 5 random entries from each. Wait for approval, and apply only the user's corrections.
- [ ] **Step 4: Gather dev confusions.** Build the index from the current examples and run `evaluate_multilabel` on **dev v2** with the round-one defaults (3, 0.0). Save the confusion table to the scratch workspace, not to the repo. This is a dev-only run.
- [ ] **Step 5: Top up the examples.** Dispatch the **example author** with the dev confusion table and the category descriptions. It must not open the v2 test file. It appends about 4 examples to each of books, jobs, news, resource, business, events, media, patent and social_media in `data/category_examples.json`, with the same record format and continuing ids. It must not edit any existing record. Validate with `load_category_examples`, and add a test, `test_round_one_examples_are_unchanged`, asserting that the first 160 records equal the committed `041ca0c0` version (load it with `git show 041ca0c0:data/category_examples.json`).
- [ ] **Step 6: Leakage before measuring.** Run `leakage --examples data/category_examples.json --dev data/eval/category_selection_dev_v2.json --against data/eval/category_selection_test_v2.json`. For each flagged test query, send the test-set author a rewrite request (the flagged query and its offending match only), then re-validate and re-run until there are 0 flags.
- [ ] **Step 7: Freeze.** Commit, with `git add -f` where needed:
  - `data/eval/category_selection_test_v2.json`, message `data: v2 held-out category selection test set (frozen)`;
  - `data/eval/category_selection_dev_v2.json`;
  - the topped-up examples and the unchanged-examples test, message `data: v2 dev relabel and example top-up`.

  From the test-set commit on, the test file is frozen.

### Task 3: Tune on dev, then score the test set once

- [ ] **Step 1:** Rebuild `data/category_index` from the topped-up examples, then run `tune-v2 --labels data/eval/category_selection_dev_v2.json --report data/eval/category_selection_v2_tuning.json`.
- [ ] **Step 2:** Set `DEFAULT_CATEGORY_TOP_K` and `DEFAULT_CATEGORY_SELECT_MARGIN` in `src/shared_configs/intent.py` to the winner. Update any test that pins the defaults.
- [ ] **Step 3:** Run `evaluate-v2 --labels data/eval/category_selection_test_v2.json --report data/eval/category_selection_v2.json` **exactly once**. Commit the tuning report, the defaults and the v2 report (`git add -f`) with the message `data: v2 tuning (dev) and pre-registered test result`. Do not change examples, the grid, the defaults or the labels afterwards, whatever the result.
- [ ] **Step 4:** Confirm the round-one result still reproduces: rebuilding with the **original 160 examples** (from `041ca0c0`) and the round-one `evaluate` gives 0.729 / 1.00. This is a sanity check, not committed. It shows round one is untouched.

### Task 4: CI, docs, review and PR

- [ ] **Step 1: CI.** In `.github/workflows/eval-gate.yml`'s category step:
  - also run `evaluate-v2` into `/tmp`;
  - warn (`::warning::`) if hit, precision or mean picks drift by more than 0.005 from the committed `data/eval/category_selection_v2.json`;
  - enforce `leakage --against data/eval/category_selection_test_v2.json --dev data/eval/category_selection_dev_v2.json` (no allowlist);
  - leave the bar unenforced and never write to the committed report.

  Keep the YAML valid.
- [ ] **Step 2: Docs.**
  - **`docs/search-engine.md`:** a v2 section beside v1 covering the hit rate, pooled precision, mean picks, top-1, the bar and pass/fail, the chosen defaults, how the labels and examples were made (three blind subagents, user spot-checks, leakage before measuring), and the spec's caveats.
  - **`docs/configuration.md`:** the new defaults.
  - **README:** update the clause about auto selection to match the v2 outcome.
- [ ] **Step 3: Verify:** `pytest -q`, Ruff, `git diff --check`, the docs link check and a YAML parse.
- [ ] **Step 4: Review.** One fresh whole-branch reviewer, on the most capable model, against `origin/main`, the spec and the Review Focus. Integrity of the history is part of the review: confirm the test file was frozen before `tune-v2`, and that no examples, defaults or labels changed after the v2 report commit. Fix findings RED→GREEN and record rulings.
- [ ] **Step 5: PR.** Check the branch and `origin/main`, push, and open the PR with the v2 result table and the history-integrity statement. Watch CI on the final commit, mark it ready, and report. Do not merge.

## Validation record

Spec approved by the user on 2026-10-12. Plan awaiting review. Implementation is blocked until #689 merges (see Prerequisite).
