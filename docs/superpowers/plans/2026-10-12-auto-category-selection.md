# Automatic Category Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** `domains: ["auto"]` picks 1–3 search-source categories for a query by nearest examples (e5), runs the normal multi-category search, and is measured against a pre-registered bar on held-out labels.

**Architecture:**

- **Selector:** a new torch-free package, `src/model/pre_training/categories/`, holds the example loader, `CategoryIndex` (top-k mean scoring plus margin selection), and a CLI that builds, tunes and evaluates. It reuses `encode_texts`, `DEFAULT_ENCODER` and `fingerprint` from the intent package without modifying it.
- **Serving adapter:** `src/internal/tools/category_selection.py` loads and caches the index from settings, and degrades to `["web"]`.
- **Resolver:** `resolve_auto_domains`, in `src/internal/tools/search.py`, runs before `normalize_search_domains` at every entry point that accepts `domains`.

**Tech Stack:** Python 3.10+, numpy, sentence-transformers (optional; already used by the intent router), pytest, FastAPI, React/TypeScript, Vitest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-12-auto-category-selection-design.md`

## Global Constraints

- **Opt-in only:** `auto` is accepted only as `domains == ["auto"]`. `auto` mixed with named categories, or `domain="auto"`, gives `InvalidToolInput` (HTTP 400 or a tool error). Requests that don't ask for `auto` are byte-identical to today.
- **Encoder:** `intfloat/e5-small-v2`, through the intent package's `encode_texts`, which adds the `query:` prefix itself. Callers pass raw text.
- **Scoring:**
  - a category's score is the mean of its `top_k` highest cosines among its own examples;
  - the selection is the best category plus every category whose score is ≥ `best − select_margin`, ordered by score (ties in `DOMAIN_REGISTRY` order), truncated to 3;
  - at least one category is always selected.
- **Tuning grid:** `top_k ∈ {1, 2, 3, 5}` and `select_margin ∈ {0.00, 0.01, 0.02, 0.03, 0.05}`, chosen on the dev split only. The winner has the highest coverage, then the fewest mean picks, then the smaller `top_k`, then the smaller `select_margin`. The chosen values become the code defaults.
- **Splits:** dev is the first label of each category in `data/eval/source_category_queries.json`, in file order (16 queries). Test is the other 48, scored once.
- **Pre-registered bar:** test coverage ≥ 0.80 **and** mean picks ≤ 2.0. Coverage means the labelled category is in the selected set. If the bar is missed, the feature still ships opt-in and the docs report the miss.
- **Fallback:**
  - **When:** `AGENTIC_SEARCH_CATEGORY_INDEX_PATH` is empty (the default), the index fails to load (the failure is cached and never retried), its encoder differs from the default, or encoding raises.
  - **Result:** `(["web"], {"selected": ["web"], "status": "unavailable"})`. The search still runs.
- **Off the event loop:** encoding runs through `asyncio.to_thread`.
- **Settings**, in the typed app configuration and documented in `docs/configuration.md`:
  - `AGENTIC_SEARCH_CATEGORY_INDEX_PATH`;
  - `AGENTIC_SEARCH_CATEGORY_TOP_K` (positive);
  - `AGENTIC_SEARCH_CATEGORY_SELECT_MARGIN` (in [0, 1]).
- **Example authoring:** a subagent never shown the eval labels writes the examples, and a test enforces no normalized-text overlap with eval queries.
- **Torch-free:** unit tests stub `encode_texts` and run without torch; only the eval-gate workflow uses the real encoder. Nothing under `src/internal/` imports `src.model.pre_training.categories` at module import time; imports happen inside functions.
- **Process:** every new test is mutation-checked. Separate PR against main; verify CI on the final commit; do not merge. Check the current branch before every commit, and check `git log origin/main -3` before pushing (the user pushes to main directly).

## Review Focus

1. **A request without `auto`:** provider queries and response shape are identical to before, and `auto_domains` is null. Pinned by the existing suites, unmodified, plus explicit null assertions in Task 4.
2. **Index missing or broken in production:** `auto` searches `web` with status `unavailable`, with no exception, and the load is attempted only once. Pinned in Task 3.
3. **Auto picks one category:** the single-category path is used exactly, with no `categories` metadata. Pinned in Task 4.
4. **Leakage:** example texts never equal an eval query after normalization. Pinned in Task 2.
5. **Slow or cold encoder:** the event loop is not blocked. Pinned in Task 3 by asserting `asyncio.to_thread` is used.

---

### Task 1: Category index (loader, scoring, selection)

**Files:** create `src/model/pre_training/categories/__init__.py`, `src/model/pre_training/categories/model.py`; test `tests/unit/test_category_index.py`.

**Interfaces (produces):**

```python
@dataclass(frozen=True)
class CategoryExample: id: str; category: str; text: str

def load_category_examples(path: Path) -> list[CategoryExample]
class CategoryIndex:
    def __init__(self, *, examples: list[CategoryExample], vectors: np.ndarray, encoder: str, fingerprint: str)
    @property
    def encoder(self) -> str
    def scores(self, query_vector: np.ndarray, *, top_k: int) -> dict[str, float]   # every category present
    def select(self, query_vector: np.ndarray, *, top_k: int, select_margin: float) -> list[str]
    def save(self, path: Path) -> None
    @classmethod
    def load(cls, path: Path) -> "CategoryIndex"
def build_category_index(examples_path: Path, *, encode=None) -> CategoryIndex
INDEX_FILENAME = "index.npz"
DEFAULT_TOP_K, DEFAULT_SELECT_MARGIN  # re-exported from src/shared_configs/intent.py
                                      # (DEFAULT_CATEGORY_TOP_K=3, DEFAULT_CATEGORY_SELECT_MARGIN=0.02
                                      # until Task 2's tuning sets them)
MAX_SELECTED = 3
```

- [x] **Step 1: Write the failing tests.** Create `tests/unit/test_category_index.py`. Use one-hot vectors so the cosines are exact, following `tests/unit/test_ml_intent.py`:

```python
"""CategoryIndex: top-k mean per category, margin selection, at most three."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.model.pre_training.categories.model import (
    CategoryExample,
    CategoryIndex,
    build_category_index,
    load_category_examples,
)
from src.internal.tools.search import DOMAIN_REGISTRY


def _index(rows):
    """rows: [(category, vector)] -> index with L2-normalized vectors."""
    examples = [CategoryExample(f"e{i}", c, f"text {i}") for i, (c, _) in enumerate(rows)]
    vectors = np.array([v for _, v in rows], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return CategoryIndex(examples=examples, vectors=vectors, encoder="intfloat/e5-small-v2", fingerprint="f")


def _unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_scores_are_top_k_means_per_category():
    idx = _index([("news", _unit(1, 0)), ("news", _unit(0, 1)), ("code", _unit(1, 0))])
    s = idx.scores(_unit(1, 0), top_k=1)
    assert s["news"] == pytest.approx(1.0) and s["code"] == pytest.approx(1.0)
    s = idx.scores(_unit(1, 0), top_k=2)
    assert s["news"] == pytest.approx(0.5)


def test_select_takes_categories_within_the_margin_capped_at_three():
    rows = [("news", _unit(1, 0, 0, 0)), ("code", _unit(0.99, 0.14, 0, 0)),
            ("legal", _unit(0.98, 0.2, 0, 0)), ("books", _unit(0.97, 0.24, 0, 0)),
            ("jobs", _unit(0, 0, 0, 1))]
    idx = _index(rows)
    q = _unit(1, 0, 0, 0)
    assert idx.select(q, top_k=1, select_margin=0.0) == ["news"]
    assert idx.select(q, top_k=1, select_margin=0.05) == ["news", "code", "legal"]


def test_select_breaks_ties_in_registry_order():
    idx = _index([("news", _unit(1, 0)), ("academic", _unit(1, 0))])
    order = list(DOMAIN_REGISTRY)
    picked = idx.select(_unit(1, 0), top_k=1, select_margin=0.0)
    assert picked == sorted(["news", "academic"], key=order.index)[:1]


def test_save_and_load_round_trip(tmp_path):
    idx = _index([("news", _unit(1, 0)), ("code", _unit(0, 1))])
    idx.save(tmp_path / "index.npz")
    back = CategoryIndex.load(tmp_path / "index.npz")
    assert back.encoder == idx.encoder
    assert back.select(_unit(0, 1), top_k=1, select_margin=0.0) == ["code"]


def _write(tmp_path, records):
    p = tmp_path / "ex.json"
    p.write_text(json.dumps(records))
    return p


@pytest.mark.parametrize("records,msg", [
    ([{"id": "a", "category": "nope", "text": "x"}], "category"),
    ([{"id": "a", "category": "news", "text": "x"}, {"id": "a", "category": "code", "text": "y"}], "id"),
    ([{"id": "a", "category": "news", "text": "Same  text"}, {"id": "b", "category": "code", "text": "same text"}], "text"),
])
def test_loader_rejects_bad_records(tmp_path, records, msg):
    with pytest.raises(ValueError, match=msg):
        load_category_examples(_write(tmp_path, records), require_all_categories=False)


def test_loader_requires_every_category_by_default(tmp_path):
    with pytest.raises(ValueError, match="no examples"):
        load_category_examples(_write(tmp_path, [{"id": "a", "category": "news", "text": "x"}]))


def test_build_uses_the_injected_encoder(tmp_path):
    records = [{"id": f"{c}-1", "category": c, "text": f"about {c}"} for c in DOMAIN_REGISTRY]
    p = _write(tmp_path, records)
    calls = []
    def fake_encode(texts):
        calls.append(list(texts))
        v = np.eye(len(texts), dtype=np.float32)
        return v
    idx = build_category_index(p, encode=fake_encode)
    assert calls == [[r["text"] for r in records]]
    assert set(idx.scores(np.eye(len(records), dtype=np.float32)[0], top_k=1)) == set(DOMAIN_REGISTRY)
```

(The signature of `load_category_examples` gains `require_all_categories: bool = True`, so the rejection tests can use small files.)

- [x] **Step 2: Run to verify it fails.** Run `pytest tests/unit/test_category_index.py -q`. Expect an ImportError.

- [x] **Step 3: Implement** `src/model/pre_training/categories/model.py`:
  - **Imports:** import only numpy, json, hashlib, pathlib and dataclasses at module level. Get `DOMAIN_REGISTRY` with `from src.internal.tools.search import DOMAIN_REGISTRY`, and confirm this import doesn't pull torch, checking the way the intent test does. Get `fingerprint`, `DEFAULT_ENCODER` and `encode_texts` from `src.model.pre_training.intents` (`data.fingerprint`, `model.DEFAULT_ENCODER`, `model.encode_texts`).
  - **`load_category_examples(path, *, require_all_categories=True)`:**
    - The file must be a JSON list of `{"id", "category", "text"}` with non-empty strings.
    - `category` must be in `DOMAIN_REGISTRY`.
    - ids must be unique.
    - Normalized text (`" ".join(text.lower().split())`) must be unique.
    - When `require_all_categories` is set, every registry category needs at least one example.
    - Each failure raises `ValueError` with a message containing "category", "id", "text" or "no examples" respectively.
  - **`CategoryIndex`:**
    - The constructor checks that `vectors` has one row per example, is float32 and is L2-normalized (`np.allclose(norms, 1, atol=1e-3)`). It precomputes, for each category in `DOMAIN_REGISTRY` order, the rows belonging to that category.
    - `scores(q, top_k)`: `sims = vectors @ q`. Per category, take the mean of the highest `min(top_k, n)` values among its rows, or 0.0 when it has none.
    - `select(q, top_k, select_margin)`:

      ```python
      s = self.scores(q, top_k=top_k)
      order = list(DOMAIN_REGISTRY)
      ranked = sorted(s, key=lambda c: (-s[c], order.index(c)))
      best = s[ranked[0]]
      return [c for c in ranked if s[c] >= best - select_margin][:MAX_SELECTED]
      ```

    - `save` and `load` use the npz format with `vectors`, `examples` (JSON), `encoder` and `fingerprint`, following `IntentIndex.save`/`load`. `load` raises `FileNotFoundError`, with a hint to run the CLI, when the file is missing.
  - **`build_category_index(examples_path, *, encode=None)`:** loads the examples, sets `encode = encode or (lambda texts: encode_texts(texts))`, encodes the texts in file order, and returns an index with `fingerprint(examples_path)` and `DEFAULT_ENCODER`.
  - Define `DEFAULT_CATEGORY_TOP_K = 3` and `DEFAULT_CATEGORY_SELECT_MARGIN = 0.02` in `src/shared_configs/intent.py`, beside the intent defaults. `model.py` re-exports them as `DEFAULT_TOP_K` and `DEFAULT_SELECT_MARGIN`. This is mandatory: `app_configs` must import these defaults from `shared_configs`, never from `categories.model`, which imports `tools.search` and would create an import cycle through the configs package.
  - `__init__.py` exports the public names.

- [x] **Step 4: Run.** Run `pytest tests/unit/test_category_index.py -q`. Expect PASS. Confirm torch isn't imported: `python -c "import sys, src.model.pre_training.categories.model; print('torch' in sys.modules)"` prints False.
- [x] **Step 5: Mutation checks.** Each must turn red, then restore:
  - (a) drop the `[:MAX_SELECTED]` cap;
  - (b) use the best single similarity instead of the top-k mean;
  - (c) drop the normalized-text uniqueness check.
- [x] **Step 6: Commit.** Message: `feat: category index for automatic source selection`.

### Task 2: Examples, CLI (build, tune, evaluate), leakage guard

**Files:**
- Create: `src/model/pre_training/categories/cli.py`, `src/model/pre_training/categories/evaluation.py`, and `data/category_examples.json` (with `git add -f`, written in Step 1 by a blind subagent).
- Test: `tests/unit/test_category_evaluation.py`.

**Interfaces (produces):**

```python
def split_eval_labels(path: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]   # (dev, test) of (category, query)
def evaluate_selection(index: CategoryIndex, labels, *, top_k: int, select_margin: float, encode=None) -> dict
    # {"n", "coverage", "mean_picks", "top1", "per_category": {c: {"n","coverage"}}, "confusion": {c: {picked: count}}}
def tune(index: CategoryIndex, dev, *, encode=None) -> dict   # {"top_k", "select_margin", "grid": [...]}
GRID_TOP_K = (1, 2, 3, 5); GRID_MARGIN = (0.00, 0.01, 0.02, 0.03, 0.05)
BAR_COVERAGE = 0.80; BAR_MEAN_PICKS = 2.0
```

CLI commands:
- `build --examples data/category_examples.json --output data/category_index`
- `tune --index data/category_index --labels data/eval/source_category_queries.json --report data/eval/category_selection_tuning.json`
- `evaluate --index ... --labels ... --report data/eval/category_selection.json [--top-k K --select-margin M]` (the defaults come from the model constants)

- [x] **Step 1: Draft the examples blind.** Dispatch a fresh subagent with this prompt. It must not read `data/eval/`, `docs/superpowers/`, or `.superpowers/`.

  > Write `data/category_examples.json` in /Users/linghuang/Git/Agentic-Search. Do not open anything under data/eval/, docs/superpowers/ or .superpowers/, and do not grep for eval query text. Format: a JSON list of {"id": "<category>-<n>", "category": "<category>", "text": "<query>"}. For EACH of these 20 categories write exactly 8 short, varied, realistic search queries a person would type, with no site names and without the category word itself:
  >
  > - web (general information, explanations)
  > - academic (papers, research)
  > - code (repositories, APIs, errors)
  > - social_media (opinions, discussions)
  > - resource (datasets, tools, templates)
  > - news (current or recent events)
  > - official (government or organisation announcements, documentation)
  > - legal (laws, regulations, cases)
  > - financial (markets, filings, earnings)
  > - patent (patents, trademarks)
  > - product (specs, prices, reviews)
  > - business (company profiles, industries)
  > - local (nearby places, hours)
  > - travel (flights, hotels, destinations)
  > - media (images, videos, podcasts)
  > - books (books, authors)
  > - education (courses, tutorials)
  > - events (conferences, concerts)
  > - jobs (openings, salaries)
  > - statistics (indicators, census, surveys)
  >
  > 2-space indented JSON. Do not commit or modify other files. Reply with the path and the count.

  Validate it with `load_category_examples` (160 examples, all categories present). Then commit it on its own (`git add -f data/category_examples.json`): `data: category selection examples (drafted blind to eval labels)`.

- [x] **Step 2: Write the failing tests.** Create `tests/unit/test_category_evaluation.py`:

```python
"""Category selection evaluation: splits, metrics, tuning, leakage."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.model.pre_training.categories.evaluation import (
    BAR_COVERAGE, BAR_MEAN_PICKS, GRID_MARGIN, GRID_TOP_K,
    evaluate_selection, split_eval_labels, tune,
)
from src.model.pre_training.categories.model import (
    CategoryExample, CategoryIndex, load_category_examples,
)

EVAL = "data/eval/source_category_queries.json"
EXAMPLES = "data/category_examples.json"


def _norm(t):
    return " ".join(t.lower().split())


def test_split_is_first_label_per_category_then_the_rest():
    dev, test = split_eval_labels(EVAL)
    assert len(dev) == 16 and len(test) == 48
    assert len({c for c, _ in dev}) == 16
    assert not {q for _, q in dev} & {q for _, q in test}


def test_examples_cover_every_category_and_never_copy_an_eval_query():
    examples = load_category_examples(EXAMPLES)
    dev, test = split_eval_labels(EVAL)
    eval_texts = {_norm(q) for _, q in dev + test}
    assert not {_norm(e.text) for e in examples} & eval_texts


def _toy_index():
    ex = [CategoryExample("n", "news", "n"), CategoryExample("c", "code", "c")]
    v = np.eye(2, dtype=np.float32)
    return CategoryIndex(examples=ex, vectors=v, encoder="intfloat/e5-small-v2", fingerprint="f")


def _encode_by_text(mapping):
    return lambda texts: np.array([mapping[t] for t in texts], dtype=np.float32)


def test_metrics_count_coverage_picks_top1_and_confusion():
    labels = [("news", "q1"), ("code", "q2")]
    enc = _encode_by_text({"q1": [1, 0], "q2": [1, 0]})  # q2 looks like news
    r = evaluate_selection(_toy_index(), labels, top_k=1, select_margin=0.0, encode=enc)
    assert r["n"] == 2 and r["coverage"] == 0.5 and r["top1"] == 0.5 and r["mean_picks"] == 1.0
    assert r["confusion"]["code"] == {"news": 1}


def test_tune_prefers_coverage_then_fewer_picks_then_smaller_values():
    labels = [("news", "q1")]
    enc = _encode_by_text({"q1": [1, 0]})
    r = tune(_toy_index(), labels, encode=enc)
    assert r["top_k"] == min(GRID_TOP_K) and r["select_margin"] == min(GRID_MARGIN)
    assert len(r["grid"]) == len(GRID_TOP_K) * len(GRID_MARGIN)


def test_bar_constants_are_the_preregistered_ones():
    assert (BAR_COVERAGE, BAR_MEAN_PICKS) == (0.80, 2.0)
```

- [x] **Step 3: Run to verify it fails.** Expect an ImportError.

- [x] **Step 4: Implement:**
  - **`evaluation.py`:**
    - `split_eval_labels` reads the marked source-category label file (`taxonomy == "source_categories"`). It walks `domains` in file order; the first query of each category goes to dev, and the rest go to test.
    - `evaluate_selection` encodes the queries once, using `encode or encode_texts`, runs `index.select` per query, and computes:
      - coverage: the share of queries whose labelled category is in the selection;
      - `mean_picks`;
      - `top1`: whether the first selected category equals the label;
      - `per_category`;
      - `confusion`: for each labelled category, counts of the picked categories other than the label.
    - `tune` evaluates every grid pair on dev and picks by `(-coverage, mean_picks, top_k, select_margin)`. It returns `{"top_k", "select_margin", "grid": [rows]}`.
  - **`cli.py`:** argparse with `build`, `tune` and `evaluate`. `build` writes `<output>/index.npz`. `tune` and `evaluate` write JSON reports. `evaluate` adds `"bar": {"coverage": BAR_COVERAGE, "mean_picks": BAR_MEAN_PICKS}`, `"passed": coverage >= BAR_COVERAGE and mean_picks <= BAR_MEAN_PICKS`, `"split": "test"`, and the `top_k`/`select_margin` used.

- [x] **Step 5: Run.** Run `pytest tests/unit/test_category_evaluation.py tests/unit/test_category_index.py -q`. Expect PASS.

- [x] **Step 6: Build and tune for real** (local encoder, no API calls):

  ```bash
  python -m src.model.pre_training.categories.cli build --examples data/category_examples.json --output data/category_index
  python -m src.model.pre_training.categories.cli tune --index data/category_index --labels data/eval/source_category_queries.json --report data/eval/category_selection_tuning.json
  ```

  Set `DEFAULT_CATEGORY_TOP_K` and `DEFAULT_CATEGORY_SELECT_MARGIN` in `src/shared_configs/intent.py` to the tuned values. **Do not run `evaluate` yet**; Task 6 runs it exactly once.

- [x] **Step 7: Mutation checks.** Each must turn red, then restore:
  - (a) make the split use the last query per category instead of the first;
  - (b) swap the tuning tie-break to prefer more picks.
- [x] **Step 8: Commit.** Message: `feat: category selection CLI with dev-only tuning`. Include `data/eval/category_selection_tuning.json` via `git add -f`. The index itself is not committed.

### Task 3: Settings, serving adapter, resolver

**Files:**
- Modify: `src/internal/configs/app_configs.py`, `src/internal/tools/search.py`.
- Create: `src/internal/tools/category_selection.py`.
- Test: `tests/unit/test_auto_domains.py` (create), `tests/unit/test_configs.py`.

**Interfaces (produces):**
- `AppSettings.category_index_path: Path | None`
- `AppSettings.category_top_k: int`
- `AppSettings.category_select_margin: float`
- `category_selection.select_for_query(query: str) -> tuple[list[str], str]`, returning `(selected, status)` where status is `"ok"` or `"unavailable"`
- `search.AUTO_DOMAIN = "auto"`
- `async search.resolve_auto_domains(query: str, domains: list[str] | None) -> tuple[list[str] | None, dict | None]`

- [x] **Step 1: Write the failing tests.** Create `tests/unit/test_auto_domains.py`:

```python
"""resolve_auto_domains: opt-in, falls back to web, never blocks the loop."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from src.internal.tools import category_selection
from src.internal.tools.base import InvalidToolInput
from src.internal.tools.search import resolve_auto_domains
from src.model.pre_training.categories.model import CategoryExample, CategoryIndex


@pytest.fixture(autouse=True)
def _clear_cache():
    category_selection._INDEXES.clear()
    yield
    category_selection._INDEXES.clear()


def _saved_index(tmp_path):
    ex = [CategoryExample("n", "news", "n"), CategoryExample("c", "code", "c")]
    idx = CategoryIndex(examples=ex, vectors=np.eye(2, dtype=np.float32),
                        encoder="intfloat/e5-small-v2", fingerprint="f")
    idx.save(tmp_path / "index.npz")
    return tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize("domains", [None, ["news"], ["news", "code"]])
async def test_non_auto_input_is_returned_unchanged(domains):
    assert await resolve_auto_domains("q", domains) == (domains, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("domains", [["auto", "news"], ["news", "auto"]])
async def test_auto_mixed_with_names_is_rejected(domains):
    with pytest.raises(InvalidToolInput):
        await resolve_auto_domains("q", domains)


@pytest.mark.asyncio
async def test_auto_selects_with_the_configured_index(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_INDEX_PATH", str(_saved_index(tmp_path)))
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_TOP_K", "1")
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SELECT_MARGIN", "0")
    monkeypatch.setattr(category_selection, "encode_texts", lambda texts: np.array([[0, 1]], dtype=np.float32))
    assert await resolve_auto_domains("q", ["auto"]) == (["code"], {"selected": ["code"], "status": "ok"})


@pytest.mark.asyncio
async def test_unset_path_falls_back_to_web(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_CATEGORY_INDEX_PATH", raising=False)
    assert await resolve_auto_domains("q", ["auto"]) == (["web"], {"selected": ["web"], "status": "unavailable"})


@pytest.mark.asyncio
async def test_a_failed_load_is_cached_and_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_INDEX_PATH", str(tmp_path))  # no index.npz
    loads = []
    real = category_selection._load
    monkeypatch.setattr(category_selection, "_load", lambda p: loads.append(p) or real(p))
    for _ in range(2):
        assert (await resolve_auto_domains("q", ["auto"]))[1]["status"] == "unavailable"
    assert len(loads) == 1


@pytest.mark.asyncio
async def test_a_raising_encoder_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_INDEX_PATH", str(_saved_index(tmp_path)))
    def boom(texts):
        raise RuntimeError("no model")
    monkeypatch.setattr(category_selection, "encode_texts", boom)
    assert (await resolve_auto_domains("q", ["auto"]))[0] == ["web"]


@pytest.mark.asyncio
async def test_encoding_runs_off_the_event_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_INDEX_PATH", str(_saved_index(tmp_path)))
    monkeypatch.setattr(category_selection, "encode_texts", lambda texts: np.array([[1, 0]], dtype=np.float32))
    used = []
    real = asyncio.to_thread
    async def spy(fn, *a, **k):
        used.append(fn)
        return await real(fn, *a, **k)
    monkeypatch.setattr(asyncio, "to_thread", spy)
    await resolve_auto_domains("q", ["auto"])
    assert used
```

In `tests/unit/test_configs.py`, add the defaults (path None, top_k equal to the tuned `DEFAULT_TOP_K`, margin equal to the tuned `DEFAULT_SELECT_MARGIN`) and validation: `top_k` of 0 raises, and a margin of 1.5 raises.

- [x] **Step 2: Run to verify they fail.**

- [x] **Step 3: Implement:**
  - **`app_configs.py`:** add three fields beside the intent fields:
    - `category_index_path: Path | None = None`
    - `category_top_k: int`
    - `category_select_margin: float`

    Import the defaults from `src/shared_configs/intent.py` (`DEFAULT_CATEGORY_TOP_K`, `DEFAULT_CATEGORY_SELECT_MARGIN`), exactly as the intent defaults are imported. In `load_app_settings`, read `AGENTIC_SEARCH_CATEGORY_INDEX_PATH`, `_TOP_K` and `_SELECT_MARGIN`. Validate in `__post_init__`, following the intent fields.
  - **`src/internal/tools/category_selection.py`:** follow the pattern of `src/internal/servers/web/intent/similarity.py`.
    - Module-level state: `_INDEXES: dict[Path, object | None]`, a lock, and a thin `encode_texts(texts)` wrapper with a function-local import of the intent `encode_texts`.
    - `_load(directory)` loads `CategoryIndex` with a function-local import and rejects an encoder mismatch.
    - `_index_for(settings)` caches per resolved path, including `None` for a failed load.
    - `select_for_query(query)`:

      ```python
      settings = load_app_settings()
      index = _index_for(settings)
      if index is None:
          return ["web"], "unavailable"
      try:
          vector = encode_texts([query])[0]
          return index.select(vector, top_k=settings.category_top_k,
                              select_margin=settings.category_select_margin), "ok"
      except Exception:
          logger.exception("category-index: selection failed — searching web")
          return ["web"], "unavailable"
      ```

  - **`search.py`:**

    ```python
    AUTO_DOMAIN = "auto"

    async def resolve_auto_domains(query, domains):
        """Expand domains == ["auto"] into selected categories; anything else unchanged."""
        if not domains or AUTO_DOMAIN not in [str(d).strip().lower() for d in domains]:
            return domains, None
        if len(domains) != 1:
            raise InvalidToolInput("auto cannot be combined with named categories")
        from .category_selection import select_for_query

        selected, status = await asyncio.to_thread(select_for_query, query)
        return selected, {"selected": selected, "status": status}
    ```

  `domain="auto"` already fails `normalize_search_domain` as an unknown category. Add a test that pins it.
- [x] **Step 4: Run.** Run `pytest tests/unit/test_auto_domains.py tests/unit/test_configs.py tests/unit/test_category_index.py -q`, then `pytest -q`. Expect PASS.
- [x] **Step 5: Mutation checks.** Each must turn red, then restore:
  - (a) call `select_for_query` directly instead of through `to_thread`;
  - (b) don't cache a failed load;
  - (c) let `auto` mix with names.
- [x] **Step 6: Commit.** Message: `feat: resolve domains=["auto"] through the category index`.

### Task 4: Entry points: HTTP, tools, MCP

**Files:** modify `src/internal/servers/web/app.py`, `src/internal/tools/search.py` (`MultiQueryWebSearchTool.execute`, `DomainSearch.search` and `batch_search`, the schemas), and `src/internal/mcp_server/tools/search.py`. Tests: `tests/unit/servers/web/test_search_domain_http.py`, `tests/unit/test_domain_search.py`, `tests/unit/test_domain_search_tools.py`, `tests/unit/test_domain_search_mcp.py`.

**Interfaces (produces):**
- `AgentExperienceResponse.auto_domains: dict | None = None`, which is also included in the `_terminal_events` `done` event.
- The `domains` schema `items.enum` gains `"auto"` on the tools and MCP. Validation copies leave `"auto"` as is.
- Tool and MCP metadata/results gain `auto_domains`.

**Behaviour:** at every entry point, `domains, auto_info = await resolve_auto_domains(query, domains)` runs **before** `normalize_search_domains`. The rest of the existing flow is unchanged.
- **HTTP:** for an `auto` request, `domains` echoes the resolved list, **even for one category**: `echoed_domains = selected if auto_info else (existing rule)`. The mode rule applies to the resolved categories.
- **Single domain:** when `auto` resolves to one category, the selection travels through `app.py` as a plain `str`, exactly like a single named category.

- [x] **Step 1: Write the failing tests.** For each entry point, with `resolve_auto_domains` patched where it is used (`src.internal.servers.web.app.resolve_auto_domains`, or the `search` module attribute for the tools), returning `(["news", "code"], {"selected": ["news", "code"], "status": "ok"})`:
  - **HTTP `search_tool`:** `_run_direct_search` receives `domain == ("news", "code")`; the response has `domains == ["news", "code"]` and `auto_domains == {...}`.
  - **HTTP single-category auto:** the patched resolver returns `(["news"], {...})`. The leaf receives `domain == "news"`, a `str`, and the response's `domains == ["news"]`.
  - **HTTP request without `auto`:** the response's `auto_domains` is None. Also check the stream `done` event: it has `auto_domains` for `auto` and null otherwise.
  - **Unavailable:** the resolver returns `(["web"], {"selected": ["web"], "status": "unavailable"})`. The request succeeds with `domain == "web"`, and `auto_domains.status == "unavailable"`.
  - **`auto` mixed with names:** 400 over HTTP, and an error with zero provider calls on `web_search`, `search_domain` and MCP.
  - **`web_search` tool:** `domains: ["auto"]` passes schema validation; the patched resolver's picks are searched; the metadata has `auto_domains`.
  - **`DomainSearch.search(q, domains=["auto"])`:** the result has `domains` and `auto_domains`. `tag` plus `domains=["auto"]` is still rejected.
  - **MCP `search_web(q, domains=["auto"])`:** the result has `domains` and `auto_domains`.

  Write real tests with stub providers, using each file's existing helpers.
- [x] **Step 2: Run to verify they fail.**
- [x] **Step 3: Implement:**
  - **HTTP:** in `_run_agent_impl`, before `normalize_search_domains`, run `requested, auto_info = await resolve_auto_domains(query, request.domains)`. Normalize `requested`. `echoed_domains` is `list(selected)` when `auto_info` is set, and otherwise follows today's rule. Pass `auto_domains=auto_info` through `_finalize_response` (all its call sites) and into `_terminal_events`.
  - **Tools:** in `MultiQueryWebSearchTool.execute` and `DomainSearch.search`, resolve first, then run today's code. Add `auto_domains` to the metadata/result when it is set.
  - **Schemas:** add `"auto"` to `domains.items.enum` (`list(DOMAIN_REGISTRY) + [AUTO_DOMAIN]`). Make sure `_source_validation_arguments` does not fail on `"auto"`; it already leaves unknown values as they are for validation.
  - **MCP:** `search_web` and `search_domain` resolve inside the existing `try` (or before it, following the current `domains` handling), and add `auto_domains` to their outputs.
- [x] **Step 4: Run.** Run the touched suites, then `pytest -q`. Existing tests must pass unmodified, except schema tests that pin the `domains` enum exactly; extend those with `"auto"`.
- [x] **Step 5: Mutation checks.** Each must turn red, then restore:
  - (a) skip the resolver on HTTP;
  - (b) echo a one-category auto pick as null;
  - (c) drop `auto_domains` from the done event.
- [x] **Step 6: Commit.** Message: `feat: domains=["auto"] on HTTP, tools and MCP`.

### Task 5: Assist page: the Auto option

**Files:** modify `web/src/domains.ts`, `web/src/components/SearchComposer.tsx`, `web/src/pages/AssistPage.tsx` and `web/src/types.ts`. Tests: `web/src/__tests__/domains.test.ts`, `web/src/components/__tests__/SearchComposer.test.tsx` and `web/src/pages/__tests__/AssistPage.test.tsx`.

**Interfaces:** `AUTO = "auto"`; `toggleDomain` handles `auto`; `AgentExperienceResponse.auto_domains?: {selected: string[]; status: "ok" | "unavailable"} | null`; `SSEDoneEvent.auto_domains?` has the same shape.

- [x] **Step 1: Write the failing tests:**
  - **`domains.test.ts`:**
    - `toggleDomain(["web"], "auto")` → `["auto"]`;
    - `toggleDomain(["auto"], "news")` → `["news"]`;
    - `toggleDomain(["news", "code"], "auto")` → `["auto"]`;
    - `toggleDomain(["auto"], "auto")` → `["web"]`.
  - **Composer:** "Auto" is the first checkbox in the group; ticking it calls `onDomainsChange(["auto"])`; it is never disabled by the limit.
  - **AssistPage:**
    - Auto sends `domains: ["auto"]`.
    - After a stream whose done event has `auto_domains: {selected: ["financial", "news"], status: "ok"}`, the page shows "Auto picked: financial · news".
    - With `status: "unavailable"`, it shows "Auto unavailable: searched web".
    - The selection stays `["auto"]` after the search.
- [x] **Step 2: Run to verify they fail.**
- [x] **Step 3: Implement:**
  - **`toggleDomain`:**
    - `name === AUTO` → `["auto"]` (selecting it again when it's the current selection → `["web"]`);
    - any other name while the selection is `["auto"]` → behave as if starting from `["web"]`.
  - **Composer:** render an "Auto" checkbox before the registry options, with the title "Pick up to three sources automatically". It is never limit-disabled.
  - **AssistPage:**
    - Store `autoDomains` from the done event (and the JSON response, if that path exists).
    - Render one muted line under the composer when set.
    - The request already sends `domains` for a non-single selection. `["auto"]` has length 1, so add an explicit rule: `domains[0] === AUTO ? { domains } : (single ? { domain } : { domains })`.
- [x] **Step 4: Run.** Run `cd web && npm run typecheck && npm run test:unit`. Expect PASS.
- [x] **Step 5: Mutation checks.** Each must turn red, then restore:
  - (a) send `domain: "auto"` for the single auto selection;
  - (b) let category ticks append to `["auto"]`.
- [x] **Step 6: Commit.** Message: `feat: Assist page Auto search source`.

### Task 6: Evaluation run, docs, review, PR

- [x] **Step 1: Run the pre-registered evaluation exactly once:**

  ```bash
  python -m src.model.pre_training.categories.cli build --examples data/category_examples.json --output data/category_index
  python -m src.model.pre_training.categories.cli evaluate --index data/category_index --labels data/eval/source_category_queries.json --report data/eval/category_selection.json
  ```

  Also run the eval-gate leakage check: every example's e5 cosine to every eval query must be ≤ 0.95. Report the maximum and the pair. Commit the report with `git add -f data/eval/category_selection.json`. Whatever the result, do not change the examples, the grid or the defaults afterwards.
- [x] **Step 2: CI.** Add a step to `.github/workflows/eval-gate.yml`, next to the intent index build: build `data/category_index` and run `evaluate`. Follow that workflow's pattern of skipping when the encoder is unavailable. Do not fail CI on the bar; the PR description records the result. The leakage cosine check fails the step if any pair exceeds 0.95.
- [x] **Step 3: Docs:**
  - **`docs/search-engine.md`:** an "Automatic category selection" section covering opt-in `domains: ["auto"]`; the method (e5 nearest examples, top-k mean, margin, at most 3); the examples (written blind to the eval labels); the fallback; the settings; the measured test result against the bar (coverage, mean picks, top-1, the worst confusions); and the caveats from the spec.
  - **`docs/configuration.md`:** three rows.
  - **`api-reference`, `mcp`, `frontend`:** the `auto` option.
  - **README:** one clause about opt-in automatic selection.

  Verify every claim against the code.
- [x] **Step 4: Full verification:** `pytest -q`; Ruff; `cd web && npm run typecheck && npm run test:unit`; `git diff --check`; the docs link check.
- [x] **Step 5: Whole-branch review.** One fresh reviewer, on the most capable model, against `origin/main`, the spec, this plan and the Review Focus. Fix material findings RED→GREEN and record rulings.
- [ ] **Step 6: PR.** Before each commit, check the branch. Before pushing, check `git log origin/main -3`. Push, open the PR with the evaluation result in its description, watch CI on the final commit, and report. Do not merge.

## Validation record

Spec and plan approved by the user on 2026-10-12. Executed subagent-driven, with a fresh implementer and reviewer per task.

- **Examples:** 160 examples, written blind to the eval labels (`041ca0c0`). No normalized-text overlap.
- **Tuning (dev split, 16 queries):** `top_k=3`, `select_margin=0.0`, coverage 0.938 (15/16), mean picks 1.0. Re-running the tuning reproduces the committed report byte for byte.
- **Pre-registered test result (48 queries, run once, `9bab5428`):** coverage **0.729** (35/48), mean picks 1.00, top-1 0.729. **The bar (0.80) is missed**, so the feature ships opt-in and the docs report the miss.
- **Leakage:** two independent paraphrase pairs scored above 0.95 e5 cosine, both in the test split. Excluding them gives coverage 0.717 (n=46), so the conclusion is unchanged. The examples were kept and the CI guard allowlists exactly these two pairs (a ruling). Process lesson: run the semantic leakage check *before* the test measurement.
- **Fix rounds:** Task 5 needed one, because a stale Auto pick stayed visible during the next search. Plan corrections: the tie test expected a registry-ordered pair, and defaults live in `shared_configs` to avoid an import cycle.
- **Final review:** ready to merge, with 0 critical and 0 important findings. The fix wave `4d7ba460` covered:
  - `auto` on a non-search mode is now rejected before selection;
  - non-list input is guarded;
  - CI warns when a fresh evaluation drifts from the committed numbers.

  The re-review was clean.
- **Local checks:** `pytest` 5523 passed, 1 skipped; vitest 265 passed; typecheck, Ruff, `git diff --check` and the docs link check all clean.
