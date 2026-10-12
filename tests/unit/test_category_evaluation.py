"""Category selection evaluation: splits, metrics, tuning, leakage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.model.pre_training.categories.evaluation import (
    BAR_COVERAGE,
    BAR_MEAN_PICKS,
    GRID_MARGIN,
    GRID_TOP_K,
    evaluate_selection,
    split_eval_labels,
    tune,
)
from src.model.pre_training.categories.model import (
    CategoryExample,
    CategoryIndex,
    load_category_examples,
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
    payload = json.loads(Path(EVAL).read_text())
    first = {c: entries[0]["query"] for c, entries in payload["domains"].items()}
    assert dict(dev) == first


def test_examples_cover_every_category_and_never_copy_an_eval_query():
    examples = load_category_examples(EXAMPLES)
    dev, test = split_eval_labels(EVAL)
    eval_texts = {_norm(q) for _, q in dev + test}
    assert not {_norm(e.text) for e in examples} & eval_texts


def _toy_index():
    ex = [CategoryExample("n", "news", "n"), CategoryExample("c", "code", "c")]
    v = np.eye(2, dtype=np.float32)
    return CategoryIndex(
        examples=ex, vectors=v, encoder="intfloat/e5-small-v2", fingerprint="f"
    )


def _encode_by_text(mapping):
    return lambda texts: np.array([mapping[t] for t in texts], dtype=np.float32)


def test_metrics_count_coverage_picks_top1_and_confusion():
    labels = [("news", "q1"), ("code", "q2")]
    enc = _encode_by_text({"q1": [1, 0], "q2": [1, 0]})  # q2 looks like news
    r = evaluate_selection(_toy_index(), labels, top_k=1, select_margin=0.0, encode=enc)
    assert r["n"] == 2 and r["coverage"] == 0.5 and r["top1"] == 0.5
    assert r["mean_picks"] == 1.0
    assert r["confusion"]["code"] == {"news": 1}


def test_tune_prefers_coverage_then_fewer_picks_then_smaller_values():
    labels = [("news", "q1")]
    enc = _encode_by_text({"q1": [1, 0]})
    r = tune(_toy_index(), labels, encode=enc)
    assert r["top_k"] == min(GRID_TOP_K) and r["select_margin"] == min(GRID_MARGIN)
    assert len(r["grid"]) == len(GRID_TOP_K) * len(GRID_MARGIN)


def test_bar_constants_are_the_preregistered_ones():
    assert (BAR_COVERAGE, BAR_MEAN_PICKS) == (0.80, 2.0)


def test_tune_tie_break_avoids_extra_picks_when_coverage_ties():
    close = np.array([0.99, np.sqrt(1 - 0.99**2)], dtype=np.float32)
    v = np.stack([np.array([1, 0], dtype=np.float32), close])
    ex = [CategoryExample("n", "news", "n"), CategoryExample("c", "code", "c")]
    index = CategoryIndex(
        examples=ex, vectors=v, encoder="intfloat/e5-small-v2", fingerprint="f"
    )
    enc = _encode_by_text({"q1": [1, 0]})
    r = tune(index, [("news", "q1")], encode=enc)
    assert r["select_margin"] == 0.0
    assert max(row["mean_picks"] for row in r["grid"]) == 2.0
