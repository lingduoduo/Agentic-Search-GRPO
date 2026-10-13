"""v2: multi-label loader, hit/precision metrics, constrained tuning."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.model.pre_training.categories.evaluation import (
    V2_BAR_HIT,
    V2_BAR_MEAN_PICKS,
    V2_BAR_PRECISION,
    MultiLabelQuery,
    evaluate_multilabel,
    load_multilabel_queries,
    tune_multilabel,
)
from src.model.pre_training.categories.model import CategoryExample, CategoryIndex


def _write(tmp_path, queries, marker="category_selection_v2"):
    p = tmp_path / "v2.json"
    p.write_text(json.dumps({"taxonomy": marker, "queries": queries}))
    return p


def _q(i, primary, text, acceptable):
    return {"id": f"q{i}", "primary": primary, "query": text, "acceptable": acceptable}


def test_loader_reads_valid_records(tmp_path):
    out = load_multilabel_queries(
        _write(tmp_path, [_q(1, "news", "a", ["news", "financial"])])
    )
    assert out == [MultiLabelQuery("q1", "news", "a", ("news", "financial"))]


@pytest.mark.parametrize(
    "queries,marker,msg",
    [
        ([_q(1, "news", "a", ["news"])], "other", "taxonomy"),
        ([_q(1, "nope", "a", ["nope"])], "category_selection_v2", "primary"),
        ([_q(1, "news", "a", ["financial"])], "category_selection_v2", "primary"),
        (
            [_q(1, "news", "a", ["news", "code", "legal", "books"])],
            "category_selection_v2",
            "acceptable",
        ),
        ([_q(1, "news", "a", ["news", "zzz"])], "category_selection_v2", "acceptable"),
        (
            [_q(1, "news", "Same q", ["news"]), _q(2, "code", "same  q", ["code"])],
            "category_selection_v2",
            "query",
        ),
    ],
)
def test_loader_rejects(tmp_path, queries, marker, msg):
    with pytest.raises(ValueError, match=msg):
        load_multilabel_queries(_write(tmp_path, queries, marker))


def _index():
    ex = [
        CategoryExample("n", "news", "n"),
        CategoryExample("f", "financial", "f"),
        CategoryExample("c", "code", "c"),
    ]
    v = np.eye(3, dtype=np.float32)
    return CategoryIndex(
        examples=ex, vectors=v, encoder="intfloat/e5-small-v2", fingerprint="f"
    )


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
    assert r["mean_picks"] == 1.5  # 2 picks + 1 pick
    assert r["precision"] == pytest.approx(2 / 3)  # 2 acceptable of 3 picks
    assert r["confusion"]["code"] == {"news": 1}
    assert r["top1"] == 0.5  # q1's first pick is acceptable, q2's is not
    assert r["per_primary"]["news"] == {"n": 1, "hit": 1.0}
    assert r["per_primary"]["code"] == {"n": 1, "hit": 0.0}


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
    assert (
        row["hit"] == 1.0
        and row["precision"] == pytest.approx(2 / 3)
        and row["mean_picks"] == 1.5
    )


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
        {
            "top_k": 1,
            "select_margin": 0.05,
            "hit": 1.0,
            "precision": 0.50,
            "mean_picks": 2.0,
        },  # infeasible
        {
            "top_k": 3,
            "select_margin": 0.00,
            "hit": 0.9,
            "precision": 0.90,
            "mean_picks": 1.0,
        },  # feasible
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


def _leak_files(tmp_path, example_text, test_query, dev_query):
    ex = tmp_path / "examples.json"
    ex.write_text(
        json.dumps([{"id": "e1", "category": "travel", "text": example_text}])
    )
    test = _write(tmp_path, [_q(1, "travel", test_query, ["travel"])])
    dev = tmp_path / "dev.json"
    dev.write_text(
        json.dumps(
            {
                "taxonomy": "category_selection_v2",
                "queries": [_q(2, "travel", dev_query, ["travel"])],
            }
        )
    )
    return ex, test, dev


def _same_vector_encoder(texts):
    # example and the two "near" queries share a direction; others are orthogonal.
    near = {
        "best time to visit iceland",
        "best time to visit iceland to see the northern lights",
        "dev near",
    }
    rows = [[1.0, 0.0] if t in near else [0.0, 1.0] for t in texts]
    return np.array(rows, dtype=np.float32)


def test_leakage_against_has_no_allowlist_and_checks_dev(tmp_path):
    from src.model.pre_training.categories.evaluation import find_leakage_multilabel

    # The round-one allowlisted pair is flagged against a v2 file.
    ex, test, dev = _leak_files(
        tmp_path,
        "best time to visit iceland",
        "best time to visit iceland to see the northern lights",
        "unrelated",
    )
    r = find_leakage_multilabel(ex, test, encode=_same_vector_encoder)
    assert len(r["flagged"]) == 1 and r["allowed"] == []

    # A dev query near a TEST query is flagged, and only when dev is passed.
    ex, test, dev = _leak_files(tmp_path, "unrelated ex", "test near", "dev near")
    enc = lambda texts: np.array(  # noqa: E731
        [[1.0, 0.0] if t in {"dev near", "test near"} else [0.0, 1.0] for t in texts],
        dtype=np.float32,
    )
    assert find_leakage_multilabel(ex, test, encode=enc)["flagged"] == []
    r = find_leakage_multilabel(ex, test, dev_path=dev, encode=enc)
    assert [(p["source"], p["kind"], p["query"]) for p in r["flagged"]] == [
        ("dev near", "dev", "test near")
    ]
    # An example near a DEV query (test unrelated) is not flagged.
    ex, test, dev = _leak_files(tmp_path, "dev near", "unrelated", "dev near")
    r = find_leakage_multilabel(ex, test, dev_path=dev, encode=_same_vector_encoder)
    assert r["flagged"] == []
    # Normalized equality is flagged even when the encoder says they are far apart.
    ex, test, dev = _leak_files(tmp_path, "Same  Text", "same text", "other")
    keyed = _enc(
        {"Same  Text": [1.0, 0.0], "same text": [0.0, 1.0], "other": [1.0, 0.0]}
    )
    r = find_leakage_multilabel(ex, test, dev_path=dev, encode=keyed)
    assert len(r["flagged"]) == 1
    assert r["flagged"][0]["source"] == "Same  Text"
    assert r["flagged"][0]["kind"] == "example"
    assert r["flagged"][0]["cosine"] == 0.0


def test_loader_rejects_duplicate_ids(tmp_path):
    qs = [_q(1, "news", "a", ["news"]), {**_q(1, "code", "b", ["code"])}]
    with pytest.raises(ValueError, match="id"):
        load_multilabel_queries(_write(tmp_path, qs))


def test_top1_accepts_any_acceptable_and_per_primary_hit_is_a_rate():
    qs = [
        MultiLabelQuery("1", "news", "q1", ("news", "financial")),  # picks financial
        MultiLabelQuery("2", "code", "q2", ("code",)),  # picks code
        MultiLabelQuery("3", "code", "q3", ("code",)),  # picks news: miss
    ]
    enc = _enc({"q1": _unit(0, 1, 0), "q2": _unit(0, 0, 1), "q3": _unit(1, 0, 0)})
    r = evaluate_multilabel(_index(), qs, top_k=1, select_margin=0.0, encode=enc)
    assert r["top1"] == pytest.approx(2 / 3)
    assert r["per_primary"]["code"] == {"n": 2, "hit": 0.5}


def test_round_one_examples_are_unchanged():
    # v2 only appends examples; the 160 round-one records stay byte-identical
    # so the round-one result (0.729) remains reproducible.
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    original = json.loads(
        subprocess.check_output(
            ["git", "show", "041ca0c0:data/category_examples.json"], cwd=repo
        )
    )
    current = json.loads((repo / "data" / "category_examples.json").read_text())
    assert len(original) == 160
    assert current[:160] == original
    assert len(current) > 160


def test_v2_files_load_and_test_set_has_three_per_category():
    from collections import Counter
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    test = load_multilabel_queries(repo / "data/eval/category_selection_test_v2.json")
    dev = load_multilabel_queries(repo / "data/eval/category_selection_dev_v2.json")
    assert len(test) == 60 and set(Counter(q.primary for q in test).values()) == {3}
    assert len(dev) == 64
