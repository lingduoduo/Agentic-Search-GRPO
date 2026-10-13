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
    # so the round-one result (0.729) remains reproducible. Fingerprint of the
    # round-one example set as merged in #689 (ebd01873), so the check needs no
    # git history (CI checkouts are shallow).
    import hashlib
    from pathlib import Path

    round_one_sha256 = (
        "1ec71f7a766e267611b44120019acb4828571efc98c11a4e1b4d560dfa44bf89"
    )
    repo = Path(__file__).resolve().parents[2]
    current = json.loads((repo / "data" / "category_examples.json").read_text())
    original = current[:160]
    assert len(original) == 160
    digest = hashlib.sha256(
        json.dumps(original, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == round_one_sha256
    assert len(current) > 160


def test_v2_files_load_and_test_set_has_three_per_category():
    from collections import Counter
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    test = load_multilabel_queries(repo / "data/eval/category_selection_test_v2.json")
    dev = load_multilabel_queries(repo / "data/eval/category_selection_dev_v2.json")
    assert len(test) == 60 and set(Counter(q.primary for q in test).values()) == {3}
    assert len(dev) == 64


def _collisions(examples, dev, test):
    """Example or dev texts that normalize equal to a test query."""
    from src.model.pre_training.categories.model import _normalize

    keys = {_normalize(t) for t in test}
    return [
        (kind, text)
        for kind, texts in (("example", examples), ("dev", dev))
        for text in texts
        if _normalize(text) in keys
    ]


def test_no_example_or_dev_query_equals_a_test_query():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    examples = [
        r["text"]
        for r in json.loads((repo / "data" / "category_examples.json").read_text())
    ]
    dev = [
        q.query
        for q in load_multilabel_queries(
            repo / "data/eval/category_selection_dev_v2.json"
        )
    ]
    test = [
        q.query
        for q in load_multilabel_queries(
            repo / "data/eval/category_selection_test_v2.json"
        )
    ]
    assert _collisions(examples, dev, test) == []
    # Positive control: a test query copied (re-cased) into the examples is caught.
    leaked = examples + ["  " + test[0].upper() + " "]
    assert _collisions(leaked, dev, test) == [("example", leaked[-1])]


@pytest.mark.parametrize(
    "record,msg",
    [
        ({"primary": "news", "query": "a", "acceptable": ["news"]}, "id"),
        ({"id": 1, "primary": "news", "query": "a", "acceptable": ["news"]}, "id"),
        ({"id": "q1", "query": "a", "acceptable": ["news"]}, "primary"),
        (
            {"id": "q1", "primary": ["news"], "query": "a", "acceptable": ["news"]},
            "primary",
        ),
        ({"id": "q1", "primary": "news", "acceptable": ["news"]}, "query"),
        (
            {"id": "q1", "primary": "news", "query": None, "acceptable": ["news"]},
            "query",
        ),
        ({"id": "q1", "primary": "news", "query": "a"}, "acceptable"),
        (
            {"id": "q1", "primary": "news", "query": "a", "acceptable": "news"},
            "acceptable",
        ),
        (
            {
                "id": "q1",
                "primary": "news",
                "query": "a",
                "acceptable": ["news", "news"],
            },
            "acceptable",
        ),
    ],
)
def test_loader_rejects_malformed_records_with_value_error(tmp_path, record, msg):
    with pytest.raises(ValueError, match=msg):
        load_multilabel_queries(_write(tmp_path, [record]))


def _cli_setup(tmp_path, monkeypatch, ids):
    """Tiny one-hot index on disk + v2 labels; encoder maps query text -> vector."""
    from src.model.pre_training.categories import evaluation
    from src.model.pre_training.categories.model import INDEX_FILENAME

    _index().save(tmp_path / "index" / INDEX_FILENAME)
    vec = {"qa": [1.0, 0.0, 0.0], "qb": [0.0, 0.0, 1.0]}
    monkeypatch.setattr(
        evaluation,
        "encode_texts",
        lambda texts: np.array([vec[t] for t in texts], dtype=np.float32),
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "taxonomy": "category_selection_v2",
                "queries": [
                    {
                        "id": ids[0],
                        "primary": "news",
                        "query": "qa",
                        "acceptable": ["news"],
                    },
                    {
                        "id": ids[1],
                        "primary": "code",
                        "query": "qb",
                        "acceptable": ["code"],
                    },
                ],
            }
        )
    )
    return tmp_path / "index", labels


def _sha256(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tune_v2_refuses_test_set_ids(tmp_path, monkeypatch, capsys):
    from src.model.pre_training.categories import cli

    index, labels = _cli_setup(tmp_path, monkeypatch, ["d-news-1", "t-code-1"])
    report = tmp_path / "tuning.json"
    argv = ["tune-v2", "--index", str(index), "--labels", str(labels)]
    with pytest.raises(SystemExit) as exc:
        cli.main(argv + ["--report", str(report)])
    assert exc.value.code != 0
    assert "test set" in capsys.readouterr().err
    assert not report.exists()


def test_tune_v2_refuses_the_committed_test_file(tmp_path, monkeypatch, capsys):
    from pathlib import Path

    from src.model.pre_training.categories import cli

    index, _ = _cli_setup(tmp_path, monkeypatch, ["d-1", "d-2"])
    repo = Path(__file__).resolve().parents[2]
    test_file = repo / "data/eval/category_selection_test_v2.json"
    with pytest.raises(SystemExit) as exc:
        cli.main(
            ["tune-v2", "--index", str(index), "--labels", str(test_file)]
            + ["--report", str(tmp_path / "r.json")]
        )
    assert exc.value.code != 0
    assert "test set" in capsys.readouterr().err


def test_tune_v2_on_dev_ids_runs_and_records_label_provenance(tmp_path, monkeypatch):
    from src.model.pre_training.categories import cli

    index, labels = _cli_setup(tmp_path, monkeypatch, ["d-news-1", "d-code-1"])
    report = tmp_path / "tuning.json"
    argv = ["tune-v2", "--index", str(index), "--labels", str(labels)]
    assert cli.main(argv + ["--report", str(report)]) == 0
    r = json.loads(report.read_text())
    assert r["split"] == "dev_v2" and r["feasible"] is True
    assert r["labels_path"] == str(labels)
    assert r["labels_sha256"] == _sha256(labels)


def test_evaluate_v2_report_split_bar_and_inclusive_pass(tmp_path, monkeypatch):
    from src.model.pre_training.categories import cli

    index, labels = _cli_setup(tmp_path, monkeypatch, ["t-news-1", "t-code-1"])
    report = tmp_path / "eval.json"
    argv = ["evaluate-v2", "--index", str(index), "--labels", str(labels)]
    argv += ["--report", str(report), "--top-k", "1", "--select-margin", "0"]
    assert cli.main(argv) == 0
    r = json.loads(report.read_text())
    assert r["split"] == "test_v2" and (r["top_k"], r["select_margin"]) == (1, 0.0)
    assert r["bar"] == {"hit": 0.85, "precision": 0.60, "mean_picks": 2.0}
    assert (r["hit"], r["precision"], r["mean_picks"]) == (1.0, 1.0, 1.0)
    assert r["passed"] is True
    assert r["labels_sha256"] == _sha256(labels)
    # Bars equal to the achieved values still pass (all three bounds inclusive).
    monkeypatch.setattr(cli, "V2_BAR_HIT", 1.0)
    monkeypatch.setattr(cli, "V2_BAR_PRECISION", 1.0)
    monkeypatch.setattr(cli, "V2_BAR_MEAN_PICKS", 1.0)
    assert cli.main(argv) == 0
    assert json.loads(report.read_text())["passed"] is True
    monkeypatch.setattr(cli, "V2_BAR_MEAN_PICKS", 0.99)
    assert cli.main(argv) == 0
    assert json.loads(report.read_text())["passed"] is False


def test_leakage_cli_against_uses_v2_check_and_labels_uses_round_one(
    tmp_path, monkeypatch
):
    from src.model.pre_training.categories import cli, evaluation

    monkeypatch.setattr(evaluation, "encode_texts", _same_vector_encoder)
    example = "best time to visit iceland"
    query = "best time to visit iceland to see the northern lights"
    ex, test, _ = _leak_files(tmp_path, example, query, "unrelated")
    # v2: the round-one allowlisted pair is flagged.
    assert cli.main(["leakage", "--examples", str(ex), "--against", str(test)]) == 1
    # Round one: the same pair is allowlisted.
    labels = tmp_path / "round_one.json"
    labels.write_text(
        json.dumps(
            {"taxonomy": "source_categories", "domains": {"travel": [{"query": query}]}}
        )
    )
    assert cli.main(["leakage", "--examples", str(ex), "--labels", str(labels)]) == 0


def test_leakage_cli_needs_labels_or_against(tmp_path):
    from src.model.pre_training.categories import cli

    ex, _, _ = _leak_files(tmp_path, "a", "b", "c")
    with pytest.raises(SystemExit) as exc:
        cli.main(["leakage", "--examples", str(ex)])
    assert exc.value.code != 0


def test_tune_v2_refuses_the_test_path_even_with_dev_ids(tmp_path, monkeypatch):
    from src.model.pre_training.categories import cli

    index, labels = _cli_setup(tmp_path, monkeypatch, ["d-news-1", "d-code-1"])
    monkeypatch.setattr(cli, "V2_TEST_LABELS", labels.resolve())
    with pytest.raises(SystemExit) as exc:
        cli.main(
            ["tune-v2", "--index", str(index), "--labels", str(labels)]
            + ["--report", str(tmp_path / "r.json")]
        )
    assert exc.value.code != 0
