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
    examples = [
        CategoryExample(f"e{i}", c, f"text {i}") for i, (c, _) in enumerate(rows)
    ]
    vectors = np.array([v for _, v in rows], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return CategoryIndex(
        examples=examples,
        vectors=vectors,
        encoder="intfloat/e5-small-v2",
        fingerprint="f",
    )


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
    rows = [
        ("news", _unit(1, 0, 0, 0)),
        ("code", _unit(0.99, 0.14, 0, 0)),
        ("legal", _unit(0.98, 0.2, 0, 0)),
        ("books", _unit(0.97, 0.24, 0, 0)),
        ("jobs", _unit(0, 0, 0, 1)),
    ]
    idx = _index(rows)
    q = _unit(1, 0, 0, 0)
    assert idx.select(q, top_k=1, select_margin=0.0) == ["news"]
    assert idx.select(q, top_k=1, select_margin=0.05) == ["news", "code", "legal"]


def test_select_breaks_ties_in_registry_order():
    idx = _index([("news", _unit(1, 0)), ("academic", _unit(1, 0))])
    order = list(DOMAIN_REGISTRY)
    picked = idx.select(_unit(1, 0), top_k=1, select_margin=0.0)
    assert picked == sorted(["news", "academic"], key=order.index)


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


@pytest.mark.parametrize(
    "records,msg",
    [
        ([{"id": "a", "category": "nope", "text": "x"}], "category"),
        (
            [
                {"id": "a", "category": "news", "text": "x"},
                {"id": "a", "category": "code", "text": "y"},
            ],
            "id",
        ),
        (
            [
                {"id": "a", "category": "news", "text": "Same  text"},
                {"id": "b", "category": "code", "text": "same text"},
            ],
            "text",
        ),
    ],
)
def test_loader_rejects_bad_records(tmp_path, records, msg):
    with pytest.raises(ValueError, match=msg):
        load_category_examples(_write(tmp_path, records), require_all_categories=False)


def test_loader_requires_every_category_by_default(tmp_path):
    with pytest.raises(ValueError, match="no examples"):
        load_category_examples(
            _write(tmp_path, [{"id": "a", "category": "news", "text": "x"}])
        )


def test_build_uses_the_injected_encoder(tmp_path):
    records = [
        {"id": f"{c}-1", "category": c, "text": f"about {c}"} for c in DOMAIN_REGISTRY
    ]
    p = _write(tmp_path, records)
    calls = []

    def fake_encode(texts):
        calls.append(list(texts))
        v = np.eye(len(texts), dtype=np.float32)
        return v

    idx = build_category_index(p, encode=fake_encode)
    assert calls == [[r["text"] for r in records]]
    assert set(idx.scores(np.eye(len(records), dtype=np.float32)[0], top_k=1)) == set(
        DOMAIN_REGISTRY
    )
