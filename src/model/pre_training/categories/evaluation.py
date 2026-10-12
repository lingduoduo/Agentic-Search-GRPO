"""Category selection metrics, the dev/test split of the labels, and dev tuning."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.model.pre_training.intents.model import encode_texts

from .model import CategoryIndex

GRID_TOP_K = (1, 2, 3, 5)
GRID_MARGIN = (0.00, 0.01, 0.02, 0.03, 0.05)
BAR_COVERAGE = 0.80
BAR_MEAN_PICKS = 2.0

Labels = list[tuple[str, str]]


def split_eval_labels(path: Path | str) -> tuple[Labels, Labels]:
    """First query of each category -> dev; the rest -> test."""
    payload = json.loads(Path(path).read_text())
    if payload.get("taxonomy") != "source_categories":
        raise ValueError(f"{path}: not a source_categories label file")
    dev: Labels = []
    test: Labels = []
    for category, entries in payload["domains"].items():
        for i, entry in enumerate(entries):
            (dev if i == 0 else test).append((category, entry["query"]))
    return dev, test


def evaluate_selection(
    index: CategoryIndex,
    labels: Labels,
    *,
    top_k: int,
    select_margin: float,
    encode: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> dict:
    encode = encode or (lambda texts: encode_texts(texts))
    vectors = np.asarray(encode([q for _, q in labels]), dtype=np.float32)
    hits = top1 = picks = 0
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confusion: dict[str, Counter] = defaultdict(Counter)
    for (category, _), vector in zip(labels, vectors):
        chosen = index.select(vector, top_k=top_k, select_margin=select_margin)
        hit = category in chosen
        hits += hit
        top1 += chosen[0] == category
        picks += len(chosen)
        per[category][0] += 1
        per[category][1] += hit
        for picked in chosen:
            if picked != category:
                confusion[category][picked] += 1
    n = len(labels)
    return {
        "n": n,
        "coverage": hits / n if n else 0.0,
        "mean_picks": picks / n if n else 0.0,
        "top1": top1 / n if n else 0.0,
        "per_category": {
            c: {"n": a, "coverage": b / a} for c, (a, b) in sorted(per.items())
        },
        "confusion": {c: dict(v) for c, v in sorted(confusion.items())},
    }


def tune(
    index: CategoryIndex,
    dev: Labels,
    *,
    encode: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> dict:
    encode = encode or (lambda texts: encode_texts(texts))
    rows = []
    for top_k in GRID_TOP_K:
        for margin in GRID_MARGIN:
            r = evaluate_selection(
                index, dev, top_k=top_k, select_margin=margin, encode=encode
            )
            rows.append(
                {
                    "top_k": top_k,
                    "select_margin": margin,
                    "coverage": r["coverage"],
                    "mean_picks": r["mean_picks"],
                    "top1": r["top1"],
                }
            )
    best = min(
        rows,
        key=lambda r: (-r["coverage"], r["mean_picks"], r["top_k"], r["select_margin"]),
    )
    return {
        "top_k": best["top_k"],
        "select_margin": best["select_margin"],
        "grid": rows,
    }
