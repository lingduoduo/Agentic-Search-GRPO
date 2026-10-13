"""Category selection metrics, the dev/test split of the labels, and dev tuning."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.internal.tools.search import DOMAIN_REGISTRY
from src.model.pre_training.intents.model import encode_texts

from .model import CategoryIndex, load_category_examples

GRID_TOP_K = (1, 2, 3, 5)
GRID_MARGIN = (0.00, 0.01, 0.02, 0.03, 0.05)
BAR_COVERAGE = 0.80
BAR_MEAN_PICKS = 2.0

LEAKAGE_THRESHOLD = 0.95
# Example/eval pairs above the threshold that were found by the one-time test
# run (data/eval/category_selection.json) and kept. Both are in the test split
# and both are independent paraphrases: the examples were written blind to the
# labels, and were not edited after the run. Excluding the two queries, test
# coverage is 0.717 (n=46) against 0.729 (n=48), so the result does not rest on
# them. Any other pair above the threshold fails the eval gate.
LEAKAGE_ALLOWLIST = frozenset(
    {
        # e5 cosine 0.956, travel
        (
            "best time to visit iceland",
            "best time to visit iceland to see the northern lights",
        ),
        # e5 cosine 0.973, statistics
        ("us unemployment rate by month", "us unemployment rate historical monthly"),
    }
)

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


def find_leakage(
    examples_path: Path | str,
    labels_path: Path | str,
    *,
    encode: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> dict:
    """Example/eval-query pairs whose cosine exceeds ``LEAKAGE_THRESHOLD``."""
    encode = encode or (lambda texts: encode_texts(texts))
    examples = [
        e.text
        for e in load_category_examples(
            Path(examples_path), require_all_categories=False
        )
    ]
    dev, test = split_eval_labels(labels_path)
    queries = [q for _, q in dev + test]
    sims = (
        np.asarray(encode(examples), dtype=np.float32)
        @ np.asarray(encode(queries), dtype=np.float32).T
    )
    flagged, allowed = [], []
    for i, j in zip(*np.nonzero(sims > LEAKAGE_THRESHOLD)):
        pair = {
            "example": examples[i],
            "query": queries[j],
            "cosine": float(sims[i, j]),
        }
        key = (
            " ".join(examples[i].lower().split()),
            " ".join(queries[j].lower().split()),
        )
        (allowed if key in LEAKAGE_ALLOWLIST else flagged).append(pair)
    i, j = np.unravel_index(int(np.argmax(sims)), sims.shape)
    return {
        "max_cosine": float(sims[i, j]),
        "max_pair": {"example": examples[i], "query": queries[j]},
        "allowed": allowed,
        "flagged": flagged,
    }


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


V2_MARKER = "category_selection_v2"
V2_BAR_HIT, V2_BAR_PRECISION, V2_BAR_MEAN_PICKS = 0.85, 0.60, 2.0


@dataclass(frozen=True)
class MultiLabelQuery:
    id: str
    primary: str
    query: str
    acceptable: tuple[str, ...]


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def load_multilabel_queries(path: Path | str) -> list[MultiLabelQuery]:
    payload = json.loads(Path(path).read_text())
    if payload.get("taxonomy") != V2_MARKER:
        raise ValueError(f"{path}: taxonomy is not {V2_MARKER}")
    out: list[MultiLabelQuery] = []
    seen: set[str] = set()
    seen_ids: set[str] = set()
    for rec in payload["queries"]:
        qid, primary = rec["id"], rec["primary"]
        if qid in seen_ids:
            raise ValueError(f"duplicate id {qid!r}")
        seen_ids.add(qid)
        acceptable = tuple(rec["acceptable"])
        if primary not in DOMAIN_REGISTRY:
            raise ValueError(f"{qid}: primary {primary!r} is not a category")
        if primary not in acceptable:
            raise ValueError(f"{qid}: primary {primary!r} is not in acceptable")
        if not 1 <= len(acceptable) <= 3 or any(
            c not in DOMAIN_REGISTRY for c in acceptable
        ):
            raise ValueError(f"{qid}: acceptable must be 1-3 known categories")
        key = _normalize(rec["query"])
        if key in seen:
            raise ValueError(f"{qid}: duplicate query {rec['query']!r}")
        seen.add(key)
        out.append(MultiLabelQuery(qid, primary, rec["query"], acceptable))
    return out


def _evaluate_vectors(
    index: CategoryIndex,
    queries: Sequence[MultiLabelQuery],
    vectors: np.ndarray,
    top_k: int,
    select_margin: float,
) -> dict:
    hits = top1 = picks = good = 0
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confusion: dict[str, Counter] = defaultdict(Counter)
    for q, vector in zip(queries, vectors):
        chosen = index.select(vector, top_k=top_k, select_margin=select_margin)
        ok = [c for c in chosen if c in q.acceptable]
        hit = bool(ok)
        hits += hit
        top1 += chosen[0] in q.acceptable
        picks += len(chosen)
        good += len(ok)
        per[q.primary][0] += 1
        per[q.primary][1] += hit
        for picked in chosen:
            if picked not in q.acceptable:
                confusion[q.primary][picked] += 1
    n = len(queries)
    return {
        "n": n,
        "hit": hits / n if n else 0.0,
        "precision": good / picks if picks else 0.0,
        "mean_picks": picks / n if n else 0.0,
        "top1": top1 / n if n else 0.0,
        "per_primary": {c: {"n": a, "hit": b / a} for c, (a, b) in sorted(per.items())},
        "confusion": {c: dict(v) for c, v in sorted(confusion.items())},
    }


def evaluate_multilabel(
    index: CategoryIndex,
    queries: Sequence[MultiLabelQuery],
    *,
    top_k: int,
    select_margin: float,
    encode: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> dict:
    encode = encode or (lambda texts: encode_texts(texts))
    vectors = np.asarray(encode([q.query for q in queries]), dtype=np.float32)
    return _evaluate_vectors(index, queries, vectors, top_k, select_margin)


def _feasible(row: dict) -> bool:
    return (
        row["precision"] >= V2_BAR_PRECISION and row["mean_picks"] <= V2_BAR_MEAN_PICKS
    )


def _choose_tuning(grid: Sequence[dict]) -> tuple[dict, bool]:
    feasible = [r for r in grid if _feasible(r)]
    if feasible:
        best = min(
            feasible,
            key=lambda r: (
                -r["hit"],
                -r["precision"],
                r["mean_picks"],
                r["top_k"],
                r["select_margin"],
            ),
        )
        return best, True
    best = min(
        grid,
        key=lambda r: (-r["hit"], r["mean_picks"], r["top_k"], r["select_margin"]),
    )
    return best, False


def tune_multilabel(
    index: CategoryIndex,
    queries: Sequence[MultiLabelQuery],
    *,
    encode: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> dict:
    encode = encode or (lambda texts: encode_texts(texts))
    vectors = np.asarray(encode([q.query for q in queries]), dtype=np.float32)
    grid = []
    for top_k in GRID_TOP_K:
        for margin in GRID_MARGIN:
            r = _evaluate_vectors(index, queries, vectors, top_k, margin)
            grid.append(
                {
                    "top_k": top_k,
                    "select_margin": margin,
                    "hit": r["hit"],
                    "precision": r["precision"],
                    "mean_picks": r["mean_picks"],
                }
            )
    best, feasible = _choose_tuning(grid)
    return {
        "top_k": best["top_k"],
        "select_margin": best["select_margin"],
        "feasible": feasible,
        "grid": grid,
    }


def find_leakage_multilabel(
    examples_path: Path | str,
    v2_path: Path | str,
    *,
    dev_path: Path | str | None = None,
    encode: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> dict:
    """Examples (and dev queries) against v2 test queries; no allowlist."""
    encode = encode or (lambda texts: encode_texts(texts))
    examples = [
        e.text
        for e in load_category_examples(
            Path(examples_path), require_all_categories=False
        )
    ]
    right = [q.query for q in load_multilabel_queries(v2_path)]
    left = list(examples)
    kinds = ["example"] * len(left)
    if dev_path is not None:
        dev = [q.query for q in load_multilabel_queries(dev_path)]
        left += dev
        kinds += ["dev"] * len(dev)
    sims = (
        np.asarray(encode(left), dtype=np.float32)
        @ np.asarray(encode(right), dtype=np.float32).T
    )
    flagged = [
        {
            "source": left[i],
            "kind": kinds[i],
            "query": right[j],
            "cosine": float(sims[i, j]),
        }
        for i in range(len(left))
        for j in range(len(right))
        if sims[i, j] > LEAKAGE_THRESHOLD or _normalize(left[i]) == _normalize(right[j])
    ]
    i, j = np.unravel_index(int(np.argmax(sims)), sims.shape)
    return {
        "max_cosine": float(sims[i, j]),
        "max_pair": {"source": left[i], "kind": kinds[i], "query": right[j]},
        "allowed": [],
        "flagged": flagged,
    }
