"""Build, tune and evaluate the category index."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .evaluation import (
    BAR_COVERAGE,
    BAR_MEAN_PICKS,
    V2_BAR_HIT,
    V2_BAR_MEAN_PICKS,
    V2_BAR_PRECISION,
    evaluate_multilabel,
    evaluate_selection,
    find_leakage,
    find_leakage_multilabel,
    load_multilabel_queries,
    split_eval_labels,
    tune,
    tune_multilabel,
)
from .model import (
    DEFAULT_SELECT_MARGIN,
    DEFAULT_TOP_K,
    INDEX_FILENAME,
    CategoryIndex,
    build_category_index,
)


# The pre-registered v2 test set; tuning must never see it.
V2_TEST_LABELS = Path(__file__).resolve().parents[4] / (
    "data/eval/category_selection_test_v2.json"
)
TEST_ID_PREFIX = "t-"


def _provenance(labels: Path) -> dict:
    return {
        "labels_path": str(labels),
        "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="categories")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="encode the examples into an index")
    b.add_argument("--examples", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)

    lk = sub.add_parser("leakage", help="fail on an example too close to an eval query")
    lk.add_argument("--examples", type=Path, required=True)
    lk.add_argument("--labels", type=Path)
    lk.add_argument("--against", type=Path, help="v2 file; no allowlist")
    lk.add_argument("--dev", type=Path, help="v2 dev file, checked with --against")

    for name in ("tune", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--index", type=Path, required=True)
        p.add_argument("--labels", type=Path, required=True)
        p.add_argument("--report", type=Path, required=True)
        if name == "evaluate":
            p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
            p.add_argument("--select-margin", type=float, default=DEFAULT_SELECT_MARGIN)

    for name in ("tune-v2", "evaluate-v2"):
        p = sub.add_parser(name)
        p.add_argument("--index", type=Path, required=True)
        p.add_argument("--labels", type=Path, required=True)
        p.add_argument("--report", type=Path, required=True)
        if name == "evaluate-v2":
            p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
            p.add_argument("--select-margin", type=float, default=DEFAULT_SELECT_MARGIN)

    args = parser.parse_args(argv)

    if args.command == "build":
        index = build_category_index(args.examples)
        index.save(args.output / INDEX_FILENAME)
        print(f"wrote {args.output / INDEX_FILENAME}")
        return 0

    if args.command == "leakage":
        if args.against is not None:
            result = find_leakage_multilabel(
                args.examples, args.against, dev_path=args.dev
            )
        elif args.labels is None:
            parser.error("leakage needs --labels or --against")
        else:
            result = find_leakage(args.examples, args.labels)
        print(f"max cosine {result['max_cosine']:.3f}: {result['max_pair']}")
        for pair in result["allowed"]:
            print(f"allowlisted: {pair}")
        for pair in result["flagged"]:
            print(f"LEAKAGE: {pair}")
        return 1 if result["flagged"] else 0

    if args.command == "tune-v2":
        queries = load_multilabel_queries(args.labels)
        if args.labels.resolve() == V2_TEST_LABELS or any(
            q.id.startswith(TEST_ID_PREFIX) for q in queries
        ):
            parser.error(
                f"tune-v2 refuses {args.labels}: it is the v2 test set (or holds "
                f"{TEST_ID_PREFIX!r} test ids); tune on the dev file"
            )
        index = CategoryIndex.load(args.index / INDEX_FILENAME)
        result = tune_multilabel(index, queries)
        result["split"] = "dev_v2"
        result.update(_provenance(args.labels))
        _write(args.report, result)
        print(
            f"top_k={result['top_k']} select_margin={result['select_margin']} "
            f"feasible={result['feasible']}"
        )
        return 0

    index = CategoryIndex.load(args.index / INDEX_FILENAME)
    if args.command == "evaluate-v2":
        result = evaluate_multilabel(
            index,
            load_multilabel_queries(args.labels),
            top_k=args.top_k,
            select_margin=args.select_margin,
        )
        result.update(
            {
                "split": "test_v2",
                "top_k": args.top_k,
                "select_margin": args.select_margin,
                "bar": {
                    "hit": V2_BAR_HIT,
                    "precision": V2_BAR_PRECISION,
                    "mean_picks": V2_BAR_MEAN_PICKS,
                },
                "passed": result["hit"] >= V2_BAR_HIT
                and result["precision"] >= V2_BAR_PRECISION
                and result["mean_picks"] <= V2_BAR_MEAN_PICKS,
                **_provenance(args.labels),
            }
        )
        _write(args.report, result)
        print(
            f"hit={result['hit']:.3f} precision={result['precision']:.3f} "
            f"mean_picks={result['mean_picks']:.2f}"
        )
        return 0

    dev, test = split_eval_labels(args.labels)
    if args.command == "tune":
        result = tune(index, dev)
        result["split"] = "dev"
        result.update(_provenance(args.labels))
        _write(args.report, result)
        print(f"top_k={result['top_k']} select_margin={result['select_margin']}")
        return 0

    result = evaluate_selection(
        index, test, top_k=args.top_k, select_margin=args.select_margin
    )
    result.update(
        {
            "split": "test",
            "top_k": args.top_k,
            "select_margin": args.select_margin,
            "bar": {"coverage": BAR_COVERAGE, "mean_picks": BAR_MEAN_PICKS},
            "passed": result["coverage"] >= BAR_COVERAGE
            and result["mean_picks"] <= BAR_MEAN_PICKS,
            **_provenance(args.labels),
        }
    )
    _write(args.report, result)
    print(f"coverage={result['coverage']:.3f} mean_picks={result['mean_picks']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
