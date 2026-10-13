"""Build, tune and evaluate the category index."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .evaluation import (
    BAR_COVERAGE,
    BAR_MEAN_PICKS,
    evaluate_selection,
    find_leakage,
    split_eval_labels,
    tune,
)
from .model import (
    DEFAULT_SELECT_MARGIN,
    DEFAULT_TOP_K,
    INDEX_FILENAME,
    CategoryIndex,
    build_category_index,
)


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
    lk.add_argument("--labels", type=Path, required=True)

    for name in ("tune", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--index", type=Path, required=True)
        p.add_argument("--labels", type=Path, required=True)
        p.add_argument("--report", type=Path, required=True)
        if name == "evaluate":
            p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
            p.add_argument("--select-margin", type=float, default=DEFAULT_SELECT_MARGIN)

    args = parser.parse_args(argv)

    if args.command == "build":
        index = build_category_index(args.examples)
        index.save(args.output / INDEX_FILENAME)
        print(f"wrote {args.output / INDEX_FILENAME}")
        return 0

    if args.command == "leakage":
        result = find_leakage(args.examples, args.labels)
        print(f"max cosine {result['max_cosine']:.3f}: {result['max_pair']}")
        for pair in result["allowed"]:
            print(f"allowlisted: {pair}")
        for pair in result["flagged"]:
            print(f"LEAKAGE: {pair}")
        return 1 if result["flagged"] else 0

    index = CategoryIndex.load(args.index / INDEX_FILENAME)
    dev, test = split_eval_labels(args.labels)
    if args.command == "tune":
        result = tune(index, dev)
        result["split"] = "dev"
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
        }
    )
    _write(args.report, result)
    print(f"coverage={result['coverage']:.3f} mean_picks={result['mean_picks']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
