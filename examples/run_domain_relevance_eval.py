"""Compare general and domain search arms on labelled queries.

Answers one question: does selecting a topic domain move results toward that
topic's authoritative sources? It measures source alignment, not answer
quality -- an arxiv.org result is not automatically a better answer than a
well-written blog post.

The statistics live here rather than in src/internal/retrieval/domain_eval.py
because importing src.model.post_training pulls torch in through package
__init__ side effects, and the measurement core must stay torch-free.

Usage:
    python -m examples.run_domain_relevance_eval --dry-run
    python -m examples.run_domain_relevance_eval
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

from src.internal.retrieval.domain_eval import (
    DEFAULT_LABELS_PATH,
    DiskCache,
    QueryOutcome,
    cache_key,
    load_labels,
    run_query,
)
from src.internal.tools.search import prepare_domain_query, search_tool
from src.model.post_training.eval.stats import cliffs_delta, paired_permutation_p

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "eval" / "cache"
DEFAULT_REPORT_PATH = REPO_ROOT / "data" / "eval" / "domain_relevance.json"

# Below this share of usable queries the remainder is not worth reporting.
MIN_USABLE_SHARE = 2 / 3


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(outcomes: list[QueryOutcome]) -> dict:
    """Reduce per-query outcomes to the reported statistics.

    `general` is a control, not data: its hint is empty so its deltas are
    structurally zero, and including them would shrink the mean difference
    toward the null. Excluded queries leave for the same reason -- a delta
    against a failed arm measures the failure, not the domain.
    """
    control = [o for o in outcomes if o.domain == "general"]
    topic = [o for o in outcomes if o.domain != "general"]
    usable = [o for o in topic if not o.excluded]
    deltas = [o.delta for o in usable]

    interpretable = bool(topic) and len(usable) >= MIN_USABLE_SHARE * len(topic)

    if deltas and interpretable:
        p_value = paired_permutation_p(deltas, alternative="two-sided")
        effect = cliffs_delta(deltas, [0.0] * len(deltas))
    else:
        p_value = None
        effect = None

    buckets: dict[str, dict] = {}
    for outcome in topic:
        bucket = buckets.setdefault(
            outcome.domain, {"deltas": [], "jaccards": [], "excluded": 0}
        )
        if outcome.excluded:
            bucket["excluded"] += 1
        else:
            bucket["deltas"].append(outcome.delta)
            bucket["jaccards"].append(outcome.jaccard)
    # Descriptive only: three queries cannot support a per-domain claim, so no
    # p-value is computed here at any point.
    per_domain = {
        name: {
            "mean_delta": _mean(b["deltas"]),
            "mean_jaccard": _mean(b["jaccards"]),
            "n_usable": len(b["deltas"]),
            "excluded": b["excluded"],
        }
        for name, b in buckets.items()
    }

    return {
        "interpretable": interpretable,
        "excluded_count": sum(1 for o in topic if o.excluded),
        "empty_rate_general": _mean([float(o.general.empty) for o in topic]),
        "empty_rate_domain": _mean([float(o.domain_arm.empty) for o in topic]),
        "paired": {
            "n": len(deltas),
            "mean_delta": _mean(deltas),
            "p_value": p_value,
            "cliffs_delta": effect,
        },
        "control": {
            "n": len(control),
            "max_abs_delta": max((abs(o.delta) for o in control), default=0.0),
            "min_jaccard": min((o.jaccard for o in control), default=1.0),
        },
        "per_domain": per_domain,
    }


def _finite(node):
    """Replace non-finite floats with None before the report is written."""
    if isinstance(node, dict):
        return {k: _finite(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_finite(v) for v in node]
    if isinstance(node, float) and not math.isfinite(node):
        return None
    return node


def _pending_calls(labels, cache: DiskCache, provider: str) -> int:
    """How many live provider calls a run would still need."""
    pending: set[str] = set()
    for label in labels:
        for query in {label.query, prepare_domain_query(label.query, label.domain)}:
            if cache.get(cache_key(query, provider=provider)) is None:
                pending.add(query)
    return len(pending)


async def _run(labels, cache: DiskCache, provider: str) -> dict:
    outcomes = []
    for index, label in enumerate(labels, start=1):
        outcomes.append(
            await run_query(
                label, search_fn=search_tool, cache=cache, provider=provider
            )
        )
        print(f"  [{index:>2}/{len(labels)}] {label.domain:<14} {label.query[:48]}")
    return summarise(outcomes)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--provider", default="serpapi")
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        help=(
            "A query to run as a `general` control. Its hint is empty, so both "
            "arms issue the same query and the second is a cache hit."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many live provider calls the run would make, then exit.",
    )
    args = parser.parse_args(argv)

    from src.internal.retrieval.domain_eval import LabelledQuery

    labels = load_labels(args.labels)
    labels += [LabelledQuery("general", q, {"example.com"}) for q in args.control]
    cache = DiskCache(args.cache_dir)

    pending = _pending_calls(labels, cache, args.provider)
    print(f"{len(labels)} labelled queries; {pending} live provider calls needed.")
    if args.dry_run:
        return

    summary = asyncio.run(_run(labels, cache, args.provider))
    report = _finite(summary)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("paired", "control")}, indent=2))
    print(f"report written to {args.report}")


if __name__ == "__main__":
    main()
