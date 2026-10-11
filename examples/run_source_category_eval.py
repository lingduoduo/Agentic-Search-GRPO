"""Compare plain, suffix, and category (site-scoped) search arms.

Answers one question: does restricting a category's web search to its source
sites move results toward authoritative hosts, compared with today's suffix
hint? Source alignment, not answer quality. The decision rule is fixed in
docs/superpowers/specs/2026-10-10-category-source-sites-design.md.

Usage:
    python -m examples.run_source_category_eval --dry-run
    python -m examples.run_source_category_eval --control "what is a bond"
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from examples.run_domain_relevance_eval import (
    DEFAULT_CACHE_DIR,
    MIN_USABLE_SHARE,
    _finite,
    _mean,
)
from src.internal.retrieval.domain_eval import (
    DEFAULT_SOURCE_LABELS_PATH,
    DiskCache,
    SourceLabelledQuery,
    SourceQueryOutcome,
    cache_key,
    load_source_labels,
    run_source_query,
)
from src.internal.tools.search import (
    prepare_domain_query,
    restricted_query,
    search_tool,
)
from src.model.post_training.eval.stats import cliffs_delta, paired_permutation_p

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = REPO_ROOT / "data" / "eval" / "source_category_relevance.json"
ALPHA = 0.05


def _paired(deltas: list[float], interpretable: bool) -> dict:
    if deltas and interpretable:
        p_value = paired_permutation_p(deltas, alternative="two-sided")
        effect = cliffs_delta(deltas, [0.0] * len(deltas))
    else:
        p_value = effect = None
    return {
        "n": len(deltas),
        "mean_delta": _mean(deltas),
        "p_value": p_value,
        "cliffs_delta": effect,
    }


def summarise_source(outcomes: list[SourceQueryOutcome]) -> dict:
    """`web` rows are controls; excluded rows leave the paired comparison."""
    control = [o for o in outcomes if o.domain == "web"]
    data = [o for o in outcomes if o.domain != "web"]
    usable = [o for o in data if not o.excluded]
    interpretable = bool(data) and len(usable) >= MIN_USABLE_SHARE * len(data)
    vs_suffix = _paired([o.delta_vs_suffix for o in usable], interpretable)
    vs_plain = _paired([o.delta_vs_plain for o in usable], interpretable)

    decision = None
    if vs_suffix["p_value"] is not None:
        if vs_suffix["p_value"] < ALPHA and vs_suffix["mean_delta"] < 0:
            decision = "worse"
        elif vs_suffix["p_value"] < ALPHA and vs_suffix["mean_delta"] > 0:
            decision = "better"
        else:
            decision = "null"

    fired = [o for o in data if o.fallback_fired]
    live_fallback_seconds = [
        o.suffix.seconds for o in fired if o.suffix.seconds is not None
    ]
    per_domain: dict[str, dict] = {}
    for o in data:
        b = per_domain.setdefault(
            o.domain, {"deltas": [], "fallback": 0, "n": 0, "excluded": 0, "share": []}
        )
        b["n"] += 1
        b["fallback"] += int(o.fallback_fired)
        b["share"].append(o.on_list_share)
        if o.excluded:
            b["excluded"] += 1
        else:
            b["deltas"].append(o.delta_vs_suffix)
    return {
        "interpretable": interpretable,
        "excluded_count": len(data) - len(usable),
        "paired_vs_suffix": vs_suffix,
        "paired_vs_plain": vs_plain,
        "empty_rate": {
            arm: _mean([float(getattr(o, arm).empty) for o in data])
            for arm in ("plain", "suffix", "category")
        },
        "fallback_rate": _mean([float(o.fallback_fired) for o in data]),
        "fallback_added_seconds": _mean(live_fallback_seconds)
        if live_fallback_seconds
        else None,
        "on_list_share": _mean([o.on_list_share for o in data]),
        "control": {
            "n": len(control),
            "max_abs_delta": max(
                (abs(o.delta_vs_suffix) for o in control), default=0.0
            ),
            "min_jaccard": min((o.jaccard_vs_suffix for o in control), default=1.0),
        },
        # Descriptive only: four queries cannot support a per-category claim.
        "per_domain": {
            name: {
                "mean_delta_vs_suffix": _mean(b["deltas"]),
                "fallback_rate": b["fallback"] / b["n"],
                "on_list_share": _mean(b["share"]),
                "n_usable": len(b["deltas"]),
                "excluded": b["excluded"],
            }
            for name, b in per_domain.items()
        },
        "decision": decision,
    }


def _pending_calls(labels, cache: DiskCache, provider: str) -> int:
    pending: set[str] = set()
    for label in labels:
        queries = {label.query, prepare_domain_query(label.query, label.domain)}
        scoped = restricted_query(label.query, label.domain)
        if scoped is not None:
            queries.add(scoped)
        pending |= {
            q for q in queries if cache.get(cache_key(q, provider=provider)) is None
        }
    return len(pending)


async def _run(labels, cache: DiskCache, provider: str) -> dict:
    outcomes = []
    for index, label in enumerate(labels, start=1):
        outcomes.append(
            await run_source_query(
                label, search_fn=search_tool, cache=cache, provider=provider
            )
        )
        print(f"  [{index:>2}/{len(labels)}] {label.domain:<14} {label.query[:48]}")
    return summarise_source(outcomes)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=str(DEFAULT_SOURCE_LABELS_PATH))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--provider", default="serpapi")
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        help="A query run as a `web` control: all three arms are identical.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many live provider calls the run would make, then exit.",
    )
    args = parser.parse_args(argv)

    labels = load_source_labels(args.labels)
    labels += [SourceLabelledQuery("web", q, {"example.com"}) for q in args.control]
    cache = DiskCache(args.cache_dir)
    print(
        f"{len(labels)} labelled queries; {_pending_calls(labels, cache, args.provider)} live provider calls needed."
    )
    if args.dry_run:
        return
    report = _finite(asyncio.run(_run(labels, cache, args.provider)))
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("paired_vs_suffix", "control", "decision")},
            indent=2,
        )
    )
    print(f"report written to {args.report}")


if __name__ == "__main__":
    main()
