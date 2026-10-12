"""Compare plain, suffix, and category (site-scoped) search arms.

Answers one question: does restricting a category's web search to its source
sites move results toward authoritative hosts, compared with today's suffix
hint? Source alignment, not answer quality. The decision rule is fixed in
docs/superpowers/specs/2026-10-10-category-source-sites-design.md.

Usage:
    python -m examples.run_source_category_eval --dry-run
    python -m examples.run_source_category_eval --control "what is a bond"
    # pilot: first label per category, no plain arm (33 calls with one control)
    python -m examples.run_source_category_eval --per-category 1 --no-plain --control "what is a bond"
    # confirmatory: the 48 labels the pilot did not use
    python -m examples.run_source_category_eval --skip-per-category 1 --no-plain --control "what is a bond"
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from examples.run_domain_relevance_eval import (
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
# Own cache, so all three arms are fetched in one time window.
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "eval" / "source_cache"
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
    # None when the plain arm was skipped (--no-plain): reported as not run.
    plain_deltas = [o.delta_vs_plain for o in usable if o.delta_vs_plain is not None]
    ran_plain = any(o.plain is not None for o in data)
    vs_plain = _paired(plain_deltas, interpretable) if ran_plain else None

    control_stats = {
        "n": len(control),
        "max_abs_delta": max((abs(o.delta_vs_suffix) for o in control), default=0.0),
        "min_jaccard": min((o.jaccard_vs_suffix for o in control), default=1.0),
    }
    # An absent, failed or disagreeing control withholds the decision; a
    # failed control scores delta 0 and Jaccard 1, so it must be ruled out.
    control_ok = (
        control_stats["n"] >= 1
        and all(not o.excluded for o in control)
        and control_stats["max_abs_delta"] == 0
        and control_stats["min_jaccard"] == 1.0
    )
    decision = None
    if control_ok and vs_suffix["p_value"] is not None:
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
            "plain": _mean([float(o.plain.empty) for o in data if o.plain is not None])
            if ran_plain
            else None,
            "suffix": _mean([float(o.suffix.empty) for o in data]),
            "category": _mean([float(o.category.empty) for o in data]),
        },
        "fallback_rate": _mean([float(o.fallback_fired) for o in data]),
        "fallback_added_seconds": _mean(live_fallback_seconds)
        if live_fallback_seconds
        else None,
        "fallback_added_seconds_n": len(live_fallback_seconds),
        "on_list_share": _mean([o.on_list_share for o in data]),
        "control": control_stats,
        "control_ok": control_ok,
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


def select_labels(labels, per_category: int | None, skip: int = 0):
    """Labels of each category in file order: drop the first ``skip``, then
    keep at most ``per_category`` (None keeps the rest).

    Mechanical on purpose: a subset chosen before any result exists cannot be
    cherry-picked. ``skip=1`` holds out the one-per-category pilot, so a
    confirmatory run never re-scores queries already analysed.
    """
    seen: dict[str, int] = {}
    kept: dict[str, int] = {}
    picked = []
    for label in labels:
        seen[label.domain] = seen.get(label.domain, 0) + 1
        if seen[label.domain] <= skip:
            continue
        if per_category is not None and kept.get(label.domain, 0) >= per_category:
            continue
        kept[label.domain] = kept.get(label.domain, 0) + 1
        picked.append(label)
    return picked


def _pending_calls(
    labels, cache: DiskCache, provider: str, include_plain: bool = True
) -> int:
    pending: set[str] = set()
    for label in labels:
        queries = {prepare_domain_query(label.query, label.domain)}
        if include_plain:
            queries.add(label.query)
        scoped = restricted_query(label.query, label.domain)
        if scoped is not None:
            queries.add(scoped)
        pending |= {
            q for q in queries if cache.get(cache_key(q, provider=provider)) is None
        }
    return len(pending)


async def _run(
    labels, cache: DiskCache, provider: str, include_plain: bool = True
) -> dict:
    outcomes = []
    for index, label in enumerate(labels, start=1):
        outcomes.append(
            await run_source_query(
                label,
                search_fn=search_tool,
                cache=cache,
                provider=provider,
                include_plain=include_plain,
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
        "--per-category",
        type=int,
        default=None,
        help="Pilot: keep only the first N labels of each category, in file order.",
    )
    parser.add_argument(
        "--skip-per-category",
        type=int,
        default=0,
        help="Drop the first K labels of each category first (1 holds out the pilot).",
    )
    parser.add_argument(
        "--no-plain",
        action="store_true",
        help="Skip the plain arm; the pre-registered rule compares category with suffix only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many live provider calls the run would make, then exit.",
    )
    args = parser.parse_args(argv)

    if args.per_category is not None and args.per_category < 1:
        parser.error("--per-category must be at least 1")
    if args.skip_per_category < 0:
        parser.error("--skip-per-category must not be negative")
    include_plain = not args.no_plain
    labels = select_labels(
        load_source_labels(args.labels), args.per_category, args.skip_per_category
    )
    labels += [SourceLabelledQuery("web", q, {"example.com"}) for q in args.control]
    cache = DiskCache(args.cache_dir)
    print(
        f"{len(labels)} labelled queries; {_pending_calls(labels, cache, args.provider, include_plain)} live provider calls needed."
    )
    if args.dry_run:
        return
    report = _finite(asyncio.run(_run(labels, cache, args.provider, include_plain)))
    # Record the sample that produced the numbers, so a pilot is never read
    # as the full pre-registered run.
    report["sample"] = {
        "per_category": args.per_category,
        "skip_per_category": args.skip_per_category,
        "plain_arm": include_plain,
        "n_labels": len(labels) - len(args.control),
        "provider": args.provider,
    }
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
