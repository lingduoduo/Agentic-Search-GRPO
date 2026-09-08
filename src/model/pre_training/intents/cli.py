"""Build and evaluate the canonical routing index.

Three commands. ``seed`` proposes module labels for the existing training
examples so the canonical set can be curated rather than authored from nothing.
``build`` encodes the canonical set into an index. ``evaluate`` scores an index
against the held-out query sets.

This module is only an entry point: the build lives in ``data``, the scoring in
``evaluation``. The seed cues below stay here because ``seed`` is the one
command with no other home — it is a curation aid, not part of the model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from .data import build_index, load_intent_examples
from .evaluation import run_index_evaluation
from .model import DEFAULT_ENCODER, modules_for_route

# ---------------------------------------------------------------------------
# Seed — propose module labels for existing training examples.
#
# Curating ~270 canonical examples is a review pass over machine proposals, not
# 270 acts of authorship. The cues below are the same ones
# ``src/internal/servers/web/intent/rules.py`` already routes on, which is why
# the taxonomy has these fourteen modules and no others — so a proposal agrees
# with the router by construction, and a disagreement is worth looking at.
#
# Every proposal is a draft. The committed canonical file is the reviewed result.
# ---------------------------------------------------------------------------
_CUES: dict[str, tuple[re.Pattern[str], ...]] = {
    "current_info": (
        re.compile(
            r"\b(latest|current|recent|news|price|stock|weather|today|now|"
            r"this week|right now)\b",
            re.IGNORECASE,
        ),
    ),
    "lookup_document": (
        re.compile(
            r"\b(doc|document|report|runbook|postmortem|checklist|spec|readme|"
            r"guide|notes|deck|spreadsheet|page|wiki|policy)\b",
            re.IGNORECASE,
        ),
    ),
    "lookup_fact": (
        re.compile(
            r"\b(which|who|when|where|how many|how much|what is the|value of|"
            r"number|setting|config|version)\b",
            re.IGNORECASE,
        ),
    ),
    "summarize": (re.compile(r"\b(summari[sz]e|tl;?dr|recap|overview of)\b", re.I),),
    "compare": (
        re.compile(r"\b(compare|versus|vs\.?|difference between|better than)\b", re.I),
    ),
    "generate": (
        re.compile(
            r"\b(write|draft|translate|rephrase|reword|rewrite|brainstorm|"
            r"compose|generate)\b",
            re.IGNORECASE,
        ),
    ),
    "converse": (
        re.compile(r"\b(hello|hi there|thanks|thank you|joke|poem|haiku)\b", re.I),
    ),
    "explain": (
        re.compile(
            r"\b(explain|why|how does|how do|what is|describe|tell me about)\b", re.I
        ),
    ),
    "create": (re.compile(r"\b(create|open|file|add|new)\b", re.IGNORECASE),),
    "send": (re.compile(r"\b(send|email|notify|post|message|share)\b", re.IGNORECASE),),
    "schedule": (
        re.compile(r"\b(schedule|book|remind|calendar|meeting|invite)\b", re.I),
    ),
    "modify": (
        re.compile(r"\b(update|change|delete|remove|cancel|close|rename|edit)\b", re.I),
    ),
    "execute": (
        re.compile(r"\b(run|execute|deploy|trigger|invoke|rerun|kick off)\b", re.I),
    ),
}

# Used when no cue fires, so every example still gets a valid starting label.
_DEFAULT_MODULE = {
    "search": "lookup_fact",
    "chat": "explain",
    "tool": "execute",
}


def propose_modules(text: str, route: str) -> tuple[str, ...]:
    """Propose one or more modules for *text* within *route*.

    Multi-label by design: "compare the current prices of BTC and ETH" is
    genuinely both a comparison and a request for current information.
    """
    candidates = tuple(
        module
        for module in modules_for_route(route)
        if module in _CUES and any(cue.search(text) for cue in _CUES[module])
    )
    return candidates or (_DEFAULT_MODULE[route],)


def write_seed_canonical(examples_path: Path, output_path: Path) -> int:
    """Write a proposed canonical file from labeled training examples."""
    examples = load_intent_examples(examples_path)
    records = [
        {
            "id": f"canon-{position:03d}",
            "text": example.text,
            "route": example.label,
            "modules": list(propose_modules(example.text, example.label)),
        }
        for position, example in enumerate(examples, start=1)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return len(records)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and evaluate the canonical routing index. `seed` proposes "
            "module labels for existing training examples, `build` encodes the "
            "canonical set into an index, and `evaluate` scores an index "
            "against the held-out query sets."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser(
        "seed", help="propose module labels for existing training examples"
    )
    seed.add_argument("--examples", required=True, type=Path)
    seed.add_argument("--output", required=True, type=Path)

    build = subparsers.add_parser("build", help="encode the canonical set")
    build.add_argument("--canonical", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--model", default=DEFAULT_ENCODER)

    evaluate = subparsers.add_parser("evaluate", help="score an index")
    evaluate.add_argument("--index", required=True, type=Path)
    evaluate.add_argument("--eval-queries", required=True, type=Path)
    evaluate.add_argument("--hard-queries", type=Path)
    evaluate.add_argument("--out-of-scope", type=Path)
    evaluate.add_argument("--canonical", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a seed, build, or evaluate command."""
    try:
        args = _build_parser().parse_args(argv)
        if args.command == "seed":
            count = write_seed_canonical(args.examples, args.output)
            print(f"wrote {count} proposed canonical examples to {args.output}")
            return 0
        if args.command == "build":
            index = build_index(args.canonical, args.output, model_name=args.model)
            print(f"built index of {index.size} examples at {args.output}")
            low = index.low_support_modules()
            print(f"low support modules ({len(low)}): {', '.join(low) or 'none'}")
            return 0

        report = run_index_evaluation(
            index_path=args.index,
            eval_queries_path=args.eval_queries,
            hard_queries_path=args.hard_queries,
            out_of_scope_path=args.out_of_scope,
            canonical_path=args.canonical,
            output_path=args.output,
        )
        print(json.dumps(report["headline"], indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"intent index command failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
