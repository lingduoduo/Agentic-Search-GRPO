"""Simulate multi-day corpus churn: incremental hierarchy update vs full rebuild.

Holds out documents from a JSONL corpus, builds a cluster-mode hierarchy from the
rest, then for each day adds held-out documents and deletes random live ones.
Each day runs both an incremental ``update`` (against the previous incremental
tree) and a full ``build`` of the same snapshot, through call-counting adapter
wrappers, and records per-day and cumulative embedding and description-model
call counts for both. With --decision-model and --queries it also runs the
description evaluator on the final incremental and rebuilt trees; queries whose
gold passages are absent from a tree are reported as skipped.

This is a manual offline measurement with real local models. It reports counts
and evaluator output only; there is no pass threshold.

Run:

    python -m examples.simulate_hierarchy_churn --corpus-path corpus.jsonl \
        --days 5 --daily-add 20 --daily-delete 10 \
        --embedding-model intfloat/multilingual-e5-base \
        --description-model /models/qwen --output churn.json --seed 0
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

COST_KEYS = ("embedded_documents", "description_model_calls")


class _CountingEmbedder:
    """Delegates to an embedder and counts every text it encodes."""

    def __init__(self, inner):
        self._inner = inner
        self.embedded_documents = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def encode(self, texts):
        self.embedded_documents += len(texts)
        return self._inner.encode(texts)


class _CountingDescriber:
    """Delegates to a describer and counts every description model call."""

    def __init__(self, inner):
        self._inner = inner
        self.description_model_calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def describe(self, label, documents):
        self.description_model_calls += 1
        return self._inner.describe(label, documents)

    def describe_with_provenance(self, label, documents, **kwargs):
        self.description_model_calls += 1
        return self._inner.describe_with_provenance(label, documents, **kwargs)


def _counted(adapters):
    embedder, describer = adapters
    return (
        _CountingEmbedder(embedder),
        _CountingDescriber(describer) if describer is not None else None,
    )


def _counts(embedder, describer, before):
    now = dict(
        embedded_documents=embedder.embedded_documents,
        description_model_calls=describer.description_model_calls if describer else 0,
    )
    return {k: now[k] - before[k] for k in COST_KEYS}, now


def simulate(
    initial,
    days,
    config,
    *,
    incremental,
    rebuild,
    max_description_calls=None,
):
    """Run day 0 plus each ``(added_documents, deleted_ids)`` day both ways.

    ``incremental`` and ``rebuild`` are ``(embedder, describer)`` pairs; each is
    used only for its own track so their call counts stay separate.
    """
    from src.internal.document_index.hierarchy import build_hierarchy

    tracks = dict(incremental=_counted(incremental), rebuild=_counted(rebuild))
    totals = {name: dict.fromkeys(COST_KEYS, 0) for name in tracks}
    snapshot = list(initial)
    previous = rebuilt = None
    records = []
    day0_groups = None
    for day in range(len(days) + 1):
        added, deleted = ([], []) if day == 0 else days[day - 1]
        if day:
            gone = set(deleted)
            snapshot = [d for d in snapshot if d["id"] not in gone] + list(added)
        record = dict(
            day=day, corpus_size=len(snapshot), added=len(added), deleted=len(deleted)
        )
        for name, (embedder, describer) in tracks.items():
            before = totals[name]
            update = name == "incremental" and previous is not None
            artifact = build_hierarchy(
                snapshot,
                config,
                embedder=embedder,
                describer=describer,
                previous=previous if update else None,
                **(dict(max_description_calls=max_description_calls) if update else {}),
            )
            spent, totals[name] = _counts(embedder, describer, before)
            diagnostics = artifact.diagnostics
            record[name] = dict(
                **spent,
                groups=len(artifact.groups),
                structure_operations=diagnostics.get("structure_operations", []),
                update_cost=diagnostics.get("update_cost"),
                rebuild_recommended=diagnostics.get("rebuild_recommended"),
                poor_fit_documents=diagnostics.get("poor_fit_documents"),
            )
            if name == "incremental":
                previous = artifact
            else:
                rebuilt = artifact
        if day == 0:
            day0_groups = copy.deepcopy(previous.groups)
        records.append(record)
    return dict(
        days=records,
        cumulative=totals,
        day0_groups=day0_groups,
        incremental_artifact=previous,
        rebuilt_artifact=rebuilt,
    )


def plan_days(documents, days, daily_add, daily_delete, seed):
    """Hold out ``days * daily_add`` documents and draw a seeded churn schedule."""
    if days * daily_add >= len(documents):
        raise ValueError("held-out documents must leave a nonempty initial corpus")
    rng = random.Random(seed)
    shuffled = list(documents)
    rng.shuffle(shuffled)
    held = days * daily_add
    held_out, initial = shuffled[:held], shuffled[held:]
    alive = [d["id"] for d in initial]
    schedule = []
    for day in range(days):
        added = held_out[day * daily_add : (day + 1) * daily_add]
        deleted = rng.sample(alive, min(daily_delete, len(alive) - 1))
        gone = set(deleted)
        alive = [i for i in alive if i not in gone] + [d["id"] for d in added]
        schedule.append((added, deleted))
    return initial, schedule


def evaluate_tree(artifact, model, cases, seed):
    """Evaluate the stored descriptions on cases whose gold is in the tree."""
    from src.internal.document_index.decision_retriever import (
        DecisionRetriever,
        DecisionRetrieverConfig,
    )
    from src.internal.document_index.description_eval import evaluate_descriptions

    config = DecisionRetrieverConfig()
    paths = DecisionRetriever(
        artifact.documents, model, config, hierarchy=artifact
    ).evidence_paths()
    kept = [c for c in cases if all(g in paths for g in c["gold_passage_ids"])]
    skipped = [c["id"] for c in cases if c not in kept]
    if not kept:
        return dict(skipped=skipped, report=None)
    report = evaluate_descriptions(artifact, model, kept, {}, config, seed=seed)
    return dict(skipped=skipped, report=report)


def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
    return rows


def _parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus-path", required=True, help="JSONL corpus")
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--daily-add", type=int, required=True)
    parser.add_argument("--daily-delete", type=int, required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument(
        "--description-model", required=True, help="Local description model path"
    )
    parser.add_argument("--max-description-calls", type=int, default=None)
    parser.add_argument("--max-fanout", type=int, default=None)
    parser.add_argument("--decision-model", help="Local decision model path")
    parser.add_argument("--queries", help="Evaluator cases JSONL")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if (args.decision_model is None) != (args.queries is None):
            raise ValueError("--decision-model and --queries go together")
        for name in ("days", "daily_add", "daily_delete"):
            if getattr(args, name) < 0:
                raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
        # Import only after argparse: --help does not import ML libraries.
        from src.internal.document_index.hierarchy import HierarchyConfig
        from src.internal.document_index.hierarchy_models import (
            LocalHierarchyDescriber,
            SentenceEmbedding,
        )

        config = HierarchyConfig(
            mode="cluster",
            embedding_model=args.embedding_model,
            description_model=args.description_model,
            max_fanout=args.max_fanout,
        )
        initial, days = plan_days(
            _read_jsonl(args.corpus_path),
            args.days,
            args.daily_add,
            args.daily_delete,
            args.seed,
        )
        embedder = SentenceEmbedding(args.embedding_model, device=args.device)
        describer = LocalHierarchyDescriber(args.description_model, device=args.device)
        # One loaded model per kind; the counting wrappers keep tracks separate.
        result = simulate(
            initial,
            days,
            config,
            incremental=(embedder, describer),
            rebuild=(embedder, describer),
            max_description_calls=args.max_description_calls,
        )
        output = dict(
            note="Call counts and evaluator output only; there is no pass threshold.",
            settings=vars(args),
            days=result["days"],
            cumulative=result["cumulative"],
        )
        if args.decision_model:
            from src.internal.document_index.decision_model import LocalDecisionModel

            model = LocalDecisionModel.from_pretrained(
                args.decision_model, device=args.device
            )
            cases = _read_jsonl(args.queries)
            output["evaluation"] = {
                name: evaluate_tree(result[key], model, cases, args.seed)
                for name, key in (
                    ("incremental", "incremental_artifact"),
                    ("rebuild", "rebuilt_artifact"),
                )
            }
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(output["cumulative"], sort_keys=True))
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
