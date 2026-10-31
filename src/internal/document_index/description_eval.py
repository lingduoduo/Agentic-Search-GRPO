"""Paired, fixed-tree description experiments with passage-level gold evidence.

Labels are used only after navigation. Character and protocol counters are cost
proxies, not tokenizer or hardware compute measurements.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import asdict

from .decision_retriever import DecisionRetriever, DecisionRetrieverConfig


def _fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def builtin_variants(artifact):
    """Build description-only controls without embedding or changing membership."""
    from .hierarchy_descriptions import select_description_samples

    children = {}
    for group in artifact.groups:
        children.setdefault(group["parent_id"], []).append(group)
    documents = {doc["id"]: doc for doc in artifact.documents}

    def descendants(group):
        ids = list(group["document_ids"])
        for child in children.get(group["id"], []):
            ids.extend(descendants(child))
        return ids

    extractive = {}
    for group in artifact.groups:
        docs = [documents[i] for i in descendants(group)]
        selected = select_description_samples(docs, limit=12)
        titles = [documents[item["document_id"]]["title"][:120] for item in selected]
        extractive[group["id"]] = "; ".join(titles)
    return {
        "stored": {},
        "label_only": {g["id"]: "" for g in artifact.groups},
        "extractive": extractive,
    }


def _ratio(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def _query_metrics(paths, gold, navigation):
    selected = {}
    group_depth = 0
    for stage in navigation.stages:
        key = stage.kind
        if key == "group":
            key = f"group_depth_{group_depth}"
            group_depth += 1
        selected.setdefault(key, set()).update(stage.selected_ids)
    depth = max(sum(node.startswith("g/") for node in paths[g]) for g in gold)
    keys = [f"group_depth_{i}" for i in range(depth)] + [
        "document",
        "section",
        "passage",
        "choice",
    ]
    counters = {key: [0, 0, 0, 0] for key in keys}
    ancestor_survivors = passage_survivors = 0
    for passage in gold:
        path = paths[passage]
        groups = [node for node in path if node.startswith("g/")]
        required = [(f"group_depth_{i}", node) for i, node in enumerate(groups)]
        required += [
            (
                "document",
                next(n for n in path if re.fullmatch(r"d[0-9]+", n)),
            ),
            ("section", next(n for n in path if re.fullmatch(r"d[0-9]+/s[0-9]+", n))),
            ("passage", passage),
            ("choice", passage),
        ]
        survived = True
        for key, node in required:
            if key == "passage":
                ancestor_survivors += int(survived)
            hit = node in selected.get(key, set())
            counts = counters[key]
            counts[1] += 1
            counts[0] += int(survived and hit)
            counts[3] += int(survived)
            counts[2] += int(survived and hit)
            survived = survived and hit
            if key == "passage":
                passage_survivors += int(survived)
    stages = {
        key: {"unconditional": _ratio(c[0], c[1]), "conditional": _ratio(c[2], c[3])}
        for key, c in counters.items()
    }
    final = {item["document"]["id"] for item in navigation.results}
    metrics = {
        "complete_ancestor_path_survival": ancestor_survivors / len(gold),
        "complete_path_survival": passage_survivors / len(gold),
        "final_recall_at_topk": len(final.intersection(gold)) / len(gold),
        "any_gold_final_hit": float(bool(final.intersection(gold))),
        "any_gold_ancestor_path_survival": float(ancestor_survivors > 0),
        "any_gold_complete_path_survival": float(passage_survivors > 0),
    }
    for key, values in stages.items():
        for mode, value in values.items():
            metrics[f"{key}/{mode}"] = value["value"]
    return stages, metrics


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _paired(left, right, samples, seed):
    names = sorted(set().union(*(r["metrics"] for r in left + right)))
    output = {}
    for name in names:
        differences = [
            right_record["metrics"].get(name) - left_record["metrics"].get(name)
            for left_record, right_record in zip(left, right)
            if left_record["metrics"].get(name) is not None
            and right_record["metrics"].get(name) is not None
        ]
        rng = random.Random(seed)
        draws = (
            sorted(
                sum(rng.choice(differences) for _ in differences) / len(differences)
                for _ in range(samples)
            )
            if differences
            else []
        )
        output[name] = {
            "mean": _mean(differences),
            "count": len(differences),
            "ci95": [draws[int((len(draws) - 1) * p)] for p in (0.025, 0.975)]
            if draws
            else None,
        }
    return output


def evaluate_descriptions(
    artifact,
    model,
    cases,
    variants,
    config=None,
    *,
    topk=5,
    bootstrap_samples=1000,
    seed=42,
):
    """Evaluate paired variants using one exact model instance and budget policy."""
    config = config or DecisionRetrieverConfig()
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(topk) is not int or not 1 <= topk <= config.shortlist_size:
        raise ValueError("topk must be within the shortlist size")
    if not isinstance(variants, dict):
        raise ValueError("variants must be a mapping")
    group_ids = {g["id"] for g in artifact.groups}
    for name, overrides in variants.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(overrides, dict)
        ):
            raise ValueError("variant names must be nonempty and overrides mappings")
        if any(
            k not in group_ids or not isinstance(v, str) for k, v in overrides.items()
        ):
            raise ValueError(
                "variant overrides require known group IDs and string text"
            )
    if variants.get("stored"):
        raise ValueError("stored baseline cannot be overridden")
    variants = {"stored": {}, **variants}
    baseline = DecisionRetriever(artifact.documents, model, config, hierarchy=artifact)
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    ids = set()
    case_paths = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("case must be an object")
        identifier, query, gold = (
            case.get(k) for k in ("id", "query", "gold_passage_ids")
        )
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or identifier in ids
        ):
            raise ValueError("case IDs must be unique nonempty strings")
        ids.add(identifier)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("case query must be nonempty")
        if (
            not isinstance(gold, list)
            or not gold
            or any(not isinstance(g, str) or not g.strip() for g in gold)
            or len(set(gold)) != len(gold)
        ):
            raise ValueError("gold_passage_ids must be unique nonempty strings")
        filters = case.get("filters")
        if filters is not None and not isinstance(filters, dict):
            raise ValueError("filters must be an object")
        paths = baseline.evidence_paths(filters=filters)
        if any(g not in paths for g in gold):
            raise ValueError(
                "gold passage IDs must be known and visible under the fixed config"
            )
        case_paths.append(paths)
    reports = {}
    for name, overrides in variants.items():
        retriever = DecisionRetriever(
            artifact.documents,
            model,
            config,
            hierarchy=artifact,
            group_descriptions=overrides,
        )
        records = []
        for case, paths in zip(cases, case_paths):
            start = time.perf_counter()
            result = retriever.navigate(
                case["query"], topk, filters=case.get("filters")
            )
            elapsed = time.perf_counter() - start
            stages, metrics = _query_metrics(paths, case["gold_passage_ids"], result)
            metrics.update(
                {f"usage/{key}": value for key, value in result.usage.items()}
            )
            metrics["latency_seconds"] = elapsed
            records.append(
                {
                    "id": case["id"],
                    "stages": stages,
                    "metrics": metrics,
                    "usage": dict(result.usage),
                    "latency_seconds": elapsed,
                    "summaries_bypassed_for_acl": bool(
                        (case.get("filters") or {}).get("access_acl")
                    ),
                }
            )
        metric_names = sorted(set().union(*(r["metrics"] for r in records)))
        stage_names = sorted(set().union(*(r["stages"] for r in records)))
        # ACL-filtered queries route on visible titles, never on the variant's
        # summaries, so they would only dilute the description comparison.
        compared = [r for r in records if not r["summaries_bypassed_for_acl"]]
        bypassed = [r for r in records if r["summaries_bypassed_for_acl"]]

        def macro(subset):
            return {
                k: {
                    "mean": _mean([r["metrics"].get(k) for r in subset]),
                    "count": sum(r["metrics"].get(k) is not None for r in subset),
                }
                for k in metric_names
            }

        reports[name] = {
            "queries": records,
            "macro_metrics": macro(compared),
            "acl_bypassed_macro_metrics": macro(bypassed),
            "stage_counts": {
                k: {
                    mode: _ratio(
                        sum(
                            r["stages"].get(k, {}).get(mode, {}).get("numerator", 0)
                            for r in compared
                        ),
                        sum(
                            r["stages"].get(k, {}).get(mode, {}).get("denominator", 0)
                            for r in compared
                        ),
                    )
                    for mode in ("unconditional", "conditional")
                }
                for k in stage_names
            },
            "usage": {
                k: sum(r["usage"].get(k, 0) for r in records)
                for k in set().union(*(r["usage"] for r in records))
            },
            "latency_seconds": sum(r["latency_seconds"] for r in records),
            "description_fingerprint": _fingerprint(
                {
                    g["id"]: overrides.get(g["id"], g["description"])
                    for g in artifact.groups
                }
            ),
        }
    identity = getattr(getattr(model, "model", None), "name_or_path", None)
    return {
        "schema_version": 1,
        "model": {
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "identity": identity or getattr(model, "name_or_path", None),
            "max_context_tokens": getattr(model, "max_context_tokens", None),
            "inference_batch_size": getattr(model, "inference_batch_size", None),
            "tokenizer_identity": getattr(
                getattr(model, "tokenizer", None), "name_or_path", None
            ),
        },
        "tree_fingerprint": _fingerprint(
            {
                "documents": artifact.documents,
                "groups": [
                    {
                        key: group[key]
                        for key in ("id", "parent_id", "label", "document_ids")
                    }
                    for group in artifact.groups
                ],
                "segmentation": {
                    "passage_words": config.passage_words,
                    "max_passage_chars": config.max_passage_chars,
                },
            }
        ),
        "config": asdict(config),
        "topk": topk,
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "query_count": len(cases),
        "case_fingerprint": _fingerprint(cases),
        "cost_note": "Actual protocol/character proxies and wall latency; equal policy does not imply equal compute.",
        "acl_note": "Active access_acl filtering bypasses stored and override group summaries; those queries are excluded from macro_metrics, stage_counts and paired_differences and reported under acl_bypassed_macro_metrics.",
        "bootstrap_note": "Paired query resampling; a single query yields an uninformative degenerate interval."
        if len(cases) == 1
        else "Paired query resampling.",
        "determinism_note": "Model determinism is the operator responsibility.",
        "variants": reports,
        "paired_differences": {
            name: _paired(
                [
                    r
                    for r in reports["stored"]["queries"]
                    if not r["summaries_bypassed_for_acl"]
                ],
                [r for r in report["queries"] if not r["summaries_bypassed_for_acl"]],
                bootstrap_samples,
                seed,
            )
            for name, report in reports.items()
            if name != "stored"
        },
    }
