"""Which side of the pipeline an evaluation metric describes.

Every eval harness in this repo writes a flat dict, and until now the Dev
Console showed ``recall@10`` and ``exact_match`` as peers. This module is the
one place that decides whether a metric name is about the documents
(``retrieval``), the answer (``generation``), a reward rollup (``reward``), a
timing (``latency``), or none of those (``other``). Stdlib only: it sits on
the web request path.
"""

from __future__ import annotations

import math
import re

GROUPS: tuple[str, ...] = ("retrieval", "generation", "reward", "latency", "other")

_RETRIEVAL_LEAVES = re.compile(
    r"^(recall|ndcg|map|precision|hit_rate)(@\d+)?$|"
    r"^(mrr|context_precision|context_recall|routing_accuracy|"
    r"reranker_improvement_ratio)$"
)
_GENERATION_LEAVES = re.compile(
    r"^(exact_match|contains_match|faithfulness|answer_relevancy|token_f1)$|"
    r"^citation_"
)
_REWARD_LEAVES = re.compile(r"^(reward|avg_reward|dim_|side_)")
_LATENCY_LEAVES = re.compile(r"latency|_ms$")
_LATENCY_PERCENTILES = frozenset({"p50", "p90", "p95", "p99", "mean", "max", "min"})

# A parent segment that settles otherwise-ambiguous leaves such as ``n`` or
# ``num_queries``: eval_runner nests its reranked block, RAGAS-style reports
# nest a generation block.
_RETRIEVAL_PARENTS = frozenset({"retrieval", "reranked", "retriever"})
_GENERATION_PARENTS = frozenset({"generation", "generator", "answer"})


def classify_metric(name: str) -> str:
    """Return the group for a (possibly dotted) metric name."""
    parts = name.lower().split(".")
    leaf, parents = parts[-1], parts[:-1]
    if _LATENCY_LEAVES.search(leaf) or any(_LATENCY_LEAVES.search(p) for p in parents):
        if leaf in _LATENCY_PERCENTILES or _LATENCY_LEAVES.search(leaf):
            return "latency"
    if _REWARD_LEAVES.search(leaf):
        return "reward"
    if _RETRIEVAL_LEAVES.search(leaf):
        return "retrieval"
    if _GENERATION_LEAVES.search(leaf):
        return "generation"
    if any(p in _RETRIEVAL_PARENTS for p in parents):
        return "retrieval"
    if any(p in _GENERATION_PARENTS for p in parents):
        return "generation"
    return "other"


def flatten_metrics(data: dict, *, max_depth: int = 2) -> dict[str, float]:
    """Finite, non-bool numbers from *data*, nested dicts dotted up to *max_depth*."""
    flat: dict[str, float] = {}

    def _walk(node: dict, prefix: str, depth: int) -> None:
        for key, value in node.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                if math.isfinite(value):
                    flat[name] = value
            elif isinstance(value, dict) and depth < max_depth:
                _walk(value, name, depth + 1)

    _walk(data, "", 1)
    return flat


def group_metrics(flat: dict[str, float]) -> dict[str, dict[str, float]]:
    """Bucket a flat metrics dict by group, in :data:`GROUPS` order, empties omitted."""
    grouped: dict[str, dict[str, float]] = {group: {} for group in GROUPS}
    for name, value in flat.items():
        grouped[classify_metric(name)][name] = value
    return {group: values for group, values in grouped.items() if values}
