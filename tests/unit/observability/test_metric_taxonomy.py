"""One taxonomy decides whether an eval metric is about the documents
(retrieval), the answer (generation), a reward rollup, a latency, or other —
so a panel or report can never show recall@10 and exact_match as peers."""

from __future__ import annotations

import math

import pytest

from src.internal.observability.metric_taxonomy import (
    GROUPS,
    classify_metric,
    flatten_metrics,
    group_metrics,
)


@pytest.mark.parametrize(
    "name",
    [
        "recall@10",
        "ndcg@5",
        "mrr",
        "map@10",
        "precision@3",
        "hit_rate@10",
        "context_precision",
        "context_recall",
        "routing_accuracy",
        "reranker_improvement_ratio",
        "reranked.recall@10",
        "retrieval.num_queries",
        "reranked.num_queries",
    ],
)
def test_retrieval_metrics(name):
    assert classify_metric(name) == "retrieval"


@pytest.mark.parametrize(
    "name",
    [
        "exact_match",
        "contains_match",
        "faithfulness",
        "answer_relevancy",
        "token_f1",
        "citation_validity",
        "generation.exact_match",
        "generation.n",
    ],
)
def test_generation_metrics(name):
    assert classify_metric(name) == "generation"


@pytest.mark.parametrize(
    "name",
    [
        "avg_reward",
        "reward_total",
        "dim_correctness",
        "side_retrieval",
        "avg_reward_generation",
    ],
)
def test_reward_metrics(name):
    assert classify_metric(name) == "reward"


@pytest.mark.parametrize(
    "name",
    [
        "latency_ms.p99",
        "qt_latency_ms.p99",
        "latency_ms.mean",
        "retrieval_ms",
        "p95_latency",
    ],
)
def test_latency_metrics(name):
    assert classify_metric(name) == "latency"


@pytest.mark.parametrize("name", ["num_examples", "num_queries", "n", "seed"])
def test_unqualified_counts_are_other(name):
    assert classify_metric(name) == "other"


def test_flatten_keeps_finite_numbers_and_dots_nested_keys():
    data = {
        "recall@10": 0.5,
        "num_queries": 12,
        "ok": True,
        "note": "x",
        "nan": math.nan,
        "inf": math.inf,
        "reranked": {"recall@10": 0.6, "deep": {"too": {"far": 1.0}}},
        "latency_ms": {"p99": 12.5},
    }
    assert flatten_metrics(data) == {
        "recall@10": 0.5,
        "num_queries": 12,
        "reranked.recall@10": 0.6,
        "latency_ms.p99": 12.5,
    }


def test_flatten_depth_is_configurable():
    data = {"a": {"b": {"c": 1.0}}}
    assert flatten_metrics(data, max_depth=1) == {}
    assert flatten_metrics(data, max_depth=3) == {"a.b.c": 1.0}


def test_group_metrics_omits_empty_groups_and_keeps_group_order():
    flat = {
        "exact_match": 0.4,
        "recall@10": 0.5,
        "avg_reward": 0.1,
        "num_examples": 125,
    }
    grouped = group_metrics(flat)
    assert list(grouped) == ["retrieval", "generation", "reward", "other"]
    assert grouped["retrieval"] == {"recall@10": 0.5}
    assert grouped["generation"] == {"exact_match": 0.4}
    assert grouped["reward"] == {"avg_reward": 0.1}
    assert grouped["other"] == {"num_examples": 125}
    assert "latency" not in grouped


def test_groups_constant_lists_every_group_once():
    assert GROUPS == ("retrieval", "generation", "reward", "latency", "other")
