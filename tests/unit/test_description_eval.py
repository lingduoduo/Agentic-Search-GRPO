"""Offline evaluation tests: deterministic injected judges, no quality claims."""

import copy

import pytest

from src.internal.document_index.description_eval import evaluate_descriptions
from src.internal.document_index.hierarchy import HierarchyConfig, build_hierarchy
from src.internal.document_index.decision_retriever import DecisionRetrieverConfig


class Judge:
    def __init__(self):
        self.prompts = []

    def binary(self, query, candidates):
        self.prompts.append((query, [c.text for c in candidates]))
        return [1.0 for _ in candidates]

    def choice(self, query, candidates):
        return [1 / len(candidates)] * len(candidates)


def artifact(acl=None):
    return build_hierarchy(
        [
            {
                "id": "a",
                "title": "Alpha",
                "contents": "alpha evidence",
                "metadata": {"path": ["Root"], **({"acl": acl} if acl else {})},
            },
            {
                "id": "b",
                "title": "Beta",
                "contents": "beta evidence",
                "metadata": {"path": ["Root"]},
            },
        ],
        HierarchyConfig(mode="structure"),
    )


def test_fixed_tree_model_and_multigold_metrics():
    tree = artifact()
    before = copy.deepcopy(tree)
    judge = Judge()
    report = evaluate_descriptions(
        tree,
        judge,
        [
            {
                "id": "q",
                "query": "find evidence",
                "gold_passage_ids": ["a::passage-0", "b::passage-0"],
            }
        ],
        {"label_only": {g["id"]: "" for g in tree.groups}},
        DecisionRetrieverConfig(beam_width=1),
        topk=1,
        bootstrap_samples=20,
    )
    assert tree == before
    assert set(report["variants"]) == {"stored", "label_only"}
    record = report["variants"]["stored"]["queries"][0]
    assert record["metrics"]["final_recall_at_topk"] == 0.5
    assert record["stages"]["document"]["unconditional"]["denominator"] == 2
    assert record["stages"]["section"]["conditional"]["denominator"] == 1
    assert (
        report["paired_differences"]["label_only"]["final_recall_at_topk"]["mean"] == 0
    )
    assert all("gold_passage_ids" not in str(p) for p in judge.prompts)


@pytest.mark.parametrize(
    "cases",
    [
        [],
        [{"id": "q", "query": "x", "gold_passage_ids": ["unknown"]}],
        [
            {
                "id": "q",
                "query": "x",
                "gold_passage_ids": ["a::passage-0", "a::passage-0"],
            }
        ],
    ],
)
def test_invalid_cases_fail_before_model_calls(cases):
    judge = Judge()
    with pytest.raises(ValueError):
        evaluate_descriptions(artifact(), judge, cases, {}, bootstrap_samples=5)
    assert not judge.prompts


def test_ragged_paths_missing_stages_and_multigold_conditioning():
    from types import SimpleNamespace
    from src.internal.document_index.description_eval import _query_metrics
    from src.internal.document_index.decision_retriever import DecisionStage

    paths = {
        "a::passage-0": ("g/root", "d0", "d0/s0", "a::passage-0"),
        "b::passage-0": ("g/root", "g/deep", "d1", "d1/s0", "b::passage-0"),
    }
    run = SimpleNamespace(
        stages=[
            DecisionStage("group", ("g/root",), (1.0,), ("g/root",)),
            DecisionStage("group", ("g/deep",), (0.0,), ()),
            DecisionStage("document", ("d0",), (1.0,), ("d0",)),
            # Missing section/passage/choice stages are loss, not omitted.
        ],
        results=[],
    )
    stages, metrics = _query_metrics(paths, list(paths), run)
    assert stages["group_depth_1"]["unconditional"] == {
        "numerator": 0,
        "denominator": 1,
        "value": 0.0,
    }
    assert stages["document"]["conditional"]["denominator"] == 1
    assert stages["document"]["unconditional"]["value"] == 0.5
    assert stages["section"]["unconditional"]["value"] == 0
    assert stages["section"]["conditional"]["denominator"] == 1
    assert stages["passage"]["conditional"]["value"] is None
    assert metrics["complete_ancestor_path_survival"] == 0
    assert metrics["final_recall_at_topk"] == 0


def test_paired_bootstrap_is_seeded_and_excludes_not_applicable():
    from src.internal.document_index.description_eval import _paired

    left = [{"metrics": {"x": 0}}, {"metrics": {"x": 1}}, {"metrics": {"x": None}}]
    right = [{"metrics": {"x": 1}}, {"metrics": {"x": 0}}, {"metrics": {"x": 1}}]
    one = _paired(left, right, 100, 42)
    assert one == _paired(left, right, 100, 42)
    assert one["x"]["mean"] == 0
    assert one["x"]["count"] == 2


def test_override_validation_and_bootstrap_validation_before_calls():
    tree = artifact()
    judge = Judge()
    cases = [{"id": "q", "query": "find", "gold_passage_ids": ["a::passage-0"]}]
    for variants in (
        {"stored": {tree.groups[0]["id"]: "changed"}},
        {"bad": {"unknown": "text"}},
        {"bad": {tree.groups[0]["id"]: 4}},
    ):
        with pytest.raises(ValueError):
            evaluate_descriptions(tree, judge, cases, variants)
    for count in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            evaluate_descriptions(tree, judge, cases, {}, bootstrap_samples=count)
    assert judge.prompts == []


def test_acl_hidden_gold_rejected_and_summary_comparison_flagged():
    tree = artifact(acl=["private"])
    judge = Judge()
    hidden = [
        {
            "id": "q",
            "query": "find",
            "gold_passage_ids": ["a::passage-0"],
            "filters": {"access_acl": ["public"]},
        }
    ]
    with pytest.raises(ValueError, match="visible"):
        evaluate_descriptions(tree, judge, hidden, {}, topk=1)
    assert judge.prompts == []


def test_structure_group_ids_do_not_impersonate_section_paths():
    tree = artifact()
    assert any(g["id"].startswith("structure-") for g in tree.groups)
    cases = [{"id": "q", "query": "find", "gold_passage_ids": ["a::passage-0"]}]
    report = evaluate_descriptions(
        tree, Judge(), cases, {}, topk=2, bootstrap_samples=5
    )
    metrics = report["variants"]["stored"]["queries"][0]["metrics"]
    assert metrics["complete_ancestor_path_survival"] == 1
    assert metrics["complete_path_survival"] == 1
    assert metrics["section/unconditional"] == 1
    assert metrics["section/conditional"] == 1
    assert len(report["case_fingerprint"]) == 64
    changed = copy.deepcopy(cases)
    changed[0]["filters"] = {"access_acl": ["public"]}
    other = evaluate_descriptions(
        tree, Judge(), changed, {}, topk=2, bootstrap_samples=5
    )
    assert report["case_fingerprint"] != other["case_fingerprint"]


def test_bootstrap_conditional_macro_excludes_zero_denominator_pairs():
    from src.internal.document_index.description_eval import _paired

    left = [
        {"metrics": {"section/conditional": None}},
        {"metrics": {"section/conditional": 0.25}},
        {"metrics": {"section/conditional": 1.0}},
    ]
    right = [
        {"metrics": {"section/conditional": 1.0}},
        {"metrics": {"section/conditional": 0.75}},
        {"metrics": {"section/conditional": 0.5}},
    ]
    metric = _paired(left, right, 1000, 42)["section/conditional"]
    assert metric == {"mean": 0.0, "count": 2, "ci95": [-0.5, 0.5]}


def test_tree_fingerprint_excludes_description_generation_metadata():
    tree = artifact()
    changed = copy.deepcopy(tree)
    for group in changed.groups:
        group["description"] = "A changed but valid routing description"
        group["description_source"] = "another-local-describer"
        group.pop("description_provenance", None)
        group.pop("description_quality", None)
        group["centroid"] = [0.25, 0.75]
    cases = [{"id": "q", "query": "find", "gold_passage_ids": ["a::passage-0"]}]
    original = evaluate_descriptions(
        tree, Judge(), cases, {}, topk=1, bootstrap_samples=5
    )
    updated = evaluate_descriptions(
        changed, Judge(), cases, {}, topk=1, bootstrap_samples=5
    )
    assert original["tree_fingerprint"] == updated["tree_fingerprint"]
    assert (
        original["variants"]["stored"]["description_fingerprint"]
        != updated["variants"]["stored"]["description_fingerprint"]
    )


def test_acl_bypassed_queries_are_excluded_from_description_comparison():
    tree = artifact()
    cases = [
        {"id": "open", "query": "find", "gold_passage_ids": ["a::passage-0"]},
        {
            "id": "acl",
            "query": "find",
            "gold_passage_ids": ["a::passage-0"],
            "filters": {"access_acl": ["public"]},
        },
    ]
    report = evaluate_descriptions(
        tree,
        Judge(),
        cases,
        {"label_only": {g["id"]: "" for g in tree.groups}},
        topk=1,
        bootstrap_samples=10,
    )
    paired = report["paired_differences"]["label_only"]
    assert paired["final_recall_at_topk"]["count"] == 1
    variant = report["variants"]["label_only"]
    assert variant["macro_metrics"]["final_recall_at_topk"]["count"] == 1
    assert variant["acl_bypassed_macro_metrics"]["final_recall_at_topk"]["count"] == 1
    assert len(variant["queries"]) == 2
