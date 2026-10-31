from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from src.internal.document_index.decision_retriever import (
    DecisionRetriever,
    DecisionRetrieverConfig,
)
from src.internal.document_index.hierarchy import HierarchyConfig, build_hierarchy
from src.internal.servers.retrieval.decision import create_app


class Judge:
    def __init__(self):
        self.inputs = []

    def binary(self, query, candidates):
        self.inputs.extend(c.text for c in candidates)
        return [
            0.95
            if "quantum sensor calibration" in c.text.lower()
            else 0.8
            if "quantum" in c.text.lower()
            else 0.1
            for c in candidates
        ]

    def choice(self, query, candidates):
        return [1 / len(candidates)] * len(candidates)


def tree():
    return build_hierarchy(
        [
            {
                "id": "gold",
                "title": "Quantum sensor calibration",
                "text": "Quantum sensor calibration measures phase bias.",
                "metadata": {"hierarchy_path": ["A"]},
            },
            {
                "id": "other",
                "title": "Quantum software",
                "text": "Quantum software controls scheduling.",
                "metadata": {"hierarchy_path": ["B"]},
            },
        ],
        HierarchyConfig(mode="structure"),
    )


def overrides(artifact):
    return {
        g["id"]: "Various documents"
        if "gold" in g["document_ids"]
        else "Quantum software scheduling controls and programming reference"
        for g in artifact.groups
    }


def test_bounded_previews_rescue_weak_branch_without_extra_judgments():
    artifact = tree()
    baseline = DecisionRetriever(
        artifact.documents,
        Judge(),
        DecisionRetrieverConfig(beam_width=1),
        hierarchy=artifact,
        group_descriptions=overrides(artifact),
    ).navigate("q", 1)
    recovered = DecisionRetriever(
        artifact.documents,
        Judge(),
        DecisionRetrieverConfig(
            beam_width=1,
            preview_max_nodes=2,
            preview_total_chars=400,
            preview_chars_per_node=200,
        ),
        hierarchy=artifact,
        group_descriptions=overrides(artifact),
    ).navigate("q", 1)
    assert baseline.results[0]["document"]["metadata"]["document_id"] == "other"
    assert recovered.results[0]["document"]["metadata"]["document_id"] == "gold"
    assert (
        recovered.usage["binary_judgments"] == baseline.usage["binary_judgments"] == 5
    )
    assert recovered.usage["preview_nodes"] == 1
    assert 0 < recovered.usage["preview_chars"] <= 400
    assert all(len(stage.selected_ids) <= 1 for stage in recovered.stages)
    assert len(recovered.stages[0].previewed_ids) == 1


def test_preview_scan_budget_stops_before_evidence_without_unbounded_walk():
    artifact = tree()
    result = DecisionRetriever(
        artifact.documents,
        Judge(),
        DecisionRetrieverConfig(
            beam_width=1, preview_max_nodes=2, preview_scan_limit=1
        ),
        hierarchy=artifact,
        group_descriptions=overrides(artifact),
    ).navigate("q", 1)
    assert result.usage["preview_scan_nodes"] == 1
    assert result.usage["preview_nodes"] == 0
    assert result.results[0]["document"]["metadata"]["document_id"] == "other"


def test_shared_preview_node_and_character_budgets_are_hard_caps():
    artifact = tree()
    both_generic = {g["id"]: "Various documents" for g in artifact.groups}
    config = DecisionRetrieverConfig(
        beam_width=2,
        preview_max_nodes=1,
        preview_total_chars=120,
        preview_chars_per_node=100,
        preview_children=1,
    )
    result = DecisionRetriever(
        artifact.documents,
        Judge(),
        config,
        hierarchy=artifact,
        group_descriptions=both_generic,
    ).navigate("q", 2)
    assert result.usage["preview_nodes"] == 1
    assert 0 < result.usage["preview_chars"] <= 100
    assert result.usage["preview_scan_nodes"] <= config.preview_scan_limit
    assert (
        sum(stage.preview_chars for stage in result.stages)
        == result.usage["preview_chars"]
    )


def test_evidence_paths_cover_ragged_tree_and_visible_passages_only():
    artifact = build_hierarchy(
        [
            {
                "id": "open",
                "title": "Public",
                "text": "answer",
                "metadata": {"hierarchy_path": ["Root"]},
            },
            {
                "id": "closed",
                "title": "Hidden title",
                "text": "hidden evidence",
                "metadata": {"hierarchy_path": ["Root", "Nested"], "acl": ["private"]},
            },
        ],
        HierarchyConfig(mode="structure"),
    )
    model = Judge()
    retriever = DecisionRetriever(artifact.documents, model, hierarchy=artifact)
    paths = retriever.evidence_paths()
    assert set(paths) == {"open::passage-0", "closed::passage-0"}
    assert len(paths["closed::passage-0"]) == len(paths["open::passage-0"]) + 1
    assert set(retriever.evidence_paths(filters={"access_acl": ["public"]})) == {
        "open::passage-0"
    }
    assert model.inputs == []


def test_filtered_previews_do_not_leak_stored_summaries_or_private_content():
    artifact = build_hierarchy(
        [
            {
                "id": "open",
                "title": "Overview",
                "text": "Public useful evidence",
                "metadata": {"hierarchy_path": ["Root"]},
            },
            {
                "id": "closed",
                "title": "PRIVATE_INVENTION",
                "text": "PRIVATE_SECRET_TEXT",
                "metadata": {"hierarchy_path": ["Root"], "acl": ["private"]},
            },
        ],
        HierarchyConfig(mode="structure"),
    )
    judge = Judge()
    retriever = DecisionRetriever(
        artifact.documents,
        judge,
        DecisionRetrieverConfig(preview_max_nodes=3),
        hierarchy=artifact,
        group_descriptions={g["id"]: "PRIVATE_SUMMARY" for g in artifact.groups},
    )
    result = retriever.navigate("q", filters={"access_acl": ["public"]})
    assert all("PRIVATE_" not in text for text in judge.inputs)
    assert {item["document"]["metadata"]["document_id"] for item in result.results} == {
        "open"
    }


def test_group_overrides_do_not_mutate_artifact_and_label_only_is_allowed():
    artifact = tree()
    before = [g["description"] for g in artifact.groups]
    model = Judge()
    retriever = DecisionRetriever(
        artifact.documents,
        model,
        hierarchy=artifact,
        group_descriptions={g["id"]: "" for g in artifact.groups},
    )
    retriever.navigate("q")
    assert [g["description"] for g in artifact.groups] == before
    assert set(model.inputs[:2]) == {"A", "B"}
    with pytest.raises(ValueError, match="group"):
        DecisionRetriever(
            artifact.documents,
            Judge(),
            hierarchy=artifact,
            group_descriptions={"unknown": "x"},
        )


def test_trace_returns_usage_without_changing_normal_response():
    artifact = tree()
    retriever = DecisionRetriever(artifact.documents, Judge(), hierarchy=artifact)
    client = TestClient(create_app(retriever))
    assert "usage" not in client.post("/retrieve", json={"query": "q"}).json()
    response = client.post(
        "/retrieve", json={"query": "q", "include_trace": True}
    ).json()
    assert response["usage"]["binary_judgments"] > 0
    assert response["usage"]["preview_nodes"] == 0


@pytest.mark.parametrize(
    "values",
    [
        {"preview_max_nodes": -1},
        {"preview_chars_per_node": 0},
        {"preview_total_chars": 0},
        {"preview_children": True},
        {"preview_scan_limit": 0},
    ],
)
def test_invalid_preview_budget_rejected(values):
    with pytest.raises(ValueError):
        replace(DecisionRetrieverConfig(), **values)


def test_wide_group_preview_reaches_evidence_before_scan_budget_is_spent():
    artifact = build_hierarchy(
        [
            {
                "id": str(i),
                "title": f"Document {i}",
                "text": f"Visible evidence for topic {i}.",
                "metadata": {"hierarchy_path": ["Root"]},
            }
            for i in range(100)
        ],
        HierarchyConfig(mode="structure"),
    )
    judge = Judge()
    result = DecisionRetriever(
        artifact.documents,
        judge,
        DecisionRetrieverConfig(preview_max_nodes=2, preview_scan_limit=64),
        hierarchy=artifact,
        group_descriptions={g["id"]: "Various documents" for g in artifact.groups},
    ).navigate("q")
    assert result.stages[0].previewed_ids
    assert result.usage["preview_chars"] > 0
    assert result.usage["preview_scan_nodes"] < 64
    assert any("Visible evidence" in text for text in judge.inputs)
