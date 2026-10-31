from __future__ import annotations

import pytest

from src.internal.document_index.decision_retriever import (
    DecisionRetriever,
    DecisionRetrieverConfig,
)
from src.internal.document_index.hierarchy import HierarchyConfig, build_hierarchy


class Judge:
    def __init__(self):
        self.texts = []

    def binary(self, query, candidates):
        self.texts.extend(c.text for c in candidates)
        return [0.8] * len(candidates)

    def choice(self, query, candidates):
        self.texts.extend(c.text for c in candidates)
        return [1 / len(candidates)] * len(candidates)


def artifact(docs):
    return build_hierarchy(docs, HierarchyConfig(mode="structure"))


def test_ragged_groups_keep_direct_and_deep_document_evidence():
    tree = artifact(
        [
            {
                "id": "short",
                "title": "Direct",
                "text": "short answer",
                "metadata": {"hierarchy_path": ["Root"]},
            },
            {
                "id": "long",
                "title": "Nested",
                "text": "deep answer",
                "metadata": {"hierarchy_path": ["Root", "Sub", "Deep"]},
            },
        ]
    )
    run = DecisionRetriever(
        tree.documents, Judge(), DecisionRetrieverConfig(beam_width=10), hierarchy=tree
    ).navigate("q")
    assert {item["document"]["metadata"]["document_id"] for item in run.results} == {
        "short",
        "long",
    }
    assert [s.kind for s in run.stages] == [
        "group",
        "group",
        "group",
        "document",
        "section",
        "passage",
        "choice",
    ]


def test_filtered_group_descriptions_never_include_hidden_sources():
    tree = artifact(
        [
            {
                "id": "open",
                "title": "Public",
                "text": "public answer",
                "metadata": {"hierarchy_path": ["Root"]},
            },
            {
                "id": "secret",
                "title": "Secret invention",
                "text": "secret answer",
                "metadata": {"hierarchy_path": ["Root", "Deep"], "acl": ["private"]},
            },
        ]
    )
    judge = Judge()
    retriever = DecisionRetriever(tree.documents, judge, hierarchy=tree)
    rows = retriever.navigate("q", filters={"access_acl": ["public"]}).results
    assert [item["document"]["metadata"]["document_id"] for item in rows] == ["open"]
    assert all("secret" not in text.lower() for text in judge.texts)
    assert "Public" in judge.texts[0]


def test_artifact_sections_and_character_limit_handle_chinese_without_spaces():
    tree = artifact(
        [
            {
                "id": "cn",
                "title": "中文",
                "contents": "<h1>说明</h1><p>" + "中文内容" * 20 + "</p>",
                "metadata": {"format": "html"},
            }
        ]
    )
    run = DecisionRetriever(
        tree.documents,
        Judge(),
        DecisionRetrieverConfig(max_passage_chars=20),
        hierarchy=tree,
    ).navigate("q", topk=20)
    passages = sorted(
        (item["document"] for item in run.results),
        key=lambda doc: doc["metadata"]["passage_index"],
    )
    assert all(len(doc["text"]) <= 20 for doc in passages)
    assert (
        "".join("".join(doc["text"].split()) for doc in passages)
        == "说明" + "中文内容" * 20
    )
    assert all(doc["metadata"]["section"] == "说明" for doc in passages)


def test_per_kind_budget_selects_group_independently_of_document_budget():
    tree = artifact(
        [
            {
                "id": str(i),
                "title": f"Document {i}",
                "text": "answer",
                "metadata": {"hierarchy_path": [f"Group {i}"]},
            }
            for i in range(3)
        ]
    )
    run = DecisionRetriever(
        tree.documents,
        Judge(),
        DecisionRetrieverConfig(beam_width=3, beam_widths={"group": 1}),
        hierarchy=tree,
    ).navigate("q")
    assert len(run.stages[0].selected_ids) == 1
    assert len(run.results) == 1


def test_artifact_rejects_mismatching_corpus_ids():
    tree = artifact([{"id": "a", "text": "answer"}])
    with pytest.raises(ValueError, match="corpus|document"):
        DecisionRetriever([{"id": "b", "text": "different"}], Judge(), hierarchy=tree)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_passage_chars": 0},
        {"beam_widths": {"group": 0}},
        {"beam_widths": {"unknown": 1}},
    ],
)
def test_invalid_navigation_granularity_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DecisionRetrieverConfig(**kwargs)


def test_server_loads_offline_artifact_without_loading_build_models(
    tmp_path, monkeypatch
):
    import sys
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from src.internal.servers.retrieval.decision import main

    tree = artifact(
        [{"id": "a", "text": "local answer", "metadata": {"hierarchy_path": ["Docs"]}}]
    )
    path = tmp_path / "tree.json"
    tree.save(path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "decision",
            "--hierarchy",
            str(path),
            "--model",
            "/local",
            "--group-beam-width",
            "2",
            "--max-passage-chars",
            "80",
        ],
    )
    with (
        patch("src.internal.servers.retrieval.decision.load_environment"),
        patch(
            "src.internal.document_index.decision_model.LocalDecisionModel.from_pretrained",
            return_value=Judge(),
        ),
        patch("src.internal.servers.retrieval.decision.run_uvicorn_app") as serve,
        patch(
            "src.internal.document_index.hierarchy_models.SentenceEmbedding",
            side_effect=AssertionError("offline only"),
        ),
        patch(
            "src.internal.document_index.hierarchy_models.LocalHierarchyDescriber",
            side_effect=AssertionError("offline only"),
        ),
    ):
        main()
    response = TestClient(serve.call_args.args[0]).post(
        "/retrieve", json={"query": "q", "include_trace": True}
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["text"] == "local answer"
    assert response.json()["trace"][0]["kind"] == "group"


def test_default_retrieval_preserves_code_indentation_and_line_breaks():
    text = "def f():\n    return 1\n\nnext line\tcolumn"
    rows = (
        DecisionRetriever([{"id": "code", "contents": text}], Judge())
        .navigate("q")
        .results
    )
    assert rows[0]["document"]["text"] == text
