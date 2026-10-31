from __future__ import annotations

import math

import pytest

from src.internal.document_index.decision_retriever import (
    DecisionRetriever,
    DecisionRetrieverConfig,
)


class RecordingModel:
    def __init__(self):
        self.calls = []

    def binary(self, query, candidates):
        self.calls.append(("binary", query, list(candidates)))
        return [
            0.9 if "target" in candidate.text.lower() else 0.1
            for candidate in candidates
        ]

    def choice(self, query, candidates):
        self.calls.append(("choice", query, list(candidates)))
        weights = [i + 1 for i in range(len(candidates))]
        return [weight / sum(weights) for weight in weights]


def test_navigation_prunes_branches_and_jointly_reranks_leaves():
    model = RecordingModel()
    retriever = DecisionRetriever(
        [
            {
                "id": "keep",
                "title": "Target manual",
                "contents": "# Target section\nfirst target\n\n# Other section\nsecond",
                "url": "https://example.test/manual",
                "metadata": {"topic": "Target topic", "source": "Manual"},
            },
            {"id": "drop", "title": "Unrelated", "contents": "secret unused passage"},
        ],
        model,
        DecisionRetrieverConfig(beam_width=1, passage_words=3),
    )

    run = retriever.navigate("find target", topk=2)
    assert [stage.kind for stage in run.stages] == [
        "topic",
        "document",
        "section",
        "passage",
        "choice",
    ]
    assert all(
        "secret unused passage" not in candidate.text
        for _, _, candidates in model.calls
        for candidate in candidates
    )
    assert len(run.results) == 2
    assert run.results[0]["score"] > run.results[1]["score"]
    # Choice reverses the stable binary ordering; scores are not path products.
    assert run.results[0]["document"]["id"] == "keep::passage-1"
    assert run.results[0]["score"] == pytest.approx(2 / 3)
    doc = run.results[0]["document"]
    assert doc["url"] == "https://example.test/manual"
    assert doc["metadata"] == {
        "topic": "Target topic",
        "source": "Manual",
        "document_id": "keep",
        "section": "Target section",
        "passage_index": 1,
    }


def test_acl_is_applied_before_any_model_call_or_beam_selection():
    model = RecordingModel()
    docs = [
        {
            "id": "hidden",
            "title": "Target confidential",
            "contents": "target secret",
            "metadata": {"topic": "Secret topic", "tags": {"acl": ["private"]}},
        },
        {
            "id": "public",
            "title": "Public",
            "text": "public answer",
            "metadata": {"acl": ["public"]},
        },
    ]
    retriever = DecisionRetriever(docs, model, DecisionRetrieverConfig(beam_width=1))
    run = retriever.navigate("target", filters={"access_acl": ["public"]})
    assert run.results[0]["document"]["metadata"]["document_id"] == "public"
    assert all(
        "secret" not in candidate.text.lower()
        and "confidential" not in candidate.text.lower()
        for _, _, candidates in model.calls
        for candidate in candidates
    )
    assert "hidden::passage-0" not in str(run.stages)


def test_binary_requests_are_bounded_but_final_choice_is_joint():
    model = RecordingModel()
    retriever = DecisionRetriever(
        [
            {"id": str(i), "title": f"Doc {i}", "contents": f"answer {i}"}
            for i in range(7)
        ],
        model,
        DecisionRetrieverConfig(beam_width=7, batch_size=2, shortlist_size=7),
    )
    rows = retriever.retrieve(["one", "two"], topk=7)
    assert [len(row) for row in rows] == [7, 7]
    assert all(
        len(candidates) <= 2 for mode, _, candidates in model.calls if mode == "binary"
    )
    assert [
        len(candidates) for mode, _, candidates in model.calls if mode == "choice"
    ] == [7, 7]
    assert [row["document"]["metadata"]["document_id"] for row in rows[0]] == list(
        "6543210"
    )


def test_empty_corpus_and_fully_filtered_corpus_do_not_call_model():
    model = RecordingModel()
    assert DecisionRetriever([], model).retrieve(["question"], 5) == [[]]
    retriever = DecisionRetriever(
        [
            {"contents": "private", "metadata": {"acl": ["private"]}},
            {"contents": " \n "},
        ],
        model,
    )
    assert retriever.retrieve(["question"], 5, filters={"access_acl": ["public"]}) == [
        []
    ]
    assert model.calls == []


def test_threshold_can_abstain_without_falling_back_to_another_retriever():
    model = RecordingModel()
    run = DecisionRetriever(
        [{"contents": "unrelated"}],
        model,
        DecisionRetrieverConfig(min_relevance=0.5),
    ).navigate("question")
    assert run.results == []
    assert len(model.calls) == 1


def test_section_and_passage_splitting_preserves_content_and_ignores_fenced_headings():
    text = "Preamble.\n# Heading\nalpha beta\n```\n# Not a section\n```\ngamma delta"
    retriever = DecisionRetriever(
        [{"id": "d", "contents": text}],
        RecordingModel(),
        DecisionRetrieverConfig(beam_width=10, passage_words=3, shortlist_size=10),
    )
    run = retriever.navigate("question", topk=10)
    docs = sorted(
        (item["document"] for item in run.results),
        key=lambda doc: doc["metadata"]["passage_index"],
    )
    assert " ".join(doc["text"] for doc in docs).split() == text.split()
    assert {doc["metadata"]["section"] for doc in docs} == {"Body", "Heading"}
    assert len({doc["id"] for doc in docs}) == len(docs)
    assert all(len(doc["text"].split()) <= 3 for doc in docs)


@pytest.mark.parametrize("scores", [[], [math.nan], [math.inf], [-0.1], [1.1]])
def test_invalid_binary_output_fails_explicitly(scores):
    class Broken(RecordingModel):
        def binary(self, query, candidates):
            return scores

    with pytest.raises(ValueError, match="scores|probabilities"):
        DecisionRetriever([{"contents": "body"}], Broken()).navigate("query")


def test_invalid_choice_normalization_is_rejected():
    class Broken(RecordingModel):
        def choice(self, query, candidates):
            return [0.5]

    with pytest.raises(ValueError, match="sum to one"):
        DecisionRetriever([{"contents": "body"}], Broken()).navigate("query")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beam_width": 0},
        {"shortlist_size": -1},
        {"batch_size": 0},
        {"passage_words": 0},
        {"min_relevance": math.nan},
        {"min_relevance": 1.1},
        {"beam_width": True},
        {"batch_size": 1.5},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DecisionRetrieverConfig(**kwargs)


@pytest.mark.parametrize("query,topk", [("", 1), (" \n", 1), ("q", 0), ("q", 21)])
def test_invalid_query_or_topk_is_rejected(query, topk):
    with pytest.raises(ValueError):
        DecisionRetriever([], RecordingModel()).navigate(query, topk)


def test_duplicate_ids_are_rejected_even_for_empty_documents():
    with pytest.raises(ValueError, match="Duplicate"):
        DecisionRetriever(
            [{"id": "a"}, {"id": "a", "contents": "body"}], RecordingModel()
        )


@pytest.mark.parametrize(
    "doc",
    [
        {"contents": None},
        {"contents": 42},
        {"title": []},
        {"metadata": []},
        {"metadata": {"topic": ""}},
        {"metadata": {"topic": ["topic"]}},
    ],
)
def test_malformed_corpus_fields_fail_explicitly(doc):
    with pytest.raises(ValueError):
        DecisionRetriever([doc], RecordingModel())


@pytest.mark.parametrize(
    "filters", [None, {}, {"access_acl": []}, {"source": "manual"}]
)
def test_no_acl_filter_skips_visibility_walk(monkeypatch, filters):
    def unexpected_acl(*args):
        raise AssertionError("No ACL traversal is needed for an unrestricted query")

    monkeypatch.setattr(
        "src.internal.document_index.decision_retriever.acl_allows", unexpected_acl
    )
    run = DecisionRetriever([{"text": "answer"}], RecordingModel()).navigate(
        "q", filters=filters
    )
    assert run.results[0]["document"]["text"] == "answer"
