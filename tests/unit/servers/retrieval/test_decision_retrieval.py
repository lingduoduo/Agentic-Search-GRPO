from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.internal.document_index.decision_retriever import DecisionRetriever
from src.internal.servers.retrieval.decision import create_app, main, parse_args


class Model:
    def binary(self, query, candidates):
        assert all("classified" not in candidate.text for candidate in candidates)
        return [0.8] * len(candidates)

    def choice(self, query, candidates):
        return [1 / len(candidates)] * len(candidates)


@pytest.fixture
def client():
    return TestClient(
        create_app(
            DecisionRetriever(
                [
                    {"id": "open", "title": "Handbook", "contents": "public answer"},
                    {
                        "id": "closed",
                        "title": "classified",
                        "contents": "classified answer",
                        "metadata": {"acl": ["private"], "topic": "classified"},
                    },
                ],
                Model(),
            )
        )
    )


def test_standard_single_and_batch_response_shapes_and_trace(client):
    body = {"query": "question", "filters": {"access_acl": ["public"]}}
    single = client.post("/retrieve", json=body)
    assert single.status_code == 200
    assert single.json()["results"][0]["id"] == "open::passage-0"
    assert "trace" not in single.json()
    scored = client.post(
        "/retrieve", json={**body, "return_scores": True, "include_trace": True}
    )
    assert scored.status_code == 200
    assert scored.json()["results"][0]["score"] == 1.0
    assert [stage["kind"] for stage in scored.json()["trace"]] == [
        "topic",
        "document",
        "section",
        "passage",
        "choice",
    ]
    batch = client.post(
        "/retrieve",
        json={
            "queries": ["first", "second"],
            "filters": body["filters"],
            "include_trace": True,
        },
    )
    assert batch.status_code == 200
    assert len(batch.json()["results"]) == len(batch.json()["trace"]) == 2
    assert batch.json()["results"][0][0]["text"] == "public answer"
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"query": ""},
        {"queries": []},
        {"queries": [" "]},
        {"query": "q", "queries": ["q"]},
        {"query": "q", "topk": 0},
        {"query": "q", "topk": 21},
    ],
)
def test_invalid_requests_return_422(client, body):
    assert client.post("/retrieve", json=body).status_code == 422


def test_cli_exposes_model_and_navigation_configuration():
    with patch.object(
        sys,
        "argv",
        [
            "decision",
            "--corpus",
            "demo",
            "--model",
            "/models/local",
            "--beam-width",
            "3",
            "--shortlist-size",
            "10",
            "--passage-words",
            "100",
            "--device",
            "mps",
        ],
    ):
        args = parse_args()
    assert args.corpus == "demo"
    assert args.model == "/models/local"
    assert args.beam_width == 3
    assert args.shortlist_size == 10
    assert args.passage_words == 100
    assert args.device == "mps"


async def test_existing_search_client_can_consume_navigation_results(
    client, monkeypatch
):
    from src.context.retrieval.client import SearchClient, SearchClientConfig

    search = SearchClient(SearchClientConfig(url="http://decision.test/retrieve"))

    async def post(url, payload, action):
        assert action == "retrieve"
        response = client.post("/retrieve", json=payload)
        response.raise_for_status()
        return response.json()

    monkeypatch.setattr(search, "_post_json", post)
    monkeypatch.setattr("src.context.retrieval.client.serving_cache", lambda: None)
    rows = await search.retrieve(["question"], filters={"access_acl": ["public"]})
    assert len(rows) == len(rows[0]) == 1
    assert rows[0][0].contents == "public answer"
    assert rows[0][0].score == 1.0
    assert rows[0][0].metadata["document_id"] == "open"


def test_main_wires_corpus_model_configuration_and_http_app():
    argv = [
        "decision",
        "--corpus",
        "demo",
        "--model",
        "/models/local",
        "--beam-width",
        "3",
        "--port",
        "8123",
    ]
    with (
        patch.object(sys, "argv", argv),
        patch("src.internal.servers.retrieval.decision.load_environment"),
        patch(
            "src.internal.servers.retrieval.corpus_registry.resolve_corpus_docs",
            return_value=[{"id": "d", "contents": "public answer"}],
        ) as resolve,
        patch(
            "src.internal.document_index.decision_model.LocalDecisionModel.from_pretrained",
            return_value=Model(),
        ) as build,
        patch("src.internal.servers.retrieval.decision.run_uvicorn_app") as serve,
    ):
        main()
    resolve.assert_called_once_with("demo")
    build.assert_called_once_with(
        "/models/local",
        device="cpu",
        max_context_tokens=4096,
        inference_batch_size=1,
    )
    assert serve.call_args.kwargs == {"host": "127.0.0.1", "port": 8123}
    response = TestClient(serve.call_args.args[0]).post(
        "/retrieve", json={"query": "q"}
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["metadata"]["document_id"] == "d"
