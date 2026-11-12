"""/api/datasets: the corpora registered in data/corpora.json and their documents."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.configs import load_app_settings
from src.internal.db import AgenticSearchStore
from src.internal.servers.retrieval import corpus_registry, demo
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


def _client(monkeypatch, env=None):
    monkeypatch.setenv("AGENTIC_SEARCH_DEV_ADMIN", "false")
    return TestClient(
        create_web_app(
            SearchExperienceSettings(
                db_path=":memory:", search_url="http://localhost:8001/retrieve"
            ),
            app_settings=load_app_settings(env or {}),
            store=AgenticSearchStore(":memory:"),
        )
    )


def _sign_in(client, email):
    credentials = {"email": email, "password": "test-only-password"}
    client.post("/auth/register", json=credentials)
    client.post(
        "/auth/login",
        data={"username": credentials["email"], "password": credentials["password"]},
    )


def _admin(monkeypatch, env=None):
    client = _client(monkeypatch, env)
    _sign_in(client, "admin@example.test")  # the first account is the admin
    return client


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    present = tmp_path / "demo.jsonl"
    present.write_text(
        "\n".join(
            json.dumps(doc)
            for doc in [
                {
                    "id": "doc_001",
                    "title": "Dense Retrieval with FAISS",
                    "contents": "Dense retrieval encodes queries as vectors.",
                    "metadata": {"acl": ["public"], "source": "Vector Search Docs"},
                },
                {"id": 2, "title": "BM25", "contents": "Sparse lexical ranking."},
                {"id": "doc_003", "title": "Rerankers", "contents": "Cross-encoders."},
            ]
        )
        + "\n\n"
    )
    entries = {
        "scifact": {
            "path": str(tmp_path / "missing.jsonl"),
            "docs": 5183,
            "domain": "scientific",
            "source": "SciFact Corpus",
        },
        "demo": {"path": str(present), "docs": 3, "domain": "retrieval/ML"},
    }
    monkeypatch.setattr(corpus_registry, "load_manifest", lambda *a, **k: entries)
    return entries


def test_pages_registered_corpora_for_an_admin(monkeypatch, manifest):
    response = _admin(monkeypatch).get("/api/datasets")

    assert response.status_code == 200
    assert response.json() == {
        "list": [
            {
                "name": "demo",
                "path": manifest["demo"]["path"],
                "docs": 3,
                "domain": "retrieval/ML",
                "source": "demo",  # no label: falls back to the name
                "available": True,
                "served": None,  # a separate retrieval server answers
            },
            {
                "name": "scifact",
                "path": manifest["scifact"]["path"],
                "docs": 5183,
                "domain": "scientific",
                "source": "SciFact Corpus",
                "available": False,
                "served": None,
            },
        ],
        "paginator": {
            "current_page": 1,
            "page_size": 20,
            "total_page": 1,
            "total_record": 2,
        },
    }


def test_pages_and_searches_the_corpora(monkeypatch, manifest):
    client = _admin(monkeypatch)

    second = client.get("/api/datasets", params={"page_size": 1, "current_page": 2})
    assert [d["name"] for d in second.json()["list"]] == ["scifact"]
    assert second.json()["paginator"]["total_page"] == 2

    found = client.get("/api/datasets", params={"search_word": " Scientific "})
    assert [d["name"] for d in found.json()["list"]] == ["scifact"]
    assert found.json()["paginator"]["total_record"] == 1


def test_flags_the_corpora_the_embedded_retriever_serves(monkeypatch, manifest):
    # Skip building the in-process retriever; only its corpus spec matters here.
    monkeypatch.setattr(corpus_registry, "resolve_corpus_docs", lambda spec: [])
    monkeypatch.setattr(demo.TfidfRetriever, "from_docs", lambda docs: object())
    monkeypatch.setattr(demo, "create_app", lambda retriever: FastAPI())
    client = _admin(monkeypatch, {"AGENTIC_SEARCH_EMBEDDED_RETRIEVAL": "demo"})

    page = client.get("/api/datasets").json()

    assert {d["name"]: d["served"] for d in page["list"]} == {
        "demo": True,
        "scifact": False,
    }


def test_accepts_a_path_only_manifest_entry(monkeypatch):
    monkeypatch.setattr(
        corpus_registry, "load_manifest", lambda *a, **k: {"old": "nowhere.jsonl"}
    )

    [dataset] = _admin(monkeypatch).get("/api/datasets").json()["list"]

    assert dataset["path"] == "nowhere.jsonl"
    assert dataset["docs"] is None
    assert dataset["available"] is False


def test_reports_an_unparseable_manifest(monkeypatch):
    def broken(*a, **k):
        raise json.JSONDecodeError("Expecting value", "", 0)

    monkeypatch.setattr(corpus_registry, "load_manifest", broken)

    response = _admin(monkeypatch).get("/api/datasets")

    assert response.status_code == 500
    assert response.json()["detail"] == "The corpus manifest could not be parsed."


def test_gets_one_corpus(monkeypatch, manifest):
    client = _admin(monkeypatch)

    assert client.get("/api/datasets/scifact").json()["source"] == "SciFact Corpus"
    missing = client.get("/api/datasets/nope")
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Dataset not found."


def test_pages_a_corpus_documents(monkeypatch, manifest):
    response = _admin(monkeypatch).get(
        "/api/datasets/demo/documents", params={"page_size": 2}
    )

    assert response.status_code == 200
    assert response.json() == {
        "list": [
            {
                "id": "doc_001",
                "title": "Dense Retrieval with FAISS",
                "contents": "Dense retrieval encodes queries as vectors.",
                "source": "Vector Search Docs",  # the document's own source wins
                "acl": ["public"],
            },
            {
                "id": "2",  # ids are strings in the response
                "title": "BM25",
                "contents": "Sparse lexical ranking.",
                "source": "demo",  # falls back to the corpus label
                "acl": [],
            },
        ],
        "paginator": {
            "current_page": 1,
            "page_size": 2,
            "total_page": 2,
            "total_record": 3,
        },
    }


def test_searches_document_titles_and_contents(monkeypatch, manifest):
    client = _admin(monkeypatch)

    by_title = client.get(
        "/api/datasets/demo/documents", params={"search_word": "rerank"}
    )
    by_contents = client.get(
        "/api/datasets/demo/documents", params={"search_word": "LEXICAL"}
    )

    assert [d["id"] for d in by_title.json()["list"]] == ["doc_003"]
    assert [d["id"] for d in by_contents.json()["list"]] == ["2"]


def test_documents_of_unknown_or_missing_corpora(monkeypatch, manifest):
    client = _admin(monkeypatch)

    assert client.get("/api/datasets/nope/documents").status_code == 404
    missing = client.get("/api/datasets/scifact/documents")
    assert missing.status_code == 409
    assert missing.json()["detail"] == "The dataset's file is missing on the server."


@pytest.mark.parametrize(
    "path", ["/api/datasets", "/api/datasets/demo", "/api/datasets/demo/documents"]
)
def test_requires_an_admin(monkeypatch, manifest, path):
    client = _client(monkeypatch)
    _sign_in(client, "admin@example.test")
    client.post("/auth/logout")
    _sign_in(client, "user@example.test")

    assert client.get(path).status_code == 403


def test_requires_sign_in(monkeypatch, manifest):
    assert _client(monkeypatch).get("/api/datasets").status_code == 401
