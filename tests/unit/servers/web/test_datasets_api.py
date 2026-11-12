"""GET /api/datasets: the corpora registered in data/corpora.json."""

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


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    present = tmp_path / "demo.jsonl"
    present.write_text("{}\n")
    entries = {
        "scifact": {
            "path": str(tmp_path / "missing.jsonl"),
            "docs": 5183,
            "domain": "scientific",
            "source": "SciFact Corpus",
        },
        "demo": {"path": str(present), "docs": 20, "domain": "retrieval/ML"},
    }
    monkeypatch.setattr(corpus_registry, "load_manifest", lambda *a, **k: entries)
    return entries


def test_lists_registered_corpora_for_an_admin(monkeypatch, manifest):
    client = _client(monkeypatch)
    _sign_in(client, "admin@example.test")  # the first account is the admin

    response = client.get("/api/datasets")

    assert response.status_code == 200
    assert response.json() == [
        {
            "name": "demo",
            "path": manifest["demo"]["path"],
            "docs": 20,
            "domain": "retrieval/ML",
            "source": "demo",  # no label: falls back to the name, as at search time
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
    ]


def test_flags_the_corpora_the_embedded_retriever_serves(monkeypatch, manifest):
    # Skip building the in-process retriever; only its corpus spec matters here.
    monkeypatch.setattr(corpus_registry, "resolve_corpus_docs", lambda spec: [])
    monkeypatch.setattr(demo.TfidfRetriever, "from_docs", lambda docs: object())
    monkeypatch.setattr(demo, "create_app", lambda retriever: FastAPI())
    client = _client(monkeypatch, {"AGENTIC_SEARCH_EMBEDDED_RETRIEVAL": "demo"})
    _sign_in(client, "admin@example.test")

    served = {d["name"]: d["served"] for d in client.get("/api/datasets").json()}

    assert served == {"demo": True, "scifact": False}


def test_accepts_a_path_only_manifest_entry(monkeypatch):
    monkeypatch.setattr(
        corpus_registry, "load_manifest", lambda *a, **k: {"old": "nowhere.jsonl"}
    )
    client = _client(monkeypatch)
    _sign_in(client, "admin@example.test")

    [dataset] = client.get("/api/datasets").json()

    assert dataset["path"] == "nowhere.jsonl"
    assert dataset["docs"] is None
    assert dataset["available"] is False


def test_reports_an_unparseable_manifest(monkeypatch):
    def broken(*a, **k):
        raise json.JSONDecodeError("Expecting value", "", 0)

    monkeypatch.setattr(corpus_registry, "load_manifest", broken)
    client = _client(monkeypatch)
    _sign_in(client, "admin@example.test")

    response = client.get("/api/datasets")

    assert response.status_code == 500
    assert response.json()["detail"] == "The corpus manifest could not be parsed."


def test_requires_an_admin(monkeypatch, manifest):
    client = _client(monkeypatch)
    _sign_in(client, "admin@example.test")
    client.post("/auth/logout")
    _sign_in(client, "user@example.test")

    assert client.get("/api/datasets").status_code == 403


def test_requires_sign_in(monkeypatch, manifest):
    assert _client(monkeypatch).get("/api/datasets").status_code == 401
