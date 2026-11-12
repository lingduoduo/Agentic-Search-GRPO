"""Managing datasets through /api/datasets: the store, the exported corpus
file, the manifest entry, and the in-process index rebuild."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.configs import load_app_settings
from src.internal.db import AgenticSearchStore
from src.internal.servers.retrieval import corpus_registry, demo
from src.internal.servers.web import datasets as datasets_api
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app

LEGACY_DOCS = [
    {
        "id": "doc_001",
        "title": "Dense Retrieval",
        "contents": "Vectors.",
        "metadata": {"acl": ["public"], "source": "Vector Docs"},
    },
    {"id": "doc_002", "title": "BM25", "contents": "Sparse ranking."},
]


def _read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def files(tmp_path, monkeypatch):
    """A manifest with one registered corpus ('legacy') and one whose file is
    gone ('gone'); managed datasets export under tmp_path/datasets."""
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text("".join(json.dumps(d) + "\n" for d in LEGACY_DOCS))
    manifest = tmp_path / "corpora.json"
    manifest.write_text(
        json.dumps(
            {
                "legacy": {
                    "path": str(legacy),
                    "docs": 2,
                    "domain": "ml",
                    "source": "Legacy KB",
                },
                "gone": {"path": str(tmp_path / "gone.jsonl"), "docs": 1},
            }
        )
    )
    monkeypatch.setattr(corpus_registry, "DEFAULT_MANIFEST_PATH", str(manifest))
    monkeypatch.setattr(datasets_api, "DATASETS_DIR", str(tmp_path / "datasets"))
    return {
        "manifest": manifest,
        "legacy": legacy,
        "export": lambda name: tmp_path / "datasets" / f"{name}.jsonl",
    }


def _manifest(files):
    return json.loads(files["manifest"].read_text())


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
def admin(monkeypatch, files):
    client = _client(monkeypatch)
    _sign_in(client, "admin@example.test")  # the first account is the admin
    return client


def _documents(client, name):
    return client.get(f"/api/datasets/{name}/documents").json()["list"]


# --- datasets --------------------------------------------------------------


def test_creates_a_dataset_and_registers_its_empty_corpus(admin, files):
    response = admin.post(
        "/api/datasets",
        json={"name": "kb", "description": " Team docs ", "source": "Team KB"},
    )

    assert response.status_code == 201
    created = response.json()
    assert created["managed"] is True
    assert created["description"] == "Team docs"
    assert created["source"] == "Team KB"
    assert created["docs"] == 0
    assert created["available"] is True
    assert created["updated_at"]
    assert _manifest(files)["kb"] == {
        "path": str(files["export"]("kb")),
        "docs": 0,
        "domain": None,
        "source": "Team KB",
    }
    assert _read(files["export"]("kb")) == []
    names = [d["name"] for d in admin.get("/api/datasets").json()["list"]]
    assert names == ["gone", "kb", "legacy"]


@pytest.mark.parametrize("name", ["", "Has Caps", "a,b", "../up", "-lead", "x" * 65])
def test_rejects_a_name_that_cannot_be_a_corpus_key(admin, name):
    assert admin.post("/api/datasets", json={"name": name}).status_code == 422


def test_rejects_a_taken_name(admin):
    assert admin.post("/api/datasets", json={"name": "legacy"}).status_code == 409
    assert admin.post("/api/datasets", json={"name": "kb"}).status_code == 201
    taken = admin.post("/api/datasets", json={"name": "kb"})
    assert taken.status_code == 409
    assert taken.json()["detail"] == "That name is taken."


def test_updates_only_the_fields_sent(admin, files):
    admin.post(
        "/api/datasets",
        json={"name": "kb", "description": "Docs", "domain": "ml", "source": "KB"},
    )

    response = admin.patch("/api/datasets/kb", json={"domain": "nlp", "source": " "})

    assert response.status_code == 200
    updated = response.json()
    assert updated["description"] == "Docs"
    assert updated["domain"] == "nlp"
    assert updated["source"] == "kb"  # cleared: falls back to the name
    assert _manifest(files)["kb"]["domain"] == "nlp"
    assert _manifest(files)["kb"]["source"] == "kb"


def test_an_unmanaged_corpus_must_be_imported_before_it_changes(admin):
    for response in (
        admin.patch("/api/datasets/legacy", json={"description": "x"}),
        admin.delete("/api/datasets/legacy"),
    ):
        assert response.status_code == 409
        assert response.json()["detail"] == "Import this dataset before changing it."
    assert admin.patch("/api/datasets/nope", json={}).status_code == 404
    assert admin.delete("/api/datasets/nope").status_code == 404


def test_deletes_a_created_dataset_and_its_corpus(admin, files):
    admin.post("/api/datasets", json={"name": "kb"})

    assert admin.delete("/api/datasets/kb").status_code == 204

    assert "kb" not in _manifest(files)
    assert not files["export"]("kb").exists()
    assert admin.get("/api/datasets/kb").status_code == 404


# --- import ----------------------------------------------------------------


def test_imports_a_registered_corpus(admin, files):
    response = admin.post("/api/datasets/legacy/import")

    assert response.status_code == 200
    imported = response.json()
    assert imported["managed"] is True
    assert imported["docs"] == 2
    assert imported["source"] == "Legacy KB"
    assert imported["domain"] == "ml"
    assert imported["path"] == str(files["export"]("legacy"))
    assert _manifest(files)["legacy"]["path"] == str(files["export"]("legacy"))
    # Exported as stored; a document without metadata gets an empty one.
    assert _read(files["export"]("legacy")) == [
        LEGACY_DOCS[0],
        {**LEGACY_DOCS[1], "metadata": {}},
    ]
    assert _read(files["legacy"]) == LEGACY_DOCS  # the original is untouched
    docs = _documents(admin, "legacy")
    assert [d["id"] for d in docs] == ["doc_001", "doc_002"]
    assert all(d["document_id"] and d["enabled"] for d in docs)
    assert docs[1]["source"] == "Legacy KB"  # no source of its own


def test_import_refuses_a_managed_or_missing_corpus(admin):
    admin.post("/api/datasets/legacy/import")

    assert admin.post("/api/datasets/legacy/import").status_code == 409
    gone = admin.post("/api/datasets/gone/import")
    assert gone.status_code == 409
    assert gone.json()["detail"] == "The dataset's file is missing on the server."
    assert admin.post("/api/datasets/nope/import").status_code == 404


def test_deleting_an_imported_dataset_registers_its_original_again(admin, files):
    admin.post("/api/datasets/legacy/import")

    assert admin.delete("/api/datasets/legacy").status_code == 204

    restored = admin.get("/api/datasets/legacy").json()
    assert restored["managed"] is False
    assert restored["path"] == str(files["legacy"])
    assert restored["docs"] == 2
    assert not files["export"]("legacy").exists()


# --- documents -------------------------------------------------------------


def test_renames_disables_and_deletes_documents(admin, files):
    admin.post("/api/datasets/legacy/import")
    first, second = _documents(admin, "legacy")
    url = "/api/datasets/legacy/documents/{}".format

    renamed = admin.patch(url(first["document_id"]), json={"title": " Dense "})
    disabled = admin.patch(url(second["document_id"]), json={"enabled": False})

    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Dense"
    assert disabled.json()["enabled"] is False
    assert admin.get(url(second["document_id"])).json()["enabled"] is False
    # Search gets only the enabled documents, as renamed.
    [exported] = _read(files["export"]("legacy"))
    assert (exported["id"], exported["title"]) == ("doc_001", "Dense")
    assert _manifest(files)["legacy"]["docs"] == 1
    # The store keeps both.
    assert admin.get("/api/datasets/legacy").json()["docs"] == 2

    assert admin.delete(url(first["document_id"])).status_code == 204
    assert [d["id"] for d in _documents(admin, "legacy")] == ["doc_002"]
    assert _read(files["export"]("legacy")) == []


def test_document_routes_need_a_managed_dataset_and_a_real_document(admin):
    url = "/api/datasets/{}/documents/{}".format
    assert admin.get(url("legacy", "x")).status_code == 409
    admin.post("/api/datasets/legacy/import")

    assert admin.get(url("legacy", "x")).status_code == 404
    assert admin.patch(url("legacy", "x"), json={"enabled": False}).status_code == 404
    assert admin.delete(url("legacy", "x")).status_code == 404
    doc = _documents(admin, "legacy")[0]["document_id"]
    assert admin.patch(url("legacy", doc), json={"title": ""}).status_code == 422


def test_management_requires_an_admin(monkeypatch, files):
    client = _client(monkeypatch)
    _sign_in(client, "admin@example.test")
    client.post("/auth/logout")
    _sign_in(client, "user@example.test")

    assert client.post("/api/datasets", json={"name": "kb"}).status_code == 403
    assert client.post("/api/datasets/legacy/import").status_code == 403


# --- in-process retrieval --------------------------------------------------


def test_a_change_rebuilds_the_in_process_index_that_serves_it(monkeypatch, files):
    builds: list[list[str]] = []

    def from_docs(docs):
        builds.append([d["id"] for d in docs])
        return object()

    monkeypatch.setattr(demo.TfidfRetriever, "from_docs", from_docs)
    monkeypatch.setattr(demo, "create_app", lambda retriever: FastAPI())
    # Startup reads the same manifest the router writes.
    resolve = corpus_registry.resolve_corpus_docs
    monkeypatch.setattr(
        corpus_registry,
        "resolve_corpus_docs",
        lambda spec: resolve(spec, manifest=_manifest(files)),
    )
    client = _client(monkeypatch, {"AGENTIC_SEARCH_EMBEDDED_RETRIEVAL": "legacy"})
    _sign_in(client, "admin@example.test")
    assert builds == [["doc_001", "doc_002"]]

    client.post("/api/datasets/legacy/import")
    doc = _documents(client, "legacy")[1]["document_id"]
    client.patch(f"/api/datasets/legacy/documents/{doc}", json={"enabled": False})
    client.post("/api/datasets", json={"name": "other"})  # not served: no rebuild

    assert builds[-1] == ["doc_001"]
    assert len(builds) == 3  # startup, import, disable


def test_a_failed_rebuild_keeps_the_change(monkeypatch, files):
    monkeypatch.setattr(demo.TfidfRetriever, "from_docs", lambda docs: object())
    monkeypatch.setattr(demo, "create_app", lambda retriever: FastAPI())
    calls = {"n": 0}

    def resolve(spec):
        calls["n"] += 1
        if calls["n"] > 1:
            raise ValueError("empty vocabulary")
        return []

    monkeypatch.setattr(corpus_registry, "resolve_corpus_docs", resolve)
    client = _client(monkeypatch, {"AGENTIC_SEARCH_EMBEDDED_RETRIEVAL": "legacy"})
    _sign_in(client, "admin@example.test")

    assert client.post("/api/datasets/legacy/import").status_code == 200
    assert client.get("/api/datasets/legacy").json()["managed"] is True
