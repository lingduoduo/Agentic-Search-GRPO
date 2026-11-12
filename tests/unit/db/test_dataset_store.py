"""Managed knowledge-base datasets and documents in the SQLite store."""

from __future__ import annotations

import sqlite3

import pytest

from src.internal.db.store import AgenticSearchStore


@pytest.fixture
def store():
    with AgenticSearchStore(":memory:") as s:
        yield s


def _docs(*titles):
    return [
        {"id": f"d{i}", "title": t, "contents": f"About {t}", "metadata": {"acl": []}}
        for i, t in enumerate(titles, start=1)
    ]


def test_creates_lists_and_updates_a_dataset(store):
    created = store.create_dataset("kb", description="Docs", source="KB")

    assert created.name == "kb"
    assert created.description == "Docs"
    assert created.domain is None
    assert [d.name for d in store.list_datasets()] == ["kb"]

    assert store.update_dataset(created.id, domain="ml", description="New")
    updated = store.get_dataset_by_name("kb")
    assert (updated.description, updated.domain, updated.source) == ("New", "ml", "KB")
    assert updated.updated_at >= created.updated_at


def test_a_dataset_name_is_unique(store):
    store.create_dataset("kb")

    with pytest.raises(sqlite3.IntegrityError):
        store.create_dataset("kb")


def test_appends_documents_in_order_and_pages_them(store):
    ds = store.create_dataset("kb")
    store.add_dataset_documents(ds.id, _docs("Alpha", "Beta"))
    store.add_dataset_documents(ds.id, [{"id": 7, "text": "plain text field"}])

    page, total = store.list_dataset_documents(ds.id, offset=1, limit=2)

    assert total == 3
    assert [(d.position, d.title) for d in page] == [(1, "Beta"), (2, None)]
    assert page[1].doc_id == "7"
    assert page[1].contents == "plain text field"
    assert store.count_dataset_documents(ds.id) == 3


def test_searches_titles_and_contents_ignoring_case(store):
    ds = store.create_dataset("kb")
    store.add_dataset_documents(ds.id, _docs("Alpha", "Beta%"))

    by_title, _ = store.list_dataset_documents(ds.id, search_word=" ALPHA ")
    by_contents, total = store.list_dataset_documents(ds.id, search_word="about beta%")

    assert [d.title for d in by_title] == ["Alpha"]
    assert [d.title for d in by_contents] == ["Beta%"]  # % is literal, not LIKE
    assert total == 1


def test_renames_disables_and_deletes_a_document(store):
    ds = store.create_dataset("kb")
    store.add_dataset_documents(ds.id, _docs("Alpha", "Beta"))
    alpha, beta = store.list_dataset_documents(ds.id)[0]

    assert store.update_dataset_document(ds.id, alpha.id, title="Renamed")
    assert store.update_dataset_document(ds.id, beta.id, enabled=False)
    assert store.get_dataset_document(ds.id, alpha.id).title == "Renamed"
    assert [d.title for d in store.enabled_dataset_documents(ds.id)] == ["Renamed"]

    assert store.delete_dataset_document(ds.id, alpha.id)
    assert store.enabled_dataset_documents(ds.id) == []
    assert not store.delete_dataset_document(ds.id, alpha.id)


def test_document_calls_are_scoped_to_their_dataset(store):
    first = store.create_dataset("a")
    second = store.create_dataset("b")
    store.add_dataset_documents(first.id, _docs("Alpha"))
    [doc], _ = store.list_dataset_documents(first.id)

    assert store.get_dataset_document(second.id, doc.id) is None
    assert not store.update_dataset_document(second.id, doc.id, enabled=False)
    assert not store.delete_dataset_document(second.id, doc.id)


def test_deleting_a_dataset_deletes_its_documents(store):
    ds = store.create_dataset("kb")
    store.add_dataset_documents(ds.id, _docs("Alpha"))

    assert store.delete_dataset(ds.id)

    assert store.get_dataset_by_name("kb") is None
    assert store.count_dataset_documents(ds.id) == 0


def test_tracks_an_upload_batch_per_file(store):
    ds = store.create_dataset("kb")

    waiting = store.create_dataset_upload_batch(ds.id, ["a.txt", "b.jsonl"])

    assert [(u.filename, u.status, u.position) for u in waiting] == [
        ("a.txt", "waiting", 0),
        ("b.jsonl", "waiting", 1),
    ]
    batch = waiting[0].batch
    assert waiting[1].batch == batch
    store.set_dataset_upload_status(waiting[0].id, "completed", document_count=3)
    store.set_dataset_upload_status(waiting[1].id, "error", error="Line 2: bad")
    done = store.list_dataset_upload_batch(ds.id, batch)
    assert [(u.status, u.document_count, u.error) for u in done] == [
        ("completed", 3, None),
        ("error", 0, "Line 2: bad"),
    ]
    assert store.list_dataset_upload_batch(store.create_dataset("other").id, batch) == []


def test_records_search_test_queries_newest_first_per_dataset(store):
    store.add_dataset_query("kb", "first", top_k=5, result_count=2)
    store.add_dataset_query("other", "elsewhere", top_k=5, result_count=0)
    second = store.add_dataset_query("kb", "second", top_k=3, result_count=1)

    recent = store.list_dataset_queries("kb")

    assert [q.query for q in recent] == ["second", "first"]
    assert (recent[0].top_k, recent[0].result_count) == (3, 1)
    assert recent[0].created_at == second.created_at
    assert [q.query for q in store.list_dataset_queries("kb", limit=1)] == ["second"]
