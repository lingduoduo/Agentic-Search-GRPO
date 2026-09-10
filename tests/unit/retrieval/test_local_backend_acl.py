"""The full ``RetrievalService`` backend must honour ``access_acl`` with the
shared rule, not treat it as a metadata equality key (which matched nothing,
so a filtered request returned no documents and an unfiltered one was never
ACL-checked)."""

from __future__ import annotations

import pytest

from src.internal.retrieval.backends.local import LocalBackend


def _rows() -> list[dict]:
    return [
        {"document": {"id": "public", "contents": "p"}, "score": 0.9},
        {
            "document": {
                "id": "nested-a",
                "contents": "n",
                "metadata": {"acl": ["user:a"]},
            },
            "score": 0.8,
        },
        {
            "document": {"id": "flat-a", "contents": "f", "acl": "user:a"},
            "score": 0.7,
        },
        {
            "document": {
                "id": "nested-b",
                "contents": "b",
                "metadata": {"acl": ["user:b"]},
            },
            "score": 0.6,
        },
        {
            "document": {
                "id": "lang-fr",
                "contents": "l",
                "lang": "fr",
                "metadata": {"acl": ["user:a"]},
            },
            "score": 0.5,
        },
    ]


class _FakeSparse:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def retrieve(self, queries, topk):
        return [self._rows[:topk] for _ in queries]


@pytest.fixture
def backend(monkeypatch) -> LocalBackend:
    import src.internal.retrieval.backends.local as local_mod

    monkeypatch.setattr(
        local_mod, "_make_sparse_retriever", lambda cfg: _FakeSparse(_rows())
    )
    from src.internal.document_index.retrieval import SparseRetrieverConfig

    return LocalBackend(SparseRetrieverConfig(index_path="x", corpus_path="y"))


def test_access_acl_keeps_public_and_matching_drops_disjoint(backend):
    results = backend.search_sparse("q", top_k=10, filters={"access_acl": ["user:a"]})
    assert [r.doc_id for r in results] == ["public", "nested-a", "flat-a", "lang-fr"]


def test_access_acl_combines_with_equality_keys(backend):
    results = backend.search_sparse(
        "q", top_k=10, filters={"access_acl": ["user:a"], "lang": "fr"}
    )
    assert [r.doc_id for r in results] == ["lang-fr"]


def test_no_filters_returns_everything(backend):
    results = backend.search_sparse("q", top_k=10, filters=None)
    assert len(results) == 5
