"""`acl_allows` is the one ACL rule every serving part shares: intersect the
request's ``access_acl`` with the document's declared ``acl``; a document that
declares nothing is public. Mirrors ``SearchFilters.matches``."""

from __future__ import annotations

from src.internal.retrieval.acl import acl_allows


def test_no_filters_allows():
    assert acl_allows({"acl": ["user:b"]}, None)
    assert acl_allows({"acl": ["user:b"]}, {})


def test_filters_without_access_acl_allow():
    assert acl_allows({"acl": ["user:b"]}, {"source": "confluence"})


def test_undeclared_acl_is_public():
    assert acl_allows({}, {"access_acl": ["user:a"]})
    assert acl_allows(None, {"access_acl": ["user:a"]})
    assert acl_allows({"acl": []}, {"access_acl": ["user:a"]})


def test_string_acl_intersecting_allows():
    assert acl_allows({"acl": "user:a"}, {"access_acl": ["public", "user:a"]})


def test_disjoint_acl_denies():
    assert not acl_allows({"acl": ["user:b"]}, {"access_acl": ["public", "user:a"]})
    assert not acl_allows({"acl": "user:b"}, {"access_acl": ["user:a"]})


def test_demo_server_wrapper_reads_document_metadata():
    from src.internal.servers.retrieval.demo import _allowed_by_acl

    doc = {"id": "d1", "metadata": {"acl": ["user:b"]}}
    assert not _allowed_by_acl(doc, {"access_acl": ["user:a"]})
    assert _allowed_by_acl({"id": "d2"}, {"access_acl": ["user:a"]})
