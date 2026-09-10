"""The one ACL rule every serving part shares.

Mirrors ``SearchFilters.matches`` (``src/context/models.py``): a request's
``access_acl`` must intersect the document's declared ``acl``; a document that
declares no ACL is public. Kept torch-free and free of web-layer imports so
the retrieval servers and backends can use it directly.
"""

from __future__ import annotations


def acl_allows(metadata: dict | None, filters: dict | None) -> bool:
    """Whether a document with *metadata* is readable under *filters*."""
    if not filters:
        return True
    allowed = filters.get("access_acl")
    if not allowed:
        return True
    declared = (metadata or {}).get("acl")
    if not declared:
        return True
    if isinstance(declared, str):
        declared = [declared]
    return bool(set(declared) & set(allowed))
