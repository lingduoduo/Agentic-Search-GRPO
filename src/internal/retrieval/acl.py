"""The one ACL rule every serving part shares.

Mirrors ``SearchFilters.matches`` (``src/context/models.py``): a request's
``access_acl`` must intersect the document's declared ACL, read from
``metadata["acl"]`` and ``metadata["tags"]["acl"]``; a document that declares
no ACL is public. Kept torch-free and free of web-layer imports so the
retrieval servers and backends can use it directly.
"""

from __future__ import annotations


def _declared_acl(metadata: dict | None) -> set[str]:
    metadata = metadata or {}
    values = _as_set(metadata.get("acl"))
    tags = metadata.get("tags")
    if isinstance(tags, dict):
        values |= _as_set(tags.get("acl"))
    return values


def _as_set(value: object) -> set[str]:
    if not value:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {str(v) for v in value}
    return {str(value)}


def acl_allows(metadata: dict | None, filters: dict | None) -> bool:
    """Whether a document with *metadata* is readable under *filters*."""
    if not filters:
        return True
    allowed = filters.get("access_acl")
    if not allowed:
        return True
    declared = _declared_acl(metadata)
    if not declared:
        return True
    return bool(declared & set(allowed))
