"""Identity decides entitlement, in exactly one place."""

from __future__ import annotations

from src.internal.access.capabilities import RequestCapabilities, resolve_capabilities
from src.internal.auth.users import AuthenticatedUser


class _Store:
    """Minimal stand-in for AgenticSearchStore's memory reads."""

    def __init__(self, memories=None):
        self._memories = memories or {}

    def get_user_memories(self, user_id):
        return self._memories.get(user_id, [])


def test_anonymous_is_public_only():
    caps = resolve_capabilities(None, _Store())
    assert caps == RequestCapabilities(
        user_id=None, access_acl=["public"], memory_preamble=""
    )


def test_anonymous_user_object_is_treated_as_anonymous():
    user = AuthenticatedUser(id="anon", is_anonymous=True)
    assert resolve_capabilities(user, _Store()).user_id is None


def test_signed_in_user_gets_public_plus_their_own_entries():
    user = AuthenticatedUser(id="u1", email="a@b.c", group_ids=frozenset({"g1"}))
    caps = resolve_capabilities(user, _Store())
    assert caps.user_id == "u1"
    assert set(caps.access_acl) == {"public", "user:u1", "email:a@b.c", "group:g1"}


def test_access_acl_is_never_empty():
    # An empty list would read as "no filter" downstream, which is the hole
    # this resolver exists to close.
    for user in (None, AuthenticatedUser(id="u1")):
        assert resolve_capabilities(user, _Store()).access_acl


def test_signed_in_user_gets_their_memory():
    store = _Store({"u1": ["prefers hybrid retrieval"]})
    caps = resolve_capabilities(AuthenticatedUser(id="u1"), store)
    assert "prefers hybrid retrieval" in caps.memory_preamble


def test_anonymous_gets_no_memory_even_if_the_store_has_some():
    store = _Store({"u1": ["prefers hybrid retrieval"]})
    assert resolve_capabilities(None, store).memory_preamble == ""


def test_a_store_failure_degrades_to_no_memory():
    # The store is the source of truth (#476); a user whose row is gone must
    # not take the request down with it.
    class _Broken:
        def get_user_memories(self, user_id):
            raise RuntimeError("row is gone")

    caps = resolve_capabilities(AuthenticatedUser(id="u1"), _Broken())
    assert caps.memory_preamble == ""
    assert caps.user_id == "u1"


def test_query_and_encoder_are_forwarded_to_the_preamble(monkeypatch):
    seen: dict = {}

    def fake_preamble(store, user_id, *, query=None, encoder=None, **kw):
        seen.update(user_id=user_id, query=query, encoder=encoder)
        return "pre"

    monkeypatch.setattr(
        "src.internal.access.capabilities.memory_preamble", fake_preamble
    )
    sentinel = object()
    caps = resolve_capabilities(
        AuthenticatedUser(id="u1"), _Store(), query="thai", encoder=sentinel
    )
    assert caps.memory_preamble == "pre"
    assert seen == {"user_id": "u1", "query": "thai", "encoder": sentinel}


def test_anonymous_never_reaches_the_preamble_even_with_a_query(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("preamble must not run for anonymous")

    monkeypatch.setattr("src.internal.access.capabilities.memory_preamble", boom)
    assert resolve_capabilities(None, _Store(), query="thai").memory_preamble == ""
