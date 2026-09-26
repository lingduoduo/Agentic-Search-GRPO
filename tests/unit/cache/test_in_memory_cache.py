"""InMemoryCache expiry: the TTL semantics Redis gives, in-process."""

from __future__ import annotations

import pytest

from src.internal.cache.interface import (
    TTL_KEY_NOT_FOUND,
    TTL_NO_EXPIRY,
    InMemoryCache,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def cache(clock: FakeClock) -> InMemoryCache:
    return InMemoryCache(clock=clock)


def test_get_returns_the_value_before_expiry_and_none_after(cache, clock):
    cache.set("k", "v", ex=10)
    clock.now += 9
    assert cache.get("k") == b"v"
    clock.now += 1
    assert cache.get("k") is None


def test_key_is_expired_exactly_at_its_deadline(cache, clock):
    cache.set("k", "v", ex=10)
    clock.now += 10
    assert cache.exists("k") is False


def test_exists_and_ttl_after_expiry(cache, clock):
    cache.set("k", "v", ex=5)
    assert cache.exists("k") is True
    assert cache.ttl("k") == 5
    clock.now += 6
    assert cache.exists("k") is False
    assert cache.ttl("k") == TTL_KEY_NOT_FOUND


def test_ttl_reports_remaining_whole_seconds_rounded_up(cache, clock):
    cache.set("k", "v", ex=100)
    clock.now += 0.5
    assert cache.ttl("k") == 100


def test_an_expired_key_is_deleted_on_read(cache, clock):
    cache.set("k", "v", ex=1)
    clock.now += 2
    assert cache.get("k") is None
    assert "k" not in cache._store
    assert "k" not in cache._expires


def test_expire_on_an_existing_key_sets_its_expiry(cache, clock):
    cache.set("k", "v")
    cache.expire("k", 3)
    assert cache.ttl("k") == 3
    clock.now += 3
    assert cache.get("k") is None


def test_expire_on_a_missing_key_does_nothing(cache, clock):
    cache.expire("ghost", 3)
    assert cache.ttl("ghost") == TTL_KEY_NOT_FOUND
    cache.set("ghost", "v")
    assert cache.ttl("ghost") == TTL_NO_EXPIRY


def test_ex_none_never_expires(cache, clock):
    cache.set("k", "v", ex=None)
    clock.now += 10**9
    assert cache.get("k") == b"v"
    assert cache.ttl("k") == TTL_NO_EXPIRY


def test_set_without_ex_clears_an_existing_expiry(cache, clock):
    cache.set("k", "v1", ex=5)
    cache.set("k", "v2")
    clock.now += 10
    assert cache.get("k") == b"v2"
    assert cache.ttl("k") == TTL_NO_EXPIRY


def test_delete_clears_the_expiry(cache, clock):
    cache.set("k", "v", ex=5)
    cache.delete("k")
    cache.set("k", "v")
    clock.now += 10
    assert cache.get("k") == b"v"
    assert cache.ttl("k") == TTL_NO_EXPIRY


def test_default_clock_is_monotonic():
    cache = InMemoryCache()
    cache.set("k", "v", ex=60)
    assert cache.get("k") == b"v"
    assert 0 < cache.ttl("k") <= 60
