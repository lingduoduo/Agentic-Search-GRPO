"""The process-local TTL cache behind the serving path."""

from __future__ import annotations

import pytest

from src.internal.cache.ttl_cache import TTLCache


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_miss_then_hit():
    cache = TTLCache(ttl_seconds=10)
    assert cache.get(("k",)) is None
    cache.set(("k",), [1, 2])
    assert cache.get(("k",)) == [1, 2]


def test_entry_expires_after_ttl():
    clock = _Clock()
    cache = TTLCache(ttl_seconds=10, clock=clock)
    cache.set("k", "v")
    clock.now += 9.9
    assert cache.get("k") == "v"
    clock.now += 0.2
    assert cache.get("k") is None


def test_least_recently_used_is_evicted_when_full():
    cache = TTLCache(ttl_seconds=10, max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1  # touch a; b is now the least recently used
    cache.set("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_stats_count_hits_and_misses():
    cache = TTLCache(ttl_seconds=10)
    cache.get("k")
    cache.set("k", 1)
    cache.get("k")
    cache.get("k")
    assert cache.stats() == {"hits": 2, "misses": 1, "size": 1, "stale_hits": 0}


def test_clear_empties_the_cache():
    cache = TTLCache(ttl_seconds=10)
    cache.set("k", 1)
    cache.clear()
    assert cache.get("k") is None


def test_in_grace_entry_is_a_miss_but_kept_and_served_stale():
    clock = _Clock()
    cache = TTLCache(ttl_seconds=10, stale_seconds=100, clock=clock)
    cache.set("k", "v")
    clock.now += 50
    assert cache.get("k") is None
    assert cache.stats()["size"] == 1
    assert cache.stats()["hits"] == 0
    assert cache.get_stale("k") == "v"
    assert cache.get("k") is None  # still not fresh


def test_get_stale_returns_fresh_entries_too():
    cache = TTLCache(ttl_seconds=10, stale_seconds=100)
    cache.set("k", "v")
    assert cache.get_stale("k") == "v"
    assert cache.get_stale("missing") is None


def test_beyond_grace_both_miss_and_entry_is_deleted():
    clock = _Clock()
    cache = TTLCache(ttl_seconds=10, stale_seconds=100, clock=clock)
    cache.set("k", "v")
    cache.set("j", "w")
    clock.now += 110
    assert cache.get("k") is None
    assert cache.get_stale("j") is None
    assert cache.get_stale("k") is None
    assert cache.stats()["size"] == 0


def test_zero_stale_seconds_is_todays_behavior():
    clock = _Clock()
    cache = TTLCache(ttl_seconds=10, clock=clock)
    cache.set("k", "v")
    clock.now += 10
    assert cache.get_stale("k") is None
    cache.set("j", "w")
    clock.now += 10
    assert cache.get("j") is None
    assert cache.stats()["size"] == 0


def test_stale_entries_count_toward_the_lru_bound():
    clock = _Clock()
    cache = TTLCache(ttl_seconds=10, max_entries=2, stale_seconds=100, clock=clock)
    cache.set("a", 1)
    clock.now += 20  # a is stale
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get_stale("a") is None
    assert cache.stats()["size"] == 2


def test_stale_hits_are_counted():
    clock = _Clock()
    cache = TTLCache(ttl_seconds=10, stale_seconds=100, clock=clock)
    cache.set("k", "v")
    clock.now += 20
    cache.get_stale("k")
    cache.get_stale("k")
    cache.get_stale("missing")
    assert cache.stats()["stale_hits"] == 2


def test_negative_stale_seconds_raises():
    with pytest.raises(ValueError):
        TTLCache(ttl_seconds=10, stale_seconds=-1)
