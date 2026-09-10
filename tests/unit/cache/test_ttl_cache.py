"""The process-local TTL cache behind the serving path."""

from __future__ import annotations

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
    assert cache.stats() == {"hits": 2, "misses": 1, "size": 1}


def test_clear_empties_the_cache():
    cache = TTLCache(ttl_seconds=10)
    cache.set("k", 1)
    cache.clear()
    assert cache.get("k") is None
