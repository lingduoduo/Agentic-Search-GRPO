"""A small process-local cache: per-entry TTL, LRU-bounded, lock-guarded.

With ``stale_seconds`` an expired entry is kept that much longer, invisible to
``get`` but returned by ``get_stale``, for a caller whose live call failed."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any


class TTLCache:
    def __init__(
        self,
        ttl_seconds: float,
        *,
        max_entries: int = 1024,
        stale_seconds: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if stale_seconds < 0:
            raise ValueError("stale_seconds must not be negative")
        self._ttl = float(ttl_seconds)
        self._max = max_entries
        self._stale = float(stale_seconds)
        self._clock = clock
        self._entries: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._stale_hits = 0

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            expires_at, value = entry
            now = self._clock()
            if now >= expires_at:
                # Inside the grace window it stays for get_stale; past it, gone.
                if now >= expires_at + self._stale:
                    del self._entries[key]
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return value

    def get_stale(self, key: Hashable) -> Any | None:
        """The value if the entry is fresh or inside the grace window."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if self._clock() >= expires_at + self._stale:
                del self._entries[key]
                return None
            self._stale_hits += 1
            return value

    def set(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._entries[key] = (self._clock() + self._ttl, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._entries),
                "stale_hits": self._stale_hits,
            }
