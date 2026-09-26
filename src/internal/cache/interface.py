"""Abstract cache backend interface, in-memory implementation, and type constants."""

from __future__ import annotations

import abc
import math
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

TTL_KEY_NOT_FOUND = -2
TTL_NO_EXPIRY = -1

CACHE_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (Exception,)


class CacheBackendType(str, Enum):
    REDIS = "redis"
    POSTGRES = "postgres"


class CacheLock(abc.ABC):
    """Abstract distributed lock returned by CacheBackend.lock()."""

    @abc.abstractmethod
    def acquire(
        self,
        blocking: bool = True,
        blocking_timeout: float | None = None,
    ) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def release(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def owned(self) -> bool:
        raise NotImplementedError

    def __enter__(self) -> "CacheLock":
        if not self.acquire():
            raise RuntimeError("Failed to acquire lock")
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


class CacheBackend(abc.ABC):
    """Thin abstraction over a key-value cache with TTL, locks, and blocking lists."""

    @abc.abstractmethod
    def get(self, key: str) -> bytes | None:
        raise NotImplementedError

    @abc.abstractmethod
    def set(
        self,
        key: str,
        value: str | bytes | int | float,
        ex: int | None = None,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def expire(self, key: str, seconds: int) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def ttl(self, key: str) -> int:
        """Return remaining TTL in seconds.

        Returns TTL_NO_EXPIRY (-1) if key exists without expiry,
        TTL_KEY_NOT_FOUND (-2) if key is missing or expired.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def lock(self, name: str, timeout: float | None = None) -> CacheLock:
        raise NotImplementedError

    @abc.abstractmethod
    def rpush(self, key: str, value: str | bytes) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def blpop(self, keys: list[str], timeout: int = 0) -> tuple[bytes, bytes] | None:
        """Block until a value is available on one of *keys*, or *timeout* expires.

        Returns ``(key, value)`` or ``None`` on timeout.
        """
        raise NotImplementedError


class _InMemoryCacheLock(CacheLock):
    def acquire(
        self,
        blocking: bool = True,
        blocking_timeout: float | None = None,
    ) -> bool:
        return True

    def release(self) -> None:
        pass

    def owned(self) -> bool:
        return True


class InMemoryCache(CacheBackend):
    """Thread-unsafe in-memory cache for local / test mode.

    Keys may carry an expiry (``set(ex=...)`` / ``expire``), checked lazily on
    read against *clock* (monotonic seconds), with Redis semantics.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._store: dict[str, Any] = {}
        self._expires: dict[str, float] = {}
        self._lists: dict[str, list[bytes]] = {}
        self._clock = clock

    def _live(self, key: str) -> bool:
        """Is *key* present and unexpired? Deletes it if it has expired."""
        deadline = self._expires.get(key)
        if deadline is not None and self._clock() >= deadline:
            self.delete(key)
        return key in self._store

    def get(self, key: str) -> bytes | None:
        if not self._live(key):
            return None
        v = self._store[key]
        if v is None:
            return None
        return v if isinstance(v, bytes) else str(v).encode()

    def set(
        self,
        key: str,
        value: str | bytes | int | float,
        ex: int | None = None,
    ) -> None:
        self._store[key] = value
        if ex is None:
            self._expires.pop(key, None)
        else:
            self._expires[key] = self._clock() + ex

    def delete(self, key: str) -> None:
        self._store.pop(key, None)
        self._expires.pop(key, None)

    def exists(self, key: str) -> bool:
        return self._live(key)

    def expire(self, key: str, seconds: int) -> None:
        if self._live(key):
            self._expires[key] = self._clock() + seconds

    def ttl(self, key: str) -> int:
        if not self._live(key):
            return TTL_KEY_NOT_FOUND
        deadline = self._expires.get(key)
        if deadline is None:
            return TTL_NO_EXPIRY
        return math.ceil(deadline - self._clock())

    def lock(self, name: str, timeout: float | None = None) -> CacheLock:
        return _InMemoryCacheLock()

    def rpush(self, key: str, value: str | bytes) -> None:
        bucket = self._lists.setdefault(key, [])
        bucket.append(value if isinstance(value, bytes) else value.encode())

    def blpop(self, keys: list[str], timeout: int = 0) -> tuple[bytes, bytes] | None:
        for key in keys:
            if self._lists.get(key):
                return (key.encode(), self._lists[key].pop(0))
        return None


_default_cache: CacheBackend | None = None


def get_cache_backend() -> CacheBackend:
    """Return the active cache backend, defaulting to InMemoryCache."""
    global _default_cache
    if _default_cache is None:
        _default_cache = InMemoryCache()
    return _default_cache


def set_cache_backend(backend: CacheBackend) -> None:
    """Override the global cache backend (e.g. for tests or Redis integration)."""
    global _default_cache
    _default_cache = backend
