"""Per-process circuit breakers for serving dependencies.

A breaker remembers that a dependency is down so later requests skip straight
to the caller's existing degradation path instead of paying the full timeout.
Calls are never wrapped: a site calls ``before_call()`` and then
``record_success()`` or ``record_failure()``, and decides for itself what
counts as a failure. Policy comes from ``[circuit_breaker]`` in timeouts.toml,
read when a breaker is first created.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from src.internal.configs.timeouts import get_timeout_policies

logger = logging.getLogger(__name__)

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised by ``before_call`` while the dependency is being skipped."""

    def __init__(self, name: str, retry_in_seconds: float) -> None:
        super().__init__(f"circuit {name!r} is open; retry in {retry_in_seconds:.1f}s")
        self.name = name
        self.retry_in_seconds = retry_in_seconds


@dataclass(frozen=True)
class BreakerSnapshot:
    name: str
    state: str
    consecutive_failures: int
    opened_at: float | None
    retry_in_seconds: float


def is_failure_status(status: int) -> bool:
    """HTTP statuses that mean the dependency is unhealthy: 5xx and 429."""
    return status >= 500 or status == 429


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int,
        open_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self._threshold = failure_threshold
        self._open_seconds = open_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CLOSED
        self._failures = 0
        self._opened_at: float | None = None
        # When the current half-open probe was admitted. A probe that never
        # reports (cancelled) is replaced after open_seconds.
        self._probe_at: float | None = None

    def before_call(self) -> None:
        with self._lock:
            if self._state == CLOSED:
                return
            now = self._clock()
            since = self._opened_at if self._state == OPEN else self._probe_at
            wait = since + self._open_seconds - now
            if wait > 0:
                raise CircuitOpenError(self.name, wait if self._state == OPEN else 0.0)
            self._state = HALF_OPEN
            self._probe_at = now

    def record_success(self) -> None:
        with self._lock:
            if self._state != CLOSED:
                logger.info("Circuit %r closed: the dependency answered", self.name)
            self._state = CLOSED
            self._failures = 0
            self._opened_at = None
            self._probe_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            reopen = self._state == HALF_OPEN
            if reopen or (self._state == CLOSED and self._failures >= self._threshold):
                self._state = OPEN
                self._opened_at = self._clock()
                self._probe_at = None
                logger.warning(
                    "Circuit %r %s after %d consecutive failures; "
                    "skipping it for %.1fs",
                    self.name,
                    "re-opened" if reopen else "opened",
                    self._failures,
                    self._open_seconds,
                )

    def snapshot(self) -> BreakerSnapshot:
        with self._lock:
            retry_in = 0.0
            if self._state == OPEN:
                retry_in = max(
                    0.0, self._opened_at + self._open_seconds - self._clock()
                )
            return BreakerSnapshot(
                name=self.name,
                state=self._state,
                consecutive_failures=self._failures,
                opened_at=self._opened_at,
                retry_in_seconds=retry_in,
            )


_registry: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_breaker(name: str) -> CircuitBreaker:
    """The process-wide breaker for ``name``, created from policy on first use."""
    with _registry_lock:
        breaker = _registry.get(name)
        if breaker is None:
            policy = get_timeout_policies().circuit_breaker
            breaker = CircuitBreaker(
                name,
                failure_threshold=policy.failure_threshold,
                open_seconds=policy.open_seconds,
            )
            _registry[name] = breaker
        return breaker


def breaker_snapshots() -> list[BreakerSnapshot]:
    with _registry_lock:
        breakers = sorted(_registry.values(), key=lambda b: b.name)
    return [b.snapshot() for b in breakers]


def reset_breakers() -> None:
    """Forget every breaker. Tests only."""
    with _registry_lock:
        _registry.clear()
