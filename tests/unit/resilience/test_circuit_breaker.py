"""The circuit-breaker state machine, driven by a fake clock."""

from __future__ import annotations

import threading

import pytest

from src.internal.configs.timeouts import TIMEOUTS_PATH_ENV, reset_timeout_policies
from src.internal.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    breaker_snapshots,
    get_breaker,
    is_failure_status,
    reset_breakers,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _breaker(clock, threshold=3, open_seconds=30.0):
    return CircuitBreaker(
        "dep", failure_threshold=threshold, open_seconds=open_seconds, clock=clock
    )


def _open(breaker, threshold=3):
    for _ in range(threshold):
        breaker.before_call()
        breaker.record_failure()


def test_opens_after_exactly_threshold_consecutive_failures():
    b = _breaker(_Clock())
    for _ in range(2):
        b.before_call()
        b.record_failure()
    assert b.snapshot().state == "closed"
    assert b.snapshot().consecutive_failures == 2
    b.before_call()
    b.record_failure()
    assert b.snapshot().state == "open"
    with pytest.raises(CircuitOpenError):
        b.before_call()


def test_success_resets_the_consecutive_count():
    b = _breaker(_Clock())
    for _ in range(2):
        b.record_failure()
    b.record_success()
    for _ in range(2):
        b.record_failure()
    assert b.snapshot().state == "closed"
    assert b.snapshot().consecutive_failures == 2


def test_open_raises_and_retry_in_counts_down():
    clock = _Clock()
    b = _breaker(clock)
    _open(b)
    snap = b.snapshot()
    assert snap.opened_at == 1000.0
    assert snap.retry_in_seconds == 30.0
    clock.now += 10
    with pytest.raises(CircuitOpenError) as info:
        b.before_call()
    assert info.value.name == "dep"
    assert info.value.retry_in_seconds == pytest.approx(20.0)
    assert b.snapshot().retry_in_seconds == pytest.approx(20.0)
    assert isinstance(info.value, RuntimeError)


def test_after_open_seconds_exactly_one_probe_is_admitted():
    clock = _Clock()
    b = _breaker(clock)
    _open(b)
    clock.now += 30
    b.before_call()  # the probe
    assert b.snapshot().state == "half_open"
    assert b.snapshot().retry_in_seconds == 0.0
    with pytest.raises(CircuitOpenError):
        b.before_call()  # a concurrent second call


def test_probe_success_closes():
    clock = _Clock()
    b = _breaker(clock)
    _open(b)
    clock.now += 30
    b.before_call()
    b.record_success()
    snap = b.snapshot()
    assert (snap.state, snap.consecutive_failures, snap.opened_at) == (
        "closed",
        0,
        None,
    )
    b.before_call()  # passes


def test_probe_failure_reopens_with_a_fresh_timer():
    clock = _Clock()
    b = _breaker(clock)
    _open(b)
    clock.now += 30
    b.before_call()
    b.record_failure()
    snap = b.snapshot()
    assert snap.state == "open"
    assert snap.opened_at == 1030.0
    clock.now += 29
    with pytest.raises(CircuitOpenError):
        b.before_call()
    clock.now += 1
    b.before_call()  # next probe


def test_unreported_probe_is_replaced_after_open_seconds():
    # A cancelled probe records nothing; the breaker must not fail fast forever.
    clock = _Clock()
    b = _breaker(clock)
    _open(b)
    clock.now += 30
    b.before_call()  # probe, never reports
    clock.now += 29
    with pytest.raises(CircuitOpenError):
        b.before_call()
    clock.now += 1
    b.before_call()  # a replacement probe


def test_half_open_admits_one_probe_across_threads():
    clock = _Clock()
    b = _breaker(clock)
    _open(b)
    clock.now += 30
    admitted = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        try:
            b.before_call()
            admitted.append(1)
        except CircuitOpenError:
            pass

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(admitted) == 1


@pytest.mark.parametrize(
    "status, failure",
    [(500, True), (503, True), (429, True), (404, False), (401, False), (200, False)],
)
def test_is_failure_status(status, failure):
    assert is_failure_status(status) is failure


def test_registry_is_lazy_shared_and_uses_bundled_policy():
    assert breaker_snapshots() == []
    b = get_breaker("serpapi")
    assert get_breaker("serpapi") is b
    for _ in range(4):
        b.record_failure()
    assert b.snapshot().state == "closed"  # bundled threshold is 5
    b.record_failure()
    assert b.snapshot().state == "open"
    get_breaker("alpha")
    assert [s.name for s in breaker_snapshots()] == ["alpha", "serpapi"]
    reset_breakers()
    assert breaker_snapshots() == []


def test_override_file_changes_the_policy(tmp_path, monkeypatch):
    f = tmp_path / "t.toml"
    f.write_text("[circuit_breaker]\nfailure_threshold = 1\nopen_seconds = 2\n")
    monkeypatch.setenv(TIMEOUTS_PATH_ENV, str(f))
    reset_timeout_policies()
    b = get_breaker("rerank")
    b.record_failure()
    snap = b.snapshot()
    assert snap.state == "open"
    assert snap.retry_in_seconds == pytest.approx(2.0, abs=0.5)


def test_transitions_log(caplog):
    clock = _Clock()
    b = _breaker(clock, threshold=1)
    with caplog.at_level("INFO", logger="src.internal.resilience.circuit_breaker"):
        b.record_failure()
        clock.now += 30
        b.before_call()
        b.record_failure()
        clock.now += 30
        b.before_call()
        b.record_success()
    levels = [(r.levelname, r.getMessage()) for r in caplog.records]
    assert [lvl for lvl, _ in levels] == ["WARNING", "WARNING", "INFO"]
    assert "opened" in levels[0][1] and "re-opened" in levels[1][1]
    assert "closed" in levels[2][1]
