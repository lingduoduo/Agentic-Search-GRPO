"""Unit tests for the public_data HTTP layer. No live network."""

from __future__ import annotations

import asyncio
import json

import pytest

from src.internal.tools.public_data import _http
from src.internal.tools.public_data._http import (
    PublicDataError,
    get_json,
    guarded,
)


class _FakeResponse:
    def __init__(self, *, status=200, body="{}"):
        self.status = status
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return self._body


class _FakeSession:
    """Records the single request it is given, then replays a canned response."""

    calls: list[dict] = []

    def __init__(self, *, status=200, body="{}", raises=None):
        self._status = status
        self._body = body
        self._raises = raises

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def request(self, method, url, **kwargs):
        if self._raises is not None:
            raise self._raises
        _FakeSession.calls.append({"method": method, "url": url, **kwargs})
        return _FakeResponse(status=self._status, body=self._body)


def _install(monkeypatch, **kwargs):
    _FakeSession.calls = []

    class _Aiohttp:
        @staticmethod
        def ClientTimeout(total=None):
            return total

        @staticmethod
        def ClientSession(timeout=None):
            return _FakeSession(**kwargs)

    monkeypatch.setattr(_http, "aiohttp", _Aiohttp)


def test_get_json_parses_body_and_sends_user_agent(monkeypatch):
    _install(monkeypatch, body=json.dumps({"ok": True}))

    result = asyncio.run(get_json("https://example.test/x", params={"a": "b"}))

    assert result == {"ok": True}
    call = _FakeSession.calls[0]
    assert call["method"] == "GET"
    assert call["params"] == {"a": "b"}
    assert call["headers"]["User-Agent"] == _http.USER_AGENT


def test_get_json_caller_headers_override_default(monkeypatch):
    _install(monkeypatch, body="{}")

    asyncio.run(
        get_json("https://example.test/x", headers={"User-Agent": "Mozilla/5.0"})
    )

    assert _FakeSession.calls[0]["headers"]["User-Agent"] == "Mozilla/5.0"


def test_get_json_raises_on_http_error(monkeypatch):
    _install(monkeypatch, status=503, body="down")

    with pytest.raises(PublicDataError) as excinfo:
        asyncio.run(get_json("https://example.test/x"))

    assert "503" in str(excinfo.value)


def test_get_json_raises_on_non_json_body(monkeypatch):
    _install(monkeypatch, body="<html>nope</html>")

    with pytest.raises(PublicDataError):
        asyncio.run(get_json("https://example.test/x"))


def test_get_json_raises_on_transport_failure(monkeypatch):
    _install(monkeypatch, raises=asyncio.TimeoutError())

    with pytest.raises(PublicDataError):
        asyncio.run(get_json("https://example.test/x"))


def test_guarded_serializes_success():
    @guarded
    async def _ok(value: str):
        return {"value": value}

    assert json.loads(asyncio.run(_ok(value="hi"))) == {"value": "hi"}


def test_guarded_converts_public_data_error():
    @guarded
    async def _boom():
        raise PublicDataError("upstream is down")

    assert json.loads(asyncio.run(_boom())) == {"error": "upstream is down"}


def test_guarded_converts_unexpected_error():
    @guarded
    async def _boom():
        raise KeyError("missing")

    assert "error" in json.loads(asyncio.run(_boom()))


def test_guarded_result_is_a_coroutine_function():
    """FunctionTool.execute awaits only if iscoroutinefunction() is True."""
    import inspect

    @guarded
    async def _ok():
        return {}

    assert inspect.iscoroutinefunction(_ok)


async def _no_sleep(_seconds):
    """Backoff must not slow the suite; the delay itself is not under test."""
    return None


class _SequencedSession:
    """Replays one queued outcome per request, recording each call."""

    calls: list[dict] = []

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def request(self, method, url, **kwargs):
        _SequencedSession.calls.append({"method": method, "url": url, **kwargs})
        outcome = self._outcomes.pop(0) if self._outcomes else 200
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResponse(status=outcome, body=json.dumps({"ok": outcome}))


def _install_sequence(monkeypatch, outcomes):
    _SequencedSession.calls = []
    shared = _SequencedSession(outcomes)

    class _Aiohttp:
        @staticmethod
        def ClientTimeout(total=None):
            return total

        @staticmethod
        def ClientSession(timeout=None):
            return shared

    monkeypatch.setattr(_http, "aiohttp", _Aiohttp)
    monkeypatch.setattr(_http.asyncio, "sleep", _no_sleep)


def test_a_transient_status_is_retried_and_can_succeed(monkeypatch):
    """The whole point: a 503 that would have worked on the next try."""
    _install_sequence(monkeypatch, [503, 200])

    result = asyncio.run(get_json("https://example.org/x"))

    assert result == {"ok": 200}
    assert len(_SequencedSession.calls) == 2


def test_retries_stop_at_the_attempt_cap(monkeypatch):
    _install_sequence(monkeypatch, [503, 503, 503, 200])

    with pytest.raises(PublicDataError, match="HTTP 503"):
        asyncio.run(get_json("https://example.org/x"))

    assert len(_SequencedSession.calls) == _http._MAX_ATTEMPTS == 3


def test_a_non_transient_status_is_not_retried(monkeypatch):
    """A 404 is an answer, not a blip."""
    _install_sequence(monkeypatch, [404, 200])

    with pytest.raises(PublicDataError, match="HTTP 404"):
        asyncio.run(get_json("https://example.org/x"))

    assert len(_SequencedSession.calls) == 1


def test_a_network_error_is_retried(monkeypatch):
    _install_sequence(monkeypatch, [OSError("connection reset"), 200])

    result = asyncio.run(get_json("https://example.org/x"))

    assert result == {"ok": 200}
    assert len(_SequencedSession.calls) == 2


def test_a_post_is_never_retried(monkeypatch):
    """Retrying a write needs an idempotency guarantee this layer lacks."""
    _install_sequence(monkeypatch, [503, 200])

    with pytest.raises(PublicDataError, match="HTTP 503"):
        asyncio.run(
            _http._fetch(
                "POST", "https://example.org/x", timeout_seconds=1.0, as_json=True
            )
        )

    assert len(_SequencedSession.calls) == 1


def test_no_retry_once_the_elapsed_budget_is_spent(monkeypatch):
    """A caller with a long timeout must not pay it three times over."""
    _install_sequence(monkeypatch, [503, 200])
    # `_http.time` is the stdlib module, so this patch is global and asyncio's
    # internals call it too. A non-exhausting fake keeps those calls from
    # consuming the scripted values -- an iterator here raises StopIteration
    # from inside the event loop instead of failing the assertion.
    scripted = [0.0, _http._RETRY_BUDGET_SECONDS + 1.0]
    monkeypatch.setattr(
        _http.time, "monotonic", lambda: scripted.pop(0) if scripted else 1e9
    )

    with pytest.raises(PublicDataError, match="HTTP 503"):
        asyncio.run(get_json("https://example.org/x"))

    assert len(_SequencedSession.calls) == 1
