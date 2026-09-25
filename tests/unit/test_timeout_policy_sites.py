"""Each migrated site reads its value from the timeout policies."""

import asyncio
from contextlib import contextmanager

import pytest

from src.internal.configs.timeouts import load_timeout_policies, use_timeout_policies


@contextmanager
def overridden(overrides: dict):
    with use_timeout_policies(load_timeout_policies({}, overrides=overrides)) as p:
        yield p


def test_public_data_attempts_follow_the_policy(monkeypatch):
    from src.internal.tools.public_data import _http
    from tests.unit.test_public_data_http import _SequencedSession

    real_sleep = asyncio.sleep
    shared = _SequencedSession([503, 503, 503, 503, 503])
    _SequencedSession.calls = []

    class _Aiohttp:
        @staticmethod
        def ClientTimeout(total=None):
            return total

        @staticmethod
        def ClientSession(timeout=None):
            return shared

    monkeypatch.setattr(_http, "aiohttp", _Aiohttp)
    monkeypatch.setattr(_http.asyncio, "sleep", lambda s: real_sleep(0))
    with overridden({"tools": {"public_data": {"max_attempts": 5}}}):
        with pytest.raises(_http.PublicDataError):
            asyncio.run(_http.get_json("https://x.test"))
    assert (
        len(_SequencedSession.calls) == 5
    )  # 5 attempts, 2 backoff steps: last step reused


def test_public_data_timeout_follows_the_policy_and_kwarg_wins(monkeypatch):
    from src.internal.tools.public_data import _http

    seen = []

    async def fake_fetch(method, url, *, timeout_seconds, **kw):
        seen.append(timeout_seconds)
        return {}

    monkeypatch.setattr(_http, "_fetch", fake_fetch)
    with overridden({"tools": {"public_data": {"timeout_seconds": 3}}}):
        asyncio.run(_http.get_json("https://x.test"))
        asyncio.run(_http.get_json("https://x.test", timeout_seconds=9))
    assert seen == [3.0, 9]
