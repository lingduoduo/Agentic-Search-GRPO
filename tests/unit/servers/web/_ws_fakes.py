"""In-memory doubles for the WebSocket token paths.

Shared so the token tests and the channel tests exercise the same real
`store_ws_token` / `retrieve_ws_token_data` logic -- rate limit, TTL, atomic
single-use consumption -- rather than stubbing the functions under test.
"""

from __future__ import annotations


class FakeRedis:
    """Just enough Redis for the WS token paths, including GETDEL semantics."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.counters: dict[str, int] = {}

    def pipeline(self) -> "FakePipeline":
        return FakePipeline(self)

    async def decr(self, key: str) -> int:
        self.counters[key] = self.counters.get(key, 0) - 1
        return self.counters[key]

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.values[key] = value

    async def getdel(self, key: str) -> str | None:
        return self.values.pop(key, None)


class FakePipeline:
    def __init__(self, backing: FakeRedis) -> None:
        self._backing = backing
        self._queued: list[tuple[str, str]] = []

    def incr(self, key: str) -> None:
        self._queued.append(("incr", key))

    def expire(self, key: str, seconds: int) -> None:
        self._queued.append(("expire", key))

    async def execute(self) -> list:
        results = []
        for op, key in self._queued:
            if op == "incr":
                self._backing.counters[key] = self._backing.counters.get(key, 0) + 1
                results.append(self._backing.counters[key])
            else:
                results.append(True)
        self._queued.clear()
        return results
