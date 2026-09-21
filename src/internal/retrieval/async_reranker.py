from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
import time

from src.internal.retrieval.backends.base import RetrievalResult


class RerankerTimeoutError(RuntimeError):
    pass


class RerankerOverloaded(RerankerTimeoutError):
    """Refused before running, because it could not have finished in time.

    A subclass of the timeout it replaces, so existing callers -- which degrade
    to the pre-rerank ordering -- keep working unchanged while new code can
    tell a refusal apart from an expiry.
    """


# Weight on the newest observation. Low enough that one slow document set does
# not close the door, high enough to follow a real shift in scoring cost.
_EWMA_ALPHA = 0.2


class AsyncReranker:
    """Wraps any reranker, offloads scoring to a thread pool with a timeout.

    The timeout is measured from **submit**, not from when scoring starts, so
    queue wait spends the same budget as the work does. Past the pool's
    capacity that degrades badly: every request waits the full deadline, gives
    up, and is discarded -- having already occupied a worker, because
    ``Future.cancel()`` cannot stop a task that has begun.

    Measured with the shipped defaults (4 workers, 500ms, an 80ms scorer):
    saturation at 4 concurrent, then 92.5% of requests timing out at 32 and
    96.2% at 64, with 478ms of the 500ms budget spent queueing.

    So a submission that cannot plausibly start in time is refused immediately
    instead. The caller already degrades to the fused ordering on a timeout, so
    the outcome is identical -- reached in microseconds rather than half a
    second, and without burning a worker on a result nobody will read.
    """

    def __init__(
        self,
        base_reranker,
        *,
        timeout_ms: int = 500,
        max_workers: int = 4,
    ) -> None:
        self._base = base_reranker
        self._timeout_ms = timeout_ms
        self._max_workers = max_workers
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._guard = threading.Lock()
        self._outstanding = 0
        self._observed_seconds: float | None = None
        self.counters = {"admitted": 0, "refused": 0, "expired": 0}

    # -- admission ---------------------------------------------------------

    def _admit(self) -> bool:
        """Reserve a slot, unless the work could not finish inside the budget."""
        with self._guard:
            if self._observed_seconds is not None:
                # Whole batches ahead of this one, each occupying every worker.
                queued_batches = self._outstanding // self._max_workers
                projected = (queued_batches + 1) * self._observed_seconds
                if projected > self._timeout_ms / 1000:
                    self.counters["refused"] += 1
                    return False
            self._outstanding += 1
            self.counters["admitted"] += 1
            return True

    def _release(self, seconds: float | None) -> None:
        with self._guard:
            self._outstanding -= 1
            if seconds is None:
                return
            if self._observed_seconds is None:
                self._observed_seconds = seconds
            else:
                self._observed_seconds = (
                    1 - _EWMA_ALPHA
                ) * self._observed_seconds + _EWMA_ALPHA * seconds

    def _timed(self, query: str, results: list[RetrievalResult], top_k: int):
        """Run the wrapped reranker, recording how long scoring actually took."""
        started = time.monotonic()
        seconds: float | None = None
        try:
            scored = self._base.rerank(query, results, top_k)
            seconds = time.monotonic() - started
            return scored
        finally:
            self._release(seconds)

    def _refuse(self) -> RerankerOverloaded:
        return RerankerOverloaded(
            f"Reranker refused: work queued past the {self._timeout_ms}ms budget"
        )

    # -- entry points ------------------------------------------------------

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Sync shim: submits to thread pool, blocks with timeout."""
        if not self._admit():
            raise self._refuse()
        future = self._executor.submit(self._timed, query, results, top_k)
        try:
            return future.result(timeout=self._timeout_ms / 1000)
        except concurrent.futures.TimeoutError:
            future.cancel()
            self.counters["expired"] += 1
            raise RerankerTimeoutError(
                f"Reranker exceeded {self._timeout_ms}ms timeout"
            )

    async def arerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Async entry point: runs scorer in thread pool, awaits with timeout."""
        if not self._admit():
            raise self._refuse()
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._executor, self._timed, query, results, top_k
        )
        try:
            return await asyncio.wait_for(future, timeout=self._timeout_ms / 1000)
        except asyncio.TimeoutError:
            self.counters["expired"] += 1
            raise RerankerTimeoutError(
                f"Reranker exceeded {self._timeout_ms}ms timeout"
            )

    @classmethod
    def from_env(cls, base_reranker) -> AsyncReranker:
        return cls(
            base_reranker,
            timeout_ms=int(os.environ.get("RERANKER_TIMEOUT_MS", "500")),
            max_workers=int(os.environ.get("RERANKER_MAX_WORKERS", "4")),
        )
