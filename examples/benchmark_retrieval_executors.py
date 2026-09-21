"""Load-test the two thread pools on the retrieval path.

Both targets drive the real production method. Only the work inside a pool task
is stubbed, so task cost becomes a controlled variable and everything measured
-- queueing, thread spawn, scheduling, fan-in -- is the shipped orchestration.

    search  RetrievalService._search_one -- ThreadPoolExecutor(max_workers=2),
            constructed per call, receiving exactly two tasks. Nothing can
            queue for a worker; the cost is two fresh threads per request.

    rerank  AsyncReranker -- one persistent pool, RERANKER_MAX_WORKERS (4 by
            default), shared by every request. This one can queue, and its
            RERANKER_TIMEOUT_MS deadline is measured from submit, so queue wait
            spends the same budget as scoring.

Task modes bound real behaviour, which sits between them:

    sleep   releases the GIL -- models FAISS, pyserini-JVM, network
    cpu     holds the GIL    -- models Python-level scoring

Examples:
    python -m examples.benchmark_retrieval_executors --target search
    python -m examples.benchmark_retrieval_executors --target rerank \
        --task-ms 80 --workers 4 --timeout-ms 500
"""

from __future__ import annotations

import argparse
import threading
import time

from src.internal.retrieval.async_reranker import AsyncReranker
from src.internal.retrieval.async_reranker import RerankerTimeoutError
from src.internal.retrieval.backends.base import RetrievalBackend
from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.service import RetrievalService


def _spend(seconds: float, mode: str) -> None:
    """Occupy a worker for `seconds`, either releasing the GIL or holding it."""
    if mode == "sleep":
        time.sleep(seconds)
        return
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def _result(doc_id: str) -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id, title="t", text="x", url=None, score=1.0, metadata={}
    )


class _Timed:
    """Records when each task actually began, keyed by the call id."""

    def __init__(self) -> None:
        self.starts: dict[str, float] = {}
        self._lock = threading.Lock()

    def mark(self, call_id: str, start: float) -> None:
        with self._lock:
            self.starts.setdefault(call_id, start)


class StubBackend(RetrievalBackend, _Timed):
    """Two retrieval legs of controllable cost. The call id rides in the query."""

    def __init__(self, task_s: float, mode: str) -> None:
        _Timed.__init__(self)
        self._task_s = task_s
        self._mode = mode

    def _leg(self, leg: str, query: str) -> list[RetrievalResult]:
        self.mark(f"{query}:{leg}", time.perf_counter())
        _spend(self._task_s, self._mode)
        return [_result(f"{leg}-0")]

    def search_sparse(self, query, top_k, filters=None):
        return self._leg("sparse", query)

    def search_dense(self, query, top_k, filters=None):
        return self._leg("dense", query)


class StubReranker(_Timed):
    """A scorer of controllable cost, standing in for the cross-encoder."""

    def __init__(self, task_s: float, mode: str) -> None:
        _Timed.__init__(self)
        self._task_s = task_s
        self._mode = mode

    def rerank(self, query, results, top_k):
        self.mark(query, time.perf_counter())
        _spend(self._task_s, self._mode)
        return results[:top_k]


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(int(round(p / 100.0 * (len(ordered) - 1))), len(ordered) - 1)]


def _drive(
    call, concurrency: int, per_thread: int
) -> tuple[list[float], dict[str, float], float, int, int]:
    """Run `call(call_id)` from `concurrency` threads. Returns latencies and peak threads."""
    latencies: list[float] = []
    submitted: dict[str, float] = {}
    timeouts = 0
    lock = threading.Lock()
    peak = threading.active_count()
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, threading.active_count())
            time.sleep(0.002)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()

    def worker(wid: int) -> None:
        nonlocal timeouts
        for i in range(per_thread):
            call_id = f"w{wid}-{i}"
            t0 = time.perf_counter()
            with lock:
                submitted[call_id] = t0
            try:
                call(call_id)
            except RerankerTimeoutError:
                with lock:
                    timeouts += 1
            elapsed = time.perf_counter() - t0
            with lock:
                latencies.append(elapsed)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(concurrency)]
    started = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - started
    stop.set()
    sampler.join(timeout=1)
    return latencies, submitted, wall, peak, timeouts


def run_level(args, concurrency: int) -> dict:
    task_s = args.task_ms / 1000.0

    if args.target == "search":
        stub = StubBackend(task_s, args.mode)
        service = RetrievalService(stub)

        def call(call_id: str) -> None:
            service._search_one(call_id, over_fetch=10, filters=None)

        def wait_for(call_id: str, t0: float) -> list[float]:
            return [
                stub.starts[k] - t0
                for k in (f"{call_id}:sparse", f"{call_id}:dense")
                if k in stub.starts
            ]
    else:
        stub = StubReranker(task_s, args.mode)
        reranker = AsyncReranker(
            stub, timeout_ms=args.timeout_ms, max_workers=args.workers
        )
        docs = [_result(f"d{i}") for i in range(args.docs)]

        def call(call_id: str) -> None:
            reranker.rerank(call_id, docs, args.docs)

        def wait_for(call_id: str, t0: float) -> list[float]:
            return [stub.starts[call_id] - t0] if call_id in stub.starts else []

    latencies, submitted, wall, peak, timeouts = _drive(
        call, concurrency, args.per_thread
    )

    waits: list[float] = []
    for call_id, t0 in submitted.items():
        waits.extend(wait_for(call_id, t0))

    total = len(latencies)
    return {
        "conc": concurrency,
        "reqs": total,
        "rps": total / wall if wall else float("nan"),
        "p50": _pct(latencies, 50) * 1000,
        "p95": _pct(latencies, 95) * 1000,
        "p99": _pct(latencies, 99) * 1000,
        "wait50": _pct(waits, 50) * 1000,
        "wait99": _pct(waits, 99) * 1000,
        "timeout_pct": 100.0 * timeouts / total if total else 0.0,
        "threads": peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--target", choices=("search", "rerank"), default="search")
    parser.add_argument("--levels", default="1,2,4,8,16,32,64")
    parser.add_argument("--per-thread", type=int, default=20)
    parser.add_argument("--task-ms", type=float, default=10.0)
    parser.add_argument("--mode", choices=("sleep", "cpu"), default="sleep")
    parser.add_argument("--workers", type=int, default=4, help="rerank only: pool size")
    parser.add_argument(
        "--timeout-ms", type=int, default=500, help="rerank only: deadline from submit"
    )
    parser.add_argument("--docs", type=int, default=20, help="rerank only: candidates")
    args = parser.parse_args()

    banner = f"target={args.target}  mode={args.mode}  task={args.task_ms}ms"
    if args.target == "rerank":
        banner += f"  workers={args.workers}  timeout={args.timeout_ms}ms"
    print(banner)

    header = (
        f"{'conc':>5} {'reqs':>6} {'rps':>9} {'p50':>9} {'p95':>9} {'p99':>9} "
        f"{'wait50':>9} {'wait99':>9} {'tmo%':>7} {'thr':>5}"
    )
    print(header)
    print("-" * len(header))
    for level in (int(x) for x in args.levels.split(",")):
        r = run_level(args, level)
        print(
            f"{r['conc']:>5} {r['reqs']:>6} {r['rps']:>9.1f} {r['p50']:>9.2f} "
            f"{r['p95']:>9.2f} {r['p99']:>9.2f} {r['wait50']:>9.2f} "
            f"{r['wait99']:>9.2f} {r['timeout_pct']:>7.1f} {r['threads']:>5}"
        )


if __name__ == "__main__":
    main()
