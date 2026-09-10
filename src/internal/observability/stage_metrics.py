"""Per-request retrieval-vs-generation telemetry.

The route-latency middleware measures the whole request and the control-flow
trace is dev-only, so until now nothing production-reachable could say whether
a slow ``/api/agent`` spent its time fetching documents or generating the
answer. This module is the always-on layer beneath the trace:

- ``start_request`` / ``finish_request`` scope a :class:`RequestStageMetrics`
  in a ContextVar (the same pattern ``request_capture`` uses), so the two
  choke points every path goes through — ``SearchClient.retrieve`` and the LLM
  backends — can ``note_*`` into it without knowing about the web layer.
- :class:`StageLatencyStats` keeps a rolling window of finished requests per
  stage, mirroring ``RouteLatencyStats``, for percentiles an admin can read.

Stdlib only; every entry point is a no-op when no request is active.
"""

from __future__ import annotations

import math
from collections import deque
from contextvars import ContextVar, Token
from dataclasses import dataclass

_DEFAULT_MAX_SAMPLES = 512


@dataclass
class RequestStageMetrics:
    retrieval_calls: int = 0
    retrieval_cache_hits: int = 0
    retrieval_ms: float = 0.0
    retrieval_docs: int = 0
    generation_calls: int = 0
    generation_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            "retrieval": {
                "calls": self.retrieval_calls,
                "cache_hits": self.retrieval_cache_hits,
                "ms": round(self.retrieval_ms, 3),
                "docs": self.retrieval_docs,
            },
            "generation": {
                "calls": self.generation_calls,
                "ms": round(self.generation_ms, 3),
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            },
        }


_current: ContextVar[RequestStageMetrics | None] = ContextVar(
    "stage_metrics", default=None
)


def current() -> RequestStageMetrics | None:
    return _current.get()


def start_request() -> Token:
    """Open a request scope; pass the token back to :func:`finish_request`."""
    return _current.set(RequestStageMetrics())


def finish_request(token: Token | None) -> RequestStageMetrics | None:
    """Close the scope opened by *token* and return what it accumulated."""
    metrics = _current.get()
    if token is not None:
        _current.reset(token)
    return metrics


def note_retrieval(*, elapsed_ms: float, docs: int, cache_hit: bool = False) -> None:
    metrics = _current.get()
    if metrics is None:
        return
    metrics.retrieval_calls += 1
    metrics.retrieval_ms += float(elapsed_ms)
    metrics.retrieval_docs += int(docs)
    if cache_hit:
        metrics.retrieval_cache_hits += 1


def note_generation(
    *,
    elapsed_ms: float,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> None:
    metrics = _current.get()
    if metrics is None:
        return
    metrics.generation_calls += 1
    metrics.generation_ms += float(elapsed_ms)
    if prompt_tokens:
        metrics.prompt_tokens += int(prompt_tokens)
    if completion_tokens:
        metrics.completion_tokens += int(completion_tokens)


def _percentile(ordered: list[float], fraction: float) -> float:
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


class StageLatencyStats:
    """Rolling window of finished requests, one bucket per stage.

    A request contributes to a stage only if it used that stage, so the
    retrieval percentiles describe requests that retrieved and the generation
    percentiles describe requests that generated.
    """

    def __init__(self, max_samples: int = _DEFAULT_MAX_SAMPLES) -> None:
        self._retrieval: deque[tuple[float, int, bool]] = deque(maxlen=max_samples)
        self._generation: deque[tuple[float, int, int]] = deque(maxlen=max_samples)

    def record(self, metrics: RequestStageMetrics | None) -> None:
        if metrics is None:
            return
        if metrics.retrieval_calls:
            self._retrieval.append(
                (
                    metrics.retrieval_ms,
                    metrics.retrieval_docs,
                    metrics.retrieval_cache_hits == metrics.retrieval_calls,
                )
            )
        if metrics.generation_calls:
            self._generation.append(
                (
                    metrics.generation_ms,
                    metrics.prompt_tokens,
                    metrics.completion_tokens,
                )
            )

    @staticmethod
    def _timing(samples: list[float]) -> dict[str, float | int]:
        ordered = sorted(samples)
        return {
            "count": len(ordered),
            "p50_ms": round(_percentile(ordered, 0.50), 3),
            "p95_ms": round(_percentile(ordered, 0.95), 3),
            "max_ms": round(ordered[-1], 3),
        }

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        retrieval: dict[str, float | int] = {"count": 0}
        if self._retrieval:
            rows = list(self._retrieval)
            retrieval = self._timing([r[0] for r in rows])
            retrieval["avg_docs"] = round(sum(r[1] for r in rows) / len(rows), 3)
            retrieval["cache_hit_rate"] = sum(1 for r in rows if r[2]) / len(rows)
        generation: dict[str, float | int] = {"count": 0}
        if self._generation:
            rows = list(self._generation)
            generation = self._timing([r[0] for r in rows])
            generation["avg_prompt_tokens"] = round(
                sum(r[1] for r in rows) / len(rows), 3
            )
            generation["avg_completion_tokens"] = round(
                sum(r[2] for r in rows) / len(rows), 3
            )
        return {"retrieval": retrieval, "generation": generation}


#: Process-wide window the web app records into, mirroring ``ROUTE_LATENCY``.
STAGE_LATENCY = StageLatencyStats()
