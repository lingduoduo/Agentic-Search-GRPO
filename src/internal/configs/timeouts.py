"""Timeout and retry policies, loaded from TOML.

Precedence: an explicit kwarg at the call site > the legacy env vars >
the operator file named by AGENTIC_SEARCH_TIMEOUTS_PATH > the bundled
timeouts.toml. Sites read ``get_timeout_policies()`` when they construct or
call, never at import time.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from importlib import resources
from typing import Any, get_type_hints

try:
    import tomllib
except ImportError:  # Python 3.10, or an import blocked for testing
    import tomli as tomllib

TIMEOUTS_PATH_ENV = "AGENTIC_SEARCH_TIMEOUTS_PATH"

# Env vars that predate the file keep working and win over it.
_ENV_OVERRIDES = {
    "TOOL_APPROVAL_TIMEOUT_SECONDS": "tool_loop.approval_timeout_seconds",
    "AGENTIC_SEARCH_GENERATION_TIMEOUT": "llm.local_generation_timeout_seconds",
    "LLM_SOCKET_READ_TIMEOUT": "llm.socket_read_timeout_seconds",
    "AGENTIC_SEARCH_SSE_HEARTBEAT_SECONDS": "sse.heartbeat_seconds",
}
# 0 means "disabled" or "no retry" here; everywhere else a timeout must be
# positive and a count is an attempt count (0 would make no request).
_ZERO_ALLOWED = frozenset(
    {
        "sse.heartbeat_seconds",
        "llm.local_generation_timeout_seconds",
        "llm.grounded_max_retries",
        "tool_loop.recovery.max_retries",
    }
)


@dataclass(frozen=True)
class PublicDataPolicy:
    timeout_seconds: float
    max_attempts: int
    backoff_seconds: tuple[float, ...]
    retry_budget_seconds: float
    overpass_timeout_seconds: float
    overpass_query_timeout_seconds: float


@dataclass(frozen=True)
class WebSearchPolicy:
    google_timeout_seconds: float
    serpapi_timeout_seconds: float
    serper_timeout_seconds: float
    fetch_page_timeout_seconds: float
    fetch_pages_timeout_seconds: float
    query_timeout_seconds: float


@dataclass(frozen=True)
class SearchRouterPolicy:
    timeout_seconds: float
    max_retries: int


@dataclass(frozen=True)
class OpenAPIPolicy:
    timeout_seconds: float


@dataclass(frozen=True)
class MCPPolicy:
    timeout_seconds: float
    sse_read_timeout_seconds: float


@dataclass(frozen=True)
class ToolsPolicies:
    public_data: PublicDataPolicy
    web_search: WebSearchPolicy
    search_router: SearchRouterPolicy
    openapi: OpenAPIPolicy
    mcp: MCPPolicy


@dataclass(frozen=True)
class RetrievalClientPolicy:
    timeout_seconds: float
    max_retries: int
    backoff_base_seconds: float


@dataclass(frozen=True)
class SearchRunnerPolicy:
    timeout_seconds: float
    max_retries: int


@dataclass(frozen=True)
class WebHybridPolicy:
    provider_timeout_seconds: float
    provider_max_retries: int
    provider_wait_seconds: float


@dataclass(frozen=True)
class RetrievalPolicies:
    client: RetrievalClientPolicy
    search_runner: SearchRunnerPolicy
    web_hybrid: WebHybridPolicy


@dataclass(frozen=True)
class RerankPolicy:
    timeout_seconds: float


@dataclass(frozen=True)
class LLMPolicy:
    socket_read_timeout_seconds: float
    remote_total_timeout_seconds: float
    local_generation_timeout_seconds: float
    local_heartbeat_seconds: float
    sufficiency_timeout_seconds: float
    grounded_max_retries: int


@dataclass(frozen=True)
class RecoveryPolicyConfig:
    max_retries: int
    backoff_seconds: tuple[float, ...]
    retry_after_cap_seconds: float
    retry_budget_seconds: float


@dataclass(frozen=True)
class ToolLoopPolicy:
    approval_timeout_seconds: float
    escalation_timeout_seconds: float
    max_escalations: int
    tool_evidence_timeout_seconds: float
    recovery: RecoveryPolicyConfig


@dataclass(frozen=True)
class SSEPolicy:
    heartbeat_seconds: float


@dataclass(frozen=True)
class CircuitBreakerPolicy:
    failure_threshold: int
    open_seconds: float


@dataclass(frozen=True)
class TimeoutPolicies:
    tools: ToolsPolicies
    retrieval: RetrievalPolicies
    rerank: RerankPolicy
    llm: LLMPolicy
    tool_loop: ToolLoopPolicy
    sse: SSEPolicy
    circuit_breaker: CircuitBreakerPolicy


def _read_bundled() -> dict[str, Any]:
    text = resources.files(__package__).joinpath("timeouts.toml").read_text("utf-8")
    return tomllib.loads(text)


def _read_override(path: str) -> dict[str, Any]:
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ValueError(f"{TIMEOUTS_PATH_ENV}={path!r}: file not found") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{TIMEOUTS_PATH_ENV}={path!r}: invalid TOML: {exc}") from exc


def _merge(
    base: dict[str, Any], over: Mapping[str, Any], prefix: str = ""
) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        path = f"{prefix}{key}"
        if key not in base:
            raise ValueError(f"unknown timeout policy key {path!r}")
        if isinstance(base[key], dict):
            if not isinstance(value, Mapping):
                raise ValueError(f"timeout policy {path!r} must be a table")
            out[key] = _merge(base[key], value, f"{path}.")
        elif isinstance(value, Mapping):
            raise ValueError(f"timeout policy {path!r} must be a value, not a table")
        else:
            out[key] = value
    return out


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _coerce(hint: Any, value: Any, path: str) -> Any:
    zero_ok = path in _ZERO_ALLOWED
    if hint is float:
        if not _is_number(value) or value < 0 or (value == 0 and not zero_ok):
            bound = ">= 0" if zero_ok else "> 0"
            raise ValueError(f"timeout policy {path!r} must be a finite number {bound}")
        return float(value)
    if hint is int:
        minimum = 0 if zero_ok else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"timeout policy {path!r} must be an integer >= {minimum}")
        return value
    if hint == tuple[float, ...]:
        if (
            not isinstance(value, list)
            or not value
            or not all(_is_number(v) and v >= 0 for v in value)
        ):
            raise ValueError(
                f"timeout policy {path!r} must be a non-empty list of numbers >= 0"
            )
        return tuple(float(v) for v in value)
    raise TypeError(f"unsupported timeout policy type for {path!r}: {hint!r}")


def _build(cls: type, data: Mapping[str, Any], prefix: str = "") -> Any:
    hints = get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    unknown = sorted(set(data) - set(names))
    if unknown:
        raise ValueError(f"unknown timeout policy key {prefix + unknown[0]!r}")
    missing = [n for n in names if n not in data]
    if missing:
        raise ValueError(f"missing timeout policy key {prefix + missing[0]!r}")
    kwargs: dict[str, Any] = {}
    for name in names:
        path, hint, value = prefix + name, hints[name], data[name]
        if is_dataclass(hint):
            if not isinstance(value, Mapping):
                raise ValueError(f"timeout policy {path!r} must be a table")
            kwargs[name] = _build(hint, value, f"{path}.")
        else:
            kwargs[name] = _coerce(hint, value, path)
    return cls(**kwargs)


def _apply_env(data: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    for name, path in _ENV_OVERRIDES.items():
        raw = env.get(name)
        if raw is None or not raw.strip():
            continue
        zero_ok = path in _ZERO_ALLOWED
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number.") from exc
        if not math.isfinite(value) or value < 0 or (value == 0 and not zero_ok):
            raise ValueError(
                f"{name} must be {'non-negative' if zero_ok else 'positive'}."
            )
        *parents, leaf = path.split(".")
        overlay: dict[str, Any] = {leaf: value}
        for parent in reversed(parents):
            overlay = {parent: overlay}
        data = _merge(data, overlay)
    return data


def load_timeout_policies(
    env: Mapping[str, str] | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> TimeoutPolicies:
    """Bundled file, then the operator file, then ``overrides``, then env vars."""
    source = os.environ if env is None else env
    data = _read_bundled()
    path = (source.get(TIMEOUTS_PATH_ENV) or "").strip()
    if path:
        data = _merge(data, _read_override(path))
    if overrides:
        data = _merge(data, overrides)
    data = _apply_env(data, source)
    return _build(TimeoutPolicies, data)


_cached: TimeoutPolicies | None = None


def get_timeout_policies() -> TimeoutPolicies:
    """The process-wide policies, loaded from ``os.environ`` on first use."""
    global _cached
    if _cached is None:
        _cached = load_timeout_policies()
    return _cached


def reset_timeout_policies() -> None:
    global _cached
    _cached = None


@contextmanager
def use_timeout_policies(policies: TimeoutPolicies) -> Iterator[TimeoutPolicies]:
    global _cached
    previous = _cached
    _cached = policies
    try:
        yield policies
    finally:
        _cached = previous
