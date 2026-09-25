# Timeout and Retry Policies in a Configuration File — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every agent-facing timeout/retry policy is read from one TOML file (bundled defaults + optional operator override), with today's values and no behaviour change.

**Architecture:** `src/internal/configs/timeouts.py` loads the bundled `timeouts.toml`, deep-merges an optional operator file, applies four legacy env vars, validates, and caches a frozen `TimeoutPolicies` tree. Every migrated site replaces its literal with a read of `get_timeout_policies()` at construct/call time; explicit kwargs still win.

**Tech Stack:** Python ≥3.10, `tomllib` (3.11+) / `tomli` (3.10), dataclasses, pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-timeout-retry-config-design.md`

## Global Constraints

- No behaviour change at default values: every site keeps the exact value it uses today.
- Precedence: explicit constructor/call kwarg > env var > operator file (`AGENTIC_SEARCH_TIMEOUTS_PATH`) > bundled `src/internal/configs/timeouts.toml`.
- The four env vars keep their names: `TOOL_APPROVAL_TIMEOUT_SECONDS`, `AGENTIC_SEARCH_GENERATION_TIMEOUT`, `LLM_SOCKET_READ_TIMEOUT`, `AGENTIC_SEARCH_SSE_HEARTBEAT_SECONDS`.
- Sites read policies at construct or call time, never at import time (no module-level `get_timeout_policies()` calls).
- `tomli; python_version < "3.11"` in `requirements.txt` and `pyproject.toml`; `tomli>=2.0.0` unconditional in `requirements-unit-test.txt`.
- `pyproject.toml` `[tool.setuptools.package-data]` ships `*.toml` as well as `*.json`.
- Validation (plan ruling, amending spec §1): `*_seconds` > 0 except `sse.heartbeat_seconds` and `llm.local_generation_timeout_seconds`, which accept 0 (= disabled, as today). Integer counts ≥ 1 except `tool_loop.recovery.max_retries` and `llm.grounded_max_retries`, which accept 0 — every other `max_retries` is an attempt count (`SearchClient` loops `range(max_retries)`), so 0 would make no request.
- Branch `feat/timeout-retry-config`; never commit to `main`; commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- The unit suite must pass with torch unimportable.

## Review Focus

1. **An operator raises an attempt count above the backoff list length** (`tools.public_data.max_attempts = 5` with two backoff steps) — the retry loop must reuse the last step, not raise `IndexError` (`_http._fetch` indexes `_RETRY_BACKOFF_SECONDS[attempt]` unclamped today; `RecoveryPolicy` already clamps) — test in Task 2.
2. **An operator sets an attempt-count key to 0** (`retrieval.client.max_retries = 0`) — must fail at load naming the key, not silently make zero requests — test in Task 1.
3. **A test or process changes an env var after policies were first read** (existing tests monkeypatch `AGENTIC_SEARCH_SSE_HEARTBEAT_SECONDS` per test) — the cache must not leak between tests: an autouse fixture resets it — Task 1; SSE test in Task 6.
4. **The installed package (wheel / container) has no repo checkout** — the bundled TOML must load through `importlib.resources` and be in the wheel — test in Task 1.
5. **A typo in the operator file** (`[llm] socket_read_timout_seconds = 30`) — load fails naming `llm.socket_read_timout_seconds`, instead of silently keeping 120 — test in Task 1.

---

## File Structure

- Create `src/internal/configs/timeouts.toml` — bundled defaults, commented.
- Create `src/internal/configs/timeouts.py` — policy dataclasses, loader, cache.
- Modify `src/internal/configs/__init__.py` — export `TimeoutPolicies`, `get_timeout_policies`, `load_timeout_policies`.
- Modify `src/internal/configs/app_configs.py` — `AppSettings.timeouts`; approval/generation fields filled from it.
- Modify sites (Tasks 2–6), delete dead copies (Task 7).
- Create `tests/unit/test_timeout_policies.py` (loader), `tests/unit/test_timeout_policy_sites.py` (per-site), `tests/unit/test_timeout_policy_drift.py` (guard).
- Create `docs/configuration/timeouts.md`.

Test helper used by every site test (defined in Task 1 inside `tests/unit/test_timeout_policy_sites.py`'s header and re-used by appending):

```python
from contextlib import contextmanager

from src.internal.configs.timeouts import load_timeout_policies, use_timeout_policies


@contextmanager
def overridden(overrides: dict):
    with use_timeout_policies(load_timeout_policies({}, overrides=overrides)) as p:
        yield p
```

---

### Task 1: Loader, bundled file, packaging, AppSettings

**Files:**
- Create: `src/internal/configs/timeouts.toml`, `src/internal/configs/timeouts.py`
- Modify: `src/internal/configs/__init__.py`, `src/internal/configs/app_configs.py:221-282,375-383`, `pyproject.toml`, `requirements.txt`, `requirements-unit-test.txt:33-34`, `tests/conftest.py`
- Test: `tests/unit/test_timeout_policies.py` (create), `tests/unit/test_timeout_policy_sites.py` (create with helper header only)

**Interfaces:**
- Produces: `TimeoutPolicies` (attrs `tools`, `retrieval`, `rerank`, `llm`, `tool_loop`, `sse`, nested exactly as the TOML tables), `load_timeout_policies(env: Mapping[str, str] | None = None, *, overrides: Mapping[str, Any] | None = None) -> TimeoutPolicies`, `get_timeout_policies() -> TimeoutPolicies`, `reset_timeout_policies() -> None`, `use_timeout_policies(policies) -> ContextManager[TimeoutPolicies]`, `TIMEOUTS_PATH_ENV = "AGENTIC_SEARCH_TIMEOUTS_PATH"`, `AppSettings.timeouts: TimeoutPolicies`.

- [ ] **Step 1: Write the failing tests** — `tests/unit/test_timeout_policies.py`:

```python
import dataclasses
import math

import pytest

from src.internal.configs.timeouts import (
    TIMEOUTS_PATH_ENV,
    get_timeout_policies,
    load_timeout_policies,
    reset_timeout_policies,
    use_timeout_policies,
)

TODAY = {
    "tools": {
        "public_data": {
            "timeout_seconds": 10.0, "max_attempts": 3, "backoff_seconds": (0.4, 0.8),
            "retry_budget_seconds": 15.0, "overpass_timeout_seconds": 30.0,
            "overpass_query_timeout_seconds": 25.0,
        },
        "web_search": {
            "google_timeout_seconds": 15.0, "serpapi_timeout_seconds": 15.0,
            "serper_timeout_seconds": 10.0, "fetch_page_timeout_seconds": 15.0,
            "fetch_pages_timeout_seconds": 10.0, "query_timeout_seconds": 15.0,
        },
        "search_router": {"timeout_seconds": 15.0, "max_retries": 3},
        "openapi": {"timeout_seconds": 15.0},
        "mcp": {"timeout_seconds": 30.0, "sse_read_timeout_seconds": 300.0},
    },
    "retrieval": {
        "client": {"timeout_seconds": 10.0, "max_retries": 3, "backoff_base_seconds": 0.5},
        "search_runner": {"timeout_seconds": 15.0, "max_retries": 3},
        "web_hybrid": {
            "provider_timeout_seconds": 5.0, "provider_max_retries": 1,
            "provider_wait_seconds": 8.0,
        },
    },
    "rerank": {"timeout_seconds": 10.0},
    "llm": {
        "socket_read_timeout_seconds": 120.0, "remote_total_timeout_seconds": 120.0,
        "local_generation_timeout_seconds": 120.0, "local_heartbeat_seconds": 10.0,
        "sufficiency_timeout_seconds": 5.0, "grounded_max_retries": 1,
    },
    "tool_loop": {
        "approval_timeout_seconds": 60.0, "escalation_timeout_seconds": 120.0,
        "max_escalations": 3, "tool_evidence_timeout_seconds": 5.0,
        "recovery": {
            "max_retries": 2, "backoff_seconds": (0.5, 1.0),
            "retry_after_cap_seconds": 4.0, "retry_budget_seconds": 10.0,
        },
    },
    "sse": {"heartbeat_seconds": 15.0},
}


def test_bundled_defaults_are_todays_values():
    assert dataclasses.asdict(load_timeout_policies({})) == TODAY


def test_bundled_file_loads_through_package_resources():
    from importlib import resources

    assert resources.files("src.internal.configs").joinpath("timeouts.toml").is_file()


def test_partial_override_file_deep_merges(tmp_path):
    f = tmp_path / "t.toml"
    f.write_text("[llm]\nsocket_read_timeout_seconds = 30\n")
    p = load_timeout_policies({TIMEOUTS_PATH_ENV: str(f)})
    assert p.llm.socket_read_timeout_seconds == 30.0
    assert p.llm.remote_total_timeout_seconds == 120.0
    assert p.tools.public_data.max_attempts == 3


def test_unknown_key_names_its_dotted_path(tmp_path):
    f = tmp_path / "t.toml"
    f.write_text("[llm]\nsocket_read_timout_seconds = 30\n")
    with pytest.raises(ValueError, match=r"llm\.socket_read_timout_seconds"):
        load_timeout_policies({TIMEOUTS_PATH_ENV: str(f)})


def test_unknown_table_names_its_path():
    with pytest.raises(ValueError, match=r"'nope'"):
        load_timeout_policies({}, overrides={"nope": {"x": 1}})


def test_missing_override_file_errors(tmp_path):
    with pytest.raises(ValueError, match="AGENTIC_SEARCH_TIMEOUTS_PATH"):
        load_timeout_policies({TIMEOUTS_PATH_ENV: str(tmp_path / "absent.toml")})


def test_unparsable_override_file_errors(tmp_path):
    f = tmp_path / "t.toml"
    f.write_text("[llm\n")
    with pytest.raises(ValueError, match="invalid TOML"):
        load_timeout_policies({TIMEOUTS_PATH_ENV: str(f)})


def test_empty_path_env_means_bundled_only():
    assert load_timeout_policies({TIMEOUTS_PATH_ENV: "  "}) == load_timeout_policies({})


@pytest.mark.parametrize(
    "overrides, path",
    [
        ({"rerank": {"timeout_seconds": 0}}, "rerank.timeout_seconds"),
        ({"rerank": {"timeout_seconds": -1}}, "rerank.timeout_seconds"),
        ({"rerank": {"timeout_seconds": True}}, "rerank.timeout_seconds"),
        ({"rerank": {"timeout_seconds": "10"}}, "rerank.timeout_seconds"),
        ({"rerank": {"timeout_seconds": math.inf}}, "rerank.timeout_seconds"),
        ({"retrieval": {"client": {"max_retries": 0}}}, "retrieval.client.max_retries"),
        ({"retrieval": {"client": {"max_retries": 2.5}}}, "retrieval.client.max_retries"),
        ({"tools": {"public_data": {"max_attempts": 0}}}, "tools.public_data.max_attempts"),
        ({"tool_loop": {"max_escalations": 0}}, "tool_loop.max_escalations"),
        ({"tool_loop": {"recovery": {"max_retries": -1}}}, "tool_loop.recovery.max_retries"),
        ({"tool_loop": {"recovery": {"backoff_seconds": []}}}, "tool_loop.recovery.backoff_seconds"),
        ({"tool_loop": {"recovery": {"backoff_seconds": [0.5, -1]}}}, "tool_loop.recovery.backoff_seconds"),
        ({"llm": 5}, "llm"),
    ],
)
def test_invalid_values_error_naming_the_key(overrides, path):
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        load_timeout_policies({}, overrides=overrides)


def test_zero_is_allowed_where_it_means_disabled_or_no_retry():
    p = load_timeout_policies(
        {},
        overrides={
            "sse": {"heartbeat_seconds": 0},
            "llm": {"local_generation_timeout_seconds": 0, "grounded_max_retries": 0},
            "tool_loop": {"recovery": {"max_retries": 0}},
        },
    )
    assert p.sse.heartbeat_seconds == 0.0
    assert p.llm.local_generation_timeout_seconds == 0.0
    assert p.llm.grounded_max_retries == 0
    assert p.tool_loop.recovery.max_retries == 0


@pytest.mark.parametrize(
    "env, attr, value",
    [
        ({"TOOL_APPROVAL_TIMEOUT_SECONDS": "12.5"}, ("tool_loop", "approval_timeout_seconds"), 12.5),
        ({"AGENTIC_SEARCH_GENERATION_TIMEOUT": "0"}, ("llm", "local_generation_timeout_seconds"), 0.0),
        ({"LLM_SOCKET_READ_TIMEOUT": "45"}, ("llm", "socket_read_timeout_seconds"), 45.0),
        ({"AGENTIC_SEARCH_SSE_HEARTBEAT_SECONDS": "7"}, ("sse", "heartbeat_seconds"), 7.0),
    ],
)
def test_env_var_beats_the_file(tmp_path, env, attr, value):
    f = tmp_path / "t.toml"
    f.write_text(
        "[llm]\nsocket_read_timeout_seconds = 1\nlocal_generation_timeout_seconds = 1\n"
        "[tool_loop]\napproval_timeout_seconds = 1\n[sse]\nheartbeat_seconds = 1\n"
    )
    p = load_timeout_policies({TIMEOUTS_PATH_ENV: str(f), **env})
    assert getattr(getattr(p, attr[0]), attr[1]) == value


@pytest.mark.parametrize("raw", ["0", "-1", "inf", "nan", "abc"])
def test_bad_approval_env_names_the_env_var(raw):
    with pytest.raises(ValueError, match="TOOL_APPROVAL_TIMEOUT_SECONDS"):
        load_timeout_policies({"TOOL_APPROVAL_TIMEOUT_SECONDS": raw})


def test_cache_and_use_and_reset(monkeypatch):
    monkeypatch.setenv("LLM_SOCKET_READ_TIMEOUT", "33")
    reset_timeout_policies()
    assert get_timeout_policies().llm.socket_read_timeout_seconds == 33.0
    custom = load_timeout_policies({}, overrides={"rerank": {"timeout_seconds": 2}})
    with use_timeout_policies(custom):
        assert get_timeout_policies().rerank.timeout_seconds == 2.0
    assert get_timeout_policies().llm.socket_read_timeout_seconds == 33.0


def test_tomli_fallback_loads_on_python_310_path():
    import subprocess
    import sys

    code = (
        "import sys, importlib.abc\n"
        "class B(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name == 'tomllib': raise ImportError('blocked')\n"
        "sys.meta_path.insert(0, B())\n"
        "from src.internal.configs.timeouts import load_timeout_policies, tomllib\n"
        "assert tomllib.__name__ == 'tomli', tomllib.__name__\n"
        "print(load_timeout_policies({}).rerank.timeout_seconds)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "10.0"


def test_app_settings_carry_the_same_policies():
    from src.internal.configs import load_app_settings

    s = load_app_settings({"TOOL_APPROVAL_TIMEOUT_SECONDS": "12.5"})
    assert s.timeouts.tool_loop.approval_timeout_seconds == 12.5
    assert s.tool_approval_timeout_seconds == 12.5
    assert s.generation_timeout_seconds == s.timeouts.llm.local_generation_timeout_seconds == 120.0
```

Create `tests/unit/test_timeout_policy_sites.py` with only the module docstring `"""Each migrated site reads its value from the timeout policies."""` and the `overridden` helper from File Structure.

- [ ] **Step 2: Run to verify failure** — `pytest -q tests/unit/test_timeout_policies.py` → FAIL (`ModuleNotFoundError: src.internal.configs.timeouts`).

- [ ] **Step 3: Write the bundled file** — `src/internal/configs/timeouts.toml`:

```toml
# Timeout and retry policies for everything an agent request waits on.
# Operators override any subset with a partial file named by
# AGENTIC_SEARCH_TIMEOUTS_PATH. Env vars noted below win over both files.
# Reference: docs/configuration/timeouts.md

[tools.public_data]
timeout_seconds = 10.0             # one keyless public-data HTTP call
max_attempts = 3                   # GET only; a POST is never retried
backoff_seconds = [0.4, 0.8]       # sleep before attempt 2, 3 (last step reused)
retry_budget_seconds = 15.0        # no new attempt starts after this wall time
overpass_timeout_seconds = 30.0    # search_nearby_places HTTP call
overpass_query_timeout_seconds = 25.0  # Overpass server-side [timeout:N]

[tools.web_search]
google_timeout_seconds = 15.0      # google_custom_search called directly
serpapi_timeout_seconds = 15.0     # serpapi_search and the cascade SerpAPI leg
serper_timeout_seconds = 10.0      # serper_dev_search
fetch_page_timeout_seconds = 15.0  # fetch_url, one page
fetch_pages_timeout_seconds = 10.0 # fetch_pages_concurrently, per page
query_timeout_seconds = 15.0       # web_search tool, per query

[tools.search_router]
timeout_seconds = 15.0             # search_tool, any provider
max_retries = 3                    # attempts; also the corpus search tool's reported attempts

[tools.openapi]
timeout_seconds = 15.0             # one OpenAPI-registered tool call

[tools.mcp]
timeout_seconds = 30.0             # MCP streamable-HTTP request
sse_read_timeout_seconds = 300.0   # MCP stream read

[retrieval.client]
timeout_seconds = 10.0             # SearchClient default, retrieval_search, search agents
max_retries = 3                    # attempts
backoff_base_seconds = 0.5         # sleep base * 2**attempt between attempts

[retrieval.search_runner]
timeout_seconds = 15.0             # context retrieval search_runner
max_retries = 3                    # attempts

[retrieval.web_hybrid]
provider_timeout_seconds = 5.0     # /api/agent hybrid search, per provider request
provider_max_retries = 1           # attempts per provider
provider_wait_seconds = 8.0        # wall clock per provider leg

[rerank]
timeout_seconds = 10.0             # HTTP cross-encoder reranker call

[llm]
socket_read_timeout_seconds = 120.0      # max gap between streamed chunks; env LLM_SOCKET_READ_TIMEOUT
remote_total_timeout_seconds = 120.0     # remote (OpenAI-compatible) server call
local_generation_timeout_seconds = 120.0 # local generation wall clock, 0 = none; env AGENTIC_SEARCH_GENERATION_TIMEOUT
local_heartbeat_seconds = 10.0           # local generation progress heartbeat
sufficiency_timeout_seconds = 5.0        # AgenticRAG sufficiency judge
grounded_max_retries = 1                 # grounded-answer regenerations, 0 = none

[tool_loop]
approval_timeout_seconds = 60.0    # wait for a user's tool approval; env TOOL_APPROVAL_TIMEOUT_SECONDS
escalation_timeout_seconds = 120.0 # wait for a user's failure escalation decision
max_escalations = 3                # escalations per run before it stops unresolved
tool_evidence_timeout_seconds = 5.0 # one tool call feeding answer evidence

[tool_loop.recovery]
max_retries = 2                    # extra attempts after the first, read-only transient only
backoff_seconds = [0.5, 1.0]       # jittered x U(0.5, 1.5); last step reused
retry_after_cap_seconds = 4.0      # cap on a provider's Retry-After
retry_budget_seconds = 10.0        # total retry sleep per run

[sse]
heartbeat_seconds = 15.0           # keepalive on an idle stream, 0 = off; env AGENTIC_SEARCH_SSE_HEARTBEAT_SECONDS
```

- [ ] **Step 4: Write the loader** — `src/internal/configs/timeouts.py`:

```python
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
except ModuleNotFoundError:  # Python 3.10
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
class TimeoutPolicies:
    tools: ToolsPolicies
    retrieval: RetrievalPolicies
    rerank: RerankPolicy
    llm: LLMPolicy
    tool_loop: ToolLoopPolicy
    sse: SSEPolicy


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


def _merge(base: dict[str, Any], over: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
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
            raise ValueError(f"{name} must be {'non-negative' if zero_ok else 'positive'}.")
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
```

- [ ] **Step 5: Wire exports, AppSettings, packaging, deps, test isolation**

`src/internal/configs/__init__.py`: add `from .timeouts import TimeoutPolicies, get_timeout_policies, load_timeout_policies` and the three names to `__all__` if the module has one.

`src/internal/configs/app_configs.py`: import `from .timeouts import TimeoutPolicies, get_timeout_policies, load_timeout_policies` (and `field` from dataclasses if not imported). Add as the last `AppSettings` field:

```python
    timeouts: TimeoutPolicies = field(default_factory=get_timeout_policies)
```

In `load_app_settings`, replace the `tool_approval_timeout_seconds = get_env_float(...)` block and its finiteness/positivity check (lines ~271-278) with:

```python
    timeouts = load_timeout_policies(source)
```

and at the `AppSettings(...)` construction pass `tool_approval_timeout_seconds=timeouts.tool_loop.approval_timeout_seconds`, `generation_timeout_seconds=timeouts.llm.local_generation_timeout_seconds` (replacing the `get_env_float(source, "AGENTIC_SEARCH_GENERATION_TIMEOUT", 120.0)` call) and `timeouts=timeouts`. Remove `math`/`get_env_float` imports only if this left them unused.

`pyproject.toml`: `dependencies = ["PyJWT[crypto]>=2.10.1,<3", "tomli>=2.0.0; python_version < '3.11'"]`; `[tool.setuptools.package-data]` `"*" = ["*.json", "*.toml"]`.

`requirements.txt`: append `tomli>=2.0.0; python_version < "3.11"  # timeouts.toml on 3.10`.

`requirements-unit-test.txt:33-34`: replace the comment + marker line with:

```
# Unconditional so the timeouts loader's tomli fallback (the Python 3.10
# path) is exercised on every CI interpreter, and for TOML packaging tests.
tomli>=2.0.0
```

`tests/conftest.py`: add

```python
@pytest.fixture(autouse=True)
def _fresh_timeout_policies():
    """Policies are cached per process; tests that set env vars must not leak."""
    from src.internal.configs.timeouts import reset_timeout_policies

    reset_timeout_policies()
    yield
    reset_timeout_policies()
```

(import `pytest` there if absent).

- [ ] **Step 6: Run tests** — `pytest -q tests/unit/test_timeout_policies.py tests/unit/test_configs.py` → PASS (the four existing `tool_approval_timeout` tests in `test_configs.py` pass unchanged). Then build a wheel and check the file ships: `python -m pip wheel . --no-deps -w /tmp/whl -q && python -c "import zipfile,glob; z=zipfile.ZipFile(glob.glob('/tmp/whl/*.whl')[0]); assert 'src/internal/configs/timeouts.toml' in z.namelist(); print('ok')"`.

- [ ] **Step 7: Mutation-check** — rename a key in `timeouts.toml` → `test_bundled_defaults_are_todays_values` FAILS; delete `"llm.grounded_max_retries"` from `_ZERO_ALLOWED` → `test_zero_is_allowed_...` FAILS; swap the `_apply_env` call before the operator-file merge → `test_env_var_beats_the_file` FAILS. Restore.

- [ ] **Step 8: Commit** — `git add` the files above; message `configs: timeout and retry policies load from a bundled TOML file`.

---

### Task 2: Public-data tools read their policy

**Files:**
- Modify: `src/internal/tools/public_data/_http.py:34-47,110-150,167-216`, `src/internal/tools/public_data/geo.py:19,294,306`
- Modify: `tests/unit/test_public_data_http.py:216,259` (they read the deleted constants)
- Test: `tests/unit/test_timeout_policy_sites.py` (append)

**Interfaces:**
- Consumes: `get_timeout_policies().tools.public_data.{timeout_seconds,max_attempts,backoff_seconds,retry_budget_seconds,overpass_timeout_seconds,overpass_query_timeout_seconds}`.
- Produces: `get_json/get_text/<third helper at :204>` accept `timeout_seconds: float | None = None`.

- [ ] **Step 1: Write the failing tests** (append). Reuse the `_SequencedSession` fake from `tests/unit/test_public_data_http.py` by importing it (read that file first; it scripts statuses and records `.calls`):

```python
import asyncio

import pytest


def test_public_data_attempts_follow_the_policy(monkeypatch):
    from src.internal.tools.public_data import _http
    from tests.unit.test_public_data_http import _SequencedSession  # read its API first

    real_sleep = asyncio.sleep
    _SequencedSession.script([503, 503, 503, 503, 503])
    monkeypatch.setattr(_http.aiohttp, "ClientSession", _SequencedSession)
    monkeypatch.setattr(_http.asyncio, "sleep", lambda s: real_sleep(0))
    with overridden({"tools": {"public_data": {"max_attempts": 5}}}):
        with pytest.raises(_http.PublicDataError):
            asyncio.run(_http.get_json("https://x.test"))
    assert len(_SequencedSession.calls) == 5  # 5 attempts, 2 backoff steps: last step reused


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
```

Adapt `_SequencedSession.script(...)` to that fake's real scripting API (it may be a class attribute list); keep the assertions.

- [ ] **Step 2: Run** — `pytest -q tests/unit/test_timeout_policy_sites.py -k public_data` → FAIL (first: `IndexError` or 3 calls; second: `seen == [10.0, 9]`).

- [ ] **Step 3: Implement.** In `_http.py` delete `DEFAULT_TIMEOUT_SECONDS`, `_MAX_ATTEMPTS`, `_RETRY_BACKOFF_SECONDS`, `_RETRY_BUDGET_SECONDS` (keep their explanatory comments, moved above the reads in `_fetch`), import `from src.internal.configs.timeouts import get_timeout_policies`, and in `_fetch`:

```python
    policy = get_timeout_policies().tools.public_data
    attempts = policy.max_attempts if method.upper() == "GET" else 1
    ...
        if time.monotonic() - started >= policy.retry_budget_seconds:
            break
        backoff = policy.backoff_seconds
        await asyncio.sleep(backoff[min(attempt, len(backoff) - 1)])
```

Each public helper (`get_json`, `get_text`, and the one at line ~204) takes `timeout_seconds: float | None = None` and passes `timeout_seconds=timeout_seconds if timeout_seconds is not None else get_timeout_policies().tools.public_data.timeout_seconds`. In `geo.py` delete `OVERPASS_TIMEOUT_SECONDS`; pass `timeout_seconds=get_timeout_policies().tools.public_data.overpass_timeout_seconds` and build the query prefix as `f"[out:json][timeout:{int(policy.overpass_query_timeout_seconds)}];\n"` (read `policy` once in that function). Update `tests/unit/test_public_data_http.py:216` to `== 3` via `get_timeout_policies().tools.public_data.max_attempts` and `:259` to `get_timeout_policies().tools.public_data.retry_budget_seconds + 1.0`. Grep `grep -rn "OVERPASS_TIMEOUT_SECONDS\|DEFAULT_TIMEOUT_SECONDS" src tests` must show only `servers/web_search/serp.py` (out of scope).

- [ ] **Step 4: Run** — `pytest -q tests/unit/test_timeout_policy_sites.py tests/unit/test_public_data_http.py tests/unit -k "public_data or geo or nearby"` → PASS.

- [ ] **Step 5: Mutation-check** — restore `_RETRY_BACKOFF_SECONDS[attempt]` indexing → first test FAILS (`IndexError`); hard-code `10.0` in `get_json` → second test FAILS. Restore.

- [ ] **Step 6: Commit** — `tools: public-data HTTP and Overpass read the timeout policies`.

---

### Task 3: Web search, search router, corpus tool, OpenAPI, MCP

**Files:**
- Modify: `src/internal/tools/search.py:282,321,372,411-412,447-448,535,633,655,731`, `src/internal/tools/routing_tools.py:17-19,81`, `src/internal/tools/api.py:225`, `src/internal/tools/mcp_client.py:139`
- Test: `tests/unit/test_timeout_policy_sites.py` (append)

**Interfaces:**
- Consumes: `tools.web_search.*`, `tools.search_router.{timeout_seconds,max_retries}`, `retrieval.client.{timeout_seconds,max_retries}`, `tools.openapi.timeout_seconds`, `tools.mcp.*`.
- Produces: every listed search.py parameter becomes `timeout_seconds: float | None = None` (and `max_retries: int | None = None`), resolved at the top of the function body.

Mapping (default → key): `google_custom_search` → `tools.web_search.google_timeout_seconds`; `serpapi_search` and `_cascade` → `serpapi_timeout_seconds`; `serper_dev_search` → `serper_timeout_seconds`; `retrieval_search` → `retrieval.client.timeout_seconds` / `retrieval.client.max_retries`; `search_tool` → `tools.search_router.timeout_seconds` / `max_retries`; `fetch_url` → `fetch_page_timeout_seconds`; `fetch_pages_concurrently` → `fetch_pages_timeout_seconds`; `MultiQueryWebSearchTool.__init__` → `query_timeout_seconds` (resolve in `__init__`, store in `self._timeout_seconds`). The cascade browser leg keeps calling `browser_fn` without a timeout, so it uses `search_tool`'s resolved default — unchanged.

- [ ] **Step 1: Write the failing tests** (append):

```python
def test_search_tool_uses_router_policy_and_kwarg_wins(monkeypatch):
    from src.internal.tools import search

    seen = []

    async def fake_retrieval(query, **kw):
        seen.append((kw["timeout_seconds"], kw["max_retries"]))
        return []

    monkeypatch.setattr(search, "retrieval_search", fake_retrieval)
    with overridden({"tools": {"search_router": {"timeout_seconds": 4, "max_retries": 2}}}):
        asyncio.run(search.search_tool("q"))
        asyncio.run(search.search_tool("q", timeout_seconds=6, max_retries=1))
    assert seen == [(4.0, 2), (6, 1)]


def test_retrieval_search_uses_client_policy(monkeypatch):
    from src.internal.tools import search

    configs = []

    class FakeClient:
        def __init__(self, config):
            configs.append(config)

        async def search(self, *a, **k):
            return []

        async def aclose(self):
            pass

    monkeypatch.setattr(search, "SearchClient", FakeClient)
    with overridden({"retrieval": {"client": {"timeout_seconds": 7, "max_retries": 2}}}):
        asyncio.run(search.retrieval_search("q", search_url="http://x"))
    assert (configs[0].timeout_seconds, configs[0].max_retries) == (7.0, 2)


def test_google_search_uses_web_search_policy(monkeypatch):
    from src.internal.tools import search

    seen = []

    async def fake_get_json(url, *, params, timeout_seconds, **kw):
        seen.append(timeout_seconds)
        return {"items": []}

    monkeypatch.setattr(search, "_get_json", fake_get_json)
    with overridden({"tools": {"web_search": {"google_timeout_seconds": 3}}}):
        asyncio.run(search.google_custom_search("q", api_key="k", cse_id="c"))
    assert seen == [3.0]


def test_corpus_search_reports_router_attempts(monkeypatch):
    from src.internal.tools import routing_tools
    from src.internal.tools.search import SearchPage

    async def failing(*a, **k):
        return [SearchPage(error="down")]

    monkeypatch.setattr(routing_tools, "search_tool", failing)
    tool = routing_tools.build_search_routing_tool(search_url="http://x", top_k=3)
    with overridden({"tools": {"search_router": {"max_retries": 5}}}):
        text, _raw, meta = asyncio.run(tool.execute("i", {"query": "q"}))
    assert meta["failure"].provider_attempts == 5


def test_openapi_tool_timeout_follows_policy(monkeypatch):
    from src.internal.tools import api as api_tools

    seen = []
    real = api_tools.aiohttp.ClientTimeout
    monkeypatch.setattr(
        api_tools.aiohttp, "ClientTimeout", lambda total: seen.append(total) or real(total=total)
    )
    # Build one ApiRequestTool the way tests/unit/test_api_tools*.py do (read one first)
    # and execute it against a stubbed session; assert seen[0] == 2.0 under
    # overridden({"tools": {"openapi": {"timeout_seconds": 2}}}).
```

Complete `test_openapi_tool_timeout_follows_policy` using the existing OpenAPI tool test harness (find it with `grep -rln "ApiRequestTool\|register_from_openapi" tests/unit`), and add `test_mcp_client_passes_policy_timeouts` asserting the kwargs `streamablehttp_client` receives under `overridden({"tools": {"mcp": {"timeout_seconds": 3, "sse_read_timeout_seconds": 9}}})` are `timeout=3.0, sse_read_timeout=9.0` (monkeypatch `mcp.client.streamable_http.streamablehttp_client` with an async context manager that records kwargs and raises after recording; follow `tests/unit/test_mcp_client.py`'s existing fakes). If `SearchClient`'s real method names differ from `search`/`aclose`, match them — read `src/context/retrieval/client.py` and `retrieval_search`.

- [ ] **Step 2: Run** — `pytest -q tests/unit/test_timeout_policy_sites.py -k "search or corpus or openapi or mcp"` → FAIL.

- [ ] **Step 3: Implement** the mapping above. Pattern at the top of each function body:

```python
    if timeout_seconds is None:
        timeout_seconds = get_timeout_policies().tools.web_search.google_timeout_seconds
```

`routing_tools.py`: delete `_SEARCH_ATTEMPTS` and its comment; at the failure site pass `provider_attempts=get_timeout_policies().tools.search_router.max_retries` with a comment `# search_tool's attempts, read from the same policy it uses`. `api.py:225`: `aiohttp.ClientTimeout(total=get_timeout_policies().tools.openapi.timeout_seconds)`. `mcp_client.py:139`: read `policy = get_timeout_policies().tools.mcp` and pass `timeout=policy.timeout_seconds, sse_read_timeout=policy.sse_read_timeout_seconds`.

- [ ] **Step 4: Run** — the site tests plus `pytest -q tests/unit -k "search or routing or api_tool or openapi or mcp or web_search or domain"` → PASS.

- [ ] **Step 5: Mutation-check** — restore `_SEARCH_ATTEMPTS = 3` → corpus test FAILS; restore `timeout_seconds: int = 15` on `search_tool` → router test FAILS. Restore.

- [ ] **Step 6: Commit** — `tools: web search, corpus search, OpenAPI and MCP read the timeout policies`.

---

### Task 4: Retrieval client, search runner, search agents, web hybrid, reranker

**Files:**
- Modify: `src/context/retrieval/client.py:54-55,105`, `src/context/retrieval/search_runner.py:26-27,84-85,108-109`, `src/agents/search/search.py:255-256`, `src/agents/generation/single_turn.py:81-82`, `src/internal/servers/web/app.py:2935-2936,2973`, `src/internal/search/stages.py:139`
- Test: `tests/unit/test_timeout_policy_sites.py` (append)

**Interfaces:**
- Consumes: `retrieval.client.*`, `retrieval.search_runner.*`, `retrieval.web_hybrid.*`, `rerank.timeout_seconds`.

Dataclass fields become policy-backed factories:

```python
from dataclasses import field

from src.internal.configs.timeouts import get_timeout_policies


def _client_policy():
    return get_timeout_policies().retrieval.client

    # in SearchClientConfig:
    timeout_seconds: float = field(default_factory=lambda: _client_policy().timeout_seconds)
    max_retries: int = field(default_factory=lambda: _client_policy().max_retries)
```

`client.py:105` becomes `await asyncio.sleep(_client_policy().backoff_base_seconds * (2**attempt))`. The agent config dataclasses (`search_timeout_seconds`, `search_max_retries` in `agents/search/search.py` and `agents/generation/single_turn.py`) use the same two factories. `search_runner.py`'s three functions take `timeout_seconds: float | None = None, max_retries: int | None = None` resolved from `retrieval.search_runner`. `app.py:2935-2936` pass `timeout_seconds=hybrid.provider_timeout_seconds, max_retries=hybrid.provider_max_retries` and `:2973` `timeout=hybrid.provider_wait_seconds`, with `hybrid = get_timeout_policies().retrieval.web_hybrid` read once at the top of the enclosing function. `stages.py:139` `timeout: float | None = None` → `self._timeout = timeout if timeout is not None else get_timeout_policies().rerank.timeout_seconds`.

Check each dataclass's field order before converting: a `field(default_factory=...)` is still a defaulted field, so ordering rules are unchanged.

- [ ] **Step 1: Write the failing tests** (append):

```python
def test_search_client_config_defaults_follow_policy():
    from src.context.retrieval.client import SearchClientConfig

    with overridden({"retrieval": {"client": {"timeout_seconds": 7, "max_retries": 2}}}):
        cfg = SearchClientConfig(url="http://x")
    assert (cfg.timeout_seconds, cfg.max_retries) == (7.0, 2)
    assert SearchClientConfig(url="http://x", timeout_seconds=1).timeout_seconds == 1


def test_search_runner_defaults_follow_policy(monkeypatch):
    from src.context.retrieval import search_runner

    seen = []

    async def fake_search_tool(query, **kw):
        seen.append((kw["timeout_seconds"], kw["max_retries"]))
        return []

    # Read search_runner.py: patch whatever name its first function calls with
    # timeout_seconds/max_retries (search_tool or SearchClient) and call that
    # function with only its required arguments.
    monkeypatch.setattr(search_runner, "search_tool", fake_search_tool, raising=False)
    with overridden({"retrieval": {"search_runner": {"timeout_seconds": 4, "max_retries": 2}}}):
        ...  # call each of the three public functions once with required args
    assert seen and all(s == (4.0, 2) for s in seen)


def test_agent_search_configs_follow_client_policy():
    from src.agents.generation.single_turn import SingleTurnConfig  # read the real class name
    from src.agents.search.search import SearchAgentConfig  # read the real class name

    with overridden({"retrieval": {"client": {"timeout_seconds": 7, "max_retries": 2}}}):
        for cfg in (SearchAgentConfig(), SingleTurnConfig()):
            assert (cfg.search_timeout_seconds, cfg.search_max_retries) == (7.0, 2)


def test_rerank_stage_timeout_follows_policy():
    from src.internal.search.stages import RerankHTTPRankingStage

    with overridden({"rerank": {"timeout_seconds": 2}}):
        stage = RerankHTTPRankingStage("http://r", document_contents=lambda c: "")
    assert stage._timeout == 2.0
```

Fill the two `...`/class-name spots from the real code (the dataclass names at `agents/search/search.py:~250` and `single_turn.py:~75`; `search_runner`'s three function names and callee). Add `test_web_hybrid_uses_policy`: patch `app.search_tool` (the name imported in `app.py`) with a recorder, drive the hybrid path through the smallest existing test that reaches `app.py:2930` (find with `grep -rln "hybrid" tests/unit/servers/web`), and assert `(5.0, 1)` becomes `(3.0, 2)` under `overridden({"retrieval": {"web_hybrid": {"provider_timeout_seconds": 3, "provider_max_retries": 2}}})`. If no test reaches it cheaply, extract nothing — instead assert via AST in Task 7's drift guard that `app.py` has no `timeout_seconds=5`/`max_retries=1`/`timeout=8.0` literals, and say so in the report.

- [ ] **Step 2: Run** — `pytest -q tests/unit/test_timeout_policy_sites.py -k "client or runner or agent_search or rerank or hybrid"` → FAIL.

- [ ] **Step 3: Implement** as specified above.

- [ ] **Step 4: Run** — site tests + `pytest -q tests/unit -k "retrieval or search_runner or search_agent or single_turn or rerank or hybrid or stages"` + `tests/unit/servers/web/test_web_experience_app.py` → PASS.

- [ ] **Step 5: Mutation-check** — restore `timeout_seconds: int = 10` in `SearchClientConfig` → first test FAILS; restore `timeout: float = 10.0` in the rerank stage → rerank test FAILS. Restore.

- [ ] **Step 6: Commit** — `retrieval: client, runner, agents, web hybrid and reranker read the timeout policies`.

---

### Task 5: LLM timeouts and grounded-generation retries

**Files:**
- Modify: `src/internal/llm/multi_llm.py:59,317-319`, `src/model/serving.py:252-257,423-435`, `src/agents/search/agentic_rag.py:129`, `src/context/models.py:285`, `src/context/pipeline.py:200`
- Test: `tests/unit/test_timeout_policy_sites.py` (append)

**Interfaces:**
- Consumes: `llm.*`.

Changes:
- `multi_llm.py`: delete module constant `LLM_SOCKET_READ_TIMEOUT` (only reader is line 319); `if timeout is None: self._timeout = get_timeout_policies().llm.socket_read_timeout_seconds`. The env var is honoured by the loader.
- `serving.py` `OpenAIServerManager.__init__`: `timeout_seconds: float | None = None` → `get_timeout_policies().llm.remote_total_timeout_seconds`. `LocalServerManager.__init__`: `generation_timeout_seconds: float | None = <sentinel>` is subtle — today `None` means "no timeout". Use a module sentinel `_FROM_POLICY = object()`: default `generation_timeout_seconds: float | None | object = _FROM_POLICY`; if it is `_FROM_POLICY`, read `llm.local_generation_timeout_seconds` (0 keeps meaning "none", as the existing `> 0` check at line ~491 handles). `generation_heartbeat_seconds: float | None = None` → `llm.local_heartbeat_seconds`.
- `agentic_rag.py:129`: `sufficiency_timeout_s: float = field(default_factory=lambda: get_timeout_policies().llm.sufficiency_timeout_seconds)`.
- `context/models.py:285`: `max_retries: int = field(default_factory=lambda: get_timeout_policies().llm.grounded_max_retries)`.
- `pipeline.py:200`: `cap = get_timeout_policies().llm.grounded_max_retries` then `max_attempts = 1 + min(max(request.grounded_generation.max_retries, 0), cap)`.

- [ ] **Step 1: Write the failing tests** (append):

```python
def test_llm_socket_read_timeout_follows_policy():
    from src.internal.llm import multi_llm

    # Construct the class defined around multi_llm.py:300 with the minimum
    # arguments its existing tests use (grep tests for its name), timeout omitted.
    with overridden({"llm": {"socket_read_timeout_seconds": 33}}):
        llm = _make_default_llm(multi_llm)
    assert llm._timeout == 33.0


def test_remote_server_manager_timeout_follows_policy():
    from src.model.serving import OpenAIServerManager

    with overridden({"llm": {"remote_total_timeout_seconds": 44}}):
        m = OpenAIServerManager(server_url="http://x")  # match its real required args
    assert m.timeout_seconds == 44.0


def test_local_generation_timeout_policy_and_explicit_none():
    from src.model.serving import LocalServerManager

    with overridden({"llm": {"local_generation_timeout_seconds": 50, "local_heartbeat_seconds": 2}}):
        m = LocalServerManager.__new__(LocalServerManager)
        LocalServerManager.__init__(m, model_path="m")  # if __init__ loads a model, patch the loader as its tests do
    assert (m.generation_timeout_seconds, m.generation_heartbeat_seconds) == (50.0, 2.0)
    # explicit None still means "no timeout"


def test_sufficiency_and_grounded_retries_follow_policy():
    from src.agents.search.agentic_rag import AgenticRAGConfig  # real name at agentic_rag.py:~120
    from src.context.models import GroundedGenerationConfig

    with overridden({"llm": {"sufficiency_timeout_seconds": 2, "grounded_max_retries": 0}}):
        assert AgenticRAGConfig().sufficiency_timeout_s == 2.0
        assert GroundedGenerationConfig().max_retries == 0
```

Define `_make_default_llm` in the test file after reading `multi_llm.py`'s class and its existing tests. For `test_local_generation_timeout_policy_and_explicit_none`, add a second construction with `generation_timeout_seconds=None` asserting `None`. Add `test_grounded_clamp_reads_policy`: with `grounded_max_retries = 0` and a request whose `grounded_generation.max_retries = 1`, the pipeline makes exactly one generation attempt — drive it through the smallest existing grounded-generation test in `tests/unit` (`grep -rln "grounded_generation" tests/unit`).

- [ ] **Step 2: Run** — `pytest -q tests/unit/test_timeout_policy_sites.py -k "llm or server_manager or local_generation or sufficiency or grounded"` → FAIL.

- [ ] **Step 3: Implement** as specified.

- [ ] **Step 4: Run** — site tests + `pytest -q tests/unit -k "multi_llm or serving or server_manager or agentic_rag or grounded or pipeline"` + `tests/unit/servers/web/test_web_experience_app.py` → PASS.

- [ ] **Step 5: Mutation-check** — restore `timeout_seconds: int = 120` on `OpenAIServerManager` → its test FAILS; restore `min(..., 1)` in `pipeline.py` → clamp test FAILS. Restore.

- [ ] **Step 6: Commit** — `llm: socket, remote, local generation and grounded retries read the timeout policies`.

---

### Task 6: Tool loop, brokers, recovery, tool evidence, SSE

**Files:**
- Modify: `src/agents/tool/tool_calling.py:228-230`, `src/agents/tool/recovery.py:32-35`, `src/internal/servers/web/tool_agent_runner.py:214-216`, `src/internal/servers/web/tool_approval.py:257,295`, `src/internal/servers/web/app.py:1576-1579`, `src/context/tool_evidence.py:61`, `src/context/pipeline.py:434`, `src/internal/servers/sse.py:48-53`
- Test: `tests/unit/test_timeout_policy_sites.py` (append)

**Interfaces:**
- Consumes: `tool_loop.*`, `tool_loop.recovery.*`, `sse.heartbeat_seconds`.

Changes:
- `ToolAgentLoopConfig`: `approval_timeout_seconds`, `escalation_timeout_seconds`, `max_escalations` become `field(default_factory=lambda: get_timeout_policies().tool_loop.<key>)`.
- `RecoveryPolicy`: `max_retries`, `backoff` (from `recovery.backoff_seconds`), `retry_after_cap` (from `retry_after_cap_seconds`), `retry_budget` (from `retry_budget_seconds`) become `default_factory` fields. Field names stay.
- `tool_agent_runner.py:214-216`: pass `approval_timeout_seconds=resolved.tool_approval_timeout_seconds` (delete the `getattr(..., 60.0)` fallback; `resolved` is an `AppSettings`, read the surrounding code to confirm — if `resolved` can be a non-`AppSettings` object, use `get_timeout_policies().tool_loop.approval_timeout_seconds` as the fallback instead of the literal).
- `ToolApprovalBroker.__init__(self, timeout_seconds: float | None = None)` → `tool_loop.approval_timeout_seconds`; `ToolEscalationBroker.__init__(self, timeout_seconds: float | None = None)` → `tool_loop.escalation_timeout_seconds`. `app.py:1579`: `ToolEscalationBroker(resolved.timeouts.tool_loop.escalation_timeout_seconds)`.
- `collect_tool_evidence(..., timeout_seconds: float | None = None)` and `pipeline.py:434` `tool_timeout_seconds: float | None = None`, both resolved from `tool_loop.tool_evidence_timeout_seconds`; keep the existing `<= 0` `ValueError` check after resolution.
- `sse.heartbeat_seconds()` returns `get_timeout_policies().sse.heartbeat_seconds`; docstring keeps "Zero disables it." and names the env var and file key.

- [ ] **Step 1: Write the failing tests** (append):

```python
def test_tool_loop_config_and_recovery_follow_policy():
    from src.agents.tool.recovery import RecoveryPolicy
    from src.agents.tool.tool_calling import ToolAgentLoopConfig

    with overridden(
        {
            "tool_loop": {
                "approval_timeout_seconds": 11, "escalation_timeout_seconds": 22,
                "max_escalations": 4,
                "recovery": {"max_retries": 1, "backoff_seconds": [0.1], "retry_budget_seconds": 3},
            }
        }
    ):
        cfg = ToolAgentLoopConfig()
        policy = RecoveryPolicy()
    assert (cfg.approval_timeout_seconds, cfg.escalation_timeout_seconds, cfg.max_escalations) == (11.0, 22.0, 4)
    assert (policy.max_retries, policy.backoff, policy.retry_budget) == (1, (0.1,), 3.0)
    assert RecoveryPolicy(max_retries=0).max_retries == 0


def test_brokers_follow_policy():
    from src.internal.servers.web.tool_approval import ToolApprovalBroker, ToolEscalationBroker

    with overridden({"tool_loop": {"approval_timeout_seconds": 11, "escalation_timeout_seconds": 22}}):
        assert ToolApprovalBroker()._timeout_seconds == 11.0
        assert ToolEscalationBroker()._timeout_seconds == 22.0


def test_escalation_timeout_is_overridable_end_to_end(tmp_path, monkeypatch):
    from src.internal.configs.timeouts import TIMEOUTS_PATH_ENV

    f = tmp_path / "t.toml"
    f.write_text("[tool_loop]\nescalation_timeout_seconds = 42\n")
    monkeypatch.setenv(TIMEOUTS_PATH_ENV, str(f))
    # Build the web app the way tests/unit/servers/web/test_tool_admin_api.py's
    # _make_app does, enter its lifespan (TestClient context), and assert:
    # app.state.tool_escalation_broker._timeout_seconds == 42.0


def test_tool_evidence_timeout_follows_policy(monkeypatch):
    import src.context.tool_evidence as te

    seen = []
    real_wait_for = asyncio.wait_for

    async def spy(aw, timeout):
        seen.append(timeout)
        return await real_wait_for(aw, timeout)

    monkeypatch.setattr(te.asyncio, "wait_for", spy)
    # Call te.collect_tool_evidence with a stub registry/selector as its existing
    # tests do (grep tests for collect_tool_evidence) under
    # overridden({"tool_loop": {"tool_evidence_timeout_seconds": 2}}).
    assert seen and set(seen) == {2.0}


def test_sse_heartbeat_env_still_wins(monkeypatch):
    from src.internal.servers import sse

    monkeypatch.setenv("AGENTIC_SEARCH_SSE_HEARTBEAT_SECONDS", "0")
    assert sse.heartbeat_seconds() == 0.0
```

Complete the two commented tests from the named harnesses.

- [ ] **Step 2: Run** — `pytest -q tests/unit/test_timeout_policy_sites.py -k "tool_loop or brokers or escalation or evidence or sse"` → FAIL.

- [ ] **Step 3: Implement** as specified.

- [ ] **Step 4: Run** — site tests + `pytest -q tests/unit -k "tool_calling or recovery or approval or escalation or tool_evidence or sse or tool_agent_runner or pipeline"` + `tests/unit/servers/web` → PASS.

- [ ] **Step 5: Mutation-check** — restore `escalation_timeout_seconds: float = 120.0` → config test FAILS; make `ToolEscalationBroker.__init__` fall back to a literal `120.0` → end-to-end test FAILS; make `sse.heartbeat_seconds` return `15.0` → SSE test FAILS. Restore.

- [ ] **Step 6: Commit** — `tool loop: approval, escalation, recovery, tool evidence and SSE read the timeout policies`.

---

### Task 7: Delete dead copies, drift guard, docs

**Files:**
- Modify: `src/internal/configs/chat_configs.py:32`, `src/internal/configs/default_config.py:76`, `src/internal/configs/agent_configs.py` (the `AGENT_TIMEOUT_*` constants)
- Create: `tests/unit/test_timeout_policy_drift.py`, `docs/configuration/timeouts.md`
- Modify: `.claude/CLAUDE.md` (Configuration section, ~line 240), `.env.example`

- [ ] **Step 1: Prove the deletions are unreachable** — for each name, an AST import scan (not grep) over `src`, `examples`, `tests`:

```bash
python - <<'EOF'
import ast, pathlib
names = {"LLM_SOCKET_READ_TIMEOUT"} | {
    n.id for n in ast.walk(ast.parse(pathlib.Path("src/internal/configs/agent_configs.py").read_text()))
    if isinstance(n, ast.Name) and n.id.startswith("AGENT_TIMEOUT")
}
hits = []
for p in [*pathlib.Path("src").rglob("*.py"), *pathlib.Path("examples").rglob("*.py"), *pathlib.Path("tests").rglob("*.py")]:
    tree = ast.parse(p.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(a.name in names for a in node.names):
            hits.append((str(p), node.lineno))
        if isinstance(node, ast.Attribute) and node.attr in names:
            hits.append((str(p), node.lineno))
print(len(names), "names;", hits or "no importers")
EOF
```

Expected: `no importers` (after Task 5 removed `multi_llm`'s own constant). If there are hits, keep those names and record them in the report.

- [ ] **Step 2: Write the drift guard** — `tests/unit/test_timeout_policy_drift.py`:

```python
"""Schema-owned timeout/retry values must not reappear as literals."""

import ast
import pathlib

import pytest

MIGRATED = [
    "src/internal/tools/public_data/_http.py",
    "src/internal/tools/public_data/geo.py",
    "src/internal/tools/search.py",
    "src/internal/tools/routing_tools.py",
    "src/internal/tools/api.py",
    "src/internal/tools/mcp_client.py",
    "src/context/retrieval/client.py",
    "src/context/retrieval/search_runner.py",
    "src/agents/search/search.py",
    "src/agents/generation/single_turn.py",
    "src/internal/search/stages.py",
    "src/internal/llm/multi_llm.py",
    "src/model/serving.py",
    "src/agents/search/agentic_rag.py",
    "src/context/models.py",
    "src/agents/tool/tool_calling.py",
    "src/agents/tool/recovery.py",
    "src/internal/servers/web/tool_approval.py",
    "src/context/tool_evidence.py",
    "src/context/pipeline.py",
    "src/internal/servers/sse.py",
]
POLICY_WORDS = ("timeout", "retries", "attempts", "backoff", "retry_budget", "heartbeat")


def _is_policy_name(name: str) -> bool:
    lowered = name.lower()
    return any(w in lowered for w in POLICY_WORDS)


def _numeric(node) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ) or (isinstance(node, ast.Tuple) and node.elts and all(_numeric(e) for e in node.elts))


def _violations(path: str) -> list[str]:
    tree = ast.parse(pathlib.Path(path).read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            pos = args.posonlyargs + args.args
            pairs = list(zip(pos[len(pos) - len(args.defaults):], args.defaults))
            pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
            found += [f"{path}:{d.lineno} {a.arg}" for a, d in pairs if _is_policy_name(a.arg) and _numeric(d)]
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if _is_policy_name(node.target.id) and node.value is not None and _numeric(node.value):
                found.append(f"{path}:{node.lineno} {node.target.id}")
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper() and _is_policy_name(t.id) and _numeric(node.value):
                    found.append(f"{path}:{node.lineno} {t.id}")
    return found


@pytest.mark.parametrize("path", MIGRATED)
def test_no_literal_policy_defaults(path):
    assert _violations(path) == []


def test_guard_catches_a_literal(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("def f(*, timeout_seconds: int = 15):\n    pass\n")
    assert _violations(str(f)) == [f"{f}:1 timeout_seconds"]
```

- [ ] **Step 3: Run** — `pytest -q tests/unit/test_timeout_policy_drift.py`. Any violation that is NOT a schema-owned value (e.g. a local `max_retries` unrelated to this spec, or a value the spec puts out of scope) is listed in the report and added to an explicit `ALLOWED = {"path:name", ...}` set in the test with a one-line reason each — never widened silently. A violation that IS schema-owned means a site was missed: migrate it.

- [ ] **Step 4: Delete dead copies** — `chat_configs.py:32` `LLM_SOCKET_READ_TIMEOUT` line; `default_config.py:76` entry becomes a comment `# LLM_SOCKET_READ_TIMEOUT: see src/internal/configs/timeouts.toml (llm.socket_read_timeout_seconds, 120)` if the dict is documentation, else delete the entry; every `AGENT_TIMEOUT_*` assignment in `agent_configs.py` (keep any other content of the module).

- [ ] **Step 5: Docs** — `docs/configuration/timeouts.md`: one table per TOML table (key, default, what it bounds, env override if any), the precedence line, the `AGENTIC_SEARCH_TIMEOUTS_PATH` example:

```toml
# /etc/agentic/timeouts.toml — only the keys you change
[llm]
socket_read_timeout_seconds = 60.0

[tool_loop.recovery]
max_retries = 1
```

and the validation rules (which keys accept 0). `.claude/CLAUDE.md` Configuration bullet list gains `- \`AGENTIC_SEARCH_TIMEOUTS_PATH\` — partial TOML overriding \`src/internal/configs/timeouts.toml\` (timeouts/retries; see docs/configuration/timeouts.md)`. `.env.example` gains a commented `# AGENTIC_SEARCH_TIMEOUTS_PATH=/etc/agentic/timeouts.toml`.

- [ ] **Step 6: Run** — `pytest -q` (whole default suite) and the torch-blocked unit run:

```bash
cat > /tmp/notorch.py <<'EOF'
import sys, importlib.abc
class B(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in {"torch", "sentence_transformers", "transformers"}:
            raise ImportError(f"blocked {name}")
sys.meta_path.insert(0, B())
if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", *sys.argv[1:]]))
EOF
python /tmp/notorch.py tests/unit
```

→ both PASS.

- [ ] **Step 7: Mutation-check** — re-add `timeout_seconds: int = 15` to `search_tool` → drift guard FAILS naming `search.py`. Restore.

- [ ] **Step 8: Commit** — `configs: delete dead timeout copies, guard against literal drift, document the policy file`.
