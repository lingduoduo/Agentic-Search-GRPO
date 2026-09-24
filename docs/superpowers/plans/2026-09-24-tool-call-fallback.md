# Tool-call Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give tool-call failures three ordered recovery levels — retry, degrade gracefully, escalate to the chat user — owned by `ToolAgentLoop`, visible on every entry point.

**Architecture:** Adapters return typed failures (`ToolErrorText`, a `str` subclass carrying a `ToolFailure`); `ToolRegistry.invoke_detailed` surfaces them. A pure `RecoveryPolicy` decides retry / unavailable / escalate / feed-back per failure and tool effect. `ToolAgentLoop._call_tool` applies the decision with per-run state; escalations call an `on_escalation` callback that the web layer backs with a `ToolEscalationBroker` (a sibling of the approval broker on a shared `DecisionBroker` base), an `escalation_required` event, a decide endpoint, a WebSocket message and a `ToolEscalationCard`.

**Tech Stack:** Python 3.10+, asyncio, FastAPI, pytest/pytest-asyncio, React 19 + TypeScript + Vitest.

**Spec:** `docs/superpowers/specs/2026-09-24-tool-call-fallback-design.md`

## Global Constraints

- Retry, degrade and escalate apply only after approval: a denied/expired approval stays `SKIPPED` and is never retried, degraded or escalated.
- Automatic retry only for `ToolEffect.READ_ONLY`; `SIDE_EFFECTING`/`UNSPECIFIED` failures escalate (never auto-replayed).
- Policy defaults: `max_retries=2`, `backoff=(0.5, 1.0)` with jitter `uniform(0.5, 1.5)`, `retry_after_cap=4.0`, `retry_budget=10.0` s per run; escalation timeout 120 s; at most 3 escalations per run.
- A failure whose provider already retried (`provider_attempts > 1`) is never retried again by the loop.
- `ToolRegistry.invoke()` keeps its signature and `(response, raw, errors)` tuple; every non-loop caller sees byte-identical text.
- Nothing classifies failure by searching text for "error".
- `asyncio.CancelledError` always propagates.
- Error codes: `invalid_input` keeps the existing `error_code="invalid_arguments"`, `not_found` keeps `"tool_not_found"`; degraded calls use `error_code="tool_unavailable"`; other failures use the category (`transient`, `permanent`, `unknown`).
- Fixed answers, verbatim: cancel → `Stopped: <tool> failed (<category>); nothing further was attempted.`; unresolved → `<tool> failed and was not retried; the action may not have completed.`; degraded note appended to a non-empty answer → `Note: <tools> was unavailable, so this answer may be incomplete.`
- Approval behaviour, events, endpoint and counters are unchanged (existing approval tests pass untouched).
- Unit tests must pass with torch unimportable. Branch `feat/tool-failure-recovery`; never commit to `main`; commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`; run `ruff check --fix` and `ruff format` on touched Python files before each commit and confirm with `git log -1` (the pre-commit hook rejects unformatted files silently). Do not add or commit `.planning/` or the untracked `docs/superpowers/{specs,plans}/2026-09-24-tool-failure-recovery*` files.

**Spec narrowing (decided while planning):** the loop's `tool_recovery.escalations` records carry `tool`, `category`, `attempts` and `decision` but not arguments — the loop cannot import the web layer's `sanitize_tool_arguments`, and the card already shows sanitized arguments through the broker view.

## Review Focus

1. **Two concurrent calls in one batch both fail transiently**: the shared retry budget must not be overspent (reserved before sleeping) — test in Task 4.
2. **A read-only tool the model calls again on a later turn after it was marked unavailable** must not execute (short-circuit `SKIPPED`). Calls within one concurrent batch all start before any of them degrades, so the guarantee is per later turn — test in Task 4.
3. **The escalation callback raises or hangs**: treated as expired → run stops `unresolved`, never success — test in Task 5.
4. **A stopped run's fixed answer must survive the runners' fallbacks**: explicit tool mode's `answer or _assistant_fallback`, the auto-route's blank→CHAT fallback and `/tool` JSON — test in Task 6.
5. **A decision posted by a different user, or twice**: 403 / 409, and the approval broker's counters are unaffected by escalations — test in Task 7.

---

### Task 1: Typed failures and `invoke_detailed`

**Files:**
- Modify: `src/internal/tools/base.py` (types after `ToolEffect`; `FunctionTool.execute`)
- Modify: `src/internal/tools/registry.py` (`invoke`, new `invoke_detailed`, `ToolInvocation`)
- Modify: `src/internal/tools/__init__.py` (export the new names)
- Test: `tests/unit/test_tool_failures.py`

**Interfaces:**
- Produces: `FailureCategory` (str Enum: `TRANSIENT="transient"`, `PERMANENT="permanent"`, `INVALID_INPUT="invalid_input"`, `NOT_FOUND="not_found"`, `UNKNOWN="unknown"`); `ToolFailure(category, message, retry_after=None, provider_attempts=0)` frozen; `ToolErrorText(text, failure)` (`str` subclass, `.failure`); `ToolInvocation(response, raw, errors, failure)` frozen; `ToolRegistry.invoke_detailed(name, arguments, *, validate=True) -> ToolInvocation`. All exported from `src.internal.tools`.

- [ ] **Step 1: Write the failing tests**

```python
import asyncio

import pytest

from src.internal.tools import (
    FailureCategory,
    FunctionTool,
    ToolErrorText,
    ToolFailure,
    ToolRegistry,
)


def _failure(category=FailureCategory.TRANSIENT):
    return ToolFailure(category, "upstream temporarily unavailable")


def test_tool_error_text_is_its_text():
    text = ToolErrorText('{"error": "boom"}', _failure())
    assert text == '{"error": "boom"}'
    assert isinstance(text, str)
    assert text.failure.category is FailureCategory.TRANSIENT


def _registry(fn):
    registry = ToolRegistry()
    registry.register(FunctionTool.from_fn(name="t", description="t")(fn))
    return registry


def test_invoke_detailed_surfaces_a_typed_failure():
    async def t():
        return ToolErrorText('{"error": "boom"}', _failure())

    got = asyncio.run(_registry(t).invoke_detailed("t", {}))
    assert got.response == '{"error": "boom"}'
    assert got.failure.category is FailureCategory.TRANSIENT
    assert got.errors == []


def test_invoke_keeps_its_tuple_and_text():
    async def t():
        return ToolErrorText('{"error": "boom"}', _failure())

    response, raw, errors = asyncio.run(_registry(t).invoke("t", {}))
    assert (response, errors) == ('{"error": "boom"}', [])


def test_invoke_detailed_success_has_no_failure():
    async def t():
        return {"ok": 1}

    got = asyncio.run(_registry(t).invoke_detailed("t", {}))
    assert got.failure is None and got.response == '{"ok": 1}'


def test_invoke_detailed_maps_not_found_and_invalid_input():
    registry = ToolRegistry()
    registry.register(
        FunctionTool.from_fn(
            name="t",
            description="t",
            parameters={
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
            },
        )(lambda n: n)
    )
    missing = asyncio.run(registry.invoke_detailed("nope", {}))
    assert missing.failure.category is FailureCategory.NOT_FOUND
    assert missing.errors
    invalid = asyncio.run(registry.invoke_detailed("t", {}))
    assert invalid.failure.category is FailureCategory.INVALID_INPUT


@pytest.mark.parametrize(
    ("exc", "category"),
    [
        (asyncio.TimeoutError(), FailureCategory.TRANSIENT),
        (ConnectionError("reset"), FailureCategory.TRANSIENT),
        (ValueError("bad"), FailureCategory.UNKNOWN),
    ],
)
def test_invoke_detailed_maps_raised_exceptions(exc, category):
    async def t():
        raise exc

    got = asyncio.run(_registry(t).invoke_detailed("t", {}))
    assert got.failure.category is category
    assert got.failure.message == type(exc).__name__  # never the raw text


def test_invoke_detailed_propagates_cancellation():
    async def t():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_registry(t).invoke_detailed("t", {}))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_tool_failures.py -q`
Expected: collection error, `ImportError: cannot import name 'FailureCategory'`.

- [ ] **Step 3: Implement**

`src/internal/tools/base.py`, after `class ToolEffect`:

```python
class FailureCategory(str, Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    INVALID_INPUT = "invalid_input"
    NOT_FOUND = "not_found"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ToolFailure:
    """Why a tool call failed, in a form the recovery policy can act on.

    ``message`` is a short fixed-vocabulary description safe to show a user;
    raw exception text and provider bodies stay in the logs.
    """

    category: FailureCategory
    message: str
    retry_after: float | None = None
    provider_attempts: int = 0


class ToolErrorText(str):
    """Error text a tool returns, carrying the typed failure behind it.

    A ``str``, so every caller that only reads text sees exactly what it did
    before; the agent loop reads ``.failure``.
    """

    failure: ToolFailure

    def __new__(cls, text: str, failure: ToolFailure) -> "ToolErrorText":
        obj = super().__new__(cls, text)
        obj.failure = failure
        return obj
```

In `FunctionTool.execute`, replace the final `return response, result, {}` with:

```python
        meta = {"failure": result.failure} if isinstance(result, ToolErrorText) else {}
        return response, result, meta
```

`src/internal/tools/registry.py`: add imports `import asyncio`, `import logging`, `from dataclasses import dataclass` (if absent) and `from .base import FailureCategory, ToolFailure`; add `logger = logging.getLogger(__name__)` if absent. Add above `class ToolRegistry`:

```python
@dataclass(frozen=True, slots=True)
class ToolInvocation:
    response: str
    raw: Any
    errors: list[str]
    failure: ToolFailure | None


def _failure_from_exception(exc: Exception) -> ToolFailure:
    transient = isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError))
    try:
        import aiohttp

        transient = transient or isinstance(exc, aiohttp.ClientError)
    except ImportError:  # pragma: no cover - aiohttp is a declared dependency
        pass
    category = FailureCategory.TRANSIENT if transient else FailureCategory.UNKNOWN
    return ToolFailure(category, type(exc).__name__)
```

Replace the body of `invoke` from `entry = self._entries.get(name)` to the end with a shared helper, and add `invoke_detailed`:

```python
    def _resolve(
        self, name: str, arguments: dict[str, Any], validate: bool
    ) -> tuple[Tool | None, list[str]]:
        entry = self._entries.get(name)
        if entry is None:
            return None, [f"Tool {name!r} not found."]
        if validate and entry.tool.schema.parameters:
            return entry.tool, validate_arguments(entry.tool.schema.parameters, arguments)
        return entry.tool, []

    async def invoke(self, name, arguments, *, validate=True):  # keep the existing signature and docstring
        tool, errors = self._resolve(name, arguments, validate)
        if tool is None or errors:
            return "", None, errors
        instance_id = await tool.create()
        try:
            response, raw, _meta = await tool.execute(instance_id, arguments)
        finally:
            await tool.release(instance_id)
        return response, raw, []

    async def invoke_detailed(
        self, name: str, arguments: dict[str, Any], *, validate: bool = True
    ) -> ToolInvocation:
        """Like ``invoke``, but every failure comes back typed instead of raised."""
        tool, errors = self._resolve(name, arguments, validate)
        if tool is None:
            return ToolInvocation(
                "", None, errors, ToolFailure(FailureCategory.NOT_FOUND, errors[0])
            )
        if errors:
            return ToolInvocation(
                "", None, errors,
                ToolFailure(FailureCategory.INVALID_INPUT, "; ".join(errors)),
            )
        instance_id = await tool.create()
        try:
            try:
                response, raw, meta = await tool.execute(instance_id, arguments)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("tool %r raised", name, exc_info=True)
                return ToolInvocation("", None, [], _failure_from_exception(exc))
        finally:
            await tool.release(instance_id)
        failure = meta.get("failure") if isinstance(meta, dict) else None
        return ToolInvocation(response, raw, [], failure)
```

(Keep `invoke`'s real signature line and docstring exactly as they are; only its body changes.)

Export `FailureCategory`, `ToolFailure`, `ToolErrorText` and `ToolInvocation` from `src/internal/tools/__init__.py` alongside `ToolEffect`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_tool_failures.py tests/unit/test_tool_registry.py tests/unit/test_tool_arg_validation.py -q`
Expected: all PASS.

- [ ] **Step 5: Mutation-check**

(a) In `FunctionTool.execute` always return `{}` as meta → `test_invoke_detailed_surfaces_a_typed_failure` FAILS. (b) In `_failure_from_exception` drop `ConnectionError` → the `ConnectionError` case FAILS. (c) Remove the `except asyncio.CancelledError: raise` → `test_invoke_detailed_propagates_cancellation` FAILS (CancelledError is a BaseException on 3.8+, so check it still fails; if it does not, record that the guard is belt-and-braces and keep it). Restore; delete `__pycache__`; re-run PASS.

- [ ] **Step 6: Commit**

```bash
git add src/internal/tools/base.py src/internal/tools/registry.py src/internal/tools/__init__.py tests/unit/test_tool_failures.py docs/superpowers/plans/2026-09-24-tool-call-fallback.md
git commit -m "tools: typed tool failures and ToolRegistry.invoke_detailed"
```

---

### Task 2: Adapters report typed failures

**Files:**
- Modify: `src/internal/tools/public_data/_http.py` (`PublicDataError`, `_fetch`, `guarded`)
- Modify: `src/internal/tools/mcp_client.py` (`_result_text`)
- Test: `tests/unit/test_tool_failures.py` (append)

**Interfaces:**
- Consumes: `FailureCategory`, `ToolFailure`, `ToolErrorText` (Task 1).
- Produces: `PublicDataError(message, *, status=None, attempts=1, retry_after=None, transport=False)`; `guarded` returns `ToolErrorText` on failure with byte-identical JSON text; `_result_text` returns `ToolErrorText` with `UNKNOWN` when `isError`.

- [ ] **Step 1: Write the failing tests** (append)

```python
import json
from types import SimpleNamespace

from src.internal.tools.mcp_client import _result_text
from src.internal.tools.public_data._http import PublicDataError, guarded


def _guarded_failure(exc):
    @guarded
    async def tool():
        raise exc

    return asyncio.run(tool())


@pytest.mark.parametrize(
    ("exc", "category", "attempts"),
    [
        (PublicDataError("x returned HTTP 503", status=503, attempts=3), FailureCategory.TRANSIENT, 3),
        (PublicDataError("request to x failed: reset", transport=True, attempts=3), FailureCategory.TRANSIENT, 3),
        (PublicDataError("x returned HTTP 404", status=404, attempts=1), FailureCategory.PERMANENT, 1),
        (PublicDataError("x returned a non-JSON body", attempts=1), FailureCategory.PERMANENT, 1),
        (RuntimeError("surprise"), FailureCategory.UNKNOWN, 0),
    ],
)
def test_guarded_classifies_and_keeps_its_json_text(exc, category, attempts):
    text = _guarded_failure(exc)
    assert isinstance(text, ToolErrorText)
    assert text.failure.category is category
    assert text.failure.provider_attempts == attempts
    assert "error" in json.loads(text)  # same body shape as before


def test_guarded_passes_retry_after_through():
    text = _guarded_failure(PublicDataError("x returned HTTP 429", status=429, attempts=3, retry_after=2.0))
    assert text.failure.retry_after == 2.0


def test_guarded_success_is_plain_text():
    @guarded
    async def tool():
        return {"ok": 1}

    text = asyncio.run(tool())
    assert not isinstance(text, ToolErrorText) and json.loads(text) == {"ok": 1}


def test_public_data_error_message_constructor_still_works():
    exc = PublicDataError("x returned HTTP 500")
    assert str(exc) == "x returned HTTP 500" and exc.status is None and exc.attempts == 1


def test_mcp_is_error_is_an_unknown_failure_with_the_same_text():
    result = SimpleNamespace(content=[SimpleNamespace(text="quota exceeded")], isError=True)
    text = _result_text(result)
    assert text == "Error: quota exceeded"
    assert text.failure.category is FailureCategory.UNKNOWN


def test_mcp_success_is_plain_text():
    result = SimpleNamespace(content=[SimpleNamespace(text="fine")], isError=False)
    assert not isinstance(_result_text(result), ToolErrorText)
```

Also add to `tests/unit/test_public_data_http.py` (read its existing fake-session pattern first and reuse it) one test per branch that `_fetch` stamps the attributes: a 503 three times → raised error has `status=503`, `attempts=3`; a 404 → `status=404`, `attempts=1`; a 429 with header `Retry-After: 2` → `retry_after=2.0`; a connection exception three times → `transport=True`, `attempts=3`.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_tool_failures.py tests/unit/test_public_data_http.py -q`
Expected: FAIL (`PublicDataError` takes no `status` keyword; `guarded` returns plain `str`).

- [ ] **Step 3: Implement**

`_http.py`:

```python
class PublicDataError(Exception):
    """An upstream call failed. ``guarded`` turns this into {"error": ...}."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        attempts: int = 1,
        retry_after: float | None = None,
        transport: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.attempts = attempts
        self.retry_after = retry_after
        self.transport = transport


def _retry_after(value: str | None) -> float | None:
    """Delta-seconds only; an HTTP-date or garbage is ignored."""
    try:
        seconds = float(value) if value is not None else None
    except ValueError:
        return None
    return seconds if seconds is not None and seconds >= 0 else None
```

In `_fetch`:
- the HTTP-error raise becomes `raise PublicDataError(f"{url} returned HTTP {response.status}", status=response.status, retry_after=_retry_after(response.headers.get("Retry-After")))`;
- in `except PublicDataError as exc:` use `exc.status` instead of `_status_of(exc)`, and before re-raising a non-retryable status set `exc.attempts = attempt + 1`;
- the transport branch becomes `last_error = PublicDataError(f"request to {url} failed: {exc}", transport=True)`;
- before the final `raise last_error or ...`, set `last_error.attempts = attempt + 1` when `last_error` is not None.

Delete `_status_of` if nothing else uses it (`grep -n "_status_of" -r src tests`); if a test uses it, keep it.

`guarded`:

```python
_TRANSIENT_MESSAGE = "upstream temporarily unavailable"
_PERMANENT_MESSAGE = "upstream refused or could not answer the request"
_UNKNOWN_MESSAGE = "tool failed unexpectedly"


def _classify(exc: PublicDataError) -> ToolFailure:
    transient = exc.transport or exc.status in _RETRYABLE_STATUSES
    return ToolFailure(
        FailureCategory.TRANSIENT if transient else FailureCategory.PERMANENT,
        _TRANSIENT_MESSAGE if transient else _PERMANENT_MESSAGE,
        retry_after=exc.retry_after,
        provider_attempts=exc.attempts,
    )
```

and in `_wrapped`:

```python
        except PublicDataError as exc:
            return ToolErrorText(json.dumps({"error": str(exc)}), _classify(exc))
        except Exception as exc:  # noqa: BLE001 - a tool must never raise
            logger.debug("tool %s failed", getattr(fn, "__name__", "?"), exc_info=True)
            return ToolErrorText(
                json.dumps({"error": f"{type(exc).__name__}: {exc}"}),
                ToolFailure(FailureCategory.UNKNOWN, _UNKNOWN_MESSAGE),
            )
```

`mcp_client._result_text`: when `isError`, `return ToolErrorText(f"Error: {text}" if text else "Error: tool call failed.", ToolFailure(FailureCategory.UNKNOWN, "remote tool reported an error"))`.

Imports: `from src.internal.tools.base import FailureCategory, ToolErrorText, ToolFailure` in both modules (use the module's existing import style).

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_tool_failures.py tests/unit/test_public_data_http.py tests/unit/test_public_data_knowledge.py tests/unit -q -k "public_data or mcp or tool_failures"`
Expected: all PASS.

- [ ] **Step 5: Mutation-check**

(a) In `_classify` drop `exc.transport or` → the transport case FAILS. (b) In `_fetch` do not set `attempts` on the final raise → the `attempts=3` `_fetch` test FAILS. (c) Return plain `str` from `_result_text` on `isError` → the MCP test FAILS. Restore, re-run.

- [ ] **Step 6: Commit**

```bash
git add src/internal/tools/public_data/_http.py src/internal/tools/mcp_client.py tests/unit/test_tool_failures.py tests/unit/test_public_data_http.py
git commit -m "tools: public-data and MCP adapters report typed failures"
```

---

### Task 3: `RecoveryPolicy`

**Files:**
- Create: `src/agents/tool/recovery.py`
- Test: `tests/unit/test_tool_recovery_policy.py`

**Interfaces:**
- Consumes: `FailureCategory`, `ToolFailure`, `ToolEffect` (`src.internal.tools`).
- Produces: `Action` (Enum: `RETRY`, `UNAVAILABLE`, `ESCALATE`, `FEED_BACK`); `Decision(action, delay=0.0)`; `RecoveryPolicy(max_retries=2, backoff=(0.5, 1.0), retry_after_cap=4.0, retry_budget=10.0)` with `decide(failure, effect, retries_so_far, budget_left, uniform=random.uniform) -> Decision`; `RecoveryState` (Task 4 fills it in; defined here).

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from src.agents.tool.recovery import Action, RecoveryPolicy
from src.internal.tools import FailureCategory, ToolEffect, ToolFailure

POLICY = RecoveryPolicy()
MID = lambda low, high: 1.0  # noqa: E731 - jitter factor 1.0


def _f(category, *, provider_attempts=0, retry_after=None):
    return ToolFailure(category, "m", retry_after=retry_after, provider_attempts=provider_attempts)


@pytest.mark.parametrize("effect", list(ToolEffect))
@pytest.mark.parametrize("category", [FailureCategory.INVALID_INPUT, FailureCategory.NOT_FOUND])
def test_input_failures_go_back_to_the_model(effect, category):
    assert POLICY.decide(_f(category), effect, 0, 10.0, MID).action is Action.FEED_BACK


@pytest.mark.parametrize("effect", [ToolEffect.SIDE_EFFECTING, ToolEffect.UNSPECIFIED])
@pytest.mark.parametrize(
    "category", [FailureCategory.TRANSIENT, FailureCategory.PERMANENT, FailureCategory.UNKNOWN]
)
def test_non_read_only_failures_escalate(effect, category):
    assert POLICY.decide(_f(category), effect, 0, 10.0, MID).action is Action.ESCALATE


def test_transient_read_only_retries_with_backoff():
    first = POLICY.decide(_f(FailureCategory.TRANSIENT), ToolEffect.READ_ONLY, 0, 10.0, MID)
    second = POLICY.decide(_f(FailureCategory.TRANSIENT), ToolEffect.READ_ONLY, 1, 10.0, MID)
    assert (first.action, first.delay) == (Action.RETRY, 0.5)
    assert (second.action, second.delay) == (Action.RETRY, 1.0)


def test_retries_run_out():
    got = POLICY.decide(_f(FailureCategory.TRANSIENT), ToolEffect.READ_ONLY, 2, 10.0, MID)
    assert got.action is Action.UNAVAILABLE


def test_provider_already_retried_is_not_retried_again():
    got = POLICY.decide(
        _f(FailureCategory.TRANSIENT, provider_attempts=3), ToolEffect.READ_ONLY, 0, 10.0, MID
    )
    assert got.action is Action.UNAVAILABLE


def test_retry_after_is_honoured_up_to_the_cap():
    got = POLICY.decide(
        _f(FailureCategory.TRANSIENT, retry_after=2.0), ToolEffect.READ_ONLY, 0, 10.0, MID
    )
    assert got.delay == 2.0
    capped = POLICY.decide(
        _f(FailureCategory.TRANSIENT, retry_after=60.0), ToolEffect.READ_ONLY, 0, 10.0, MID
    )
    assert capped.delay == 4.0


def test_a_delay_that_does_not_fit_the_budget_degrades():
    got = POLICY.decide(_f(FailureCategory.TRANSIENT), ToolEffect.READ_ONLY, 0, 0.3, MID)
    assert got.action is Action.UNAVAILABLE


@pytest.mark.parametrize("category", [FailureCategory.PERMANENT, FailureCategory.UNKNOWN])
def test_permanent_and_unknown_read_only_degrade(category):
    assert POLICY.decide(_f(category), ToolEffect.READ_ONLY, 0, 10.0, MID).action is Action.UNAVAILABLE


def test_jitter_scales_the_backoff():
    got = POLICY.decide(
        _f(FailureCategory.TRANSIENT), ToolEffect.READ_ONLY, 0, 10.0, lambda low, high: 1.5
    )
    assert got.delay == 0.75
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_tool_recovery_policy.py -q`
Expected: collection error, `ModuleNotFoundError: src.agents.tool.recovery`.

- [ ] **Step 3: Implement** `src/agents/tool/recovery.py`

```python
"""Retry → degrade → escalate: the per-call recovery decision for tool failures.

Pure policy, no I/O: ``ToolAgentLoop`` owns the state and applies the decision.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.internal.tools import FailureCategory, ToolEffect, ToolFailure


class Action(Enum):
    RETRY = "retry"
    UNAVAILABLE = "unavailable"
    ESCALATE = "escalate"
    FEED_BACK = "feed_back"


@dataclass(frozen=True)
class Decision:
    action: Action
    delay: float = 0.0


@dataclass(frozen=True)
class RecoveryPolicy:
    max_retries: int = 2
    backoff: tuple[float, ...] = (0.5, 1.0)
    retry_after_cap: float = 4.0
    retry_budget: float = 10.0

    def decide(
        self,
        failure: ToolFailure,
        effect: ToolEffect,
        retries_so_far: int,
        budget_left: float,
        uniform: Callable[[float, float], float] = random.uniform,
    ) -> Decision:
        if failure.category in (FailureCategory.INVALID_INPUT, FailureCategory.NOT_FOUND):
            return Decision(Action.FEED_BACK)
        if effect is not ToolEffect.READ_ONLY:
            # Replaying might repeat an action; only the user may authorise it.
            return Decision(Action.ESCALATE)
        if (
            failure.category is FailureCategory.TRANSIENT
            # The provider already retried: retrying again would multiply attempts.
            and failure.provider_attempts <= 1
            and retries_so_far < self.max_retries
        ):
            step = self.backoff[min(retries_so_far, len(self.backoff) - 1)]
            delay = max(
                step * uniform(0.5, 1.5),
                min(failure.retry_after or 0.0, self.retry_after_cap),
            )
            if delay <= budget_left:
                return Decision(Action.RETRY, delay)
        return Decision(Action.UNAVAILABLE)


@dataclass
class RecoveryState:
    """Per-run recovery bookkeeping, shared by the calls of one run."""

    budget_left: float
    unavailable: set[str] = field(default_factory=set)
    retries: int = 0
    degraded: list[str] = field(default_factory=list)
    escalations: list[dict[str, Any]] = field(default_factory=list)
    escalation_count: int = 0
    outcome: str | None = None  # None | "cancelled" | "unresolved"
    stop_answer: str | None = None

    def mark_unavailable(self, tool_name: str) -> None:
        if tool_name not in self.unavailable:
            self.unavailable.add(tool_name)
            self.degraded.append(tool_name)

    def stop(self, outcome: str, answer: str) -> None:
        if self.stop_answer is None:  # the first stop wins
            self.outcome, self.stop_answer = outcome, answer

    def summary(self) -> dict[str, Any] | None:
        if not (self.retries or self.degraded or self.escalations):
            return None
        outcome = self.outcome or ("degraded" if self.degraded else "recovered")
        return {
            "outcome": outcome,
            "needs_user": outcome == "unresolved",
            "retries": self.retries,
            "degraded": list(self.degraded),
            "escalations": list(self.escalations),
        }
```

(The unused-`field` import check: `field` is used by `RecoveryState`.)

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_tool_recovery_policy.py -q`
Expected: all PASS.

- [ ] **Step 5: Mutation-check**

(a) Drop `and failure.provider_attempts <= 1` → `test_provider_already_retried_is_not_retried_again` FAILS. (b) Drop the `min(..., self.retry_after_cap)` → the cap assertion FAILS. (c) Swap the `effect` check to `is ToolEffect.SIDE_EFFECTING` only → the `UNSPECIFIED` escalation cases FAIL. Restore, re-run.

- [ ] **Step 6: Commit**

```bash
git add src/agents/tool/recovery.py tests/unit/test_tool_recovery_policy.py
git commit -m "agents: RecoveryPolicy decides retry, degrade, escalate or feed back"
```

---

### Task 4: Retry and degrade in `ToolAgentLoop`

**Files:**
- Modify: `src/agents/tool/tool_calling.py` (`__init__`, `_call_tool`, `_tool_message_content`, `run`)
- Modify: `src/agents/core/base.py` (`AgentLoopOutput.tool_recovery`)
- Modify: `src/agents/core/state.py` (the `TaskStatus` docstring sentence "nothing in this repo schedules, queues or retries a tool" becomes "a tool may be retried inside one call; the status is the final outcome")
- Test: `tests/unit/test_tool_recovery_loop.py`

**Interfaces:**
- Consumes: `RecoveryPolicy`, `RecoveryState`, `Action` (Task 3); `invoke_detailed` (Task 1).
- Produces: `ToolAgentLoop(..., recovery_policy: RecoveryPolicy | None = None)`; `_call_tool(tool_call, state=None, on_escalation=None)`; `AgentLoopOutput.tool_recovery: dict | None` (shape = `RecoveryState.summary()`); constant `UNAVAILABLE_NOTE`.

- [ ] **Step 1: Write the failing tests**

```python
import asyncio
import json

from src.agents import ToolAgentLoop, ToolAgentLoopConfig
from src.agents.core.state import TaskStatus
from src.agents.tool.recovery import RecoveryPolicy
from src.internal.tools import (
    FailureCategory,
    FunctionTool,
    ToolEffect,
    ToolErrorText,
    ToolFailure,
)

FAST = RecoveryPolicy(backoff=(0.0, 0.0))


class _Tokenizer:
    chat_template = ""

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=True):
        text = "\n".join(m.get("content", "") for m in messages)
        return self.encode(text) if tokenize else text


class _Manager:
    def __init__(self, tokenizer, responses):
        self.tokenizer = tokenizer
        self.responses = iter(responses)
        self.prompts = []

    async def generate(self, request_id, prompt_ids, sampling_params):
        self.prompts.append(self.tokenizer.decode(prompt_ids))
        return self.tokenizer.encode(next(self.responses))


def _loop(tools, responses, policy=FAST):
    tokenizer = _Tokenizer()
    manager = _Manager(tokenizer, responses)
    return (
        ToolAgentLoop(
            tokenizer, manager, tools, ToolAgentLoopConfig(response_length=8192),
            recovery_policy=policy,
        ),
        manager,
    )


def _trace(output):
    return [json.loads(line) for line in (output.action_trace or "").splitlines()]


def _flaky(fail_times, category=FailureCategory.TRANSIENT):
    calls = []

    @FunctionTool.from_fn(name="lookup", effect=ToolEffect.READ_ONLY)
    async def lookup():
        calls.append(1)
        if len(calls) <= fail_times:
            return ToolErrorText('{"error": "x"}', ToolFailure(category, "upstream temporarily unavailable"))
        return {"ok": True}

    return lookup, calls


CALL = '{"name":"lookup","arguments":{}}'


def test_transient_read_only_failure_recovers_on_retry():
    tool, calls = _flaky(1)
    loop, _ = _loop([tool], [CALL, "done"])
    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {}))
    [result] = _trace(output)
    assert len(calls) == 2
    assert result["status"] == str(TaskStatus.COMPLETED)
    assert result["retry_count"] == 1
    assert output.tool_recovery == {
        "outcome": "recovered", "needs_user": False, "retries": 1, "degraded": [], "escalations": [],
    }


def test_exhausted_retries_mark_the_tool_unavailable_and_short_circuit():
    tool, calls = _flaky(99)
    loop, manager = _loop([tool], [CALL, CALL, "answered from what I have"])
    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {}))
    first, second = _trace(output)
    assert len(calls) == 3  # 1 + max_retries; the second model call never executes
    assert (first["status"], first["error_code"]) == (str(TaskStatus.FAILED), "tool_unavailable")
    assert (second["status"], second["error_code"]) == (str(TaskStatus.SKIPPED), "tool_unavailable")
    assert '"status": "unavailable"' in manager.prompts[1]
    assert output.tool_recovery["degraded"] == ["lookup"]
    assert output.tool_recovery["outcome"] == "degraded"
    assert output.final_answer == "answered from what I have"


def test_permanent_read_only_failure_degrades_without_retry():
    tool, calls = _flaky(99, FailureCategory.PERMANENT)
    loop, _ = _loop([tool], [CALL, "done"])
    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {}))
    assert len(calls) == 1
    assert output.tool_recovery["degraded"] == ["lookup"]


class _NoJitter(RecoveryPolicy):
    """Deterministic delays: the loop never passes a jitter source."""

    def decide(self, *args, **kwargs):
        return super().decide(*args, uniform=lambda low, high: 1.0)


def test_concurrent_retries_share_and_never_overspend_the_budget():
    tool, calls = _flaky(99)
    # Each retry costs exactly 0.04 s; the run has 0.06 s. The first call to
    # reserve takes 0.04, leaving 0.02 — too little for any further retry.
    policy = _NoJitter(backoff=(0.04, 0.04), retry_budget=0.06)
    loop, _ = _loop([tool], [f"[{CALL},{CALL}]", "done"], policy=policy)
    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {}))
    assert output.tool_recovery["retries"] == 1


def test_invalid_arguments_still_go_back_to_the_model():
    @FunctionTool.from_fn(
        name="lookup",
        effect=ToolEffect.READ_ONLY,
        parameters={"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
    )
    def lookup(n):
        return n

    loop, _ = _loop([lookup], [CALL, "done"])
    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {}))
    [result] = _trace(output)
    assert result["error_code"] == "invalid_arguments"
    assert output.tool_recovery is None


def test_no_failures_means_no_recovery_summary():
    tool, _ = _flaky(0)
    loop, _ = _loop([tool], [CALL, "done"])
    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {}))
    assert output.tool_recovery is None
```

Before relying on the batch syntax `[{...},{...}]`, confirm the JSON tool parser accepts a list of calls (`grep -n "def extract_tool_calls" -A30 src/agents/tool/*parser*.py`); if it takes one call per line instead, emit the two calls separated by `\n`.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_tool_recovery_loop.py -q`
Expected: FAIL — `ToolAgentLoop.__init__() got an unexpected keyword argument 'recovery_policy'`.

- [ ] **Step 3: Implement**

`src/agents/core/base.py`: add as the last `AgentLoopOutput` field

```python
    # Retry/degrade/escalate summary (RecoveryState.summary()); None when no
    # tool call needed recovery.
    tool_recovery: dict[str, Any] | None = None
```

`src/agents/tool/tool_calling.py`:

- imports: `from .recovery import Action, RecoveryPolicy, RecoveryState` and `from src.internal.tools import FailureCategory` (next to the existing `ToolEffect` import).
- module constants:

```python
UNAVAILABLE_NOTE = (
    "This tool is unavailable for the rest of this turn. Answer from what you "
    "have and say what is missing."
)
_LEGACY_ERROR_CODES = {
    FailureCategory.INVALID_INPUT: "invalid_arguments",
    FailureCategory.NOT_FOUND: "tool_not_found",
}
```

- `__init__`: add keyword parameter `recovery_policy: RecoveryPolicy | None = None` and `self._recovery = recovery_policy or RecoveryPolicy()`.
- replace `_call_tool` with:

```python
    async def _call_tool(
        self,
        tool_call: FunctionCall,
        state: RecoveryState | None = None,
        on_escalation: "ToolEscalationCallback | None" = None,
    ) -> ToolExecutionResult:
        """Execute one call with retry → degrade → escalate recovery."""
        state = state or RecoveryState(budget_left=self._recovery.retry_budget)
        start = time.perf_counter()
        args = tool_call.parsed_arguments()
        name = tool_call.name
        if name in state.unavailable:
            return self._unavailable_result(name, args, TaskStatus.SKIPPED, "unavailable", start)
        tool = self._registry.get(name)
        effect = tool.effect if tool is not None else ToolEffect.UNSPECIFIED
        retries = 0
        while True:
            outcome = await self._registry.invoke_detailed(name, args)
            failure = outcome.failure
            if failure is None:
                self._record_tool_stage(name, args, outcome.response)
                return self._result(name, args, TaskStatus.COMPLETED, start,
                                    result=outcome.response, retry_count=retries)
            decision = self._recovery.decide(failure, effect, retries, state.budget_left)
            if decision.action is Action.RETRY:
                state.budget_left -= decision.delay  # reserve before awaiting
                retries += 1
                state.retries += 1
                await asyncio.sleep(decision.delay)
                continue
            if decision.action is Action.FEED_BACK:
                return self._result(
                    name, args, TaskStatus.FAILED, start,
                    error_code=_LEGACY_ERROR_CODES[failure.category],
                    error_message="; ".join(outcome.errors) or failure.message,
                    retry_count=retries,
                )
            if decision.action is Action.UNAVAILABLE:
                state.mark_unavailable(name)
                return self._unavailable_result(
                    name, args, TaskStatus.FAILED, failure.category.value, start, retries
                )
            # Action.ESCALATE is handled in Task 5; until then escalate as unresolved.
            state.stop(
                "unresolved",
                f"{name} failed and was not retried; the action may not have completed.",
            )
            return self._result(
                name, args, TaskStatus.FAILED, start,
                error_code=failure.category.value, error_message=failure.message,
                retry_count=retries,
            )

    def _result(self, name, args, status, start, *, result=None, error_code=None,
                error_message=None, retry_count=0) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_name=name,
            status=status,
            result=result,
            arguments=args,
            performance=PerformanceMetrics(
                execution_time=time.perf_counter() - start,
                success_rate=1.0 if status is TaskStatus.COMPLETED else 0.0,
            ),
            error_code=error_code,
            error_message=error_message,
            retry_count=retry_count,
        )

    def _unavailable_result(self, name, args, status, reason, start, retries=0):
        return self._result(
            name, args, status, start,
            error_code="tool_unavailable", error_message=reason, retry_count=retries,
        )
```

- `_tool_message_content`: add as the first branch

```python
        if result.error_code == "tool_unavailable":
            return json.dumps(
                {"status": "unavailable", "tool": result.tool_name,
                 "reason": result.error_message, "note": UNAVAILABLE_NOTE}
            )
```

- `run`: after `final_answer: str | None = None` add `state = RecoveryState(budget_left=self._recovery.retry_budget)`; change the execute gather element to `self._call_tool(tc, state, on_escalation)` (add `on_escalation: "ToolEscalationCallback | None" = None` to `run`'s keyword parameters now; its type arrives in Task 5 — until then annotate it as `Any`); after `working_messages.extend(tool_responses)` add

```python
            if state.stop_answer is not None:
                final_answer = state.stop_answer
                break
```

and pass `tool_recovery=state.summary()` to the returned `AgentLoopOutput`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_tool_recovery_loop.py tests/unit/test_tool_approval.py tests/unit/test_tool_arg_validation.py tests/unit/test_tool_error_feedback.py tests/unit/test_on_turn_callback.py tests/unit/test_tool_loop_context_budget.py -q`
Expected: all PASS. If an existing test expected `error_code == "<ExceptionName>"` for a raised exception, it now sees `tool_unavailable` (read-only) or an escalation stop (other effects): update that assertion to the spec's behaviour and record it in the ledger.

- [ ] **Step 5: Mutation-check**

(a) Remove the `if name in state.unavailable` short-circuit → `test_exhausted_retries_...` FAILS (calls becomes 6). (b) Move `state.budget_left -= decision.delay` after the `await asyncio.sleep` → `test_concurrent_retries_share_...` FAILS (both calls reserve the same budget). (c) Drop `retry_count=retries` from the success `_result` → `test_transient_...` FAILS. Restore, re-run.

- [ ] **Step 6: Commit**

```bash
git add src/agents/tool/tool_calling.py src/agents/core/base.py src/agents/core/state.py tests/unit/test_tool_recovery_loop.py
git commit -m "agents: ToolAgentLoop retries transient read-only failures and degrades the rest"
```

---

### Task 5: Escalation in the loop

**Files:**
- Modify: `src/agents/tool/tool_calling.py` (types, config, `_call_tool` ESCALATE branch, `run` signature)
- Modify: `src/agents/tool/__init__.py` and `src/agents/__init__.py` (export `EscalationDecision`, `ToolEscalationRequest`, `ToolEscalationCallback`)
- Test: `tests/unit/test_tool_recovery_loop.py` (append)

**Interfaces:**
- Consumes: Task 4's `_call_tool`, `RecoveryState`.
- Produces: `EscalationDecision` (str Enum `RETRY="retry"`, `SKIP="skip"`, `CANCEL="cancel"`, `EXPIRED="expired"`); `ToolEscalationRequest(escalation_id, tool_name, arguments, category, message, attempts, created_at, expires_at)` frozen; `ToolEscalationCallback = Callable[[ToolEscalationRequest], Awaitable[EscalationDecision]]`; `ToolAgentLoopConfig.escalation_timeout_seconds: float = 120.0`, `max_escalations: int = 3`; `run(..., on_escalation: ToolEscalationCallback | None = None)`.

- [ ] **Step 1: Write the failing tests** (append)

```python
from src.agents import ApprovalDecision, EscalationDecision


def _writer(fail_times):
    calls = []

    @FunctionTool.from_fn(name="send", effect=ToolEffect.SIDE_EFFECTING)
    async def send():
        calls.append(1)
        if len(calls) <= fail_times:
            return ToolErrorText("Error: boom", ToolFailure(FailureCategory.UNKNOWN, "remote tool reported an error"))
        return "sent"

    return send, calls


SEND = '{"name":"send","arguments":{}}'


async def _approve(request):
    return ApprovalDecision.APPROVE


def _run(tool, responses, on_escalation, **config):
    tokenizer = _Tokenizer()
    manager = _Manager(tokenizer, responses)
    loop = ToolAgentLoop(
        tokenizer, manager, [tool],
        ToolAgentLoopConfig(response_length=8192, **config), recovery_policy=FAST,
    )
    return asyncio.run(
        loop.run([{"role": "user", "content": "go"}], {},
                 on_approval=_approve, on_escalation=on_escalation)
    )


def _answering(*decisions):
    seen = []
    queue = list(decisions)

    async def on_escalation(request):
        seen.append(request)
        return queue.pop(0)

    return on_escalation, seen


def test_side_effecting_failure_escalates_and_retry_runs_it_again():
    tool, calls = _writer(1)
    on_escalation, seen = _answering(EscalationDecision.RETRY)
    output = _run(tool, [SEND, "done"], on_escalation)
    assert len(calls) == 2 and len(seen) == 1
    assert seen[0].tool_name == "send" and seen[0].category == "unknown" and seen[0].attempts == 1
    assert _trace(output)[0]["status"] == str(TaskStatus.COMPLETED)
    assert output.tool_recovery["escalations"] == [
        {"tool": "send", "category": "unknown", "attempts": 1, "decision": "retry"}
    ]


def test_skip_degrades_and_the_run_continues():
    tool, calls = _writer(99)
    on_escalation, _ = _answering(EscalationDecision.SKIP)
    output = _run(tool, [SEND, "carried on"], on_escalation)
    assert len(calls) == 1
    assert output.final_answer == "carried on"
    assert output.tool_recovery["degraded"] == ["send"]


def test_cancel_stops_with_the_fixed_answer():
    tool, _ = _writer(99)
    on_escalation, _ = _answering(EscalationDecision.CANCEL)
    output = _run(tool, [SEND, "never generated"], on_escalation)
    assert output.final_answer == "Stopped: send failed (unknown); nothing further was attempted."
    assert output.tool_recovery["outcome"] == "cancelled"
    assert output.tool_recovery["needs_user"] is False


def test_expired_stops_unresolved():
    tool, _ = _writer(99)
    on_escalation, _ = _answering(EscalationDecision.EXPIRED)
    output = _run(tool, [SEND, "never generated"], on_escalation)
    assert output.final_answer == "send failed and was not retried; the action may not have completed."
    assert output.tool_recovery["needs_user"] is True


def test_no_callback_stops_unresolved_without_asking():
    tool, calls = _writer(99)
    output = _run(tool, [SEND, "never generated"], None)
    assert len(calls) == 1
    assert output.tool_recovery["outcome"] == "unresolved"
    assert output.tool_recovery["escalations"][0]["decision"] == "no_callback"


def test_callback_that_raises_or_hangs_is_unresolved():
    tool, _ = _writer(99)

    async def boom(request):
        raise RuntimeError("socket gone")

    assert _run(tool, [SEND, "x"], boom).tool_recovery["needs_user"] is True

    async def hang(request):
        await asyncio.sleep(10)

    output = _run(tool, [SEND, "x"], hang, escalation_timeout_seconds=0.05)
    assert output.tool_recovery["needs_user"] is True


def test_escalations_are_capped_per_run():
    tool, calls = _writer(99)
    on_escalation, seen = _answering(*[EscalationDecision.RETRY] * 10)
    output = _run(tool, [SEND, "x"], on_escalation, max_escalations=3)
    assert len(seen) == 3
    assert len(calls) == 4  # first attempt + 3 user-authorised retries
    assert output.tool_recovery["outcome"] == "unresolved"
    assert output.tool_recovery["escalations"][-1]["decision"] == "cap"


def test_denied_approval_is_never_escalated():
    tool, calls = _writer(99)
    on_escalation, seen = _answering(EscalationDecision.RETRY)
    tokenizer = _Tokenizer()
    manager = _Manager(tokenizer, [SEND, "done"])
    loop = ToolAgentLoop(tokenizer, manager, [tool], ToolAgentLoopConfig(response_length=8192))

    async def deny(request):
        return ApprovalDecision.DENY

    asyncio.run(loop.run([{"role": "user", "content": "go"}], {},
                         on_approval=deny, on_escalation=on_escalation))
    assert calls == [] and seen == []
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_tool_recovery_loop.py -q -k "escalat or cancel or skip or expired or callback or denied"`
Expected: `ImportError: cannot import name 'EscalationDecision'`.

- [ ] **Step 3: Implement**

Types next to `ApprovalDecision` in `tool_calling.py`:

```python
class EscalationDecision(str, Enum):
    RETRY = "retry"
    SKIP = "skip"
    CANCEL = "cancel"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class ToolEscalationRequest:
    escalation_id: str
    tool_name: str
    arguments: dict[str, Any]
    category: str
    message: str
    attempts: int
    created_at: datetime
    expires_at: datetime


ToolEscalationCallback = Callable[[ToolEscalationRequest], Awaitable[EscalationDecision]]
```

`ToolAgentLoopConfig`: add `escalation_timeout_seconds: float = 120.0` and `max_escalations: int = 3`.

Replace Task 4's temporary ESCALATE tail of `_call_tool` with:

```python
            # Action.ESCALATE
            decision_value = await self._escalate(name, args, failure, retries, state, on_escalation)
            state.escalations.append(
                {"tool": name, "category": failure.category.value,
                 "attempts": retries + 1, "decision": decision_value}
            )
            if decision_value == EscalationDecision.RETRY.value:
                retries += 1  # user-authorised: does not draw on the retry budget
                continue
            if decision_value == EscalationDecision.SKIP.value:
                state.mark_unavailable(name)
                return self._unavailable_result(
                    name, args, TaskStatus.FAILED, failure.category.value, start, retries
                )
            if decision_value == EscalationDecision.CANCEL.value:
                state.stop(
                    "cancelled",
                    f"Stopped: {name} failed ({failure.category.value}); "
                    "nothing further was attempted.",
                )
            else:  # expired, no_callback, cap
                state.stop(
                    "unresolved",
                    f"{name} failed and was not retried; the action may not have completed.",
                )
            return self._result(
                name, args, TaskStatus.FAILED, start,
                error_code=failure.category.value, error_message=failure.message,
                retry_count=retries,
            )
```

and add the helper:

```python
    async def _escalate(self, name, args, failure, retries, state, on_escalation) -> str:
        """Ask the user; return a decision value, or "no_callback" / "cap"."""
        if on_escalation is None:
            return "no_callback"
        if state.escalation_count >= self.tool_config.max_escalations:
            return "cap"
        state.escalation_count += 1  # counted before awaiting: concurrent calls see it
        created_at = datetime.now(timezone.utc)
        timeout = self.tool_config.escalation_timeout_seconds
        request = ToolEscalationRequest(
            escalation_id=uuid4().hex,
            tool_name=name,
            arguments=args,
            category=failure.category.value,
            message=failure.message,
            attempts=retries + 1,
            created_at=created_at,
            expires_at=created_at + timedelta(seconds=timeout),
        )
        try:
            decision = await asyncio.wait_for(on_escalation(request), timeout=timeout)
        except asyncio.TimeoutError:
            return EscalationDecision.EXPIRED.value
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Escalation callback failed for tool %r", name)
            return EscalationDecision.EXPIRED.value
        return EscalationDecision(decision).value
```

Change `run`'s `on_escalation` annotation to `ToolEscalationCallback | None`, and export the three new names from `src/agents/tool/__init__.py` and `src/agents/__init__.py` next to `ApprovalDecision`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_tool_recovery_loop.py tests/unit/test_tool_approval.py -q`
Expected: all PASS.

- [ ] **Step 5: Mutation-check**

(a) Change the cap check to `>` → `test_escalations_are_capped_per_run` FAILS (4 escalations). (b) Return `RETRY` from the `except Exception` branch → `test_callback_that_raises_or_hangs_is_unresolved` FAILS. (c) Remove the `break` on `state.stop_answer` from `run` → `test_cancel_stops_with_the_fixed_answer` FAILS. (d) Pass `on_escalation` only when approval was granted is already structural; instead drop the `SKIP` → `mark_unavailable` call → `test_skip_degrades_and_the_run_continues` FAILS. Restore, re-run.

- [ ] **Step 6: Commit**

```bash
git add src/agents/tool/tool_calling.py src/agents/tool/__init__.py src/agents/__init__.py tests/unit/test_tool_recovery_loop.py
git commit -m "agents: escalate unsafe tool failures to the user (retry, skip, cancel, expire)"
```

---

### Task 6: Runners surface recovery

**Files:**
- Modify: `src/internal/servers/web/tool_agent_runner.py` (`_run_tool_agent`)
- Modify: `src/internal/servers/web/app.py` (`_run_auto_routed`, `_run_agent_impl`: thread `on_escalation`)
- Modify: `src/internal/servers/query_and_chat/tool_backend.py` (`_run` returns `tool_recovery`; JSON response and `done` event carry it)
- Modify: `src/internal/servers/query_and_chat/models.py` (`ToolAgentMessageResponse.tool_recovery: dict | None = None`)
- Modify: `tests/unit/servers/web/test_loop_runners.py` (`_SpyLoop.run` gains `on_escalation=None`)
- Test: `tests/unit/servers/web/test_tool_recovery_runner.py`

**Interfaces:**
- Consumes: `AgentLoopOutput.tool_recovery` (Task 4), `on_escalation` (Task 5).
- Produces: `_run_tool_agent(..., on_escalation=None)` returning `extra["tool_recovery"]` when not None; `_run_auto_routed(..., on_escalation=None)`; `_run_agent_impl(..., on_escalation=None)`; `ToolAgentMessageResponse.tool_recovery`.

- [ ] **Step 1: Write the failing tests**

```python
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.internal.servers.web.app as web_app
from src.agents.core.base import AgentLoopOutput


def _output(answer, recovery):
    return AgentLoopOutput(
        prompt_ids=[], response_ids=[], response_mask=[], num_turns=1,
        final_answer=answer, action_trace="",
        trajectory_messages=[{"role": "assistant", "content": answer}],
        tool_recovery=recovery,
    )


async def _run(monkeypatch, output, **kwargs):
    run = AsyncMock(return_value=output)
    monkeypatch.setattr("src.agents.tool.tool_calling.ToolAgentLoop.run", run)
    result = await web_app._run_tool_agent(
        "q", manager=MagicMock(), tokenizer=MagicMock(), search_url="http://x/retrieve",
        history=[], resolved=types.SimpleNamespace(tool_agent_parser="json"),
        on_turn=None, with_search_tool=False, **kwargs,
    )
    return result, run


DEGRADED = {"outcome": "degraded", "needs_user": False, "retries": 2,
            "degraded": ["get_weather"], "escalations": []}


@pytest.mark.asyncio
async def test_degraded_answer_gets_the_fixed_note_and_metadata(monkeypatch):
    (answer, _c, _d, _i, extra), _ = await _run(monkeypatch, _output("Sunny, probably.", DEGRADED))
    assert answer == (
        "Sunny, probably.\n\nNote: get_weather was unavailable, "
        "so this answer may be incomplete."
    )
    assert extra["tool_recovery"] == DEGRADED


@pytest.mark.asyncio
async def test_empty_degraded_answer_gets_no_note(monkeypatch):
    # An empty answer must stay empty so the auto-route can still fall back.
    (answer, *_), _ = await _run(monkeypatch, _output("", DEGRADED))
    assert answer == ""


@pytest.mark.asyncio
async def test_no_recovery_leaves_answer_and_metadata_alone(monkeypatch):
    (answer, _c, _d, _i, extra), _ = await _run(monkeypatch, _output("fine", None))
    assert answer == "fine" and "tool_recovery" not in extra


@pytest.mark.asyncio
async def test_on_escalation_reaches_the_loop(monkeypatch):
    async def on_escalation(request):
        return None

    _, run = await _run(monkeypatch, _output("fine", None), on_escalation=on_escalation)
    assert run.await_args.kwargs["on_escalation"] is on_escalation


@pytest.mark.asyncio
async def test_stopped_answer_survives_the_explicit_mode_fallback(monkeypatch):
    stopped = {"outcome": "cancelled", "needs_user": False, "retries": 0, "degraded": [],
               "escalations": [{"tool": "send", "category": "unknown", "attempts": 1, "decision": "cancel"}]}
    text = "Stopped: send failed (unknown); nothing further was attempted."
    (answer, _c, _d, _i, extra), _ = await _run(monkeypatch, _output(text, stopped))
    assert (answer or extra.pop("_assistant_fallback", "")) == text
```

Add to the same file a `/tool/send-tool-message` JSON test: monkeypatch `_run_tool_agent` in `tool_backend`'s namespace (read how existing `tests/unit/test_tool_backend.py` builds the app and patches the runner; reuse it) to return `("Stopped: …", [], [], "tool", {"tool_calls": [], "num_turns": 1, "truncated": False, "tool_recovery": {...}})` and assert the JSON body's `answer` is the stop text and `tool_recovery` equals the dict; and a streaming variant asserting the `done` event carries `tool_recovery`. Add an `/api/agent` explicit `mode="tool_agent"` test the same way (read `test_web_experience_app.py`'s explicit-mode tests for the harness) asserting the response's `hook_metadata["tool_recovery"]` equals the dict and the answer is the stop text.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/servers/web/test_tool_recovery_runner.py -q`
Expected: FAIL — `_run_tool_agent() got an unexpected keyword argument 'on_escalation'` and missing note.

- [ ] **Step 3: Implement**

`tool_agent_runner._run_tool_agent`: add parameter `on_escalation=None` (after `on_approval`); pass `on_escalation=on_escalation` to `loop.run(...)`; after computing `extra`, add:

```python
    answer = output.final_answer or ""
    recovery = getattr(output, "tool_recovery", None)
    if recovery is not None:
        extra["tool_recovery"] = recovery
        if recovery.get("outcome") == "degraded" and recovery.get("degraded") and answer.strip():
            tools = ", ".join(recovery["degraded"])
            answer = (
                f"{answer}\n\nNote: {tools} was unavailable, "
                "so this answer may be incomplete."
            )
```

and return `answer` instead of `output.final_answer or ""`.

`app.py`: add `on_escalation=None` to `_run_auto_routed` and `_run_agent_impl` signatures (next to `on_approval`), and pass `on_escalation=on_escalation` everywhere `on_approval=on_approval` is passed into `_run_tool_agent` or `_run_auto_routed` (the TOOL branch of `_run_auto_routed`, the auto path and the explicit `tool_agent` mode of `_run_agent_impl`).

`tool_backend.py`: `_run(on_turn=None, on_approval=None, on_escalation=None)` passes `on_escalation` to `_run_tool_agent` and returns a 5th element `extra.get("tool_recovery")`; the JSON branch unpacks 5 values and sets `tool_recovery=tool_recovery` on `ToolAgentMessageResponse`; the stream branch unpacks 5 values and adds `"tool_recovery": tool_recovery` to the `done` event. `models.py`: `tool_recovery: dict | None = None` on `ToolAgentMessageResponse`.

`test_loop_runners.py`: `_SpyLoop.run(self, messages, sampling_params, *, on_turn=None, on_approval=None, on_escalation=None)`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/servers/web/test_tool_recovery_runner.py tests/unit/servers/web/test_loop_runners.py tests/unit/test_tool_backend.py tests/unit/servers/web/test_web_experience_app.py -q`
Expected: all PASS.

- [ ] **Step 5: Mutation-check**

(a) Drop the `answer.strip()` guard → `test_empty_degraded_answer_gets_no_note` FAILS. (b) Stop passing `on_escalation` to `loop.run` → `test_on_escalation_reaches_the_loop` FAILS. (c) Drop `tool_recovery` from the `/tool` JSON response → its test FAILS. Restore, re-run.

- [ ] **Step 6: Commit**

```bash
git add src/internal/servers/web/tool_agent_runner.py src/internal/servers/web/app.py src/internal/servers/query_and_chat/tool_backend.py src/internal/servers/query_and_chat/models.py tests/unit/servers/web/test_tool_recovery_runner.py tests/unit/servers/web/test_loop_runners.py
git commit -m "web: surface tool recovery in answers, metadata and the /tool responses"
```

---

### Task 7: Escalation broker, event, endpoint and WebSocket message

**Files:**
- Modify: `src/internal/servers/web/tool_approval.py` (`DecisionBroker` base; `ToolApprovalBroker` subclass; `ToolEscalationView`; `ToolEscalationBroker`)
- Modify: `src/internal/servers/web/app.py` (broker on `app.state`; `_request_human_decision`; `_request_tool_escalation`; escalation endpoint + models; SSE and WS `on_escalation`; WS `submit_escalation`)
- Modify: `src/internal/servers/web/ws_channel.py` (`serve(..., submit_escalation=...)`, dispatch `"escalation.submit"`)
- Modify: `src/internal/servers/query_and_chat/tool_backend.py` (stream `on_escalation`)
- Test: `tests/unit/servers/web/test_tool_escalation.py`

**Interfaces:**
- Consumes: `ToolEscalationRequest`, `EscalationDecision` (Task 5); `on_escalation` threading (Task 6).
- Produces: `DecisionBroker`; `ToolEscalationBroker.request(owner_user_id, request: ToolEscalationRequest, on_registered=None) -> EscalationDecision`, `.decide(escalation_id, owner_user_id, decision: EscalationDecision)`; `ToolEscalationView(id, tool_name, arguments, category, message, attempts, expires_at)`; event `{"type": "escalation_required", "escalation": asdict(view)}`; `POST /api/agent/escalations/{escalation_id}` body `{"decision": "retry"|"skip"|"cancel"}` → `{"id", "decision"}`; WS message `{"type": "escalation.submit", "escalation_id", "decision"}`.

- [ ] **Step 1: Write the failing tests**

```python
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from src.agents import EscalationDecision, ToolEscalationRequest
from src.internal.servers.web.tool_approval import (
    ApprovalConflict,
    ApprovalForbidden,
    ApprovalNotFound,
    ToolApprovalBroker,
    ToolEscalationBroker,
)


def _request(escalation_id="esc-1", expires_in=1.0):
    now = datetime.now(timezone.utc)
    return ToolEscalationRequest(
        escalation_id=escalation_id, tool_name="send_email",
        arguments={"to": "a@b.c", "password": "hunter2"},
        category="unknown", message="remote tool reported an error", attempts=1,
        created_at=now, expires_at=now + timedelta(seconds=expires_in),
    )


async def _pending(broker):
    for _ in range(100):
        if broker.pending_count:
            return
        await asyncio.sleep(0)
    raise AssertionError("not registered")


@pytest.mark.asyncio
async def test_escalation_round_trip_with_sanitized_view():
    broker = ToolEscalationBroker(timeout_seconds=5)
    views = []
    task = asyncio.create_task(broker.request("alice", _request(), on_registered=views.append))
    await _pending(broker)
    assert "password" not in views[0].arguments
    assert (views[0].category, views[0].attempts) == ("unknown", 1)
    await broker.decide("esc-1", "alice", EscalationDecision.SKIP)
    assert await task is EscalationDecision.SKIP
    assert broker.counters["skip"] == 1


@pytest.mark.asyncio
async def test_escalation_ownership_and_double_decision():
    broker = ToolEscalationBroker(timeout_seconds=5)
    task = asyncio.create_task(broker.request("alice", _request()))
    await _pending(broker)
    with pytest.raises(ApprovalForbidden):
        await broker.decide("esc-1", "mallory", EscalationDecision.RETRY)
    await broker.decide("esc-1", "alice", EscalationDecision.RETRY)
    with pytest.raises((ApprovalConflict, ApprovalNotFound)):
        await broker.decide("esc-1", "alice", EscalationDecision.CANCEL)
    assert await task is EscalationDecision.RETRY


@pytest.mark.asyncio
async def test_escalation_times_out_as_expired():
    broker = ToolEscalationBroker(timeout_seconds=0.05)
    assert await broker.request("alice", _request()) is EscalationDecision.EXPIRED
    assert broker.counters["expired"] == 1


@pytest.mark.asyncio
async def test_expired_cannot_be_submitted_as_a_decision():
    broker = ToolEscalationBroker(timeout_seconds=5)
    task = asyncio.create_task(broker.request("alice", _request()))
    await _pending(broker)
    with pytest.raises(ApprovalConflict):
        await broker.decide("esc-1", "alice", EscalationDecision.EXPIRED)
    await broker.decide("esc-1", "alice", EscalationDecision.CANCEL)
    await task


@pytest.mark.asyncio
async def test_approval_and_escalation_brokers_share_no_state():
    approvals, escalations = ToolApprovalBroker(timeout_seconds=5), ToolEscalationBroker(timeout_seconds=5)
    task = asyncio.create_task(escalations.request("alice", _request()))
    await _pending(escalations)
    assert approvals.pending_count == 0
    await escalations.decide("esc-1", "alice", EscalationDecision.SKIP)
    await task
    assert approvals.counters["requested"] == 0
```

Endpoint, SSE and WebSocket tests go in the same file, reusing the existing harnesses (read `tests/unit/servers/web/test_ws_channel.py`, `test_ws_dispatch.py`, `_ws_fakes.py` and the approval-endpoint tests in `test_web_experience_app.py` first — grep `approvals/`):

- `POST /api/agent/escalations/{id}`: with a pending escalation registered directly on `app.state.tool_escalation_broker` for the signed-in user → 200 `{"id": ..., "decision": "skip"}` and the request task resolves to `SKIP`; another user → 403; unknown id → 404; a second post → 409; `{"decision": "expired"}` → 422.
- `_request_tool_escalation(broker, owner, request, queue)` puts exactly one `{"type": "escalation_required", "escalation": {...}}` on the queue and returns the decision.
- WebSocket: an `{"type": "escalation.submit", "escalation_id": ..., "decision": "retry"}` message reaches the broker; an unknown decision → error event 400.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/servers/web/test_tool_escalation.py -q`
Expected: `ImportError: cannot import name 'ToolEscalationBroker'`.

- [ ] **Step 3: Implement**

`tool_approval.py` — move the body of `ToolApprovalBroker.__init__`, `pending_count`, `request` and `decide` into:

```python
class DecisionBroker:
    """Process-local request → human decision → future, with owner checks and a timeout.

    Approval and escalation are separate subclasses with separate instances:
    they share these mechanics, never state.
    """

    kind = "decision"

    def __init__(self, timeout_seconds, *, expired, allowed, counter_names) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout_seconds = timeout_seconds
        self._expired = expired
        self._allowed = frozenset(allowed)
        self._counter_names = dict(counter_names)  # decision -> counter key
        self._pending: dict[str, _PendingApproval] = {}
        self._lock = asyncio.Lock()
        self.counters = {"requested": 0, **{k: 0 for k in self._counter_names.values()},
                         "expired": 0, "cancelled": 0, "errors": 0}

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    async def _request(self, owner_user_id, request_id, tool_name, expires_at, view, on_registered):
        ...  # the existing ToolApprovalBroker.request body from `future = ...` on,
             # with request.approval_id -> request_id, request.expires_at -> expires_at,
             # request.tool_name -> tool_name, ApprovalDecision.EXPIRED -> self._expired,
             # and the log line "Tool %s completed id=%s ..." using self.kind

    async def decide(self, request_id, owner_user_id, decision) -> None:
        ...  # the existing decide body, with ApprovalDecision.EXPIRED -> self._expired,
             # the allowed check `decision not in self._allowed`, and
             # self.counters[self._counter_names[decision]] += 1
```

(Copy the existing bodies verbatim and make only those substitutions; `_PendingApproval.future` is typed `asyncio.Future[Any]`.)

```python
class ToolApprovalBroker(DecisionBroker):
    kind = "approval"

    def __init__(self, timeout_seconds: float = 60.0) -> None:
        super().__init__(
            timeout_seconds,
            expired=ApprovalDecision.EXPIRED,
            allowed=(ApprovalDecision.APPROVE, ApprovalDecision.DENY),
            counter_names={ApprovalDecision.APPROVE: "approved", ApprovalDecision.DENY: "denied"},
        )

    async def request(self, owner_user_id, request: ToolApprovalRequest, on_registered=None):
        view = ToolApprovalView(
            id=request.approval_id, tool_name=request.tool_name,
            arguments=sanitize_tool_arguments(request.arguments),
            expires_at=_iso(request.expires_at),
        )
        return await self._request(owner_user_id, request.approval_id, request.tool_name,
                                   request.expires_at, view, on_registered)


@dataclass(frozen=True, slots=True)
class ToolEscalationView:
    id: str
    tool_name: str
    arguments: dict[str, object]
    category: str
    message: str
    attempts: int
    expires_at: str


class ToolEscalationBroker(DecisionBroker):
    kind = "escalation"

    def __init__(self, timeout_seconds: float = 120.0) -> None:
        super().__init__(
            timeout_seconds,
            expired=EscalationDecision.EXPIRED,
            allowed=(EscalationDecision.RETRY, EscalationDecision.SKIP, EscalationDecision.CANCEL),
            counter_names={EscalationDecision.RETRY: "retry", EscalationDecision.SKIP: "skip",
                           EscalationDecision.CANCEL: "cancel"},
        )

    async def request(self, owner_user_id, request: ToolEscalationRequest, on_registered=None):
        view = ToolEscalationView(
            id=request.escalation_id, tool_name=request.tool_name,
            arguments=sanitize_tool_arguments(request.arguments),
            category=request.category, message=request.message,
            attempts=request.attempts, expires_at=_iso(request.expires_at),
        )
        return await self._request(owner_user_id, request.escalation_id, request.tool_name,
                                   request.expires_at, view, on_registered)
```

with `_iso(dt) = dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")` (the existing expression, factored out). Import `EscalationDecision, ToolEscalationRequest` from `src.agents.tool`.

`app.py`:
- next to `app.state.tool_approval_broker = ToolApprovalBroker(...)`: `app.state.tool_escalation_broker = ToolEscalationBroker()`.
- rename the body of `_request_tool_approval` into `_request_human_decision(broker, owner_user_id, request, queue, *, event_type: str, key: str)` whose `publish` puts `{"type": event_type, key: asdict(view)}`; keep `_request_tool_approval(broker, owner, request, queue)` as a one-line call with `event_type="approval_required", key="approval"`, and add `_request_tool_escalation(...)` with `event_type="escalation_required", key="escalation"`.
- Pydantic models next to the approval ones: `ToolEscalationDecisionRequest(decision: Literal["retry", "skip", "cancel"])` and `ToolEscalationDecisionResponse(id: str, decision: str)`; endpoint `POST /api/agent/escalations/{escalation_id}` mirroring `decide_tool_approval` (same `_require_auth`, same 403/404/409/410 mapping) against `app.state.tool_escalation_broker` with `EscalationDecision(request.decision)`.
- `stream_agent`: define `on_escalation` next to `on_approval` (calls `_request_tool_escalation(http_request.app.state.tool_escalation_broker, auth_user.id, request, driver.queue)`) and pass `on_escalation=on_escalation if auth_user is not None else None` to `_run_agent_impl`.
- `agent_ws.start_run`: the same `on_escalation` using `session.user_id` and `driver.queue`, passed to `_run_agent_impl`; add `submit_escalation(session, message)` mirroring `submit_approval` (reads `escalation_id`, `decision`; `EscalationDecision(decision)` with `ValueError` → 400; same exception → code mapping) and pass `submit_escalation=submit_escalation` to `serve`.

`ws_channel.serve`: add keyword parameter `submit_escalation: Callable[[WsSession, dict], Awaitable[None]] | None = None` and dispatch `elif event_type == "escalation.submit" and submit_escalation is not None: await submit_escalation(session, message)`.

`tool_backend` stream: next to `on_approval`, build `on_escalation` from `http_request.app.state.tool_escalation_broker` with `_request_tool_escalation` under the same non-anonymous-user condition, and pass it into `_run`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/servers/web/test_tool_escalation.py tests/unit/servers/web/test_tool_approval_broker.py tests/unit/servers/web/test_ws_channel.py tests/unit/servers/web/test_ws_dispatch.py tests/unit/servers/web/test_sse_streaming.py tests/unit/test_tool_backend.py -q`
Expected: all PASS, including every pre-existing approval-broker test unchanged.

- [ ] **Step 5: Mutation-check**

(a) Drop the owner check in `DecisionBroker.decide` → the 403 test FAILS (and the approval-broker ownership test too). (b) Allow `EXPIRED` in `ToolEscalationBroker`'s `allowed` → `test_expired_cannot_be_submitted_as_a_decision` FAILS. (c) Make `ToolEscalationBroker` reuse the approval instance's `_pending` (e.g. a class-level dict) → `test_approval_and_escalation_brokers_share_no_state` FAILS. Restore, re-run.

- [ ] **Step 6: Commit**

```bash
git add src/internal/servers/web/tool_approval.py src/internal/servers/web/app.py src/internal/servers/web/ws_channel.py src/internal/servers/query_and_chat/tool_backend.py tests/unit/servers/web/test_tool_escalation.py
git commit -m "web: escalation broker, event, endpoint and WebSocket message"
```

---

### Task 8: `ToolEscalationCard` on the Assist and Tools pages

**Files:**
- Modify: `web/src/types.ts` (`ToolEscalationView`, event unions)
- Modify: `web/src/api.ts` (`submitToolEscalation`)
- Create: `web/src/components/ToolEscalationCard.tsx`
- Modify: `web/src/pages/AssistPage.tsx`, `web/src/components/ToolAgentView.tsx`
- Test: `web/src/components/__tests__/ToolEscalationCard.test.tsx` (and extend `ToolAgentView.test.tsx`)

**Interfaces:**
- Consumes: the `escalation_required` event and `POST /api/agent/escalations/{id}` (Task 7).
- Produces: `ToolEscalationView { id; tool_name; arguments: Record<string, unknown>; category: string; message: string; attempts: number; expires_at: string }`; `submitToolEscalation(escalationId, decision: "retry" | "skip" | "cancel", init?)`.

- [ ] **Step 1: Write the failing tests**

`web/src/components/__tests__/ToolEscalationCard.test.tsx` — model it on `ToolApprovalCard.test.tsx` (read it first and reuse its render/user-event setup):

```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ToolEscalationCard } from "../ToolEscalationCard";

const escalation = {
  id: "esc-1",
  tool_name: "send_email",
  arguments: { to: "a@b.c" },
  category: "unknown",
  message: "remote tool reported an error",
  attempts: 1,
  expires_at: new Date(Date.now() + 60_000).toISOString(),
};

describe("ToolEscalationCard", () => {
  it("shows the failure and warns the action may have happened", () => {
    render(<ToolEscalationCard escalation={escalation} onDecision={vi.fn()} />);
    expect(screen.getByText("send_email")).toBeInTheDocument();
    expect(screen.getByText(/remote tool reported an error/)).toBeInTheDocument();
    expect(screen.getByText(/may or may not have happened/)).toBeInTheDocument();
  });

  it.each(["retry", "skip", "cancel"] as const)("submits %s once", async (decision) => {
    const onDecision = vi.fn().mockResolvedValue(undefined);
    render(<ToolEscalationCard escalation={escalation} onDecision={onDecision} />);
    const label = { retry: "Retry", skip: "Skip", cancel: "Cancel" }[decision];
    await userEvent.click(screen.getByRole("button", { name: label }));
    await userEvent.click(screen.getByRole("button", { name: label }));
    expect(onDecision).toHaveBeenCalledTimes(1);
    expect(onDecision).toHaveBeenCalledWith(decision);
  });
});
```

Extend `ToolAgentView.test.tsx` (following its existing `approval_required` test) with a stream that emits `escalation_required` → the card renders, clicking Skip calls `submitToolEscalation("esc-1", "skip", …)`, and the card disappears on `done`.

- [ ] **Step 2: Run to verify failure**

Run: `cd web && npx vitest run src/components/__tests__/ToolEscalationCard.test.tsx`
Expected: FAIL, cannot resolve `../ToolEscalationCard`.

- [ ] **Step 3: Implement**

`types.ts`: add `ToolEscalationView` (fields above), `SSEEscalationRequiredEvent { type: "escalation_required"; escalation: ToolEscalationView }`, add it to the SSE event union that includes `SSEApprovalRequiredEvent`, and add `| { type: "escalation_required"; escalation: ToolEscalationView }` to `ToolStreamEvent`.

`api.ts`, next to `submitToolApproval`:

```ts
export function submitToolEscalation(
  escalationId: string,
  decision: "retry" | "skip" | "cancel",
  init?: Pick<RequestInit, "signal">,
): Promise<unknown> {
  return requestJson<unknown>(`/api/agent/escalations/${escalationId}`, {
    method: "POST",
    body: JSON.stringify({ decision }),
    signal: init?.signal,
  });
}
```

`ToolEscalationCard.tsx`: copy `ToolApprovalCard.tsx`'s structure (countdown, `argumentText`, idle/submitting/decided/error state, locking after one decision) with these differences: props `escalation: ToolEscalationView` and `onDecision: (decision: "retry" | "skip" | "cancel") => Promise<unknown> | unknown`; eyebrow `Tool failed`; aria-label `` `Tool failure: ${escalation.tool_name}` ``; a line `{escalation.message} · attempt {escalation.attempts}`; a warning paragraph `This action may or may not have happened. Retry runs it again.`; three buttons Retry / Skip / Cancel; reuse the `tool-approval-*` CSS classes.

`AssistPage.tsx` and `ToolAgentView.tsx`: mirror every `pendingApprovals` touch point with a `pendingEscalations` state — handle `event.type === "escalation_required"` (dedupe by id), clear it wherever `setPendingApprovals([])` is called, a `handleEscalationDecision(id, decision)` that calls `submitToolEscalation` then removes the card, and render `<ToolEscalationCard>` next to the approval cards.

- [ ] **Step 4: Run tests**

Run: `cd web && npx vitest run && npm run typecheck`
Expected: all PASS, no type errors.

- [ ] **Step 5: Mutation-check**

Remove the "already submitting/decided" guard in the card's `decide` → the "submits once" cases FAIL. Restore, re-run.

- [ ] **Step 6: Commit**

```bash
git add web/src/types.ts web/src/api.ts web/src/components/ToolEscalationCard.tsx web/src/pages/AssistPage.tsx web/src/components/ToolAgentView.tsx web/src/components/__tests__/ToolEscalationCard.test.tsx web/src/components/__tests__/ToolAgentView.test.tsx
git commit -m "web: ToolEscalationCard lets the user retry, skip or cancel a failed tool"
```
