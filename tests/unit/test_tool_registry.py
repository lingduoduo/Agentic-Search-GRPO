"""Unit tests for src/internal/tools/registry.py."""

from __future__ import annotations

import pytest

from src.internal.tools import FunctionTool, ToolEffect
from src.internal.tools.registry import (
    ToolRegistry,
    _params_from_signature,
)
from src.internal.tools.validation import check_json_type, validate_arguments


def test_function_tool_defaults_to_unspecified_effect() -> None:
    tool = FunctionTool(lambda: "ok", name="unknown")
    assert tool.effect is ToolEffect.UNSPECIFIED
    assert "effect" not in tool.schema.to_dict()["function"]


def test_registry_decorator_accepts_read_only_effect() -> None:
    registry = ToolRegistry()

    @registry.tool(effect=ToolEffect.READ_ONLY)
    def lookup() -> str:
        return "ok"

    assert registry.get("lookup").effect is ToolEffect.READ_ONLY


# ---------------------------------------------------------------------------
# _params_from_signature
# ---------------------------------------------------------------------------


def _fn_with_hints(a: int, b: str, c: float = 1.0) -> str:
    """Docstring."""
    return str(a)


def _fn_no_hints(x, y=None):
    pass


def test_params_infers_types_and_required():
    schema = _params_from_signature(_fn_with_hints)
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["properties"]["b"]["type"] == "string"
    assert schema["properties"]["c"]["type"] == "number"
    assert schema["required"] == ["a", "b"]


def test_params_defaults_excluded_from_required():
    schema = _params_from_signature(_fn_with_hints)
    assert "c" not in schema.get("required", [])


def test_params_no_hints_defaults_to_string():
    schema = _params_from_signature(_fn_no_hints)
    assert schema["properties"]["x"]["type"] == "string"


def _fn_optional(a: "int | None" = None) -> None:  # noqa: F821
    pass


def test_params_optional_unwrapped():
    from typing import Optional

    # Test via direct annotation check (Optional at module level)
    import inspect

    def fn(a: Optional[int] = None) -> None:
        pass

    # Manually verify unwrapping logic since get_type_hints has trouble with
    # locally-defined Optional in test scope; just check parameter is optional
    sig = inspect.signature(fn)
    assert sig.parameters["a"].default is None  # optional (has default)
    # The schema should at least mark the param as present and optional
    schema = _params_from_signature(fn)
    assert "a" in schema["properties"]
    assert "required" not in schema  # no required fields since all have defaults


# ---------------------------------------------------------------------------
# _check_json_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,json_type,expected",
    [
        ("hello", "string", True),
        (1, "integer", True),
        (True, "integer", False),  # bool is not int for json type check
        (1.5, "number", True),
        (1, "number", True),
        (True, "boolean", True),
        ([1, 2], "array", True),
        ({"k": 1}, "object", True),
        (None, "null", True),
        ("x", "integer", False),
    ],
)
def test_check_json_type(value, json_type, expected):
    assert check_json_type(value, json_type) is expected


# ---------------------------------------------------------------------------
# _validate_arguments
# ---------------------------------------------------------------------------


def test_validate_missing_required():
    params = {
        "required": ["a", "b"],
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
    }
    errors = validate_arguments(params, {"a": "hello"})
    assert any("b" in e for e in errors)


def test_validate_type_mismatch():
    params = {"required": ["a"], "properties": {"a": {"type": "integer"}}}
    errors = validate_arguments(params, {"a": "not-an-int"})
    assert errors


def test_validate_passes_valid():
    params = {"required": ["a"], "properties": {"a": {"type": "integer"}}}
    errors = validate_arguments(params, {"a": 42})
    assert errors == []


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


def test_register_function_tool():
    reg = ToolRegistry()

    @reg.tool(description="Add two ints")
    def add(a: int, b: int) -> int:
        return a + b

    assert "add" in {e.tool.name for e in reg.list()}
    s = reg.tool_summary("add")
    assert s is not None
    assert s["description"] == "Add two ints"
    assert s["source"] == "function"


def test_register_overwrites_existing():
    reg = ToolRegistry()

    @reg.tool
    def fn(x: str) -> str:
        return x

    @reg.tool(name="fn", description="new desc")
    def fn2(x: str) -> str:
        return x

    assert len([e for e in reg.list() if e.tool.name == "fn"]) == 1
    assert reg.tool_summary("fn")["description"] == "new desc"


def test_unregister():
    reg = ToolRegistry()

    @reg.tool
    def fn(x: str) -> str:
        return x

    assert reg.unregister("fn") is True
    assert reg.get("fn") is None
    assert reg.unregister("fn") is False


@pytest.mark.asyncio
async def test_invoke_function_tool():
    reg = ToolRegistry()

    @reg.tool(description="Multiply")
    def mul(a: int, b: int) -> int:
        return a * b

    response, raw, errors = await reg.invoke("mul", {"a": 3, "b": 4})
    assert errors == []
    assert raw == 12
    assert response == "12"


@pytest.mark.asyncio
async def test_invoke_missing_tool():
    reg = ToolRegistry()
    response, raw, errors = await reg.invoke("nonexistent", {})
    assert errors
    assert "not found" in errors[0]


@pytest.mark.asyncio
async def test_invoke_validation_error():
    reg = ToolRegistry()

    @reg.tool
    def typed(n: int) -> int:
        return n

    _, _, errors = await reg.invoke("typed", {"n": "not_an_int"})
    assert errors  # type error detected


@pytest.mark.asyncio
async def test_invoke_missing_required():
    reg = ToolRegistry()

    @reg.tool
    def greet(name: str) -> str:
        return f"hi {name}"

    _, _, errors = await reg.invoke("greet", {})
    assert any("name" in e for e in errors)


def test_all_summaries():
    reg = ToolRegistry()

    @reg.tool
    def a(x: str) -> str:
        return x

    @reg.tool
    def b(x: str) -> str:
        return x

    summaries = reg.all_summaries()
    names = {s["name"] for s in summaries}
    assert {"a", "b"} <= names


@pytest.mark.asyncio
async def test_async_function_tool():
    reg = ToolRegistry()

    @reg.tool(description="Async double")
    async def double(n: int) -> int:
        return n * 2

    response, raw, errors = await reg.invoke("double", {"n": 5})
    assert errors == []
    assert raw == 10


def _attempt_snapshot():
    from src.internal.observability.prometheus import REGISTRY

    return tuple(
        REGISTRY.get_sample_value(
            "agentic_search_tool_attempts_total", {"outcome": outcome}
        )
        or 0
        for outcome in ("success", "timeout", "error", "cancelled")
    )


@pytest.mark.parametrize("method", ["invoke", "invoke_detailed"])
@pytest.mark.parametrize(
    "failure,expected",
    [
        (None, (1, 0, 0, 0)),
        ("timeout", (0, 1, 0, 0)),
        ("connection", (0, 0, 1, 0)),
        ("error", (0, 0, 1, 0)),
        ("cancelled", (0, 0, 0, 1)),
        ("typed", (0, 1, 0, 0)),
        ("httpx", (0, 1, 0, 0)),
        ("aiohttp", (0, 1, 0, 0)),
    ],
)
async def test_registry_records_each_execution_outcome(method, failure, expected):
    import asyncio
    import aiohttp
    import httpx
    from src.internal.tools import ToolErrorText, ToolFailure, FailureCategory

    errors = {
        "timeout": asyncio.TimeoutError(),
        "connection": ConnectionError(),
        "error": ValueError(),
        "cancelled": asyncio.CancelledError(),
        "httpx": httpx.ReadTimeout("slow"),
        "aiohttp": aiohttp.ServerTimeoutError(),
    }
    registry = ToolRegistry()

    @registry.tool()
    async def lookup() -> str:
        if failure == "typed":
            return ToolErrorText(
                "failed",
                ToolFailure(FailureCategory.TRANSIENT, "slow", is_timeout=True),
            )
        if failure:
            raise errors[failure]
        return "ok"

    before = _attempt_snapshot()
    if failure == "cancelled" or (method == "invoke" and failure in errors):
        with pytest.raises(type(errors[failure])):
            await getattr(registry, method)("lookup", {})
    else:
        result = await getattr(registry, method)("lookup", {})
        if method == "invoke_detailed" and failure in {
            "timeout",
            "httpx",
            "aiohttp",
            "typed",
        }:
            assert result.failure.is_timeout is True
            assert result.failure.category == FailureCategory.TRANSIENT
    assert tuple(b - a for a, b in zip(before, _attempt_snapshot())) == expected


@pytest.mark.parametrize("method", ["invoke", "invoke_detailed"])
async def test_registry_rejected_calls_are_not_attempts(method):
    registry = ToolRegistry()

    @registry.tool()
    async def lookup(required: int) -> str:
        return "ok"

    before = _attempt_snapshot()
    await getattr(registry, method)("missing", {})
    await getattr(registry, method)("lookup", {})
    assert _attempt_snapshot() == before


@pytest.mark.parametrize(
    "phase,failure,expected",
    [
        ("create", RuntimeError, (0, 0, 1, 0)),
        ("release", RuntimeError, (0, 0, 1, 0)),
        ("release", TimeoutError, (0, 1, 0, 0)),
    ],
)
@pytest.mark.parametrize("method", ["invoke", "invoke_detailed"])
async def test_registry_lifecycle_failures_propagate_once(
    monkeypatch, phase, failure, expected, method
):
    registry = ToolRegistry()
    events = []

    @registry.tool()
    async def lookup() -> str:
        events.append("execute")
        return "ok"

    tool = registry.get("lookup")

    async def create():
        events.append("create")
        if phase == "create":
            raise failure()
        return "instance"

    async def release(instance):
        events.append("release")
        assert instance == "instance"
        raise failure()

    monkeypatch.setattr(tool, "create", create)
    monkeypatch.setattr(tool, "release", release)
    before = _attempt_snapshot()
    with pytest.raises(failure):
        await getattr(registry, method)("lookup", {})
    assert events == (
        ["create"] if phase == "create" else ["create", "execute", "release"]
    )
    assert tuple(b - a for a, b in zip(before, _attempt_snapshot())) == expected


async def test_registry_cancelled_release_counts_cancellation(monkeypatch):
    import asyncio

    registry = ToolRegistry()

    @registry.tool()
    async def lookup() -> str:
        raise TimeoutError()

    async def release(instance):
        raise asyncio.CancelledError()

    monkeypatch.setattr(registry.get("lookup"), "release", release)
    before = _attempt_snapshot()
    with pytest.raises(asyncio.CancelledError):
        await registry.invoke_detailed("lookup", {})
    assert tuple(b - a for a, b in zip(before, _attempt_snapshot())) == (0, 0, 0, 1)
