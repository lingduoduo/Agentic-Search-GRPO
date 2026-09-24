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
