import asyncio
import json
from types import SimpleNamespace

import pytest

from src.internal.tools import (
    FailureCategory,
    FunctionTool,
    ToolErrorText,
    ToolFailure,
    ToolRegistry,
)
from src.internal.tools.mcp_client import _result_text
from src.internal.tools.public_data._http import PublicDataError, guarded


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


def _guarded_failure(exc):
    @guarded
    async def tool():
        raise exc

    return asyncio.run(tool())


@pytest.mark.parametrize(
    ("exc", "category", "attempts"),
    [
        (
            PublicDataError("x returned HTTP 503", status=503, attempts=3),
            FailureCategory.TRANSIENT,
            3,
        ),
        (
            PublicDataError("request to x failed: reset", transport=True, attempts=3),
            FailureCategory.TRANSIENT,
            3,
        ),
        (
            PublicDataError("x returned HTTP 404", status=404, attempts=1),
            FailureCategory.INVALID_INPUT,
            1,
        ),
        (
            PublicDataError("x returned a non-JSON body", attempts=1, upstream=True),
            FailureCategory.PERMANENT,
            1,
        ),
        (
            PublicDataError("invalid ticker symbol 'APPL'"),
            FailureCategory.INVALID_INPUT,
            1,
        ),
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
    text = _guarded_failure(
        PublicDataError("x returned HTTP 429", status=429, attempts=3, retry_after=2.0)
    )
    assert text.failure.retry_after == 2.0


def test_guarded_success_is_plain_text():
    @guarded
    async def tool():
        return {"ok": 1}

    text = asyncio.run(tool())
    assert not isinstance(text, ToolErrorText) and json.loads(text) == {"ok": 1}


def test_public_data_error_message_constructor_still_works():
    exc = PublicDataError("x returned HTTP 500")
    assert (
        str(exc) == "x returned HTTP 500" and exc.status is None and exc.attempts == 1
    )


def test_mcp_is_error_is_an_unknown_failure_with_the_same_text():
    result = SimpleNamespace(
        content=[SimpleNamespace(text="quota exceeded")], isError=True
    )
    text = _result_text(result)
    assert text == "Error: quota exceeded"
    assert text.failure.category is FailureCategory.UNKNOWN


def test_mcp_success_is_plain_text():
    result = SimpleNamespace(content=[SimpleNamespace(text="fine")], isError=False)
    assert not isinstance(_result_text(result), ToolErrorText)
