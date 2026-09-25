import asyncio

import aiohttp
import pytest

from src.internal.tools import (
    FailureCategory,
    FunctionTool,
    InvalidToolInput,
    ResultKind,
    ToolEffect,
    ToolRegistry,
)
from src.internal.tools.registry import _failure_from_exception, validate_tool_contract


def _tool(**kwargs):
    return FunctionTool(lambda: "ok", name="t", description="t", **kwargs)


def test_a_fully_declared_tool_has_no_violations():
    tool = _tool(effect=ToolEffect.READ_ONLY, result_kind=ResultKind.JSON)
    assert validate_tool_contract(tool, source="function") == []


@pytest.mark.parametrize(
    ("kwargs", "source", "fragment"),
    [
        ({"effect": ToolEffect.READ_ONLY}, "function", "result_kind"),
        ({"result_kind": ResultKind.JSON}, "function", "effect"),
        (
            {
                "effect": ToolEffect.READ_ONLY,
                "result_kind": ResultKind.TEXT,
                "citeable": True,
            },
            "function",
            "citeable",
        ),
    ],
)
def test_each_rule_reports_a_violation(kwargs, source, fragment):
    problems = validate_tool_contract(_tool(**kwargs), source=source)
    assert problems and any(fragment in p for p in problems)


def test_unspecified_is_allowed_only_for_mcp():
    tool = _tool(result_kind=ResultKind.TEXT)
    assert validate_tool_contract(tool, source="mcp") == []
    assert validate_tool_contract(tool, source="function")


def test_strict_registry_rejects_and_names_the_tool():
    with pytest.raises(ValueError, match=r"tool t: .*result_kind"):
        ToolRegistry(strict=True).register(_tool(effect=ToolEffect.READ_ONLY))


def test_default_registry_still_accepts_undeclared_tools():
    registry = ToolRegistry()
    registry.register(_tool())
    assert registry.get("t") is not None


def test_decorator_forwards_the_declarations():
    registry = ToolRegistry(strict=True)

    @registry.tool(
        effect=ToolEffect.READ_ONLY,
        result_kind=ResultKind.JSON,
        retries_internally=True,
    )
    def double(n: int) -> int:
        return n * 2

    tool = registry.get("double")
    assert (tool.result_kind, tool.retries_internally) == (ResultKind.JSON, True)


def test_summaries_expose_the_declarations():
    registry = ToolRegistry()
    registry.register(
        _tool(
            effect=ToolEffect.READ_ONLY, result_kind=ResultKind.DOCUMENTS, citeable=True
        )
    )
    [summary] = registry.all_summaries()
    assert {
        k: summary[k]
        for k in ("effect", "result_kind", "citeable", "retries_internally")
    } == {
        "effect": "read_only",
        "result_kind": "documents",
        "citeable": True,
        "retries_internally": False,
    }
    assert registry.tool_summary("t")["result_kind"] == "documents"


def _response_error(status):
    return aiohttp.ClientResponseError(request_info=None, history=(), status=status)


@pytest.mark.parametrize(
    ("exc", "category"),
    [
        (InvalidToolInput("bad ticker"), FailureCategory.INVALID_INPUT),
        (_response_error(400), FailureCategory.INVALID_INPUT),
        (_response_error(404), FailureCategory.INVALID_INPUT),
        (_response_error(422), FailureCategory.INVALID_INPUT),
        (_response_error(401), FailureCategory.PERMANENT),
        (_response_error(403), FailureCategory.PERMANENT),
        (_response_error(418), FailureCategory.PERMANENT),
        (_response_error(429), FailureCategory.TRANSIENT),
        (_response_error(503), FailureCategory.TRANSIENT),
        (aiohttp.ClientConnectionError(), FailureCategory.TRANSIENT),
        (asyncio.TimeoutError(), FailureCategory.TRANSIENT),
        (ConnectionError(), FailureCategory.TRANSIENT),
        (KeyError("x"), FailureCategory.UNKNOWN),
    ],
)
def test_exception_classification(exc, category):
    assert _failure_from_exception(exc).category is category


def test_invalid_tool_input_keeps_its_message_and_is_a_value_error():
    exc = InvalidToolInput("max_results must be an integer")
    assert isinstance(exc, ValueError)
    assert _failure_from_exception(exc).message == "max_results must be an integer"


def test_httpx_transport_errors_are_transient():
    httpx = pytest.importorskip("httpx")
    assert (
        _failure_from_exception(httpx.ConnectError("down")).category
        is FailureCategory.TRANSIENT
    )
