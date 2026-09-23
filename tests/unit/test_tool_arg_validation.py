"""Tool-argument validation in ToolAgentLoop._call_tool."""

from __future__ import annotations

import pytest

from src.agents.core.state import TaskStatus
from src.internal.tools import FunctionTool, ToolEffect
from src.internal.tools.validation import validate_arguments
from tests.unit.test_tool_approval import _loop, _trace

_INT_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
}


def test_validate_arguments_missing_and_wrong_type():
    assert validate_arguments(_INT_SCHEMA, {}) == ["Missing required argument: 'value'"]
    assert validate_arguments(_INT_SCHEMA, {"value": "x"}) != []
    assert validate_arguments(_INT_SCHEMA, {"value": 3}) == []
    assert validate_arguments({}, {"anything": 1}) == []  # schemaless → no errors


@pytest.mark.asyncio
async def test_missing_required_argument_not_executed():
    executions = []

    @FunctionTool.from_fn(effect=ToolEffect.READ_ONLY, parameters=_INT_SCHEMA)
    def needs_int(value: int):
        executions.append(value)
        return value

    loop, _ = _loop([needs_int], ['{"name":"needs_int","arguments":{}}', "done"])
    output = await loop.run([{"role": "user", "content": "go"}], {})
    result = _trace(output)[0]
    assert result["status"] == str(TaskStatus.FAILED)
    assert result["error_code"] == "invalid_arguments"
    assert executions == []  # the tool body never ran


@pytest.mark.asyncio
async def test_wrong_type_argument_not_executed():
    executions = []

    @FunctionTool.from_fn(effect=ToolEffect.READ_ONLY, parameters=_INT_SCHEMA)
    def needs_int(value: int):
        executions.append(value)
        return value

    loop, _ = _loop(
        [needs_int], ['{"name":"needs_int","arguments":{"value":"x"}}', "done"]
    )
    output = await loop.run([{"role": "user", "content": "go"}], {})
    result = _trace(output)[0]
    assert result["error_code"] == "invalid_arguments"
    assert executions == []


@pytest.mark.asyncio
async def test_unknown_tool_reports_not_found():
    from src.agents import ApprovalDecision

    @FunctionTool.from_fn(effect=ToolEffect.READ_ONLY, parameters=_INT_SCHEMA)
    def needs_int(value: int):
        return value

    async def approve(_request):
        return ApprovalDecision.APPROVE

    # Model calls a tool not registered in the loop; force it past approval so it
    # reaches _call_tool, where the registry reports it as not found.
    loop, _ = _loop([needs_int], ['{"name":"ghost","arguments":{}}', "done"])
    output = await loop.run(
        [{"role": "user", "content": "go"}], {}, on_approval=approve
    )
    result = _trace(output)[0]
    assert result["status"] == str(TaskStatus.FAILED)
    assert result["error_code"] == "tool_not_found"


_BOUNDED = {
    "type": "object",
    "properties": {
        "max_length": {"type": "integer", "minimum": 1, "maximum": 50000},
        "domain": {"type": "string", "enum": ["general", "finance"]},
        "queries": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}


def test_range_is_enforced():
    assert validate_arguments(_BOUNDED, {"max_length": 0}) == [
        "Argument 'max_length': 0 is less than the minimum of 1"
    ]


def test_enum_is_enforced():
    (error,) = validate_arguments(_BOUNDED, {"domain": "bogus"})
    assert error.startswith("Argument 'domain': 'bogus' is not one of")


def test_array_items_are_enforced_with_nested_path():
    (error,) = validate_arguments(_BOUNDED, {"queries": ["ok", 2]})
    assert error == "Argument 'queries.1': 2 is not of type 'string'"


def test_unknown_key_is_rejected_when_schema_is_closed():
    (error,) = validate_arguments(_BOUNDED, {"limti": 5})
    assert error.startswith("Arguments: Additional properties are not allowed")
    assert "'limti'" in error


def test_errors_are_sorted():
    errors = validate_arguments(_BOUNDED, {"max_length": 0, "domain": "bogus"})
    assert [e.split(":")[0] for e in errors] == [
        "Argument 'domain'",
        "Argument 'max_length'",
    ]


def test_nested_required_uses_the_same_message():
    schema = {
        "type": "object",
        "properties": {
            "item": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        },
    }
    assert validate_arguments(schema, {"item": {}}) == [
        "Missing required argument: 'item.query'"
    ]


def test_schema_without_root_type_still_enforced():
    params = {"required": ["a"], "properties": {"a": {"type": "integer", "minimum": 1}}}
    assert validate_arguments(params, {}) == ["Missing required argument: 'a'"]
    assert validate_arguments(params, {"a": 0}) != []


def test_integer_semantics():
    assert validate_arguments(_INT_SCHEMA, {"value": 3.0}) == []
    assert validate_arguments(_INT_SCHEMA, {"value": True}) != []


def test_malformed_schema_falls_back_to_shallow_check(caplog):
    params = {"type": 5, "properties": {"a": {"type": "string"}}, "required": ["a"]}
    assert validate_arguments(params, {}) == ["Missing required argument: 'a'"]
    assert "falling back" in caplog.text


def test_unresolvable_ref_falls_back():
    params = {
        "type": "object",
        "properties": {"a": {"$ref": "#/components/schemas/Missing"}},
        "required": ["a"],
    }
    assert validate_arguments(params, {"a": 1}) == []
    assert validate_arguments(params, {}) == ["Missing required argument: 'a'"]


@pytest.mark.asyncio
async def test_registry_invoke_rejects_out_of_range_without_executing():
    from src.internal.tools.registry import ToolRegistry

    executions = []

    @FunctionTool.from_fn(effect=ToolEffect.READ_ONLY, parameters=_BOUNDED)
    def bounded(max_length: int = 10):
        executions.append(max_length)
        return max_length

    registry = ToolRegistry()
    registry.register(bounded)
    _, _, errors = await registry.invoke("bounded", {"max_length": 0})
    assert errors and executions == []
